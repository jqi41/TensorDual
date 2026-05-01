#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classical QAOA, TensorHyper-QAOA, and TensorDual-QAOA on MaxCut
with multi-channel noise, ZNE, and REM.

Revised version:
- QAOA depth is generalized to p >= 1
- Default depth is p = 4
- All models output 2p angles:
    [gamma_1, ..., gamma_p, beta_1, ..., beta_p]
- The classical QAOA baseline is now a true directly optimized QAOA model
  evaluated under the SAME noise / ZNE / REM conditions as the tensorized models.

Noise channels:
  - Single-qubit: depolarizing (depol), dephasing (dephase), Pauli X/Y/Z
  - Two-qubit Pauli on edges: p_twopauli ∈ {XX, YY, ZZ}
  - Mixer over-rotation: overrot_sigma (Gaussian on beta)
  - Readout error: p_readout (symmetric bit-flip model)

Mitigations:
  - ZNE: odd scales S ∈ {1,3,5} with folding on both cost phase and mixer
  - REM: analytic inverse correction on <Z_i Z_j>

Models:
  - Classical QAOA (direct angle optimization)
  - TensorHyper-QAOA
  - TensorDual-QAOA
"""

import math
import argparse
from typing import Sequence

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize


# =========================
# 0) Utility helpers
# =========================
def parse_int_list(x: str) -> list[int]:
    x = str(x).strip()
    if x.startswith("[") and x.endswith("]"):
        x = x[1:-1]
    if not x:
        return []
    return [int(t.strip()) for t in x.split(",") if t.strip()]


def make_uniform_tt_ranks(num_modes: int, rank: int) -> list[int]:
    if num_modes < 1:
        raise ValueError("num_modes must be >= 1")
    return [1] + [rank] * (num_modes - 1) + [1]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# =========================
# 1) Tensor-Train modules
# =========================
class TensorTrainLayer(nn.Module):
    def __init__(self, input_dims, output_dims, tt_ranks):
        super().__init__()
        assert len(input_dims) == len(output_dims), "input_dims and output_dims must have same length"
        assert len(tt_ranks) == len(input_dims) + 1, "tt_ranks length must be len(input_dims)+1"

        self.input_dims = list(input_dims)
        self.output_dims = list(output_dims)
        self.tt_ranks = list(tt_ranks)
        self.input_dim = int(np.prod(self.input_dims))
        self.output_dim = int(np.prod(self.output_dims))

        self.tt_cores = nn.ParameterList()
        for k in range(len(self.input_dims)):
            r0, r1 = self.tt_ranks[k], self.tt_ranks[k + 1]
            n_k, m_k = self.input_dims[k], self.output_dims[k]
            core = nn.Parameter(torch.randn(r0, n_k, m_k, r1) * 0.1)
            self.tt_cores.append(core)

        self.bias = nn.Parameter(torch.zeros(self.output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        if x.size(1) != self.input_dim:
            raise ValueError(f"Expected input dim {self.input_dim}, got {x.size(1)}")

        x_rs = x.view(bsz, *self.input_dims)

        batch = 'b'
        letters = [chr(i) for i in range(ord('a'), ord('z') + 1) if chr(i) != batch]
        d = len(self.input_dims)
        iL = letters[:d]
        oL = letters[d: 2 * d]
        rL = letters[2 * d: 2 * d + d + 1]

        inp = batch + ''.join(iL)
        cores = [f"{rL[k]}{iL[k]}{oL[k]}{rL[k+1]}" for k in range(d)]
        outp = batch + ''.join(oL)
        eins = inp + ',' + ','.join(cores) + '->' + outp

        out = torch.einsum(eins, x_rs, *self.tt_cores)
        return out.reshape(bsz, -1) + self.bias


class MetaTTQAOA(nn.Module):
    """
    TensorHyper-QAOA: TT directly maps graph features -> 2p angles
    Output ordering:
      [gamma_1, ..., gamma_p, beta_1, ..., beta_p]
    """
    def __init__(self, input_dims, output_dims, tt_ranks):
        super().__init__()
        self.tt = TensorTrainLayer(input_dims, output_dims, tt_ranks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.tt(x)
        return raw.view(-1)


class TensorDualQAOA(nn.Module):
    """
    TensorDual-QAOA:
      shared TT backbone -> two outputs
        (1) direct angle head -> base 2p angles
        (2) feature head -> TT hypernetwork -> residual 2p angles

      final angles = base + residual
      ordering = [gamma_1,...,gamma_p,beta_1,...,beta_p]
    """
    def __init__(
        self,
        input_dims=(2, 5),
        hidden_dims=(2, 4),
        feature_dims=(2, 4),
        angle_dims=(2, 4),   # product=8 for p=4
        tt_ranks_backbone=(1, 4, 1),
        tt_ranks_feature=(1, 4, 1),
        tt_ranks_angle=(1, 4, 1),
        tt_ranks_hyper=(1, 4, 1),
    ):
        super().__init__()

        assert len(input_dims) == len(hidden_dims) == len(feature_dims) == len(angle_dims), \
            "All TensorDual mode lists must have same number of modes."

        self.backbone = TensorTrainLayer(input_dims, hidden_dims, tt_ranks_backbone)
        self.feature_head = TensorTrainLayer(hidden_dims, feature_dims, tt_ranks_feature)
        self.angle_head = TensorTrainLayer(hidden_dims, angle_dims, tt_ranks_angle)
        self.hyper = TensorTrainLayer(feature_dims, angle_dims, tt_ranks_hyper)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.tanh(self.backbone(x))
        h = torch.tanh(self.feature_head(z))
        base = self.angle_head(z)
        residual = self.hyper(h)
        raw = base + residual
        return raw.view(-1)


# ==============================
# 2) Graph → Feature conversion
# ==============================
def graph_to_features(graph: nx.Graph, hist_bins: int = 10) -> np.ndarray:
    deg_list = [d for _, d in graph.degree()]
    hist, _ = np.histogram(deg_list, bins=hist_bins, range=(0, hist_bins))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-8)
    return hist


def get_maxcut_edges(graph: nx.Graph):
    return [(i, j) for i, j in graph.edges()]


# =========================================
# 3) Gradient-safe statevector primitives
# =========================================
def _pair_indices(q: int, dim: int, device):
    idx = torch.arange(dim, device=device)
    mask0 = ((idx >> q) & 1) == 0
    idx0 = idx[mask0]
    idx1 = idx0 | (1 << q)
    return idx0, idx1


def apply_rx_layer(state: torch.Tensor, beta: torch.Tensor, n: int) -> torch.Tensor:
    dim = state.numel()
    device = state.device
    c = torch.cos(2.0 * beta)
    s = torch.sin(2.0 * beta)
    for q in range(n):
        idx0, idx1 = _pair_indices(q, dim, device)
        a = state.index_select(0, idx0)
        b = state.index_select(0, idx1)
        a_new = c * a + (-1j * s) * b
        b_new = (-1j * s) * a + c * b
        state = state.clone()
        state = state.scatter(0, idx0, a_new)
        state = state.scatter(0, idx1, b_new)
    return state


def apply_X(state: torch.Tensor, q: int, n: int) -> torch.Tensor:
    dim = state.numel()
    device = state.device
    idx0, idx1 = _pair_indices(q, dim, device)
    a = state.index_select(0, idx0)
    b = state.index_select(0, idx1)
    new_state = state.clone()
    new_state = new_state.scatter(0, idx0, b)
    new_state = new_state.scatter(0, idx1, a)
    return new_state


def apply_Z(state: torch.Tensor, q: int, n: int) -> torch.Tensor:
    dim = state.numel()
    device = state.device
    idx = torch.arange(dim, device=device)
    mask1 = ((idx >> q) & 1) == 1
    new_state = state.clone()
    new_state = new_state.scatter(0, idx[mask1], -state.index_select(0, idx[mask1]))
    return new_state


def apply_Y(state: torch.Tensor, q: int, n: int) -> torch.Tensor:
    dim = state.numel()
    device = state.device
    idx0, idx1 = _pair_indices(q, dim, device)
    a = state.index_select(0, idx0)
    b = state.index_select(0, idx1)
    a_new = 1j * b
    b_new = -1j * a
    new_state = state.clone()
    new_state = new_state.scatter(0, idx0, a_new)
    new_state = new_state.scatter(0, idx1, b_new)
    return new_state


def apply_two_qubit_pauli(state: torch.Tensor, i: int, j: int, which: str, n: int) -> torch.Tensor:
    if which == "XX":
        state = apply_X(state, i, n)
        state = apply_X(state, j, n)
    elif which == "YY":
        state = apply_Y(state, i, n)
        state = apply_Y(state, j, n)
    else:
        state = apply_Z(state, i, n)
        state = apply_Z(state, j, n)
    return state


# ===========================================
# 4) ZNE folding helpers and extrapolation
# ===========================================
def folding_pattern(scale: int) -> list[int]:
    assert scale % 2 == 1 and scale >= 1
    patt = []
    sign = +1
    for _ in range(scale):
        patt.append(sign)
        sign *= -1
    return patt


def richardson_extrapolate(scales, values, order: str = "linear"):
    s = torch.tensor(scales, dtype=torch.float64)
    v = torch.stack(values).to(dtype=torch.float64)
    if order == "linear" or len(scales) == 2:
        A = torch.stack([torch.ones_like(s), s], dim=1)
        sol = torch.linalg.lstsq(A, v).solution
        a = sol[0]
        return a.to(values[0].dtype)
    else:
        A = torch.stack([torch.ones_like(s), s, s**2], dim=1)
        sol = torch.linalg.lstsq(A, v).solution
        a = sol[0]
        return a.to(values[0].dtype)


# ===========================================================
# 5) QAOA expectation with depth p
# ===========================================================
def exact_qaoa_expectation_one_scale(
    gammas: torch.Tensor,
    betas: torch.Tensor,
    edges,
    n_qubits: int,
    noise_cfg: dict,
    mc_shots: int,
    scale: int,
    rem: bool,
    Cz_precomp: torch.Tensor = None,
    spin_matrix: torch.Tensor = None,
) -> torch.Tensor:
    assert scale % 2 == 1 and scale >= 1
    device = gammas.device
    n = n_qubits
    dim = 1 << n
    p = gammas.numel()

    depol = float(noise_cfg.get('depol', 0.0))
    deph = float(noise_cfg.get('dephase', 0.0))
    px = float(noise_cfg.get('pauli_px', 0.0))
    py = float(noise_cfg.get('pauli_py', 0.0))
    pz = float(noise_cfg.get('pauli_pz', 0.0))
    sig = float(noise_cfg.get('overrot_sigma', 0.0))
    p2 = float(noise_cfg.get('p_twopauli', 0.0))
    rerr = float(noise_cfg.get('p_readout', 0.0))

    if spin_matrix is None:
        idx = torch.arange(dim, device=device).unsqueeze(1)
        qidx = torch.arange(n, device=device).unsqueeze(0)
        bit_matrix = ((idx >> qidx) & 1).float()
        spin_matrix = 1.0 - 2.0 * bit_matrix

    if Cz_precomp is None:
        Cz = torch.zeros(dim, dtype=torch.float32, device=device)
        for (i, j) in edges:
            Cz += 0.5 * (1.0 - spin_matrix[:, i] * spin_matrix[:, j])
        Cz_precomp = Cz.to(gammas.dtype)

    patt = folding_pattern(scale)
    acc = torch.zeros((), dtype=torch.float32, device=device)

    for _ in range(mc_shots):
        state = torch.ones(dim, dtype=torch.complex64, device=device) / math.sqrt(dim)

        for layer in range(p):
            gamma_l = gammas[layer]
            beta_l = betas[layer]

            for sgn in patt:
                state = state * torch.exp(-1j * (sgn * gamma_l).view(1) * Cz_precomp)

                beta_eff = (sgn * beta_l) + (torch.randn((), device=device) * sig if sig > 0 else 0.0)
                state = apply_rx_layer(state, beta_eff, n)

                for q in range(n):
                    if torch.rand((), device=device) < px:
                        state = apply_X(state, q, n)
                    if torch.rand((), device=device) < py:
                        state = apply_Y(state, q, n)
                    if torch.rand((), device=device) < pz:
                        state = apply_Z(state, q, n)
                    if torch.rand((), device=device) < depol:
                        k = torch.randint(0, 3, (), device=device)
                        state = apply_X(state, q, n) if k == 0 else (apply_Y(state, q, n) if k == 1 else apply_Z(state, q, n))
                    if torch.rand((), device=device) < deph:
                        state = apply_Z(state, q, n)

                if p2 > 0.0:
                    for (i, j) in edges:
                        if torch.rand((), device=device) < p2:
                            which = ["XX", "YY", "ZZ"][torch.randint(0, 3, (), device=device).item()]
                            state = apply_two_qubit_pauli(state, i, j, which, n)

        probs = (state.abs() ** 2).real

        base_scale = (1.0 - 2.0 * rerr) ** 2
        use_scale = 1.0 if rem else base_scale
        inv_scale = (1.0 / max(base_scale, 1e-8)) if rem else 1.0

        exp_HC = torch.zeros((), dtype=torch.float32, device=device)
        for (i, j) in edges:
            zz = (spin_matrix[:, i] * spin_matrix[:, j]).to(probs.dtype)
            true_corr = torch.sum(probs * zz)
            corr_used = torch.clamp(true_corr * use_scale * inv_scale, -1.0, 1.0)
            exp_HC += 0.5 * (1.0 - corr_used)

        acc = acc + exp_HC

    return acc / mc_shots


def exact_qaoa_expectation(
    gammas: torch.Tensor,
    betas: torch.Tensor,
    edges,
    n_qubits: int,
    noise_cfg=None,
    mc_shots: int = 8,
    use_zne: bool = False,
    zne_scales=(1, 3, 5),
    zne_order: str = "linear",
    use_rem: bool = False,
) -> torch.Tensor:
    if noise_cfg is None:
        noise_cfg = dict(
            depol=0.0, dephase=0.0,
            pauli_px=0.0, pauli_py=0.0, pauli_pz=0.0,
            overrot_sigma=0.0, p_twopauli=0.0, p_readout=0.0
        )

    device = gammas.device
    n = n_qubits
    dim = 1 << n

    idx = torch.arange(dim, device=device).unsqueeze(1)
    qidx = torch.arange(n, device=device).unsqueeze(0)
    bit_matrix = ((idx >> qidx) & 1).float()
    spin_matrix = 1.0 - 2.0 * bit_matrix
    Cz = torch.zeros(dim, dtype=torch.float32, device=device)
    for (i, j) in edges:
        Cz += 0.5 * (1.0 - spin_matrix[:, i] * spin_matrix[:, j])
    Cz = Cz.to(gammas.dtype)

    if not use_zne:
        return exact_qaoa_expectation_one_scale(
            gammas, betas, edges, n_qubits, noise_cfg, mc_shots, scale=1, rem=use_rem,
            Cz_precomp=Cz, spin_matrix=spin_matrix
        )

    scales = list(zne_scales)
    vals = []
    for s in scales:
        v = exact_qaoa_expectation_one_scale(
            gammas, betas, edges, n_qubits, noise_cfg, mc_shots, scale=s, rem=use_rem,
            Cz_precomp=Cz, spin_matrix=spin_matrix
        )
        vals.append(v)
    return richardson_extrapolate(scales, vals, order=zne_order)


# ==============================================
# 6) Classical QAOA baseline with same noise
# ==============================================
def classical_qaoa_maxcut(
    graph: nx.Graph,
    p: int = 4,
    noise_cfg: dict | None = None,
    mc_shots: int = 8,
    use_zne: bool = False,
    zne_scales=(1, 3, 5),
    zne_order: str = "linear",
    use_rem: bool = False,
    n_restarts: int = 5,
    method: str = "COBYLA",
    seed: int = 1234,
):
    """
    Directly optimize 2p QAOA angles under the SAME noisy expectation
    used by TensorHyper-QAOA and TensorDual-QAOA.
    """
    n = graph.number_of_nodes()
    edges = get_maxcut_edges(graph)
    cfg = DEFAULT_NOISE if noise_cfg is None else noise_cfg

    rng = np.random.default_rng(seed)
    best_val = -float("inf")
    best_params = None

    def objective(params_np):
        params = torch.tensor(params_np, dtype=torch.float32)
        gammas = params[:p]
        betas = params[p:]
        val = exact_qaoa_expectation(
            gammas, betas, edges, n, cfg, mc_shots,
            use_zne=use_zne,
            zne_scales=zne_scales,
            zne_order=zne_order,
            use_rem=use_rem,
        )
        return -float(val.item())

    for r in range(n_restarts):
        x0 = rng.uniform(low=-np.pi, high=np.pi, size=2 * p)
        res = minimize(objective, x0, method=method)
        cur_val = -objective(res.x)
        if cur_val > best_val:
            best_val = cur_val
            best_params = res.x.copy()

    return float(best_val), best_params


# ==================================
# 7) Training / experiment routines
# ==================================
DEFAULT_NOISE = {
    'depol': 1e-3,
    'dephase': 1e-3,
    'pauli_px': 0.0,
    'pauli_py': 0.0,
    'pauli_pz': 0.0,
    'overrot_sigma': 0.0,
    'p_twopauli': 1e-3,
    'p_readout': 1e-2
}


def prepare_graph_feature_tensor(
    graph: nx.Graph,
    hist_bins: int,
    tt_input_dims: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    feat_np = graph_to_features(graph, hist_bins=hist_bins)
    prod_in = int(np.prod(tt_input_dims))

    if len(feat_np) < prod_in:
        pad = np.zeros(prod_in, dtype=np.float32)
        pad[:len(feat_np)] = feat_np
        feat_np = pad
    elif len(feat_np) > prod_in:
        feat_np = feat_np[:prod_in]

    return torch.tensor(feat_np, dtype=torch.float32, device=device).unsqueeze(0)


def split_angles(raw_angles: torch.Tensor, p: int):
    if raw_angles.numel() != 2 * p:
        raise ValueError(f"Expected {2*p} angles, got {raw_angles.numel()}")
    gammas = raw_angles[:p]
    betas = raw_angles[p:]
    return gammas, betas


def train_metatt_on_graph(
    graph: nx.Graph,
    p: int = 4,
    epochs: int = 100,
    lr: float = 0.02,
    hist_bins: int = 10,
    noise_cfg: dict | None = None,
    mc_shots: int = 8,
    tt_input_dims=(2, 5),
    tt_output_dims=(2, 4),
    tt_ranks=(1, 4, 1),
    use_zne: bool = False,
    zne_scales=(1, 3, 5),
    zne_order: str = "linear",
    use_rem: bool = False,
    verbose: bool = True
):
    if int(np.prod(tt_output_dims)) != 2 * p:
        raise ValueError(f"TensorHyper output dim product must equal 2p={2*p}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = graph.number_of_nodes()
    edges = get_maxcut_edges(graph)

    feat = prepare_graph_feature_tensor(graph, hist_bins, tt_input_dims, device)

    model = MetaTTQAOA(
        input_dims=list(tt_input_dims),
        output_dims=list(tt_output_dims),
        tt_ranks=list(tt_ranks)
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    cfg = DEFAULT_NOISE if noise_cfg is None else noise_cfg

    if verbose:
        print(f"  [TensorHyper] params={count_parameters(model)} dims={tt_input_dims}->{tt_output_dims} ranks={tt_ranks}")

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        raw = model(feat)
        gammas, betas = split_angles(raw, p)

        exp_hc = exact_qaoa_expectation(
            gammas, betas, edges, n, cfg, mc_shots,
            use_zne=use_zne, zne_scales=zne_scales, zne_order=zne_order, use_rem=use_rem
        )
        loss = -exp_hc
        loss.backward()
        optimizer.step()

        if verbose and (epoch % max(1, (epochs // 5)) == 0):
            print(f"  [TensorHyper] Epoch {epoch:03d}  ⟨H_C⟩={exp_hc.item():.4f}")

    with torch.no_grad():
        raw = model(feat)
        gammas, betas = split_angles(raw, p)
        final_exp = exact_qaoa_expectation(
            gammas, betas, edges, n, cfg, mc_shots,
            use_zne=use_zne, zne_scales=zne_scales, zne_order=zne_order, use_rem=use_rem
        ).item()

    return gammas.detach().cpu().numpy(), betas.detach().cpu().numpy(), final_exp


def train_tensordual_on_graph(
    graph: nx.Graph,
    p: int = 4,
    epochs: int = 100,
    lr: float = 0.02,
    hist_bins: int = 10,
    noise_cfg: dict | None = None,
    mc_shots: int = 8,
    tt_input_dims=(2, 5),
    hidden_dims=(2, 4),
    feature_dims=(2, 4),
    angle_dims=(2, 4),
    tt_ranks_backbone=(1, 4, 1),
    tt_ranks_feature=(1, 4, 1),
    tt_ranks_angle=(1, 4, 1),
    tt_ranks_hyper=(1, 4, 1),
    use_zne: bool = False,
    zne_scales=(1, 3, 5),
    zne_order: str = "linear",
    use_rem: bool = False,
    verbose: bool = True
):
    if int(np.prod(angle_dims)) != 2 * p:
        raise ValueError(f"TensorDual angle dim product must equal 2p={2*p}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = graph.number_of_nodes()
    edges = get_maxcut_edges(graph)

    feat = prepare_graph_feature_tensor(graph, hist_bins, tt_input_dims, device)

    model = TensorDualQAOA(
        input_dims=list(tt_input_dims),
        hidden_dims=list(hidden_dims),
        feature_dims=list(feature_dims),
        angle_dims=list(angle_dims),
        tt_ranks_backbone=list(tt_ranks_backbone),
        tt_ranks_feature=list(tt_ranks_feature),
        tt_ranks_angle=list(tt_ranks_angle),
        tt_ranks_hyper=list(tt_ranks_hyper),
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    cfg = DEFAULT_NOISE if noise_cfg is None else noise_cfg

    if verbose:
        print(
            f"  [TensorDual ] params={count_parameters(model)} "
            f"in={tt_input_dims} hid={hidden_dims} feat={feature_dims} ang={angle_dims} "
            f"r_backbone={tt_ranks_backbone} r_feat={tt_ranks_feature} "
            f"r_ang={tt_ranks_angle} r_hyp={tt_ranks_hyper}"
        )

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        raw = model(feat)
        gammas, betas = split_angles(raw, p)

        exp_hc = exact_qaoa_expectation(
            gammas, betas, edges, n, cfg, mc_shots,
            use_zne=use_zne, zne_scales=zne_scales, zne_order=zne_order, use_rem=use_rem
        )
        loss = -exp_hc
        loss.backward()
        optimizer.step()

        if verbose and (epoch % max(1, (epochs // 5)) == 0):
            print(f"  [TensorDual ] Epoch {epoch:03d}  ⟨H_C⟩={exp_hc.item():.4f}")

    with torch.no_grad():
        raw = model(feat)
        gammas, betas = split_angles(raw, p)
        final_exp = exact_qaoa_expectation(
            gammas, betas, edges, n, cfg, mc_shots,
            use_zne=use_zne, zne_scales=zne_scales, zne_order=zne_order, use_rem=use_rem
        ).item()

    return gammas.detach().cpu().numpy(), betas.detach().cpu().numpy(), final_exp


def compare_across_graphs(
    n_graphs: int = 5,
    n_nodes: int = 12,
    p_edge: float = 0.5,
    p_qaoa: int = 4,
    noise_cfg: dict | None = None,
    mc_shots: int = 8,
    epochs: int = 100,
    lr: float = 0.02,
    use_zne: bool = False,
    zne_scales=(1, 3, 5),
    zne_order: str = "linear",
    use_rem: bool = False,
    seed: int = 1234,
    classical_restarts: int = 5,
    # TensorHyper configs
    th_input_dims=(2, 5),
    th_output_dims=(2, 4),
    th_tt_ranks=(1, 4, 1),
    # TensorDual configs
    td_input_dims=(2, 5),
    td_hidden_dims=(2, 4),
    td_feature_dims=(2, 4),
    td_angle_dims=(2, 4),
    td_ranks_backbone=(1, 4, 1),
    td_ranks_feature=(1, 4, 1),
    td_ranks_angle=(1, 4, 1),
    td_ranks_hyper=(1, 4, 1),
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    classical_results, tensorhyper_results, tensordual_results = [], [], []
    cfg = DEFAULT_NOISE if noise_cfg is None else noise_cfg

    print("\n===== Experiment Configuration =====")
    print(f"Graphs               : {n_graphs}")
    print(f"Nodes per graph      : {n_nodes}")
    print(f"Edge probability     : {p_edge}")
    print(f"QAOA depth p         : {p_qaoa}")
    print(f"Epochs               : {epochs}")
    print(f"Learning rate        : {lr}")
    print(f"MC shots             : {mc_shots}")
    print(f"Use ZNE              : {use_zne}")
    print(f"ZNE scales           : {zne_scales}")
    print(f"ZNE order            : {zne_order}")
    print(f"Use REM              : {use_rem}")
    print(f"Noise cfg            : {cfg}")
    print(f"Classical restarts   : {classical_restarts}")
    print(f"TensorHyper dims     : in={th_input_dims}, out={th_output_dims}, ranks={th_tt_ranks}")
    print(
        f"TensorDual dims      : in={td_input_dims}, hid={td_hidden_dims}, "
        f"feat={td_feature_dims}, ang={td_angle_dims}"
    )
    print(
        f"TensorDual ranks     : backbone={td_ranks_backbone}, feature={td_ranks_feature}, "
        f"angle={td_ranks_angle}, hyper={td_ranks_hyper}"
    )

    for idx in range(1, n_graphs + 1):
        G = nx.erdos_renyi_graph(n=n_nodes, p=p_edge, seed=seed + idx)
        if not nx.is_connected(G):
            comp = max(nx.connected_components(G), key=len)
            G = G.subgraph(comp).copy()

        cut_classical, classical_params = classical_qaoa_maxcut(
            G,
            p=p_qaoa,
            noise_cfg=cfg,
            mc_shots=mc_shots,
            use_zne=use_zne,
            zne_scales=zne_scales,
            zne_order=zne_order,
            use_rem=use_rem,
            n_restarts=classical_restarts,
            seed=seed + idx,
        )
        classical_results.append(cut_classical)
        classical_gammas = np.round(classical_params[:p_qaoa], 3)
        classical_betas = np.round(classical_params[p_qaoa:], 3)
        print(f"\nGraph {idx:02d} – Classical QAOA ⇒ gammas={classical_gammas}, betas={classical_betas}, ⟨H_C⟩≈{cut_classical:>5.3f}")

        gammas_tt, betas_tt, exp_tt = train_metatt_on_graph(
            G,
            p=p_qaoa,
            epochs=epochs,
            lr=lr,
            noise_cfg=cfg,
            mc_shots=mc_shots,
            tt_input_dims=th_input_dims,
            tt_output_dims=th_output_dims,
            tt_ranks=th_tt_ranks,
            use_zne=use_zne,
            zne_scales=zne_scales,
            zne_order=zne_order,
            use_rem=use_rem,
            verbose=True,
        )
        tensorhyper_results.append(exp_tt)
        print(f"          TensorHyper-QAOA ⇒ gammas={np.round(gammas_tt, 3)}, betas={np.round(betas_tt, 3)}, ⟨H_C⟩≈{exp_tt:.3f}")

        gammas_td, betas_td, exp_td = train_tensordual_on_graph(
            G,
            p=p_qaoa,
            epochs=epochs,
            lr=lr,
            noise_cfg=cfg,
            mc_shots=mc_shots,
            tt_input_dims=td_input_dims,
            hidden_dims=td_hidden_dims,
            feature_dims=td_feature_dims,
            angle_dims=td_angle_dims,
            tt_ranks_backbone=td_ranks_backbone,
            tt_ranks_feature=td_ranks_feature,
            tt_ranks_angle=td_ranks_angle,
            tt_ranks_hyper=td_ranks_hyper,
            use_zne=use_zne,
            zne_scales=zne_scales,
            zne_order=zne_order,
            use_rem=use_rem,
            verbose=True,
        )
        tensordual_results.append(exp_td)
        print(f"          TensorDual-QAOA  ⇒ gammas={np.round(gammas_td, 3)}, betas={np.round(betas_td, 3)}, ⟨H_C⟩≈{exp_td:.3f}")

    print("\n==== Final Averages over all graphs ====")
    print(f"  Avg Classical QAOA          ≈ {float(np.mean(classical_results)):.4f}")
    print(f"  Avg TensorHyper-QAOA ⟨H_C⟩  ≈ {float(np.mean(tensorhyper_results)):.4f}")
    print(f"  Avg TensorDual-QAOA  ⟨H_C⟩  ≈ {float(np.mean(tensordual_results)):.4f}")


# ==========
# 8) CLI
# ==========
def build_argparser():
    p = argparse.ArgumentParser(
        description="Classical QAOA + TensorHyper-QAOA + TensorDual-QAOA with multi-channel noise + ZNE/REM"
    )

    p.add_argument("--n_graphs", type=int, default=10)
    p.add_argument("--n_nodes", type=int, default=14)
    p.add_argument("--p_edge", type=float, default=0.5)
    p.add_argument("--p_qaoa", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--mc_shots", type=int, default=8)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--classical_restarts", type=int, default=5)

    # Noise
    p.add_argument("--depol", type=float, default=0.001)
    p.add_argument("--dephase", type=float, default=0.001)
    p.add_argument("--pauli_px", type=float, default=0.00)
    p.add_argument("--pauli_py", type=float, default=0.00)
    p.add_argument("--pauli_pz", type=float, default=0.00)
    p.add_argument("--overrot_sigma", type=float, default=0.00)
    p.add_argument("--p_twopauli", type=float, default=0.001)
    p.add_argument("--p_readout", type=float, default=0.1)

    # Mitigations
    p.add_argument("--use_zne", action="store_true")
    p.add_argument("--zne_scales", type=int, nargs="+", default=[1, 3, 5])
    p.add_argument("--zne_order", type=str, default="linear", choices=["linear", "quadratic"])
    p.add_argument("--use_rem", action="store_true")

    # TensorHyper configs
    p.add_argument("--th_input_dims", type=str, default="2,5")
    p.add_argument("--th_output_dims", type=str, default="2,4")
    p.add_argument("--th_tt_rank", type=int, default=4)

    # TensorDual configs
    p.add_argument("--td_input_dims", type=str, default="2,5")
    p.add_argument("--td_hidden_dims", type=str, default="2,4")
    p.add_argument("--td_feature_dims", type=str, default="2,4")
    p.add_argument("--td_angle_dims", type=str, default="2,4")

    p.add_argument("--td_rank_backbone", type=int, default=4)
    p.add_argument("--td_rank_feature", type=int, default=4)
    p.add_argument("--td_rank_angle", type=int, default=4)
    p.add_argument("--td_rank_hyper", type=int, default=4)

    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()

    noise_cfg = {
        'depol': args.depol,
        'dephase': args.dephase,
        'pauli_px': args.pauli_px,
        'pauli_py': args.pauli_py,
        'pauli_pz': args.pauli_pz,
        'overrot_sigma': args.overrot_sigma,
        'p_twopauli': args.p_twopauli,
        'p_readout': args.p_readout
    }

    th_input_dims = tuple(parse_int_list(args.th_input_dims))
    th_output_dims = tuple(parse_int_list(args.th_output_dims))
    if len(th_input_dims) != len(th_output_dims):
        raise ValueError("th_input_dims and th_output_dims must have the same number of modes.")
    if int(np.prod(th_output_dims)) != 2 * args.p_qaoa:
        raise ValueError(f"TensorHyper output product must equal 2*p_qaoa = {2 * args.p_qaoa}")
    th_tt_ranks = tuple(make_uniform_tt_ranks(len(th_input_dims), args.th_tt_rank))

    td_input_dims = tuple(parse_int_list(args.td_input_dims))
    td_hidden_dims = tuple(parse_int_list(args.td_hidden_dims))
    td_feature_dims = tuple(parse_int_list(args.td_feature_dims))
    td_angle_dims = tuple(parse_int_list(args.td_angle_dims))

    if not (len(td_input_dims) == len(td_hidden_dims) == len(td_feature_dims) == len(td_angle_dims)):
        raise ValueError("All TensorDual dims must have the same number of modes.")
    if int(np.prod(td_angle_dims)) != 2 * args.p_qaoa:
        raise ValueError(f"TensorDual angle output product must equal 2*p_qaoa = {2 * args.p_qaoa}")

    td_ranks_backbone = tuple(make_uniform_tt_ranks(len(td_input_dims), args.td_rank_backbone))
    td_ranks_feature = tuple(make_uniform_tt_ranks(len(td_hidden_dims), args.td_rank_feature))
    td_ranks_angle = tuple(make_uniform_tt_ranks(len(td_hidden_dims), args.td_rank_angle))
    td_ranks_hyper = tuple(make_uniform_tt_ranks(len(td_feature_dims), args.td_rank_hyper))

    compare_across_graphs(
        n_graphs=args.n_graphs,
        n_nodes=args.n_nodes,
        p_edge=args.p_edge,
        p_qaoa=args.p_qaoa,
        noise_cfg=noise_cfg,
        mc_shots=args.mc_shots,
        epochs=args.epochs,
        lr=args.lr,
        use_zne=args.use_zne,
        zne_scales=tuple(args.zne_scales),
        zne_order=args.zne_order,
        use_rem=args.use_rem,
        seed=args.seed,
        classical_restarts=args.classical_restarts,
        th_input_dims=th_input_dims,
        th_output_dims=th_output_dims,
        th_tt_ranks=th_tt_ranks,
        td_input_dims=td_input_dims,
        td_hidden_dims=td_hidden_dims,
        td_feature_dims=td_feature_dims,
        td_angle_dims=td_angle_dims,
        td_ranks_backbone=td_ranks_backbone,
        td_ranks_feature=td_ranks_feature,
        td_ranks_angle=td_ranks_angle,
        td_ranks_hyper=td_ranks_hyper,
    )
