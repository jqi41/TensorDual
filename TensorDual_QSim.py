#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Classical VQE, TensorHyper-VQC, and TensorDual-VQC for LiH
=========================================================

This script extends the previous LiH TensorHyper-VQC code by adding
a TensorDual operator for variational quantum circuits.

Compared methods
----------------
1. Classical multi-start VQE
2. TensorHyper-VQC
   - TT network generates a residual correction to a classical warm-start vector
3. TensorDual-VQC
   - shared TT backbone
   - one head generates a base parameter vector
   - another head generates latent feature h
   - TT hypernetwork maps h -> residual parameter correction
   - final parameters = base + residual

Dependencies:
  pip install cirq numpy scipy torch openfermion openfermionpyscf pyscf
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import List, Tuple

import cirq
import numpy as np
import scipy.sparse.linalg as spla
from scipy.optimize import minimize
import torch
import torch.nn as nn

from openfermion import MolecularData
from openfermion.transforms import jordan_wigner, freeze_orbitals, get_fermion_operator
from openfermion.linalg import get_sparse_operator
from openfermionpyscf import run_pyscf


# ============================================================
# 1) LiH reduced Hamiltonian
# ============================================================
def get_lih_reduced_hamiltonian(bond_length: float = 1.6):
    geometry = [['Li', (0, 0, 0)], ['H', (0, 0, bond_length)]]
    molecule = MolecularData(
        geometry=geometry,
        basis='sto-3g',
        multiplicity=1,
        charge=0,
    )
    molecule = run_pyscf(molecule, run_scf=True, run_fci=True)

    interaction_op = molecule.get_molecular_hamiltonian()
    fermion_op = get_fermion_operator(interaction_op)

    frozen_fermion = freeze_orbitals(
        fermion_op,
        occupied=[0, 1],
        unoccupied=list(range(6, 12)),
    )

    qubit_ham = jordan_wigner(frozen_fermion)
    H_sparse = get_sparse_operator(qubit_ham)

    dim = H_sparse.shape[0]
    n_qubits = int(np.log2(dim))
    print(f"[INFO] Reduced LiH Hamiltonian -> {n_qubits} qubits, shape {H_sparse.shape}")
    assert H_sparse.shape == (16, 16), f"Expected (16,16), got {H_sparse.shape}"

    return H_sparse


# ============================================================
# 2) Ansatz configuration
# ============================================================
NUM_QUBITS = 4
qubits = cirq.LineQubit.range(NUM_QUBITS)
simulator = cirq.Simulator()


@dataclass
class AnsatzConfig:
    n_layers: int = 4
    hf_bitstring: str = "1100"
    use_reverse_ring: bool = True


def param_count(cfg: AnsatzConfig) -> int:
    return NUM_QUBITS * cfg.n_layers * 3


def prepare_hf_state(circuit: cirq.Circuit, hf_bitstring: str):
    assert len(hf_bitstring) == NUM_QUBITS
    for i, bit in enumerate(hf_bitstring):
        if bit == "1":
            circuit.append(cirq.X(qubits[i]))


def build_ansatz(params: np.ndarray, cfg: AnsatzConfig) -> cirq.Circuit:
    expected = param_count(cfg)
    assert len(params) == expected, f"Expected {expected} params, got {len(params)}"

    circuit = cirq.Circuit()
    prepare_hf_state(circuit, cfg.hf_bitstring)

    idx = 0
    for _ in range(cfg.n_layers):
        for q in qubits:
            circuit.append(cirq.rx(params[idx])(q))
            idx += 1
            circuit.append(cirq.ry(params[idx])(q))
            idx += 1
            circuit.append(cirq.rz(params[idx])(q))
            idx += 1

        for i in range(NUM_QUBITS):
            circuit.append(cirq.CNOT(qubits[i], qubits[(i + 1) % NUM_QUBITS]))

        if cfg.use_reverse_ring:
            for i in reversed(range(NUM_QUBITS)):
                circuit.append(cirq.CNOT(qubits[(i + 1) % NUM_QUBITS], qubits[i]))

    return circuit


def energy_from_params(params: np.ndarray, H_sparse, cfg: AnsatzConfig) -> float:
    circuit = build_ansatz(params, cfg)
    result = simulator.simulate(circuit)
    state = result.final_state_vector
    psi = state.reshape(-1, 1)
    return float(np.vdot(psi, H_sparse @ psi).real)


# ============================================================
# 3) Classical VQE with multi-start
# ============================================================
def generate_initial_points(
    dim: int,
    n_restarts: int,
    seed: int = 1234,
    include_zero: bool = True,
    scale: float = 0.2,
) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    xs = []
    if include_zero:
        xs.append(np.zeros(dim, dtype=np.float64))
    for _ in range(max(0, n_restarts - len(xs))):
        xs.append(rng.normal(loc=0.0, scale=scale, size=dim))
    return xs


def classical_vqe_multistart(
    H_sparse,
    cfg: AnsatzConfig,
    n_restarts: int = 12,
    maxiter: int = 300,
    seed: int = 1234,
    method: str = "COBYLA",
):
    dim = param_count(cfg)

    def objective(x):
        return energy_from_params(x, H_sparse, cfg)

    best_x = None
    best_e = float("inf")

    starts = generate_initial_points(dim, n_restarts=n_restarts, seed=seed, include_zero=True)

    for k, x0 in enumerate(starts, start=1):
        res = minimize(
            objective,
            x0,
            method=method,
            options={"maxiter": maxiter, "disp": False},
        )
        if res.fun < best_e:
            best_e = float(res.fun)
            best_x = res.x.copy()
        print(f"[Classical restart {k:02d}/{len(starts)}] energy = {res.fun:.8f} Ha")

    assert best_x is not None
    return best_x, best_e


# ============================================================
# 4) TT utilities
# ============================================================
class TTLinearNoInput(nn.Module):
    """
    TT tensor generator with no external input.
    Produces a vector of size prod(dims).
    """
    def __init__(self, dims: List[int], ranks: List[int], scale: float = 0.05):
        super().__init__()
        assert len(dims) + 1 == len(ranks)
        self.dims = dims
        self.ranks = ranks

        self.cores = nn.ParameterList([
            nn.Parameter(torch.randn(r1, d, r2) * scale)
            for r1, d, r2 in zip(ranks[:-1], dims, ranks[1:])
        ])

    def forward(self) -> torch.Tensor:
        res = self.cores[0][0]  # [d1, r]
        for core in self.cores[1:]:
            tmp = torch.einsum("xr,rds->xds", res, core)
            res = tmp.reshape(-1, core.shape[2])
        return res.squeeze(-1)


class TTLinear(nn.Module):
    """
    Standard TT linear map:
        x in R^{prod(in_dims)} -> y in R^{prod(out_dims)}
    """
    def __init__(
        self,
        in_dims: List[int],
        out_dims: List[int],
        tt_ranks: List[int],
        bias: bool = True,
        init_std: float = 0.05,
    ):
        super().__init__()
        assert len(in_dims) == len(out_dims), "in_dims and out_dims must have same length"
        assert len(tt_ranks) == len(in_dims) + 1, "tt_ranks must have length len(in_dims)+1"

        self.in_dims = list(in_dims)
        self.out_dims = list(out_dims)
        self.tt_ranks = list(tt_ranks)
        self.input_dim = int(np.prod(in_dims))
        self.output_dim = int(np.prod(out_dims))

        self.cores = nn.ParameterList()
        for k in range(len(in_dims)):
            r0 = tt_ranks[k]
            r1 = tt_ranks[k + 1]
            n_k = in_dims[k]
            m_k = out_dims[k]
            core = nn.Parameter(torch.randn(r0, n_k, m_k, r1) * init_std)
            self.cores.append(core)

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.output_dim))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Expected input dim {self.input_dim}, got {x.shape[1]}")

        x = x.view(bsz, *self.in_dims)

        symbols = [c for c in "abcdefghijklmnopqrstuvwxyz" if c != "b"]
        d = len(self.in_dims)
        if 3 * d + 1 > len(symbols):
            raise ValueError("Too many TT dimensions for available einsum symbols.")

        i_syms = symbols[:d]
        o_syms = symbols[d:2 * d]
        r_syms = symbols[2 * d:2 * d + d + 1]

        inp = "b" + "".join(i_syms)
        core_strings = [
            f"{r_syms[k]}{i_syms[k]}{o_syms[k]}{r_syms[k+1]}"
            for k in range(d)
        ]
        outp = "b" + "".join(o_syms)

        eins = inp + "," + ",".join(core_strings) + "->" + outp
        y = torch.einsum(eins, x, *self.cores)
        y = y.reshape(bsz, -1)

        if self.bias is not None:
            y = y + self.bias
        return y


def choose_tt_shape(total_dim: int):
    if total_dim == 48:
        return [4, 4, 3], [1, 4, 4, 1]
    if total_dim == 36:
        return [3, 4, 3], [1, 4, 4, 1]
    if total_dim == 24:
        return [4, 6], [1, 4, 1]

    for a in range(2, total_dim + 1):
        if total_dim % a == 0:
            b = total_dim // a
            return [a, b], [1, 4, 1]
    return [total_dim], [1, 1]


def flatten_module_params(module: nn.Module) -> np.ndarray:
    arrs = []
    for p in module.parameters():
        arrs.append(p.detach().cpu().numpy().reshape(-1))
    return np.concatenate(arrs, axis=0)


def set_module_params_from_flat(module: nn.Module, flat_params: np.ndarray):
    offset = 0
    for p in module.parameters():
        size = p.numel()
        vals = flat_params[offset: offset + size].reshape(p.shape)
        p.data.copy_(torch.from_numpy(vals.astype(np.float32)))
        offset += size
    assert offset == flat_params.shape[0]


# ============================================================
# 5) TensorHyper-VQC
# ============================================================
class ResidualTTNetwork(nn.Module):
    """
    TensorHyper: no-input TT residual generator.
    theta = base_theta + delta_scale * tanh(tt())
    """
    def __init__(self, dims: List[int], ranks: List[int], delta_scale: float = 0.35):
        super().__init__()
        self.tt = TTLinearNoInput(dims, ranks, scale=0.05)
        self.delta_scale = delta_scale

    def forward(self) -> torch.Tensor:
        return self.delta_scale * torch.tanh(self.tt())


def params_from_tensorhyper(tt_net: ResidualTTNetwork, base_theta: np.ndarray) -> np.ndarray:
    delta = tt_net().detach().cpu().numpy()
    return base_theta + delta


def tensorhyper_vqe_residual(
    H_sparse,
    cfg: AnsatzConfig,
    base_theta: np.ndarray,
    tt_dims: List[int],
    tt_ranks: List[int],
    delta_scale: float = 0.35,
    maxiter: int = 300,
    n_restarts: int = 8,
    seed: int = 1234,
    method: str = "COBYLA",
):
    dim = param_count(cfg)
    assert base_theta.shape[0] == dim
    assert int(np.prod(tt_dims)) == dim

    best_theta = None
    best_energy = float("inf")
    best_model = None

    rng = np.random.default_rng(seed)

    for restart in range(1, n_restarts + 1):
        model = ResidualTTNetwork(tt_dims, tt_ranks, delta_scale=delta_scale)

        if restart == 1:
            x0 = flatten_module_params(model)
        else:
            x0 = rng.normal(0.0, 0.05, size=flatten_module_params(model).shape[0])

        def objective(flat_params):
            set_module_params_from_flat(model, flat_params)
            theta = params_from_tensorhyper(model, base_theta)
            return energy_from_params(theta, H_sparse, cfg)

        res = minimize(
            objective,
            x0,
            method=method,
            options={"maxiter": maxiter, "disp": False},
        )

        set_module_params_from_flat(model, res.x)
        theta = params_from_tensorhyper(model, base_theta)
        e = energy_from_params(theta, H_sparse, cfg)

        if e < best_energy:
            best_energy = float(e)
            best_theta = theta.copy()
            best_model = model

        print(f"[TensorHyper restart {restart:02d}/{n_restarts}] energy = {e:.8f} Ha")

    assert best_theta is not None and best_model is not None
    return best_theta, best_energy, best_model


# ============================================================
# 6) TensorDual-VQC
# ============================================================
class TensorDualOperator(nn.Module):
    """
    TensorDual operator:
      z_shared = TTLinear(seed -> hidden)
      theta_base = TTLinear(hidden -> param)
      h = TTLinear(hidden -> latent)
      theta_res  = TTLinear(latent -> param)

      theta = base_theta + base_scale * tanh(theta_base)
                         + res_scale  * tanh(theta_res)
    """
    def __init__(
        self,
        seed_in_dims: List[int],
        hidden_dims: List[int],
        latent_dims: List[int],
        param_out_dims: List[int],
        tt_rank_backbone: int = 4,
        tt_rank_base: int = 4,
        tt_rank_latent: int = 4,
        tt_rank_hyper: int = 4,
        base_scale: float = 0.20,
        res_scale: float = 0.25,
    ):
        super().__init__()

        self.backbone = TTLinear(
            in_dims=seed_in_dims,
            out_dims=hidden_dims,
            tt_ranks=[1] + [tt_rank_backbone] * (len(seed_in_dims) - 1) + [1],
            init_std=0.05,
        )
        self.base_head = TTLinear(
            in_dims=hidden_dims,
            out_dims=param_out_dims,
            tt_ranks=[1] + [tt_rank_base] * (len(hidden_dims) - 1) + [1],
            init_std=0.05,
        )
        self.latent_head = TTLinear(
            in_dims=hidden_dims,
            out_dims=latent_dims,
            tt_ranks=[1] + [tt_rank_latent] * (len(hidden_dims) - 1) + [1],
            init_std=0.05,
        )
        self.hyper_head = TTLinear(
            in_dims=latent_dims,
            out_dims=param_out_dims,
            tt_ranks=[1] + [tt_rank_hyper] * (len(latent_dims) - 1) + [1],
            init_std=0.05,
        )

        self.base_scale = base_scale
        self.res_scale = res_scale
        self.seed_dim = int(np.prod(seed_in_dims))

    def forward(self, seed: torch.Tensor) -> torch.Tensor:
        z = torch.tanh(self.backbone(seed))
        theta_base = self.base_scale * torch.tanh(self.base_head(z))
        h = torch.tanh(self.latent_head(z))
        theta_res = self.res_scale * torch.tanh(self.hyper_head(h))
        return theta_base + theta_res


def params_from_tensordual(td_net: TensorDualOperator, base_theta: np.ndarray, seed_vec: np.ndarray) -> np.ndarray:
    seed = torch.from_numpy(seed_vec.astype(np.float32)).unsqueeze(0)
    delta = td_net(seed).detach().cpu().numpy().reshape(-1)
    return base_theta + delta


def tensordual_vqe_residual(
    H_sparse,
    cfg: AnsatzConfig,
    base_theta: np.ndarray,
    seed_in_dims: List[int],
    hidden_dims: List[int],
    latent_dims: List[int],
    param_out_dims: List[int],
    tt_rank_backbone: int = 4,
    tt_rank_base: int = 4,
    tt_rank_latent: int = 4,
    tt_rank_hyper: int = 4,
    base_scale: float = 0.20,
    res_scale: float = 0.25,
    maxiter: int = 300,
    n_restarts: int = 8,
    seed: int = 1234,
    method: str = "COBYLA",
):
    dim = param_count(cfg)
    assert base_theta.shape[0] == dim
    assert int(np.prod(param_out_dims)) == dim

    best_theta = None
    best_energy = float("inf")
    best_model = None

    rng = np.random.default_rng(seed)
    seed_vec = rng.normal(loc=0.0, scale=1.0, size=int(np.prod(seed_in_dims))).astype(np.float32)

    for restart in range(1, n_restarts + 1):
        model = TensorDualOperator(
            seed_in_dims=seed_in_dims,
            hidden_dims=hidden_dims,
            latent_dims=latent_dims,
            param_out_dims=param_out_dims,
            tt_rank_backbone=tt_rank_backbone,
            tt_rank_base=tt_rank_base,
            tt_rank_latent=tt_rank_latent,
            tt_rank_hyper=tt_rank_hyper,
            base_scale=base_scale,
            res_scale=res_scale,
        )

        if restart == 1:
            x0 = flatten_module_params(model)
        else:
            x0 = rng.normal(0.0, 0.05, size=flatten_module_params(model).shape[0])

        def objective(flat_params):
            set_module_params_from_flat(model, flat_params)
            theta = params_from_tensordual(model, base_theta, seed_vec)
            return energy_from_params(theta, H_sparse, cfg)

        res = minimize(
            objective,
            x0,
            method=method,
            options={"maxiter": maxiter, "disp": False},
        )

        set_module_params_from_flat(model, res.x)
        theta = params_from_tensordual(model, base_theta, seed_vec)
        e = energy_from_params(theta, H_sparse, cfg)

        if e < best_energy:
            best_energy = float(e)
            best_theta = theta.copy()
            best_model = model

        print(f"[TensorDual restart {restart:02d}/{n_restarts}] energy = {e:.8f} Ha")

    assert best_theta is not None and best_model is not None
    return best_theta, best_energy, best_model


# ============================================================
# 7) Optional post-polish
# ============================================================
def local_polish(
    theta0: np.ndarray,
    H_sparse,
    cfg: AnsatzConfig,
    maxiter: int = 120,
    method: str = "COBYLA",
):
    def objective(x):
        return energy_from_params(x, H_sparse, cfg)

    res = minimize(
        objective,
        theta0,
        method=method,
        options={"maxiter": maxiter, "disp": False},
    )
    return res.x.copy(), float(res.fun)


# ============================================================
# 8) Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Classical VQE, TensorHyper-VQC, and TensorDual-VQC for LiH")

    parser.add_argument("--bond_length", type=float, default=1.6)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--hf_bitstring", type=str, default="1100")
    parser.add_argument("--no_reverse_ring", action="store_true")

    parser.add_argument("--classical_restarts", type=int, default=12)
    parser.add_argument("--classical_maxiter", type=int, default=300)

    parser.add_argument("--tt_restarts", type=int, default=8)
    parser.add_argument("--tt_maxiter", type=int, default=300)
    parser.add_argument("--delta_scale", type=float, default=0.35)

    parser.add_argument("--td_restarts", type=int, default=8)
    parser.add_argument("--td_maxiter", type=int, default=300)
    parser.add_argument("--td_base_scale", type=float, default=0.20)
    parser.add_argument("--td_res_scale", type=float, default=0.25)

    parser.add_argument("--post_polish", action="store_true")
    parser.add_argument("--polish_maxiter", type=int, default=120)

    parser.add_argument("--seed", type=int, default=1234)

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = AnsatzConfig(
        n_layers=args.n_layers,
        hf_bitstring=args.hf_bitstring,
        use_reverse_ring=not args.no_reverse_ring,
    )

    H_sparse = get_lih_reduced_hamiltonian(bond_length=args.bond_length)
    E_exact = spla.eigsh(H_sparse, k=1, which='SA')[0][0].real

    print(f"Exact LiH ground energy: {E_exact:.8f} Ha")
    print(f"[INFO] Ansatz layers: {cfg.n_layers}")
    print(f"[INFO] HF bitstring: {cfg.hf_bitstring}")
    print(f"[INFO] Parameter count: {param_count(cfg)}")

    # Step 1: strong classical warm start
    classical_theta, classical_e = classical_vqe_multistart(
        H_sparse=H_sparse,
        cfg=cfg,
        n_restarts=args.classical_restarts,
        maxiter=args.classical_maxiter,
        seed=args.seed,
        method="COBYLA",
    )

    print(f"\nBest Classical VQE energy: {classical_e:.8f} Ha")
    print(f"Classical error: {abs(classical_e - E_exact):.8f} Ha")

    # Shared parameter tensorization
    total_dim = param_count(cfg)
    param_dims, param_ranks = choose_tt_shape(total_dim)
    print(f"[INFO] Param TT dims: {param_dims}, ranks: {param_ranks}")

    # Step 2: TensorHyper-VQC
    th_theta, th_e, _ = tensorhyper_vqe_residual(
        H_sparse=H_sparse,
        cfg=cfg,
        base_theta=classical_theta,
        tt_dims=param_dims,
        tt_ranks=param_ranks,
        delta_scale=args.delta_scale,
        maxiter=args.tt_maxiter,
        n_restarts=args.tt_restarts,
        seed=args.seed,
        method="COBYLA",
    )

    print(f"\nTensorHyper-VQC energy: {th_e:.8f} Ha")
    print(f"TensorHyper-VQC error: {abs(th_e - E_exact):.8f} Ha")

    # Step 3: TensorDual-VQC
    # Keep all mode counts equal (= len(param_dims))
    num_modes = len(param_dims)
    if total_dim == 48:
        hidden_dims = [4, 4, 3]
        latent_dims = [4, 4, 3]
        seed_in_dims = [4, 4, 3]
    else:
        hidden_dims = param_dims
        latent_dims = param_dims
        seed_in_dims = param_dims

    td_theta, td_e, _ = tensordual_vqe_residual(
        H_sparse=H_sparse,
        cfg=cfg,
        base_theta=classical_theta,
        seed_in_dims=seed_in_dims,
        hidden_dims=hidden_dims,
        latent_dims=latent_dims,
        param_out_dims=param_dims,
        tt_rank_backbone=4,
        tt_rank_base=4,
        tt_rank_latent=4,
        tt_rank_hyper=4,
        base_scale=args.td_base_scale,
        res_scale=args.td_res_scale,
        maxiter=args.td_maxiter,
        n_restarts=args.td_restarts,
        seed=args.seed,
        method="COBYLA",
    )

    print(f"\nTensorDual-VQC energy: {td_e:.8f} Ha")
    print(f"TensorDual-VQC error: {abs(td_e - E_exact):.8f} Ha")

    if args.post_polish:
        polished_th_theta, polished_th_e = local_polish(
            theta0=th_theta,
            H_sparse=H_sparse,
            cfg=cfg,
            maxiter=args.polish_maxiter,
            method="COBYLA",
        )
        print(f"\nPost-polished TensorHyper energy: {polished_th_e:.8f} Ha")
        print(f"Post-polished TensorHyper error: {abs(polished_th_e - E_exact):.8f} Ha")

        polished_td_theta, polished_td_e = local_polish(
            theta0=td_theta,
            H_sparse=H_sparse,
            cfg=cfg,
            maxiter=args.polish_maxiter,
            method="COBYLA",
        )
        print(f"Post-polished TensorDual energy: {polished_td_e:.8f} Ha")
        print(f"Post-polished TensorDual error: {abs(polished_td_e - E_exact):.8f} Ha")
    else:
        polished_th_e = None
        polished_td_e = None

    print("\nSummary")
    print("-------")
    print(f"Exact energy                 : {E_exact:.8f} Ha")
    print(f"Best classical VQE           : {classical_e:.8f} Ha")
    print(f"TensorHyper-VQC              : {th_e:.8f} Ha")
    print(f"TensorDual-VQC               : {td_e:.8f} Ha")
    if polished_th_e is not None:
        print(f"Post-polished TensorHyper    : {polished_th_e:.8f} Ha")
    if polished_td_e is not None:
        print(f"Post-polished TensorDual     : {polished_td_e:.8f} Ha")


if __name__ == "__main__":
    main()

