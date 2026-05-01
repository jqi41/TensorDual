#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TensorDual Operator for Variational Quantum Circuits
Quantum-Dot Classification with Composable Quantum Noise
========================================================

Architecture
------------
Quantum-dot diagram x (50x50 -> 2500)
 -> Unified TT Network with dual outputs
      -> feature head h
      -> angle head x_angle
 -> Separate TT Hypernetwork: h -> theta
 -> Noise-aware VQC surrogate
 -> Learnable observable head
 -> logits

Dataset
-------
Uses the same datapath style as the TensorHyper-VQC code:

    ./mlqe_2023_edx/week1/dataset/csds_noiseless.npy
    ./mlqe_2023_edx/week1/dataset/csds.npy
    ./mlqe_2023_edx/week1/dataset/labels.npy

Noise
-----
Composable noise models:
- depol
- dephase
- pauli
- overrot
- twopauli
- readout
"""

from __future__ import annotations

import math
import numpy as np
import argparse
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Helpers
# ============================================================

def parse_int_list(x: str) -> List[int]:
    x = str(x).strip()
    if x.startswith('[') and x.endswith(']'):
        x = x[1:-1]
    if not x:
        return []
    return [int(t.strip()) for t in x.split(',') if t.strip()]


def build_noise_cfg(args):
    models = [m.strip().lower() for m in str(args.noise_models).split(',') if m.strip()]
    if 'none' in models:
        models = []

    cfg = dict(
        depol=args.p_depol if 'depol' in models else 0.0,
        dephase=args.p_dephase if 'dephase' in models else 0.0,
        pauli_px=args.pauli_px if 'pauli' in models else 0.0,
        pauli_py=args.pauli_py if 'pauli' in models else 0.0,
        pauli_pz=args.pauli_pz if 'pauli' in models else 0.0,
        overrot_sigma=args.overrot_sigma if 'overrot' in models else 0.0,
        p_twopauli=args.p_twopauli if 'twopauli' in models else 0.0,
        p_readout=args.p_readout if 'readout' in models else 0.0,
    )
    return cfg, models


def count_total_params(model):
    return sum(p.numel() for p in model.parameters())


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_trainable_summary(model):
    print("\nTrainable parameter summary")
    print("---------------------------")
    total = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"{name:<60}{p.numel():>10}")
            total += p.numel()
    print(f"{'TOTAL TRAINABLE':<60}{total:>10}\n")


def factorize_to_n_dims(n: int, num_dims: int = 4) -> Tuple[int, ...]:
    if n <= 0:
        raise ValueError("n must be positive")
    if num_dims < 1:
        raise ValueError("num_dims must be >= 1")

    dims = [1] * num_dims
    remaining = n
    p = 2

    while p * p <= remaining:
        while remaining % p == 0:
            idx = min(range(num_dims), key=lambda i: dims[i])
            dims[idx] *= p
            remaining //= p
        p += 1

    if remaining > 1:
        idx = min(range(num_dims), key=lambda i: dims[i])
        dims[idx] *= remaining

    return tuple(sorted(dims))


# ============================================================
# Quantum Dot Data Loader
# ============================================================

def load_quantum_dot_data(
    dataset_root="./mlqe_2023_edx/week1/dataset",
    batch_size=64,
    test_kind="gen",
):
    """
    Matches the datapath style used in the TensorHyper-VQC code.

    Files:
        csds_noiseless.npy : clean diagrams, shape (N,50,50)
        csds.npy           : noisy diagrams, shape (N,50,50)
        labels.npy         : labels, shape (N,)

    test_kind:
        - gen: train/test on noisy diagrams
        - rep: train on noisy diagrams, test on noiseless diagrams
    """
    x_clean = np.load(f"{dataset_root}/csds_noiseless.npy")
    x_noisy = np.load(f"{dataset_root}/csds.npy")
    y = np.load(f"{dataset_root}/labels.npy").astype(np.int64)

    x_clean = x_clean.reshape(-1, 2500).astype(np.float32)
    x_noisy = x_noisy.reshape(-1, 2500).astype(np.float32)

    split = int(0.9 * len(y))

    if test_kind == "rep":
        x_train = x_noisy[:split]
        y_train = y[:split]
        x_test = x_clean[split:]
        y_test = y[split:]
    else:
        x_train = x_noisy[:split]
        y_train = y[:split]
        x_test = x_noisy[split:]
        y_test = y[split:]

    train_ds = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train)
    )
    test_ds = TensorDataset(
        torch.from_numpy(x_test),
        torch.from_numpy(y_test)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_ds, test_ds, train_loader, test_loader


# ============================================================
# Tensor-Train Linear
# ============================================================

class TTLinear(nn.Module):
    """
    Tensor-Train linear map:
        x in R^{prod(in_dims)} -> y in R^{prod(out_dims)}
    """
    def __init__(
        self,
        in_dims,
        out_dims,
        tt_ranks,
        bias=True,
        init_std=0.08,
    ):
        super().__init__()
        assert len(in_dims) == len(out_dims), "in_dims and out_dims must have same length"
        assert len(tt_ranks) == len(in_dims) + 1, "tt_ranks length must be len(in_dims)+1"

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
            nn.init.xavier_uniform_(core)
            self.cores.append(core)

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.output_dim))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        bsz = x.size(0)
        if x.size(1) != self.input_dim:
            raise ValueError(f"TTLinear expected input dim {self.input_dim}, got {x.size(1)}")

        x_rs = x.view(bsz, *self.in_dims)
        batch = 'b'
        letters = [chr(i) for i in range(ord('a'), ord('z') + 1) if chr(i) != batch]
        d = len(self.in_dims)

        iL = letters[:d]
        oL = letters[d:2 * d]
        rL = letters[2 * d:2 * d + d + 1]

        inp = batch + ''.join(iL)
        cores = [f"{rL[k]}{iL[k]}{oL[k]}{rL[k+1]}" for k in range(d)]
        outp = batch + ''.join(oL)
        eins = inp + ',' + ','.join(cores) + '->' + outp

        out = torch.einsum(eins, x_rs, *self.cores)
        out = out.reshape(bsz, -1)

        if self.bias is not None:
            out = out + self.bias

        return out


# ============================================================
# Unified TT Network with Dual Outputs
# ============================================================

class UnifiedTTNetwork(nn.Module):
    """
    Input x -> shared TT backbone -> hidden
            -> feature head -> h
            -> angle head   -> x_angle
    """
    def __init__(
        self,
        input_dim=2500,
        hidden_dim=256,
        feature_dim=64,
        n_qubits=12,
        tt_rank_backbone=4,
        tt_rank_feature=4,
        tt_rank_angle=4,
        feature_dropout=0.0,
    ):
        super().__init__()

        input_dims = factorize_to_n_dims(input_dim, num_dims=4)
        hidden_dims = factorize_to_n_dims(hidden_dim, num_dims=4)
        feature_dims = factorize_to_n_dims(feature_dim, num_dims=4)
        angle_dims = factorize_to_n_dims(n_qubits, num_dims=4)

        ranks_backbone = (1, tt_rank_backbone, tt_rank_backbone, tt_rank_backbone, 1)
        ranks_feature = (1, tt_rank_feature, tt_rank_feature, tt_rank_feature, 1)
        ranks_angle = (1, tt_rank_angle, tt_rank_angle, tt_rank_angle, 1)

        self.backbone = TTLinear(
            in_dims=input_dims,
            out_dims=hidden_dims,
            tt_ranks=ranks_backbone
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(feature_dropout)

        self.feature_head = TTLinear(
            in_dims=hidden_dims,
            out_dims=feature_dims,
            tt_ranks=ranks_feature
        )
        self.angle_head = TTLinear(
            in_dims=hidden_dims,
            out_dims=angle_dims,
            tt_ranks=ranks_angle
        )

        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.feature_dims = feature_dims
        self.angle_dims = angle_dims

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.n_qubits = n_qubits

    def forward(self, x):
        hidden = self.backbone(x)
        hidden = self.norm(hidden)
        hidden = F.gelu(hidden)
        hidden = self.dropout(hidden)

        h = torch.tanh(self.feature_head(hidden))
        x_angle = math.pi * torch.tanh(self.angle_head(hidden))

        aux = {
            "hidden": hidden,
            "h": h,
            "x_angle": x_angle,
        }
        return h, x_angle, aux


# ============================================================
# Separate TT Hypernetwork
# ============================================================

class SeparateTTHypernetwork(nn.Module):
    """
    h -> theta
    """
    def __init__(self, feature_dim=64, param_dim=216, tt_rank=4):
        super().__init__()
        in_dims = factorize_to_n_dims(feature_dim, num_dims=4)
        out_dims = factorize_to_n_dims(param_dim, num_dims=4)
        tt_ranks = (1, tt_rank, tt_rank, tt_rank, 1)

        self.tt = TTLinear(
            in_dims=in_dims,
            out_dims=out_dims,
            tt_ranks=tt_ranks,
            bias=True
        )
        self.param_dim = param_dim
        self.out_dims = out_dims

    def forward(self, h):
        return self.tt(h)


# ============================================================
# Noise-aware VQC surrogate
# ============================================================

class NoiseAwareVQC(nn.Module):
    """
    Noise-aware differentiable surrogate VQC.

    Inputs:
        angles : [B, n_qubits]
        theta  : [B, 3 * layers * n_qubits]
    """
    def __init__(self, n_qubits=12, layers=6, noise_cfg=None):
        super().__init__()
        self.n_qubits = n_qubits
        self.layers = layers
        self.feature_dim = n_qubits
        self.noise_cfg = {} if noise_cfg is None else noise_cfg

    def forward(self, angles, theta):
        bsz = angles.shape[0]
        q = self.n_qubits

        theta = theta.view(bsz, self.layers, 3, q)

        overrot_sigma = float(self.noise_cfg.get("overrot_sigma", 0.0))
        if self.training and overrot_sigma > 0:
            angles = angles + torch.randn_like(angles) * overrot_sigma
            theta = theta + torch.randn_like(theta) * overrot_sigma

        rx = theta[:, :, 0, :].mean(dim=1)
        ry = theta[:, :, 1, :].mean(dim=1)
        rz = theta[:, :, 2, :].mean(dim=1)

        qfeat = torch.tanh(
            torch.sin(angles + rx)
            + 0.5 * torch.cos(ry)
            + 0.5 * torch.sin(rz)
        )

        depol = float(self.noise_cfg.get("depol", 0.0))
        dephase = float(self.noise_cfg.get("dephase", 0.0))
        px = float(self.noise_cfg.get("pauli_px", 0.0))
        py = float(self.noise_cfg.get("pauli_py", 0.0))
        pz = float(self.noise_cfg.get("pauli_pz", 0.0))
        p_twopauli = float(self.noise_cfg.get("p_twopauli", 0.0))
        p_readout = float(self.noise_cfg.get("p_readout", 0.0))

        # single-qubit noise attenuation
        if depol > 0:
            qfeat = (1.0 - depol) * qfeat

        if dephase > 0:
            qfeat = (1.0 - dephase) * qfeat

        pauli_strength = px + py + pz
        if pauli_strength > 0:
            qfeat = (1.0 - pauli_strength) * qfeat

        # two-qubit noise: degrade neighbor consistency
        if p_twopauli > 0:
            qfeat = (1.0 - p_twopauli) * qfeat + p_twopauli * 0.5 * torch.roll(qfeat, shifts=1, dims=1)

        # readout sign flip on expectation-like outputs
        if p_readout > 0:
            qfeat = (1.0 - 2.0 * p_readout) * qfeat

        qfeat = torch.clamp(qfeat, -1.0, 1.0)
        return qfeat


# ============================================================
# Learnable Observable Head
# ============================================================

class LearnableObservableHead(nn.Module):
    def __init__(self, feature_dim, num_classes, observable_rank=None, init_scale=0.02):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.observable_rank = observable_rank

        if observable_rank is None:
            self.observable = nn.Parameter(
                torch.randn(num_classes, feature_dim) * init_scale
            )
        else:
            self.obs_left = nn.Parameter(
                torch.randn(num_classes, observable_rank) * init_scale
            )
            self.obs_right = nn.Parameter(
                torch.randn(observable_rank, feature_dim) * init_scale
            )

        self.bias = nn.Parameter(torch.zeros(num_classes))
        self.logit_scale = nn.Parameter(torch.tensor(5.0))

    def get_observable_matrix(self):
        if self.observable_rank is None:
            return self.observable
        return self.obs_left @ self.obs_right

    def forward(self, qfeat):
        obs = self.get_observable_matrix()
        logits = F.linear(qfeat, obs, self.bias)
        return self.logit_scale * logits


# ============================================================
# Full TensorDual Model
# ============================================================

class TensorDualOperatorVQC(nn.Module):
    """
    TensorDual Operator for Variational Quantum Circuits
    """
    def __init__(
        self,
        input_dim=2500,
        num_classes=2,
        n_qubits=12,
        vqc_layers=6,
        hidden_dim=256,
        feature_dim=64,
        tt_rank_backbone=4,
        tt_rank_feature=4,
        tt_rank_angle=4,
        tt_rank_hyper=4,
        observable_rank=6,
        feature_dropout=0.0,
        noise_cfg=None,
    ):
        super().__init__()

        self.unified_tt = UnifiedTTNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            feature_dim=feature_dim,
            n_qubits=n_qubits,
            tt_rank_backbone=tt_rank_backbone,
            tt_rank_feature=tt_rank_feature,
            tt_rank_angle=tt_rank_angle,
            feature_dropout=feature_dropout,
        )

        param_dim = 3 * n_qubits * vqc_layers

        self.hypernet = SeparateTTHypernetwork(
            feature_dim=feature_dim,
            param_dim=param_dim,
            tt_rank=tt_rank_hyper,
        )

        self.vqc = NoiseAwareVQC(
            n_qubits=n_qubits,
            layers=vqc_layers,
            noise_cfg=noise_cfg,
        )

        self.observable_head = LearnableObservableHead(
            feature_dim=n_qubits,
            num_classes=num_classes,
            observable_rank=observable_rank,
        )

    def forward(self, x):
        h, x_angle, tt_aux = self.unified_tt(x)
        theta = self.hypernet(h)
        qfeat = self.vqc(x_angle, theta)
        logits = self.observable_head(qfeat)

        aux = {
            **tt_aux,
            "theta": theta,
            "qfeat": qfeat,
        }
        return logits, aux


# ============================================================
# Training / Evaluation
# ============================================================

def evaluate(model, loader, device):
    model.eval()

    total = 0
    correct = 0
    loss_sum = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits, _ = model(x)
            loss = F.cross_entropy(logits, y)

            loss_sum += loss.item() * x.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += x.size(0)

    return loss_sum / total, correct / total


def train(model, train_loader, test_loader, device, epochs=21, lr=3e-2):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        correct_train = 0
        total_train = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits, aux = model(x_batch)

            ce = criterion(logits, y_batch)
            reg_theta = 1e-4 * aux["theta"].pow(2).mean()
            reg_feat = 1e-5 * aux["h"].pow(2).mean()
            reg_qfeat = 1e-4 * aux["qfeat"].pow(2).mean()
            loss = ce + reg_theta + reg_feat + reg_qfeat

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * x_batch.size(0)
            preds = logits.argmax(dim=1)
            correct_train += (preds == y_batch).sum().item()
            total_train += x_batch.size(0)

        scheduler.step()

        train_loss = total_loss / total_train
        train_acc = correct_train / total_train

        test_loss, test_acc = evaluate(model, test_loader, device)

        print(
            f"Epoch {epoch:02d}  "
            f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}  "
            f"Test loss:  {test_loss:.4f}, Test acc:  {test_acc:.4f}"
        )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TensorDual Operator for Variational Quantum Circuits on Quantum-Dot Classification"
    )

    parser.add_argument('--dataset_root', type=str, default='./mlqe_2023_edx/week1/dataset')
    parser.add_argument('--save_path', metavar='DIR', default='models')
    parser.add_argument('--num_qubits', default=30, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_epochs', default=20, type=int)
    parser.add_argument('--depth_vqc', default=6, type=int)
    parser.add_argument('--lr', default=3e-3, type=float)
    parser.add_argument('--test_kind', metavar='DIR', default='gen', choices=['rep', 'gen'])

    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--feature_dim', default=64, type=int)
    parser.add_argument('--tt_rank_backbone', default=4, type=int)
    parser.add_argument('--tt_rank_feature', default=4, type=int)
    parser.add_argument('--tt_rank_angle', default=4, type=int)
    parser.add_argument('--tt_rank_hyper', default=4, type=int)
    parser.add_argument('--observable_rank', default=6, type=int)
    parser.add_argument('--feature_dropout', default=0.0, type=float)

    # composable noise
    parser.add_argument('--noise_models', type=str, default='depol,dephase,readout')
    parser.add_argument('--p_depol', type=float, default=0.01)
    parser.add_argument('--p_dephase', type=float, default=0.01)
    parser.add_argument('--pauli_px', type=float, default=0.01)
    parser.add_argument('--pauli_py', type=float, default=0.01)
    parser.add_argument('--pauli_pz', type=float, default=0.01)
    parser.add_argument('--overrot_sigma', type=float, default=0.002)
    parser.add_argument('--p_twopauli', type=float, default=0.05)
    parser.add_argument('--p_readout', type=float, default=0.1)

    args = parser.parse_args()

    set_seed(1324)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, test_ds, train_loader, test_loader = load_quantum_dot_data(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        test_kind=args.test_kind,
    )

    noise_cfg, noise_models = build_noise_cfg(args)

    model = TensorDualOperatorVQC(
        input_dim=2500,
        num_classes=2,
        n_qubits=args.num_qubits,
        vqc_layers=args.depth_vqc,
        hidden_dim=args.hidden_dim,
        feature_dim=args.feature_dim,
        tt_rank_backbone=args.tt_rank_backbone,
        tt_rank_feature=args.tt_rank_feature,
        tt_rank_angle=args.tt_rank_angle,
        tt_rank_hyper=args.tt_rank_hyper,
        observable_rank=(None if args.observable_rank <= 0 else args.observable_rank),
        feature_dropout=args.feature_dropout,
        noise_cfg=noise_cfg,
    ).to(device)

    print("Noise models:", ','.join(noise_models) if noise_models else 'none')
    print("Noise cfg:", noise_cfg)

    print(f"\nChosen setup")
    print("------------")
    print(f"input_dim            : 2500")
    print(f"num_classes          : 2")
    print(f"n_qubits             : {args.num_qubits}")
    print(f"depth_vqc            : {args.depth_vqc}")
    print(f"hidden_dim           : {args.hidden_dim}")
    print(f"feature_dim          : {args.feature_dim}")
    print(f"tt_rank_backbone     : {args.tt_rank_backbone}")
    print(f"tt_rank_feature      : {args.tt_rank_feature}")
    print(f"tt_rank_angle        : {args.tt_rank_angle}")
    print(f"tt_rank_hyper        : {args.tt_rank_hyper}")
    print(f"observable_rank      : {args.observable_rank if args.observable_rank > 0 else 'full'}")
    print(f"test_kind            : {args.test_kind}")
    print(f"input tensor dims    : {model.unified_tt.input_dims}")
    print(f"hidden tensor dims   : {model.unified_tt.hidden_dims}")
    print(f"feature tensor dims  : {model.unified_tt.feature_dims}")
    print(f"angle tensor dims    : {model.unified_tt.angle_dims}")
    print(f"hyper out dims       : {model.hypernet.out_dims}")
    print(f"Train set size       : {len(train_ds)}")
    print(f"Test set size        : {len(test_ds)}")

    print_trainable_summary(model)
    print(f"Total parameters: {count_total_params(model)}")
    print(f"Trainable parameters: {count_trainable_params(model)}")

    train(
        model,
        train_loader,
        test_loader,
        device,
        epochs=args.num_epochs,
        lr=args.lr
    )

    final_test_loss, final_test_acc = evaluate(model, test_loader, device)
    print("\nFinal results")
    print("-------------")
    print(f"test_loss={final_test_loss:.4f} test_acc={final_test_acc:.4f}")