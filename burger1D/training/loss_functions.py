"""
loss_functions.py — PINN 物理损失函数
L_total = λ_res*L_res + λ_ic*L_ic + λ_bc*L_bc + λ_sp*L_sparse
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class LossConfig:
    w_res:      float = 1.0
    w_ic:       float = 10.0
    w_bc:       float = 5.0
    w_sparse:   float = 1e-3
    w_balance:  float = 1e-2
    w_conserve: float = 0.1
    use_ntk:    bool  = True
    equation:   str   = "burgers"
    nu:         float = 0.01 / np.pi


class PhysicsLoss:
    """统一 PINN 物理损失。支持 'burgers' 和 'kdv'。"""

    def __init__(self, config: LossConfig = None):
        self.cfg = config or LossConfig()
        self._ema: Dict[str, float] = {}

    def _sparse_weight(
        self,
        model: nn.Module,
        ntk_weights: Optional[Dict[str, float]] = None,
    ) -> float:
        if ntk_weights is not None and "sparse" in ntk_weights:
            return float(ntk_weights["sparse"])
        if hasattr(model, "sparsity_weight"):
            return float(model.sparsity_weight)
        return float(self.cfg.w_sparse)

    def _balance_weight(self, model: nn.Module) -> float:
        if hasattr(model, "balance_weight"):
            return float(model.balance_weight)
        return float(self.cfg.w_balance)

    # ────────────────────────── Burgers ─────────────────────────────────
    def burgers_total(
        self,
        model: nn.Module,
        xt_col: torch.Tensor,
        xt_ic: torch.Tensor,
        u_ic: torch.Tensor,
        xt_bc: torch.Tensor,
        u_bc: torch.Tensor,
        ntk_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        nu = self.cfg.nu

        # IC
        L_ic = ((model(xt_ic) - u_ic) ** 2).mean()

        # BC
        L_bc = ((model(xt_bc) - u_bc) ** 2).mean()

        # PDE residual
        xt_g = xt_col.clone().detach().requires_grad_(True)
        u_g = model(xt_g)
        grads = torch.autograd.grad(u_g.sum(), xt_g, create_graph=True, retain_graph=True)[0]
        u_x, u_t = grads[:, 0:1], grads[:, 1:2]
        u_xx = torch.autograd.grad(u_x.sum(), xt_g, create_graph=True, retain_graph=True)[0][:, 0:1]
        res = u_t + u_g * u_x - nu * u_xx
        L_res = (res ** 2).mean()

        # Gating regularizers (MoE only)
        L_sp = torch.zeros(1, dtype=xt_col.dtype, device=xt_col.device).squeeze()
        L_bal = torch.zeros(1, dtype=xt_col.dtype, device=xt_col.device).squeeze()
        if hasattr(model, "gating"):
            if hasattr(model, "compute_gate_weights"):
                g = model.compute_gate_weights(xt_col)
            else:
                g = model.gating(xt_col)
            L_sp = torch.pow(g.clamp(min=1e-8), 0.5).sum(dim=-1).mean()
            if hasattr(model, "load_balance_loss"):
                L_bal = model.load_balance_loss(xt_col)
            else:
                L_bal = model.gating.load_balance_loss(xt_col)

        loss_dict = {"res": L_res, "ic": L_ic, "bc": L_bc, "sparse": L_sp, "balance": L_bal}

        sparse_w = self._sparse_weight(model, ntk_weights)
        balance_w = self._balance_weight(model)
        w = ntk_weights or {"res": self.cfg.w_res, "ic": self.cfg.w_ic,
                             "bc": self.cfg.w_bc, "sparse": sparse_w, "balance": balance_w}
        total = (w.get("res", 1.0)*L_res + w.get("ic", 10.0)*L_ic
                 + w.get("bc", 5.0)*L_bc
                 + w.get("sparse", sparse_w)*L_sp
                 + w.get("balance", balance_w)*L_bal)
        return total, loss_dict

    # ────────────────────────── KdV ─────────────────────────────────────
    def kdv_total(
        self,
        model: nn.Module,
        xt_col: torch.Tensor,
        xt_ic: torch.Tensor,
        u_ic: torch.Tensor,
        xt_bc: torch.Tensor,
        u_bc: torch.Tensor,
        ntk_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        L_ic = ((model(xt_ic) - u_ic) ** 2).mean()
        L_bc = ((model(xt_bc) - u_bc) ** 2).mean()

        xt_g = xt_col.clone().detach().requires_grad_(True)
        u_g = model(xt_g)
        g1 = torch.autograd.grad(u_g.sum(), xt_g, create_graph=True, retain_graph=True)[0]
        u_x, u_t = g1[:, 0:1], g1[:, 1:2]
        u_xx = torch.autograd.grad(u_x.sum(), xt_g, create_graph=True, retain_graph=True)[0][:, 0:1]
        u_xxx = torch.autograd.grad(u_xx.sum(), xt_g, create_graph=True, retain_graph=True)[0][:, 0:1]
        res = u_t + 6.0 * u_g * u_x + u_xxx
        L_res = (res ** 2).mean()

        L_sp = torch.zeros(1, dtype=xt_col.dtype, device=xt_col.device).squeeze()
        L_bal = torch.zeros(1, dtype=xt_col.dtype, device=xt_col.device).squeeze()
        if hasattr(model, "gating"):
            if hasattr(model, "compute_gate_weights"):
                g = model.compute_gate_weights(xt_col)
            else:
                g = model.gating(xt_col)
            L_sp = torch.pow(g.clamp(min=1e-8), 0.5).sum(dim=-1).mean()
            if hasattr(model, "load_balance_loss"):
                L_bal = model.load_balance_loss(xt_col)
            else:
                L_bal = model.gating.load_balance_loss(xt_col)

        loss_dict = {"res": L_res, "ic": L_ic, "bc": L_bc, "sparse": L_sp, "balance": L_bal}
        sparse_w = self._sparse_weight(model, ntk_weights)
        balance_w = self._balance_weight(model)
        w = ntk_weights or {"res": self.cfg.w_res, "ic": self.cfg.w_ic,
                             "bc": self.cfg.w_bc, "sparse": sparse_w, "balance": balance_w}
        total = (w.get("res", 1.0)*L_res + w.get("ic", 10.0)*L_ic
                 + w.get("bc", 5.0)*L_bc
                 + w.get("sparse", sparse_w)*L_sp
                 + w.get("balance", balance_w)*L_bal)
        return total, loss_dict

    # ─────────────────────────── Dispatch ───────────────────────────────
    def compute(
        self,
        model: nn.Module,
        batch: dict,
        ntk_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.cfg.equation == "burgers":
            return self.burgers_total(model,
                batch["xt_col"], batch["xt_ic"], batch["u_ic"],
                batch["xt_bc"], batch["u_bc"], ntk_weights)
        elif self.cfg.equation == "kdv":
            return self.kdv_total(model,
                batch["xt_col"], batch["xt_ic"], batch["u_ic"],
                batch["xt_bc"], batch["u_bc"], ntk_weights)
        else:
            raise ValueError(f"Unknown equation: {self.cfg.equation}")


def l2_relative_error(u_pred: torch.Tensor, u_exact: torch.Tensor) -> float:
    with torch.no_grad():
        return ((u_pred - u_exact).norm(2) / (u_exact.norm(2) + 1e-10)).item()


def max_absolute_error(u_pred: torch.Tensor, u_exact: torch.Tensor) -> float:
    with torch.no_grad():
        return (u_pred - u_exact).abs().max().item()
