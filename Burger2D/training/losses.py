"""
Losses and metrics for Burger2D.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from Burger2D.equations.burgers2d import steep_region_mask


@dataclass
class LossConfig2D:
    w_res: float = 1.0
    w_ic: float = 10.0
    w_bc: float = 5.0
    w_sparse: float = 1e-3
    w_balance: float = 1e-2
    w_gate_supervised: float = 0.0
    w_gate_misroute: float = 0.0
    gate_misroute_power: float = 2.0
    w_branch_consistency: float = 0.0
    w_base_consistency: float = 0.0
    w_rotation_supervised: float = 0.0
    w_rotation_concentration: float = 0.0
    w_rotation_activation: float = 0.0
    nu: float = 0.01 / np.pi


class PhysicsLoss2D:
    def __init__(self, config: Optional[LossConfig2D] = None):
        self.cfg = config or LossConfig2D()

    def compute(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        xt_ic = batch["xt_ic"]
        xt_bc = batch["xt_bc"]
        xt_col = batch["xt_col"]
        u_ic = batch["u_ic"]
        u_bc = batch["u_bc"]

        l_ic = ((model(xt_ic) - u_ic) ** 2).mean()
        l_bc = ((model(xt_bc) - u_bc) ** 2).mean()

        xt_g = xt_col.detach().clone().requires_grad_(True)
        u = model(xt_g)
        grads = torch.autograd.grad(
            u.sum(),
            xt_g,
            create_graph=True,
            retain_graph=True,
        )[0]
        u_x = grads[:, 0:1]
        u_y = grads[:, 1:2]
        u_t = grads[:, 2:3]
        u_xx = torch.autograd.grad(
            u_x.sum(),
            xt_g,
            create_graph=True,
            retain_graph=True,
        )[0][:, 0:1]
        u_yy = torch.autograd.grad(
            u_y.sum(),
            xt_g,
            create_graph=True,
            retain_graph=True,
        )[0][:, 1:2]

        residual = u_t + u * (u_x + u_y) - self.cfg.nu * (u_xx + u_yy)
        res_weight = batch.get("res_weight")
        if res_weight is not None:
            res_weight = res_weight.to(device=xt_col.device, dtype=xt_col.dtype)
            l_res = (res_weight * residual.square()).mean()
        else:
            l_res = residual.square().mean()

        l_sparse = torch.zeros((), dtype=xt_col.dtype, device=xt_col.device)
        l_balance = torch.zeros((), dtype=xt_col.dtype, device=xt_col.device)
        l_gate_supervised = torch.zeros((), dtype=xt_col.dtype, device=xt_col.device)
        l_gate_misroute = torch.zeros((), dtype=xt_col.dtype, device=xt_col.device)
        l_branch_consistency = torch.zeros((), dtype=xt_col.dtype, device=xt_col.device)
        l_base_consistency = torch.zeros((), dtype=xt_col.dtype, device=xt_col.device)
        l_rotation_supervised = torch.zeros((), dtype=xt_col.dtype, device=xt_col.device)
        l_rotation_concentration = torch.zeros((), dtype=xt_col.dtype, device=xt_col.device)
        l_rotation_activation = torch.zeros((), dtype=xt_col.dtype, device=xt_col.device)
        if hasattr(model, "gating"):
            if hasattr(model, "compute_gate_weights"):
                weights = model.compute_gate_weights(xt_col)
            else:
                weights = model.gating(xt_col)
            l_sparse = torch.pow(weights.clamp(min=1e-8), 0.5).sum(dim=-1).mean()
            if hasattr(model, "load_balance_loss"):
                l_balance = model.load_balance_loss(xt_col)
            else:
                l_balance = model.gating.load_balance_loss(xt_col)

            if "xt_gate" in batch and "gate_target_probs" in batch:
                xt_gate = batch["xt_gate"]
                gate_target_probs = batch["gate_target_probs"]
                gate_probs = model.compute_gate_weights(xt_gate)
                sample_kl = torch.nn.functional.kl_div(
                    gate_probs.clamp_min(1e-8).log(),
                    gate_target_probs,
                    reduction="none",
                ).sum(dim=1)
                target_idx = gate_target_probs.argmax(dim=1)
                row_idx = torch.arange(gate_probs.shape[0], device=gate_probs.device)
                target_winner_prob = gate_probs[row_idx, target_idx]
                target_peak = gate_target_probs.max(dim=1).values
                target_agreement = (gate_probs * gate_target_probs).sum(dim=1)
                sample_misroute = (
                    0.7 * (1.0 - target_winner_prob).clamp_min(0.0)
                    + 0.3 * (1.0 - target_agreement).clamp_min(0.0)
                ).pow(self.cfg.gate_misroute_power)
                sample_misroute = sample_misroute * (0.35 + 0.65 * target_peak)
                gate_target_weights = batch.get("gate_target_weights")
                if gate_target_weights is not None:
                    gate_target_weights = gate_target_weights.reshape(-1).to(
                        device=xt_col.device,
                        dtype=xt_col.dtype,
                    )
                    l_gate_supervised = (gate_target_weights * sample_kl).sum() / gate_target_weights.sum().clamp_min(1e-8)
                    l_gate_misroute = (
                        gate_target_weights * sample_misroute
                    ).sum() / gate_target_weights.sum().clamp_min(1e-8)
                else:
                    l_gate_supervised = sample_kl.mean()
                    l_gate_misroute = sample_misroute.mean()

        if (
            hasattr(model, "get_expert_predictions")
            and "xt_branch_consistency" in batch
            and "branch_target_idx" in batch
        ):
            xt_branch = batch["xt_branch_consistency"]
            target_idx = batch["branch_target_idx"].reshape(-1).long().to(device=xt_branch.device)
            pred_branch = model(xt_branch)
            with torch.no_grad():
                expert_branch_preds = model.get_expert_predictions(xt_branch)
                row_idx = torch.arange(expert_branch_preds.shape[0], device=xt_branch.device)
                target_branch = expert_branch_preds[row_idx, target_idx]
            sample_consistency = (pred_branch - target_branch).square().mean(dim=1)
            branch_target_weights = batch.get("branch_target_weights")
            if branch_target_weights is not None:
                branch_target_weights = branch_target_weights.reshape(-1).to(
                    device=xt_branch.device,
                    dtype=xt_branch.dtype,
                )
                l_branch_consistency = (
                    sample_consistency * branch_target_weights
                ).sum() / branch_target_weights.sum().clamp_min(1e-8)
            else:
                l_branch_consistency = sample_consistency.mean()

        if hasattr(model, "base_model") and "xt_base_consistency" in batch:
            xt_base = batch["xt_base_consistency"]
            pred_mixture = model(xt_base)
            with torch.no_grad():
                pred_base = model.base_model(xt_base)
            sample_base_consistency = (pred_mixture - pred_base).square().mean(dim=1)
            base_target_weights = batch.get("base_target_weights")
            if base_target_weights is not None:
                base_target_weights = base_target_weights.reshape(-1).to(
                    device=xt_base.device,
                    dtype=xt_base.dtype,
                )
                l_base_consistency = (
                    sample_base_consistency * base_target_weights
                ).sum() / base_target_weights.sum().clamp_min(1e-8)
            else:
                l_base_consistency = sample_base_consistency.mean()

        if hasattr(model, "get_rotation_state") and "xt_rotation" in batch:
            xt_rotation = batch["xt_rotation"]
            rot_state = model.get_rotation_state(xt_rotation)
            if rot_state is not None and "rotation_target_angle" in batch:
                pred_angle = rot_state["rotation_angle"]
                target_angle = batch["rotation_target_angle"].reshape(-1).to(
                    device=xt_rotation.device,
                    dtype=xt_rotation.dtype,
                )
                rotation_target_weights = batch.get("rotation_target_weights")
                # Axial angle loss: theta and theta + pi represent the same local orientation.
                sample_rotation_supervised = 1.0 - torch.cos(2.0 * (pred_angle - target_angle))
                if rotation_target_weights is not None:
                    rotation_target_weights = rotation_target_weights.reshape(-1).to(
                        device=xt_rotation.device,
                        dtype=xt_rotation.dtype,
                    )
                    l_rotation_supervised = (
                        sample_rotation_supervised * rotation_target_weights
                    ).sum() / rotation_target_weights.sum().clamp_min(1e-8)
                else:
                    l_rotation_supervised = sample_rotation_supervised.mean()

                if "concentration" in rot_state and "rotation_target_concentration" in batch:
                    target_concentration = batch["rotation_target_concentration"].reshape(-1).to(
                        device=xt_rotation.device,
                        dtype=xt_rotation.dtype,
                    )
                    sample_rotation_concentration = (
                        rot_state["concentration"] - target_concentration
                    ).square()
                    if rotation_target_weights is not None:
                        l_rotation_concentration = (
                            sample_rotation_concentration * rotation_target_weights
                        ).sum() / rotation_target_weights.sum().clamp_min(1e-8)
                    else:
                        l_rotation_concentration = sample_rotation_concentration.mean()

                if "activation" in rot_state and "rotation_target_activation" in batch:
                    target_activation = batch["rotation_target_activation"].reshape(-1).to(
                        device=xt_rotation.device,
                        dtype=xt_rotation.dtype,
                    )
                    sample_rotation_activation = (
                        rot_state["activation"] - target_activation
                    ).square()
                    if rotation_target_weights is not None:
                        l_rotation_activation = (
                            sample_rotation_activation * rotation_target_weights
                        ).sum() / rotation_target_weights.sum().clamp_min(1e-8)
                    else:
                        l_rotation_activation = sample_rotation_activation.mean()

        total = (
            self.cfg.w_res * l_res
            + self.cfg.w_ic * l_ic
            + self.cfg.w_bc * l_bc
            + self.cfg.w_sparse * l_sparse
            + self.cfg.w_balance * l_balance
            + self.cfg.w_gate_supervised * l_gate_supervised
            + self.cfg.w_gate_misroute * l_gate_misroute
            + self.cfg.w_branch_consistency * l_branch_consistency
            + self.cfg.w_base_consistency * l_base_consistency
            + self.cfg.w_rotation_supervised * l_rotation_supervised
            + self.cfg.w_rotation_concentration * l_rotation_concentration
            + self.cfg.w_rotation_activation * l_rotation_activation
        )
        return total, {
            "res": l_res,
            "ic": l_ic,
            "bc": l_bc,
            "sparse": l_sparse,
            "balance": l_balance,
            "gate_supervised": l_gate_supervised,
            "gate_misroute": l_gate_misroute,
            "branch_consistency": l_branch_consistency,
            "base_consistency": l_base_consistency,
            "rotation_supervised": l_rotation_supervised,
            "rotation_concentration": l_rotation_concentration,
            "rotation_activation": l_rotation_activation,
        }


def l2_relative_error(u_pred: torch.Tensor, u_exact: torch.Tensor) -> float:
    with torch.no_grad():
        return ((u_pred - u_exact).norm(2) / (u_exact.norm(2) + 1e-10)).item()


def max_absolute_error(u_pred: torch.Tensor, u_exact: torch.Tensor) -> float:
    with torch.no_grad():
        return (u_pred - u_exact).abs().max().item()


def steep_region_metrics(
    u_pred: np.ndarray,
    u_ref: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    quantile: float = 0.9,
) -> dict[str, float]:
    mask, threshold = steep_region_mask(u_ref, x=x, y=y, quantile=quantile)
    abs_err = np.abs(u_pred - u_ref)
    steep_mae = float(abs_err[mask].mean())
    background_mae = float(abs_err[~mask].mean())
    return {
        "steep_threshold": threshold,
        "steep_mae": steep_mae,
        "background_mae": background_mae,
    }
