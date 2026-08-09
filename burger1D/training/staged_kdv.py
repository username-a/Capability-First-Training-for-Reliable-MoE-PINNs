"""
Staged training machinery for KdV two-soliton MoE-PINN.

The KdV MoE is a plain mixture (no base model): three experts
(dispersion / smooth / shock) + gate. Staged protocol:
    f in [0,1]  : experts pretrained to fraction f on region-focused batches
    f = 1       : gate trained on expert-error pseudo-labels, then joint FT
    f < 1       : gate introduced cold, experts+gate jointly trained (base-less)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from equations.kdv import KdVEquation
from training.loss_functions import LossConfig, PhysicsLoss, l2_relative_error
from training.trainer import Trainer


@dataclass
class StagedKdvConfig:
    expert_steps: int
    gate_steps: int
    joint_steps: int
    expert_pretrain_fraction: float = 1.0
    expert_lr: float = 1e-3
    gate_lr: float = 5e-4
    joint_lr: float = 5e-4
    expert_sup_weight: float = 8.0
    expert_res_focus: float = 3.0
    expert_sup_points: int = 1500
    gate_target_temperature: float = 8.0
    gate_steps_use: int = 0

    @classmethod
    def from_total_steps(cls, total_steps: int) -> "StagedKdvConfig":
        return cls(
            expert_steps=max(50, int(total_steps * 0.22)),
            gate_steps=max(30, int(total_steps * 0.10)),
            joint_steps=max(100, int(total_steps * 0.55)),
        )


def _region_scores(X: np.ndarray, T: np.ndarray, U: np.ndarray) -> Dict[str, np.ndarray]:
    """Region scores from the Hirota exact solution (shape (nt, nx))."""
    x = X[0, :]
    amp = np.abs(U)
    amp_n = amp / (amp.max() + 1e-8)
    ux = np.gradient(U, x, axis=1)
    uxx = np.gradient(ux, x, axis=1)
    curv_n = np.abs(uxx) / (np.abs(uxx).max() + 1e-8)
    t_vals = T[:, 0]
    time_w = (t_vals / max(float(t_vals.max()), 1e-8))[:, None]
    smooth = np.clip(1.1 - amp_n, 0.0, 1.5)
    shock = np.clip(amp_n * (0.7 + 0.5 * np.clip(curv_n, 0, 1.5)), 0.0, 1.5)
    dispersion = np.clip(curv_n * (0.3 + 0.7 * time_w), 0.0, 1.5)
    return {"smooth": smooth, "shock": shock, "dispersion": dispersion}


def _sample_from_mask(mask: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        idx = np.arange(mask.size)
    replace = len(idx) < n
    return rng.choice(idx, size=n, replace=replace)


def build_kdv_specialist_batches(
    X: np.ndarray,
    T: np.ndarray,
    U_exact: np.ndarray,
    batch: Dict[str, torch.Tensor],
    cfg: StagedKdvConfig,
    seed: int = 42,
) -> List[Dict[str, torch.Tensor]]:
    rng = np.random.default_rng(seed)
    scores = _region_scores(X, T, U_exact)
    device = batch["xt_col"].device
    dtype = batch["xt_col"].dtype
    flat = {k: v.reshape(-1) for k, v in scores.items()}
    total_col = batch["xt_col"].shape[0]
    xt_col = batch["xt_col"]

    batches = []
    for name in ["dispersion", "smooth", "shock"]:
        score = flat[name]
        threshold = float(np.quantile(score, 0.70 if name != "smooth" else 0.60))
        focus_mask = score >= threshold
        n_focus = max(1, int(total_col * 0.60))
        n_global = max(1, total_col - n_focus)
        xyt = np.stack([X.ravel(), T.ravel()], axis=-1)
        focus_idx = _sample_from_mask(focus_mask, n_focus, rng)
        global_idx = rng.choice(total_col, size=n_global, replace=False)
        all_idx = np.concatenate([focus_idx, global_idx])
        rng.shuffle(all_idx)
        focus_scores = torch.tensor(score[focus_idx][:, None], dtype=dtype, device=device)
        res_weight = torch.cat(
            [
                0.3 + cfg.expert_res_focus * focus_scores,
                torch.ones(n_global, 1, dtype=dtype, device=device),
            ],
            dim=0,
        )
        xt_col_i = torch.tensor(xyt[all_idx], dtype=dtype, device=device)
        sup_mask = focus_mask
        sup_idx = _sample_from_mask(sup_mask, cfg.expert_sup_points, rng)
        xt_sup = torch.tensor(xyt[sup_idx], dtype=dtype, device=device)
        u_sup = torch.tensor(U_exact.ravel()[sup_idx][:, None], dtype=dtype, device=device)
        batches.append(
            {
                "name": name,
                "xt_col": xt_col_i,
                "res_weight": res_weight,
                "xt_sup": xt_sup,
                "u_sup": u_sup,
            }
        )
    return batches


def _freeze(module: nn.Module | None) -> None:
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad_(False)


def _unfreeze(module: nn.Module | None) -> None:
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad_(True)


def pretrain_kdv_experts(
    model: nn.Module,
    batches: List[Dict[str, torch.Tensor]],
    cfg: StagedKdvConfig,
    eq: KdVEquation,
    xt_test: torch.Tensor,
    u_exact_flat: torch.Tensor,
    save_dir: str,
) -> Dict[str, Dict[str, List[float]]]:
    histories: Dict[str, Dict[str, List[float]]] = {}
    names = ["dispersion", "smooth", "shock"]
    for idx, (expert, batch) in enumerate(zip(model.experts, batches)):
        print(f"Stage A.{idx+1}: pretraining expert {idx} ({batch['name']})", flush=True)
        local_steps = max(1, int(cfg.expert_steps * cfg.expert_pretrain_fraction))
        opt = torch.optim.Adam(expert.parameters(), lr=cfg.expert_lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=local_steps, eta_min=cfg.expert_lr * 0.1)
        history = {"total": [], "res": [], "sup": [], "l2": []}
        for step in range(1, local_steps + 1):
            opt.zero_grad()
            xt_col = batch["xt_col"]
            xt_col.requires_grad_(True)
            u_pred = expert(xt_col)
            res = eq.pde_residual(u_pred, xt_col)
            res_loss = ((batch["res_weight"] * res**2)).mean()
            sup_loss = ((expert(batch["xt_sup"]) - batch["u_sup"]) ** 2).mean()
            total = res_loss + cfg.expert_sup_weight * sup_loss
            total.backward()
            torch.nn.utils.clip_grad_norm_(expert.parameters(), max_norm=1.0)
            opt.step()
            sched.step()
            history["total"].append(total.item())
            history["res"].append(res_loss.item())
            history["sup"].append(sup_loss.item())
            if step % max(10, local_steps // 4) == 0 or step == local_steps:
                with torch.no_grad():
                    l2 = float(l2_relative_error(expert(xt_test), u_exact_flat))
                history["l2"].append(l2)
        histories[batch["name"]] = history
        torch.save(
            {"model_state": expert.state_dict(), "history": history},
            f"{save_dir}/kdv_expert_{idx}_{batch['name']}.pt",
        )
    return histories


def build_kdv_gate_targets(
    model: nn.Module,
    xt_test: torch.Tensor,
    u_exact_flat: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    with torch.no_grad():
        preds = torch.stack([e(xt_test) for e in model.experts], dim=1).squeeze(-1)
        err = (preds - u_exact_flat).abs()
        rel = err / (err.mean(dim=1, keepdim=True) + 1e-8)
        targets = torch.softmax(-temperature * rel, dim=1)
    return targets


def train_kdv_gate(
    model: nn.Module,
    xt_test: torch.Tensor,
    targets: torch.Tensor,
    cfg: StagedKdvConfig,
) -> None:
    print("Stage B: training KdV gate", flush=True)
    _freeze_others(model, only_gate=True)
    opt = torch.optim.Adam(model.gating.parameters(), lr=cfg.gate_lr)
    n = targets.shape[0]
    for step in range(cfg.gate_steps):
        opt.zero_grad()
        idx = torch.randperm(n, device=targets.device)[: min(2048, n)]
        g = model.gating(xt_test[idx])
        loss = ((g - targets[idx]) ** 2).mean()
        loss.backward()
        opt.step()
    _unfreeze_others(model)


def _freeze_others(model: nn.Module, only_gate: bool = True) -> None:
    for e in model.experts:
        _freeze(e)
    if not only_gate:
        _freeze(model.gating)


def _unfreeze_others(model: nn.Module) -> None:
    for e in model.experts:
        _unfreeze(e)
    _unfreeze(model.gating)


def joint_finetune_kdv(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    cfg: StagedKdvConfig,
    eq: KdVEquation,
    xt_test: torch.Tensor,
    u_exact_flat: torch.Tensor,
    save_dir: str,
) -> Dict[str, List[float]]:
    print("Stage C: KdV joint fine-tuning", flush=True)
    loss_fn = PhysicsLoss(LossConfig(equation="kdv", use_ntk=False, w_res=1.0, w_ic=10.0, w_bc=5.0))

    def eval_l2() -> float:
        with torch.no_grad():
            return l2_relative_error(model(xt_test), u_exact_flat)

    trainer = Trainer(
        model, loss_fn, lr=cfg.joint_lr, n_steps=cfg.joint_steps,
        device=xt_test.device, save_dir=save_dir,
    )
    history = trainer.train(batch, eval_fn=eval_l2, eval_freq=max(50, cfg.joint_steps // 5))
    trainer.save_checkpoint("kdv_moe_staged")
    return history
