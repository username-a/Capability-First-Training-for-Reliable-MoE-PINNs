"""
Staged training utilities for Burgers MoE-PINN.

Training proceeds in three stages:
1. Pretrain each expert on a region-focused collocation set.
2. Freeze experts and train the gate from pointwise expert quality targets.
3. Jointly fine-tune the whole MoE end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from training.loss_functions import LossConfig, PhysicsLoss, l2_relative_error
from training.trainer import Trainer


@dataclass
class StagedBurgersConfig:
    expert_steps: int
    gate_steps: int
    joint_steps: int
    base_steps: int
    base_lr: float = 1e-3
    expert_lr: float = 1e-3
    gate_lr: float = 5e-4
    joint_lr: float = 5e-4
    gate_batch_size: int = 2048
    gate_target_temperature: float = 10.0
    gate_balance_weight: float = 1e-2
    shock_band_width: float = 0.14
    transition_band_width: float = 0.40
    expert_sup_points: int = 1200
    expert_sup_weight: float = 12.0
    expert_res_focus: float = 3.0
    expert_log_freq: int = 100

    @classmethod
    def from_total_steps(cls, total_steps: int) -> "StagedBurgersConfig":
        base_steps = max(200, int(total_steps * 0.18))
        expert_steps = max(100, int(total_steps * 0.18))
        gate_steps = max(80, int(total_steps * 0.10))
        joint_steps = max(200, int(total_steps * 0.55))
        return cls(
            base_steps=base_steps,
            expert_steps=expert_steps,
            gate_steps=gate_steps,
            joint_steps=joint_steps,
        )


def _sample_from_mask(
    X: np.ndarray,
    T: np.ndarray,
    mask: np.ndarray,
    n_samples: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coords = np.stack([X[mask], T[mask]], axis=-1)
    if coords.shape[0] == 0:
        raise ValueError("Sampling mask is empty.")
    replace = coords.shape[0] < n_samples
    idx = np.random.choice(coords.shape[0], size=n_samples, replace=replace)
    sample = torch.tensor(coords[idx], dtype=dtype, device=device)
    perm = torch.randperm(sample.shape[0], device=device)
    return sample[perm]


def _shock_centerline(X: np.ndarray, U_exact: np.ndarray) -> np.ndarray:
    x_vals = X[0]
    grad_x = np.abs(np.gradient(U_exact, x_vals, axis=1))
    dx = float(x_vals[1] - x_vals[0])
    margin = max(6.0 * dx, 0.12 * (x_vals.max() - x_vals.min()))
    interior = (x_vals >= x_vals.min() + margin) & (x_vals <= x_vals.max() - margin)
    grad_inner = grad_x[:, interior]
    center_idx_inner = grad_inner.argmax(axis=1)
    center_idx = np.flatnonzero(interior)[center_idx_inner]
    return x_vals[center_idx]


def _sample_supervised_from_mask(
    X: np.ndarray,
    T: np.ndarray,
    U_exact: np.ndarray,
    mask: np.ndarray,
    n_samples: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    coords = np.stack([X[mask], T[mask]], axis=-1)
    values = U_exact[mask][:, None]
    if coords.shape[0] == 0:
        raise ValueError("Supervised sampling mask is empty.")
    replace = coords.shape[0] < n_samples
    idx = np.random.choice(coords.shape[0], size=n_samples, replace=replace)
    xt = torch.tensor(coords[idx], dtype=dtype, device=device)
    u = torch.tensor(values[idx], dtype=dtype, device=device)
    perm = torch.randperm(xt.shape[0], device=device)
    return xt[perm], u[perm]


def _region_scores_np(
    coords: np.ndarray,
    t_grid: np.ndarray,
    center_x: np.ndarray,
    cfg: StagedBurgersConfig,
) -> Dict[str, np.ndarray]:
    x = coords[:, 0]
    t = coords[:, 1]
    center = np.interp(t, t_grid, center_x)
    dist = np.abs(x - center)
    active = (t >= 0.05).astype(np.float64)
    shock = active * np.exp(-np.square(dist / max(cfg.shock_band_width * 0.8, 1e-6)))
    transition_center = max(cfg.shock_band_width * 1.8, cfg.shock_band_width + 1e-6)
    transition_scale = max((cfg.transition_band_width - cfg.shock_band_width) * 0.45, 1e-6)
    transition = active * np.exp(-np.square((dist - transition_center) / transition_scale))
    smooth = 1.0 - np.exp(-np.square(dist / max(cfg.transition_band_width, 1e-6)))
    smooth = np.clip(np.where(t < 0.05, 1.0, smooth), 0.0, 1.0)
    return {
        "shock": shock,
        "smooth": smooth,
        "transition": np.clip(transition, 0.0, 1.0),
    }


def _expert_eval_metrics(
    expert: torch.nn.Module,
    xt_test: torch.Tensor,
    u_exact_flat: torch.Tensor,
    shock_mask_flat: torch.Tensor,
    *,
    base_model: torch.nn.Module | None = None,
    correction_scale: float = 1.0,
) -> Dict[str, float]:
    with torch.no_grad():
        pred = expert(xt_test)
        if base_model is not None:
            pred = base_model(xt_test) + correction_scale * pred
        abs_err = (pred - u_exact_flat).abs()
        shock_mask = shock_mask_flat.bool()
        bg_mask = ~shock_mask
        return {
            "l2": l2_relative_error(pred, u_exact_flat),
            "shock_mae": abs_err[shock_mask].mean().item(),
            "bg_mae": abs_err[bg_mask].mean().item(),
        }


def _expert_physics_loss(
    expert: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    *,
    base_model: torch.nn.Module | None,
    correction_scale: float,
    nu: float,
    res_weight: torch.Tensor,
    sup_weight: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    xt_ic = batch["xt_ic"]
    u_ic = batch["u_ic"]
    xt_bc = batch["xt_bc"]
    u_bc = batch["u_bc"]
    xt_col = batch["xt_col"]
    xt_sup = batch["xt_sup"]
    u_sup = batch["u_sup"]

    if base_model is not None:
        with torch.no_grad():
            base_ic = base_model(xt_ic)
            base_bc = base_model(xt_bc)
            base_sup = base_model(xt_sup)
        pred_ic = base_ic + correction_scale * expert(xt_ic)
        pred_bc = base_bc + correction_scale * expert(xt_bc)
        pred_sup = base_sup + correction_scale * expert(xt_sup)
    else:
        pred_ic = expert(xt_ic)
        pred_bc = expert(xt_bc)
        pred_sup = expert(xt_sup)

    L_ic = ((pred_ic - u_ic) ** 2).mean()
    L_bc = ((pred_bc - u_bc) ** 2).mean()
    L_sup = ((pred_sup - u_sup) ** 2).mean()

    xt_g = xt_col.clone().detach().requires_grad_(True)
    if base_model is not None:
        with torch.no_grad():
            base_g = base_model(xt_g)
        corr = expert(xt_g)
        u_g = base_g + correction_scale * corr
    else:
        u_g = expert(xt_g)
    grads = torch.autograd.grad(u_g.sum(), xt_g, create_graph=True, retain_graph=True)[0]
    u_x, u_t = grads[:, 0:1], grads[:, 1:2]
    u_xx = torch.autograd.grad(u_x.sum(), xt_g, create_graph=True, retain_graph=True)[0][:, 0:1]
    res = u_t + u_g * u_x - nu * u_xx
    weighted_res = res_weight * (res ** 2)
    L_res = weighted_res.mean()

    total = L_res + 10.0 * L_ic + 5.0 * L_bc + sup_weight * L_sup
    return total, {
        "res": L_res.detach(),
        "ic": L_ic.detach(),
        "bc": L_bc.detach(),
        "sup": L_sup.detach(),
    }


def build_specialist_batches(
    X: np.ndarray,
    T: np.ndarray,
    U_exact: np.ndarray,
    base_batch: Dict[str, torch.Tensor],
    *,
    cfg: StagedBurgersConfig,
) -> List[Dict[str, torch.Tensor]]:
    xt_global = base_batch["xt_col"]
    n_total = xt_global.shape[0]
    device = xt_global.device
    dtype = xt_global.dtype

    center_x = _shock_centerline(X, U_exact)
    dist = np.abs(X - center_x[:, None])
    t_mask = T >= 0.05

    shock_mask = t_mask & (dist <= cfg.shock_band_width)
    transition_mask = t_mask & (dist > cfg.shock_band_width) & (
        dist <= cfg.transition_band_width
    )
    smooth_mask = ~shock_mask & ~transition_mask
    center_t = T[:, 0]

    def make_batch(
        region_mask: np.ndarray,
        region_frac: float,
        region_name: str,
    ) -> Dict[str, torch.Tensor]:
        n_region = max(1, int(n_total * region_frac))
        n_global = max(1, n_total - n_region)
        xt_region = _sample_from_mask(
            X,
            T,
            region_mask,
            n_region,
            device=device,
            dtype=dtype,
        )
        global_idx = torch.randperm(n_total, device=device)[:n_global]
        xt_col = torch.cat([xt_region, xt_global[global_idx]], dim=0)
        xt_col = xt_col[torch.randperm(xt_col.shape[0], device=device)]
        xt_sup, u_sup = _sample_supervised_from_mask(
            X,
            T,
            U_exact,
            region_mask,
            cfg.expert_sup_points,
            device=device,
            dtype=dtype,
        )
        scores = _region_scores_np(
            xt_col.detach().cpu().numpy(),
            center_t,
            center_x,
            cfg,
        )
        res_score = scores[region_name]
        res_weight = 0.35 + cfg.expert_res_focus * res_score
        return {
            "name": region_name,
            "xt_col": xt_col,
            "xt_ic": base_batch["xt_ic"],
            "u_ic": base_batch["u_ic"],
            "xt_bc": base_batch["xt_bc"],
            "u_bc": base_batch["u_bc"],
            "xt_sup": xt_sup,
            "u_sup": u_sup,
            "res_weight": torch.tensor(res_weight[:, None], dtype=dtype, device=device),
        }

    return [
        make_batch(shock_mask, region_frac=0.8, region_name="shock"),
        make_batch(smooth_mask, region_frac=0.8, region_name="smooth"),
        make_batch(transition_mask, region_frac=0.7, region_name="transition"),
    ]


def _freeze_module(module: torch.nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad_(False)


def _unfreeze_module(module: torch.nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad_(True)


def pretrain_experts(
    moe_model: torch.nn.Module,
    expert_batches: List[Dict[str, torch.Tensor]],
    *,
    cfg: StagedBurgersConfig,
    nu: float,
    save_dir: str,
    xt_test: torch.Tensor,
    u_exact_flat: torch.Tensor,
    shock_mask_flat: torch.Tensor,
) -> Dict[str, Dict[str, List[float]]]:
    histories: Dict[str, Dict[str, List[float]]] = {}
    eval_freq = max(50, cfg.expert_steps // 5)

    for idx, (expert, batch) in enumerate(zip(moe_model.experts, expert_batches)):
        print("\n" + "-" * 60)
        print(f"Stage A.{idx + 1}: Pretraining expert {idx} ({batch['name']})")
        print("-" * 60)
        optimizer = torch.optim.Adam(expert.parameters(), lr=cfg.expert_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.expert_steps,
            eta_min=cfg.expert_lr * 0.1,
        )
        history: Dict[str, List[float]] = {
            "total": [],
            "res": [],
            "ic": [],
            "bc": [],
            "sup": [],
            "l2_error": [],
            "shock_mae": [],
            "bg_mae": [],
        }
        pbar = tqdm(range(1, cfg.expert_steps + 1), desc=f"Expert{idx}", ncols=100)

        for step in pbar:
            optimizer.zero_grad()
            total_loss, loss_dict = _expert_physics_loss(
                expert,
                batch,
                base_model=getattr(moe_model, "base_model", None),
                correction_scale=float(getattr(moe_model, "correction_scale", 1.0)),
                nu=nu,
                res_weight=batch["res_weight"],
                sup_weight=cfg.expert_sup_weight,
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(expert.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            history["total"].append(total_loss.item())
            for key in ["res", "ic", "bc", "sup"]:
                history[key].append(loss_dict[key].item())

            if step % eval_freq == 0 or step == cfg.expert_steps:
                metrics = _expert_eval_metrics(
                    expert,
                    xt_test,
                    u_exact_flat,
                    shock_mask_flat,
                    base_model=getattr(moe_model, "base_model", None),
                    correction_scale=float(getattr(moe_model, "correction_scale", 1.0)),
                )
                history["l2_error"].append(metrics["l2"])
                history["shock_mae"].append(metrics["shock_mae"])
                history["bg_mae"].append(metrics["bg_mae"])

            if step % cfg.expert_log_freq == 0 or step == cfg.expert_steps:
                latest_l2 = history["l2_error"][-1] if history["l2_error"] else float("nan")
                latest_shock = history["shock_mae"][-1] if history["shock_mae"] else float("nan")
                latest_bg = history["bg_mae"][-1] if history["bg_mae"] else float("nan")
                pbar.set_postfix(
                    {
                        "L": f"{total_loss.item():.3e}",
                        "res": f"{loss_dict['res'].item():.3e}",
                        "sup": f"{loss_dict['sup'].item():.3e}",
                        "l2": f"{latest_l2:.3e}",
                        "shock": f"{latest_shock:.3e}",
                        "bg": f"{latest_bg:.3e}",
                    }
                )

        print(
            f"[OK] Expert {idx} done | final total={history['total'][-1]:.4e} "
            f"| final l2={history['l2_error'][-1]:.4e}"
        )
        histories[f"expert_{idx}"] = history
        torch.save(
            {"model_state": expert.state_dict(), "history": history},
            f"{save_dir}/burgers_expert_{idx}_pretrain.pt",
        )
        print(f"[OK] Saved checkpoint: {save_dir}/burgers_expert_{idx}_pretrain.pt")

    return histories


def pretrain_base_model(
    moe_model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    *,
    cfg: StagedBurgersConfig,
    nu: float,
    save_dir: str,
    xt_test: torch.Tensor,
    u_exact_flat: torch.Tensor,
) -> Dict[str, List[float]]:
    if not hasattr(moe_model, "base_model"):
        return {}

    print("\n" + "-" * 60)
    print("Stage 0: Pretraining base model")
    print("-" * 60)

    loss_fn = PhysicsLoss(
        LossConfig(
            equation="burgers",
            nu=nu,
            use_ntk=False,
            w_res=1.0,
            w_ic=10.0,
            w_bc=5.0,
        )
    )

    def eval_l2() -> float:
        with torch.no_grad():
            return l2_relative_error(moe_model.base_model(xt_test), u_exact_flat)

    trainer = Trainer(
        moe_model.base_model,
        loss_fn,
        lr=cfg.base_lr,
        n_steps=cfg.base_steps,
        use_ntk=False,
        device=xt_test.device,
        save_dir=save_dir,
    )
    history = trainer.train(
        batch,
        eval_fn=eval_l2,
        eval_freq=max(100, cfg.base_steps // 4),
    )
    torch.save(
        {"model_state": moe_model.base_model.state_dict(), "history": history},
        f"{save_dir}/burgers_base_pretrain.pt",
    )
    print(f"[OK] Saved checkpoint: {save_dir}/burgers_base_pretrain.pt")
    return history


def build_gate_targets(
    moe_model: torch.nn.Module,
    xt_gate: torch.Tensor,
    u_exact_gate: torch.Tensor,
    *,
    temperature: float,
    cfg: StagedBurgersConfig,
) -> torch.Tensor:
    with torch.no_grad():
        if hasattr(moe_model, "get_expert_predictions"):
            expert_preds = moe_model.get_expert_predictions(xt_gate)
        else:
            expert_preds = torch.stack([expert(xt_gate) for expert in moe_model.experts], dim=1)
        point_errors = (expert_preds - u_exact_gate.unsqueeze(1)).abs().squeeze(-1)
        rel_errors = point_errors / (point_errors.mean(dim=1, keepdim=True) + 1e-8)
        perf_probs = torch.softmax(-temperature * rel_errors, dim=1)

        x_coord = xt_gate[:, 0]
        t_coord = xt_gate[:, 1]
        center_focus = torch.exp(-torch.square(x_coord / cfg.shock_band_width))
        ring_focus = torch.exp(
            -torch.square((torch.abs(x_coord) - cfg.shock_band_width * 1.6) / max(cfg.transition_band_width * 0.45, 1e-6))
        )
        time_focus = torch.sigmoid((t_coord - 0.08) * 10.0)

        shock_prior = 0.10 + 0.90 * center_focus * (0.30 + 0.70 * time_focus)
        smooth_prior = 0.20 + 0.80 * (1.0 - center_focus)
        transition_prior = 0.15 + 0.85 * ring_focus * (0.25 + 0.75 * time_focus)
        prior = torch.stack([shock_prior, smooth_prior, transition_prior], dim=1).clamp_min(1e-4)

        targets = perf_probs * prior.pow(1.2)
        return targets / targets.sum(dim=1, keepdim=True).clamp_min(1e-8)


def train_gate(
    moe_model: torch.nn.Module,
    xt_gate: torch.Tensor,
    target_probs: torch.Tensor,
    *,
    cfg: StagedBurgersConfig,
) -> Dict[str, List[float]]:
    print("\n" + "-" * 60)
    print("Stage B: Training gating network with frozen experts")
    print("-" * 60)

    for expert in moe_model.experts:
        _freeze_module(expert)
    _unfreeze_module(moe_model.gating)

    optimizer = torch.optim.Adam(moe_model.gating.parameters(), lr=cfg.gate_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.gate_steps,
        eta_min=cfg.gate_lr * 0.1,
    )

    history: Dict[str, List[float]] = {
        "loss": [],
        "match": [],
        "balance": [],
        "entropy": [],
        "max_gate": [],
    }

    n_samples = xt_gate.shape[0]
    batch_size = min(cfg.gate_batch_size, n_samples)
    pbar = tqdm(range(1, cfg.gate_steps + 1), desc="Gate", ncols=100)

    for step in pbar:
        idx = torch.randperm(n_samples, device=xt_gate.device)[:batch_size]
        x_batch = xt_gate[idx]
        y_batch = target_probs[idx]

        optimizer.zero_grad()
        probs = moe_model.compute_gate_weights(x_batch)
        match_loss = F.kl_div(
            probs.clamp_min(1e-8).log(),
            y_batch,
            reduction="batchmean",
        )
        balance_loss = moe_model.load_balance_loss(x_batch)
        total_loss = match_loss + cfg.gate_balance_weight * balance_loss
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1).mean().item()
        max_gate = probs.max(dim=-1).values.mean().item()

        history["loss"].append(total_loss.item())
        history["match"].append(match_loss.item())
        history["balance"].append(balance_loss.item())
        history["entropy"].append(entropy)
        history["max_gate"].append(max_gate)

        if step % max(50, cfg.gate_steps // 8) == 0:
            stats = moe_model.load_balance_stats(x_batch)
            pbar.set_postfix(
                {
                    "L": f"{total_loss.item():.3e}",
                    "match": f"{match_loss.item():.3e}",
                    "bal": f"{balance_loss.item():.3e}",
                    "gmax": f"{stats['max_gate_weight']:.2f}",
                    "gent": f"{stats['mean_entropy']:.2f}",
                }
            )

    for expert in moe_model.experts:
        _unfreeze_module(expert)
    return history


def joint_finetune(
    moe_model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    *,
    cfg: StagedBurgersConfig,
    nu: float,
    use_ntk: bool,
    device: torch.device,
    save_dir: str,
    xt_test: torch.Tensor,
    u_exact_flat: torch.Tensor,
) -> Dict[str, List[float]]:
    print("\n" + "-" * 60)
    print("Stage C: Joint fine-tuning")
    print("-" * 60)

    loss_fn = PhysicsLoss(
        LossConfig(
            equation="burgers",
            nu=nu,
            use_ntk=use_ntk,
            w_res=1.0,
            w_ic=10.0,
            w_bc=5.0,
        )
    )

    def eval_l2() -> float:
        with torch.no_grad():
            return l2_relative_error(moe_model(xt_test), u_exact_flat)

    trainer = Trainer(
        moe_model,
        loss_fn,
        lr=cfg.joint_lr,
        n_steps=cfg.joint_steps,
        ntk_update_freq=200,
        use_ntk=use_ntk,
        device=device,
        save_dir=save_dir,
    )
    history = trainer.train(
        batch,
        eval_fn=eval_l2,
        eval_freq=max(200, cfg.joint_steps // 6),
    )
    trainer.save_checkpoint("burgers_moe_staged")
    return history
