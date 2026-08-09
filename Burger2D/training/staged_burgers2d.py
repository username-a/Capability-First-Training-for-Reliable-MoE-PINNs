"""
Staged training utilities for Burger2D MoE-PINN.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from Burger2D.data.expert_dataset import EXPERT_REGION_NAMES
from Burger2D.equations.burgers2d import ReferenceSolution2D, steep_region_mask
from Burger2D.training.losses import LossConfig2D, PhysicsLoss2D, l2_relative_error
from Burger2D.training.trainer import Trainer


def _profile_value(profile: Dict[str, float] | None, expert_name: str, default: float) -> float:
    if profile is None:
        return default
    return float(profile.get(expert_name, default))


@dataclass
class StagedBurgers2DConfig:
    expert_steps: int
    gate_steps: int
    joint_steps: int
    base_steps: int
    rotation_steps: int
    rotation_lr: float = 5e-4
    base_lr: float = 1e-3
    expert_lr: float = 1e-3
    gate_lr: float = 5e-4
    joint_lr: float = 5e-4
    gate_batch_size: int = 4096
    gate_target_temperature: float = 8.0
    gate_confidence_quantile: float = 0.75
    gate_min_confidence: float = 0.12
    gate_confidence_power: float = 2.0
    gate_min_keep_ratio: float = 0.25
    gate_balance_weight: float = 0.0
    gate_region_prior_power: float = 1.15
    gate_branch_superiority_bonus: float = 0.0
    gate_branch_superiority_power: float = 1.25
    gate_branch_alignment_power: float = 1.0
    gate_preferred_expert_name: str | None = None
    gate_preferred_expert_bias: float = 0.0
    gate_preferred_expert_power: float = 1.25
    gate_preferred_sample_boost: float = 0.0
    gate_preferred_sample_power: float = 1.15
    directional_gate_bias: float = 0.0
    directional_gate_bias_power: float = 1.25
    gate_shock_bias: float = 0.85
    gate_shock_power: float = 1.25
    gate_shock_quantile: float = 0.82
    gate_min_shock_weight: float = 0.20
    joint_gate_supervised_weight: float = 0.35
    joint_gate_misroute_weight: float = 0.0
    joint_gate_misroute_power: float = 2.0
    joint_gate_supervision_points: int = 16384
    base_batch_refresh_freq: int = 30
    joint_batch_refresh_freq: int = 20
    expert_sup_points: int = 1800
    expert_sup_weight: float = 8.0
    expert_focus_ratio: float = 0.75
    expert_res_focus: float = 3.0
    expert_log_freq: int = 50
    expert_step_scale: Dict[str, float] | None = None
    expert_sup_weight_scale: Dict[str, float] | None = None
    expert_focus_ratio_scale: Dict[str, float] | None = None
    expert_res_focus_scale: Dict[str, float] | None = None
    expert_sup_points_scale: Dict[str, float] | None = None
    curated_weight_power: Dict[str, float] | None = None
    curated_confidence_bias: Dict[str, float] | None = None
    expert_dataset_dir: str | None = None
    use_gate_region_prior: bool = False
    joint_sparse_weight: float = 0.0
    joint_balance_weight: float = 0.0
    joint_train_mode: str = "all"
    joint_focus_ratio: float = 0.65
    joint_focus_quantile: float = 0.82
    joint_focus_power: float = 1.75
    joint_res_focus: float = 2.5
    joint_directional_bonus: float = 0.35
    joint_branch_consistency_weight: float = 0.0
    joint_branch_consistency_points: int = 0
    joint_branch_consistency_margin: float = 0.08
    joint_branch_consistency_power: float = 1.75
    joint_branch_consistency_focus_expert: str | None = None
    joint_branch_consistency_region_power: float = 1.25
    joint_branch_consistency_min_align: float = 0.15
    joint_base_consistency_weight: float = 0.0
    joint_base_consistency_points: int = 0
    joint_base_consistency_smooth_power: float = 1.5
    joint_base_consistency_shock_power: float = 1.25
    joint_base_consistency_min_smooth: float = 0.25
    joint_base_consistency_guard_advantage: bool = False
    joint_base_consistency_base_margin: float = 0.05
    joint_base_consistency_expert_adv_margin: float = 0.10
    joint_base_consistency_advantage_region_max: float = 0.45
    joint_base_consistency_advantage_power: float = 1.5
    joint_base_consistency_use_uncertainty: bool = False
    joint_base_consistency_uncertainty_quantile: float = 0.60
    joint_base_consistency_min_uncertainty: float = 0.30
    joint_base_consistency_uncertainty_power: float = 1.35
    joint_rotation_supervision_weight: float = 0.0
    joint_rotation_concentration_weight: float = 0.0
    joint_rotation_activation_weight: float = 0.0
    joint_rotation_supervision_points: int = 0
    joint_rotation_grad_quantile: float = 0.78
    joint_rotation_grad_power: float = 1.35
    joint_rotation_region_power: float = 1.30
    joint_rotation_min_region: float = 0.10
    joint_rotation_wave_bonus: float = 0.20
    joint_rotation_activation_floor_target: float = 0.05
    joint_rotation_activation_cap: float = 1.0
    joint_rotation_concentration_floor_target: float = 0.25
    joint_rotation_concentration_cap: float = 1.0
    joint_rotation_negative_ratio: float = 0.0
    joint_rotation_negative_max_region: float = 0.08
    joint_rotation_negative_grad_quantile: float = 0.55
    joint_rotation_negative_activation: float = 0.02
    joint_rotation_negative_concentration: float = 0.04
    joint_rotation_negative_weight_scale: float = 0.60

    @classmethod
    def from_total_steps(cls, total_steps: int) -> "StagedBurgers2DConfig":
        base_steps = max(20, int(total_steps * 0.18))
        expert_steps = max(20, int(total_steps * 0.20))
        gate_steps = max(10, int(total_steps * 0.12))
        joint_steps = max(20, total_steps - base_steps - expert_steps - gate_steps)
        return cls(
            rotation_steps=max(20, int(total_steps * 0.10)),
            base_steps=base_steps,
            expert_steps=expert_steps,
            gate_steps=gate_steps,
            joint_steps=joint_steps,
            rotation_lr=5e-4,
            use_gate_region_prior=False,
            gate_region_prior_power=1.15,
            gate_branch_superiority_bonus=0.0,
            gate_branch_superiority_power=1.25,
            gate_branch_alignment_power=1.0,
            gate_preferred_expert_name=None,
            gate_preferred_expert_bias=0.0,
            gate_preferred_expert_power=1.25,
            gate_preferred_sample_boost=0.0,
            gate_preferred_sample_power=1.15,
            directional_gate_bias=0.0,
            directional_gate_bias_power=1.25,
            gate_shock_bias=0.85,
            gate_shock_power=1.25,
            gate_shock_quantile=0.82,
            gate_min_shock_weight=0.20,
            joint_gate_supervised_weight=0.35,
            joint_gate_misroute_weight=0.0,
            joint_gate_misroute_power=2.0,
            joint_gate_supervision_points=16384,
            base_batch_refresh_freq=30,
            joint_batch_refresh_freq=20,
            joint_focus_ratio=0.65,
            joint_focus_quantile=0.82,
            joint_focus_power=1.75,
            joint_res_focus=2.5,
            joint_directional_bonus=0.35,
            joint_branch_consistency_weight=0.0,
            joint_branch_consistency_points=0,
            joint_branch_consistency_margin=0.08,
            joint_branch_consistency_power=1.75,
            joint_branch_consistency_focus_expert=None,
            joint_branch_consistency_region_power=1.25,
            joint_branch_consistency_min_align=0.15,
            joint_base_consistency_weight=0.0,
            joint_base_consistency_points=0,
            joint_base_consistency_smooth_power=1.5,
            joint_base_consistency_shock_power=1.25,
            joint_base_consistency_min_smooth=0.25,
            joint_base_consistency_guard_advantage=False,
            joint_base_consistency_base_margin=0.05,
            joint_base_consistency_expert_adv_margin=0.10,
            joint_base_consistency_advantage_region_max=0.45,
            joint_base_consistency_advantage_power=1.5,
            joint_base_consistency_use_uncertainty=False,
            joint_base_consistency_uncertainty_quantile=0.60,
            joint_base_consistency_min_uncertainty=0.30,
            joint_base_consistency_uncertainty_power=1.35,
            joint_rotation_supervision_weight=0.0,
            joint_rotation_concentration_weight=0.0,
            joint_rotation_activation_weight=0.0,
            joint_rotation_supervision_points=0,
            joint_rotation_grad_quantile=0.78,
            joint_rotation_grad_power=1.35,
            joint_rotation_region_power=1.30,
            joint_rotation_min_region=0.10,
            joint_rotation_wave_bonus=0.20,
            joint_rotation_activation_floor_target=0.05,
            joint_rotation_activation_cap=1.0,
            joint_rotation_concentration_floor_target=0.25,
            joint_rotation_concentration_cap=1.0,
            joint_rotation_negative_ratio=0.0,
            joint_rotation_negative_max_region=0.08,
            joint_rotation_negative_grad_quantile=0.55,
            joint_rotation_negative_activation=0.02,
            joint_rotation_negative_concentration=0.04,
            joint_rotation_negative_weight_scale=0.60,
            expert_step_scale={
                "smooth": 0.85,
                "iso_shock": 1.25,
                "directional_shock": 1.45,
                "wave": 1.20,
                "normal_gradient": 1.25,
                "anisotropy_directional": 1.45,
                "curvature_wave": 1.20,
            },
            expert_sup_weight_scale={
                "smooth": 0.75,
                "iso_shock": 1.20,
                "directional_shock": 1.45,
                "wave": 1.15,
                "normal_gradient": 1.20,
                "anisotropy_directional": 1.45,
                "curvature_wave": 1.15,
            },
            expert_focus_ratio_scale={
                "smooth": 0.75,
                "iso_shock": 1.10,
                "directional_shock": 1.18,
                "wave": 1.08,
                "normal_gradient": 1.10,
                "anisotropy_directional": 1.18,
                "curvature_wave": 1.08,
            },
            expert_res_focus_scale={
                "smooth": 0.80,
                "iso_shock": 1.20,
                "directional_shock": 1.40,
                "wave": 1.18,
                "normal_gradient": 1.20,
                "anisotropy_directional": 1.40,
                "curvature_wave": 1.18,
            },
            expert_sup_points_scale={
                "smooth": 0.75,
                "iso_shock": 1.20,
                "directional_shock": 1.35,
                "wave": 1.10,
                "normal_gradient": 1.20,
                "anisotropy_directional": 1.35,
                "curvature_wave": 1.10,
            },
            curated_weight_power={
                "smooth": 1.00,
                "iso_shock": 1.25,
                "directional_shock": 1.45,
                "wave": 1.20,
                "normal_gradient": 1.25,
                "anisotropy_directional": 1.45,
                "curvature_wave": 1.20,
            },
            curated_confidence_bias={
                "smooth": 0.20,
                "iso_shock": 0.45,
                "directional_shock": 0.65,
                "wave": 0.50,
                "normal_gradient": 0.45,
                "anisotropy_directional": 0.65,
                "curvature_wave": 0.50,
            },
        )


def _expert_focus_ratio(cfg: StagedBurgers2DConfig, expert_name: str) -> float:
    return float(
        np.clip(
            cfg.expert_focus_ratio * _profile_value(cfg.expert_focus_ratio_scale, expert_name, 1.0),
            0.35,
            0.96,
        )
    )


def _expert_res_focus(cfg: StagedBurgers2DConfig, expert_name: str) -> float:
    return float(max(0.6, cfg.expert_res_focus * _profile_value(cfg.expert_res_focus_scale, expert_name, 1.0)))


def _expert_sup_points(cfg: StagedBurgers2DConfig, expert_name: str) -> int:
    return int(max(128, round(cfg.expert_sup_points * _profile_value(cfg.expert_sup_points_scale, expert_name, 1.0))))


def _expert_sup_weight(cfg: StagedBurgers2DConfig, expert_name: str) -> float:
    # A zero supervision weight is a semantic switch for physics-only runs.
    # Do not silently re-enable reference supervision through a positive floor.
    raw_weight = cfg.expert_sup_weight * _profile_value(cfg.expert_sup_weight_scale, expert_name, 1.0)
    if raw_weight <= 0.0:
        return 0.0
    return float(max(0.5, raw_weight))


def _expert_steps(cfg: StagedBurgers2DConfig, expert_name: str) -> int:
    return int(max(20, round(cfg.expert_steps * _profile_value(cfg.expert_step_scale, expert_name, 1.0))))


def flatten_reference_solution(
    reference: ReferenceSolution2D,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    xx, yy = np.meshgrid(reference.x, reference.y, indexing="xy")
    coords = []
    for t_value in reference.t:
        tt = np.full_like(xx, t_value)
        coords.append(np.stack([xx, yy, tt], axis=-1).reshape(-1, 3))
    xyt = np.concatenate(coords, axis=0)
    u = reference.u.reshape(-1, 1)
    return (
        torch.tensor(xyt, dtype=dtype, device=device),
        torch.tensor(u, dtype=dtype, device=device),
    )


def _build_region_prior_tensor(
    expert_names: List[str],
    region_scores: Dict[str, np.ndarray],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    default_region_key = next(iter(region_scores))
    priors = []
    for expert_name in expert_names:
        region_key = expert_name if expert_name in region_scores else default_region_key
        priors.append(
            torch.tensor(
                region_scores[region_key].reshape(-1),
                dtype=dtype,
                device=device,
            )
        )
    return torch.stack(priors, dim=1)


def compute_region_scores(
    reference: ReferenceSolution2D,
    *,
    layout_variant: str = "categorical",
) -> Dict[str, np.ndarray]:
    u_ref = reference.u.astype(np.float64)
    x = reference.x.astype(np.float64)
    y = reference.y.astype(np.float64)
    t = reference.t.astype(np.float64)

    grad_mag = np.zeros_like(u_ref, dtype=np.float64)
    lap_abs = np.zeros_like(u_ref, dtype=np.float64)
    mixed_abs = np.zeros_like(u_ref, dtype=np.float64)

    for i in range(u_ref.shape[0]):
        u_slice = u_ref[i]
        u_y, u_x = np.gradient(u_slice, y, x, edge_order=1)
        u_xx = np.gradient(u_x, x, axis=1, edge_order=1)
        u_yy = np.gradient(u_y, y, axis=0, edge_order=1)
        u_xy = np.gradient(u_x, y, axis=0, edge_order=1)
        grad_mag[i] = np.sqrt(u_x**2 + u_y**2)
        lap_abs[i] = np.abs(u_xx + u_yy)
        mixed_abs[i] = np.abs(u_xy)

    grad_norm = grad_mag / (np.quantile(grad_mag, 0.90) + 1e-8)
    lap_norm = lap_abs / (np.quantile(lap_abs, 0.90) + 1e-8)
    mixed_norm = mixed_abs / (np.quantile(mixed_abs, 0.90) + 1e-8)
    time_weight = (t / max(float(t.max()), 1e-8))[:, None, None]

    smooth = np.clip(1.1 - grad_norm, 0.0, 1.5)
    if layout_variant == "categorical":
        iso_shock = np.clip(grad_norm * (1.2 - 0.7 * np.clip(mixed_norm, 0.0, 1.5)), 0.0, 1.5)
        directional = np.clip(
            grad_norm * np.clip(1.1 * mixed_norm, 0.0, 1.5) * (0.35 + 0.65 * time_weight),
            0.0,
            1.5,
        )
        wave = np.clip((lap_norm - 0.35 * grad_norm) * (0.25 + 0.75 * time_weight), 0.0, 1.5)
        return {
            "smooth": smooth.astype(np.float32),
            "iso_shock": iso_shock.astype(np.float32),
            "directional_shock": directional.astype(np.float32),
            "wave": wave.astype(np.float32),
        }
    if layout_variant == "attribute":
        normal_gradient = np.clip(
            grad_norm * (0.85 + 0.15 * time_weight) * (1.10 - 0.30 * np.clip(lap_norm, 0.0, 1.5)),
            0.0,
            1.8,
        )
        anisotropy = np.clip(
            grad_norm * np.clip(1.20 * mixed_norm, 0.0, 1.6) * (0.30 + 0.70 * time_weight),
            0.0,
            1.8,
        )
        curvature = np.clip(
            (0.90 * lap_norm + 0.25 * mixed_norm - 0.20 * grad_norm) * (0.25 + 0.75 * time_weight),
            0.0,
            1.8,
        )
        return {
            "smooth": smooth.astype(np.float32),
            "normal_gradient": normal_gradient.astype(np.float32),
            "anisotropy_directional": anisotropy.astype(np.float32),
            "curvature_wave": curvature.astype(np.float32),
        }
    raise ValueError(f"Unknown layout variant: {layout_variant}")


def shock_priority_from_region_scores(
    region_scores: Dict[str, np.ndarray],
    *,
    directional_bonus: float = 0.0,
    layout_variant: str = "categorical",
) -> np.ndarray:
    if layout_variant == "categorical":
        iso = region_scores["iso_shock"].reshape(-1)
        directional = region_scores["directional_shock"].reshape(-1) * (1.0 + directional_bonus)
        wave = region_scores["wave"].reshape(-1)
        return np.maximum.reduce([iso, directional, wave]).astype(np.float32)
    if layout_variant == "attribute":
        normal_gradient = region_scores["normal_gradient"].reshape(-1)
        anisotropy = region_scores["anisotropy_directional"].reshape(-1) * (1.0 + directional_bonus)
        curvature = region_scores["curvature_wave"].reshape(-1)
        return np.maximum.reduce([normal_gradient, anisotropy, curvature]).astype(np.float32)
    raise ValueError(f"Unknown layout variant: {layout_variant}")


def compute_rotation_reference_maps(reference: ReferenceSolution2D) -> Dict[str, np.ndarray]:
    u_ref = reference.u.astype(np.float64)
    x = reference.x.astype(np.float64)
    y = reference.y.astype(np.float64)

    grad_mag = np.zeros_like(u_ref, dtype=np.float64)
    grad_angle = np.zeros_like(u_ref, dtype=np.float64)
    for i in range(u_ref.shape[0]):
        u_slice = u_ref[i]
        u_y, u_x = np.gradient(u_slice, y, x, edge_order=1)
        grad_mag[i] = np.sqrt(u_x**2 + u_y**2)
        grad_angle[i] = np.arctan2(u_y, u_x)

    grad_norm = grad_mag / (np.quantile(grad_mag, 0.90) + 1e-8)
    return {
        "grad_mag": grad_mag.astype(np.float32),
        "grad_norm": grad_norm.astype(np.float32),
        "grad_angle": grad_angle.astype(np.float32),
    }


def _sample_indices_from_mask(mask: np.ndarray, n_samples: int) -> np.ndarray:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError("Sampling mask is empty.")
    replace = indices.size < n_samples
    return np.random.choice(indices, size=n_samples, replace=replace)


def _sample_indices_from_scores(scores: np.ndarray, n_samples: int, power: float = 1.0) -> np.ndarray:
    flat_scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    weights = np.clip(flat_scores, 0.0, None)
    if power != 1.0:
        weights = np.power(weights + 1e-8, power)
    weights = weights + 1e-8
    weights = weights / weights.sum()
    replace = flat_scores.size < n_samples
    return np.random.choice(flat_scores.size, size=n_samples, replace=replace, p=weights)


def build_specialist_batches(
    reference: ReferenceSolution2D,
    base_batch: Dict[str, torch.Tensor],
    expert_names: List[str],
    *,
    cfg: StagedBurgers2DConfig,
    layout_variant: str = "categorical",
) -> tuple[List[Dict[str, torch.Tensor]], Dict[str, np.ndarray], torch.Tensor, torch.Tensor]:
    xt_col_global = base_batch["xt_col"]
    device = xt_col_global.device
    dtype = xt_col_global.dtype
    total_col = xt_col_global.shape[0]

    xyt_ref, u_ref = flatten_reference_solution(reference, device=device, dtype=dtype)
    region_scores = compute_region_scores(reference, layout_variant=layout_variant)
    score_tensors = {
        key: torch.tensor(value.reshape(-1), dtype=dtype, device=device)
        for key, value in region_scores.items()
    }

    batches: List[Dict[str, torch.Tensor]] = []
    default_region_key = next(iter(region_scores))
    for expert_name in expert_names:
        region_key = expert_name if expert_name in region_scores else default_region_key
        score_flat = region_scores[region_key].reshape(-1)
        threshold = float(np.quantile(score_flat, 0.78 if region_key != "smooth" else 0.65))
        focus_mask = score_flat >= threshold
        focus_count = max(1, int(total_col * _expert_focus_ratio(cfg, expert_name)))
        global_count = max(1, total_col - focus_count)

        focus_idx = _sample_indices_from_mask(focus_mask, focus_count)
        xt_focus = xyt_ref[focus_idx]
        focus_scores = torch.tensor(score_flat[focus_idx][:, None], dtype=dtype, device=device)

        global_idx = torch.randperm(total_col, device=device)[:global_count]
        xt_global = xt_col_global[global_idx]

        xt_col = torch.cat([xt_focus, xt_global], dim=0)
        res_weight = torch.cat(
            [
                0.35 + _expert_res_focus(cfg, expert_name) * focus_scores,
                torch.ones(global_count, 1, dtype=dtype, device=device),
            ],
            dim=0,
        )
        perm = torch.randperm(xt_col.shape[0], device=device)
        xt_col = xt_col[perm]
        res_weight = res_weight[perm]

        sup_idx = _sample_indices_from_mask(focus_mask, _expert_sup_points(cfg, expert_name))
        xt_sup = xyt_ref[sup_idx]
        u_sup = u_ref[sup_idx]

        batches.append(
            {
                "name": expert_name,
                "xt_col": xt_col,
                "xt_ic": base_batch["xt_ic"],
                "u_ic": base_batch["u_ic"],
                "xt_bc": base_batch["xt_bc"],
                "u_bc": base_batch["u_bc"],
                "xt_sup": xt_sup,
                "u_sup": u_sup,
                "res_weight": res_weight,
            }
        )

    return batches, region_scores, xyt_ref, u_ref


def build_joint_training_batch(
    reference: ReferenceSolution2D,
    base_batch: Dict[str, torch.Tensor],
    *,
    cfg: StagedBurgers2DConfig,
    global_xt_col: torch.Tensor | None = None,
    layout_variant: str = "categorical",
) -> Dict[str, torch.Tensor]:
    xt_col_global = global_xt_col if global_xt_col is not None else base_batch["xt_col"]
    device = xt_col_global.device
    dtype = xt_col_global.dtype
    total_col = xt_col_global.shape[0]

    xyt_ref, _ = flatten_reference_solution(reference, device=device, dtype=dtype)
    region_scores = compute_region_scores(reference, layout_variant=layout_variant)
    shock_priority = shock_priority_from_region_scores(
        region_scores,
        directional_bonus=cfg.joint_directional_bonus,
        layout_variant=layout_variant,
    )

    focus_count = max(1, int(total_col * cfg.joint_focus_ratio))
    global_count = max(1, total_col - focus_count)

    focus_threshold = float(np.quantile(shock_priority, cfg.joint_focus_quantile))
    focus_scores = shock_priority.copy()
    focus_scores[focus_scores < focus_threshold] = 0.0
    if np.count_nonzero(focus_scores) == 0:
        focus_scores = shock_priority.copy()
    focus_idx = _sample_indices_from_scores(
        focus_scores,
        focus_count,
        power=cfg.joint_focus_power,
    )
    xt_focus = xyt_ref[focus_idx]
    focus_weight = torch.tensor(shock_priority[focus_idx][:, None], dtype=dtype, device=device)

    global_idx = torch.randperm(xt_col_global.shape[0], device=device)[:global_count]
    xt_global = xt_col_global[global_idx]

    xt_col = torch.cat([xt_focus, xt_global], dim=0)
    res_weight = torch.cat(
        [
            1.0 + cfg.joint_res_focus * focus_weight,
            torch.ones(global_count, 1, dtype=dtype, device=device),
        ],
        dim=0,
    )
    perm = torch.randperm(xt_col.shape[0], device=device)

    joint_batch = dict(base_batch)
    joint_batch["xt_col"] = xt_col[perm]
    joint_batch["res_weight"] = res_weight[perm]
    return joint_batch


def build_specialist_batches_from_curated_dataset(
    dataset_dir: str,
    base_batch: Dict[str, torch.Tensor],
    expert_names: List[str],
    *,
    cfg: StagedBurgers2DConfig,
) -> List[Dict[str, torch.Tensor]]:
    xt_col_global = base_batch["xt_col"]
    device = xt_col_global.device
    dtype = xt_col_global.dtype
    total_col = xt_col_global.shape[0]
    batches: List[Dict[str, torch.Tensor]] = []

    for expert_name in expert_names:
        if expert_name not in EXPERT_REGION_NAMES:
            raise ValueError(f"Curated dataset does not support expert '{expert_name}'.")
        data = np.load(os.path.join(dataset_dir, f"expert_{expert_name}.npz"))
        coords = torch.tensor(data["coords"], dtype=dtype, device=device)
        values = torch.tensor(data["u"][:, None], dtype=dtype, device=device)
        weights = torch.tensor(data["weight"][:, None], dtype=dtype, device=device)
        label_conf = torch.tensor(data["label_confidence"][:, None], dtype=dtype, device=device)
        sample_weight = weights.pow(_profile_value(cfg.curated_weight_power, expert_name, 1.0))
        sample_weight = sample_weight * (
            _profile_value(cfg.curated_confidence_bias, expert_name, 0.35) + label_conf
        ).clamp_min(1e-6)
        sample_prob = sample_weight.squeeze(-1)
        sample_prob = sample_prob / sample_prob.sum().clamp_min(1e-8)

        focus_count = max(1, int(total_col * _expert_focus_ratio(cfg, expert_name)))
        global_count = max(1, total_col - focus_count)
        focus_idx = torch.multinomial(sample_prob, num_samples=focus_count, replacement=True)
        xt_focus = coords[focus_idx]
        focus_weight = sample_weight[focus_idx]
        global_idx = torch.randperm(total_col, device=device)[:global_count]
        xt_global = xt_col_global[global_idx]

        xt_col = torch.cat([xt_focus, xt_global], dim=0)
        res_weight = torch.cat(
            [
                0.35 + _expert_res_focus(cfg, expert_name) * focus_weight,
                torch.ones(global_count, 1, dtype=dtype, device=device),
            ],
            dim=0,
        )
        perm = torch.randperm(xt_col.shape[0], device=device)
        xt_col = xt_col[perm]
        res_weight = res_weight[perm]

        sup_count = min(_expert_sup_points(cfg, expert_name), coords.shape[0])
        sup_idx = torch.multinomial(sample_prob, num_samples=sup_count, replacement=True)
        xt_sup = coords[sup_idx]
        u_sup = values[sup_idx]

        batches.append(
            {
                "name": expert_name,
                "xt_col": xt_col,
                "xt_ic": base_batch["xt_ic"],
                "u_ic": base_batch["u_ic"],
                "xt_bc": base_batch["xt_bc"],
                "u_bc": base_batch["u_bc"],
                "xt_sup": xt_sup,
                "u_sup": u_sup,
                "res_weight": res_weight,
            }
        )

    return batches


def _freeze_module(module: torch.nn.Module | None) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad_(False)


def _unfreeze_module(module: torch.nn.Module | None) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad_(True)


def _batched_predict(model: torch.nn.Module, coords: torch.Tensor) -> torch.Tensor:
    batch_size = int(getattr(model, "inference_batch_size", 65536))
    outputs = []
    for start in range(0, coords.shape[0], batch_size):
        outputs.append(model(coords[start:start + batch_size]))
    return torch.cat(outputs, dim=0)


def _batched_gate_weights(model: torch.nn.Module, coords: torch.Tensor) -> torch.Tensor:
    batch_size = int(getattr(model, "inference_batch_size", 65536))
    outputs = []
    for start in range(0, coords.shape[0], batch_size):
        outputs.append(model.compute_gate_weights(coords[start:start + batch_size]))
    return torch.cat(outputs, dim=0)


def _expert_physics_loss(
    expert: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    *,
    base_model: torch.nn.Module | None,
    correction_scale: float,
    nu: float,
    sup_weight: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    xt_ic = batch["xt_ic"]
    u_ic = batch["u_ic"]
    xt_bc = batch["xt_bc"]
    u_bc = batch["u_bc"]
    xt_col = batch["xt_col"]
    xt_sup = batch.get("xt_sup")
    u_sup = batch.get("u_sup")
    res_weight = batch["res_weight"]

    if base_model is not None:
        with torch.no_grad():
            base_ic = base_model(xt_ic)
            base_bc = base_model(xt_bc)
        pred_ic = base_ic + correction_scale * expert(xt_ic)
        pred_bc = base_bc + correction_scale * expert(xt_bc)
    else:
        pred_ic = expert(xt_ic)
        pred_bc = expert(xt_bc)

    l_ic = ((pred_ic - u_ic) ** 2).mean()
    l_bc = ((pred_bc - u_bc) ** 2).mean()
    if sup_weight > 0.0:
        if xt_sup is None or u_sup is None or xt_sup.shape[0] == 0:
            raise ValueError("Positive expert supervision requires a non-empty xt_sup/u_sup batch.")
        if base_model is not None:
            with torch.no_grad():
                base_sup = base_model(xt_sup)
            pred_sup = base_sup + correction_scale * expert(xt_sup)
        else:
            pred_sup = expert(xt_sup)
        l_sup = ((pred_sup - u_sup) ** 2).mean()
    else:
        l_sup = torch.zeros((), dtype=pred_ic.dtype, device=pred_ic.device)

    xt_g = xt_col.detach().clone().requires_grad_(True)
    if base_model is not None:
        with torch.no_grad():
            base_pred = base_model(xt_g)
        u = base_pred + correction_scale * expert(xt_g)
    else:
        u = expert(xt_g)

    grads = torch.autograd.grad(u.sum(), xt_g, create_graph=True, retain_graph=True)[0]
    u_x = grads[:, 0:1]
    u_y = grads[:, 1:2]
    u_t = grads[:, 2:3]
    u_xx = torch.autograd.grad(u_x.sum(), xt_g, create_graph=True, retain_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y.sum(), xt_g, create_graph=True, retain_graph=True)[0][:, 1:2]
    residual = u_t + u * (u_x + u_y) - nu * (u_xx + u_yy)
    l_res = (res_weight * residual.square()).mean()

    total = l_res + 5.0 * l_ic + 2.0 * l_bc + sup_weight * l_sup
    return total, {"res": l_res.detach(), "ic": l_ic.detach(), "bc": l_bc.detach(), "sup": l_sup.detach()}


def _expert_eval_metrics(
    expert: torch.nn.Module,
    xyt_test: torch.Tensor,
    u_exact_flat: torch.Tensor,
    steep_mask_flat: torch.Tensor,
    *,
    base_model: torch.nn.Module | None,
    correction_scale: float,
) -> Dict[str, float]:
    with torch.no_grad():
        pred = expert(xyt_test)
        if base_model is not None:
            pred = base_model(xyt_test) + correction_scale * pred
        abs_err = (pred - u_exact_flat).abs()
        steep_mask = steep_mask_flat.bool()
        bg_mask = ~steep_mask
        return {
            "l2": l2_relative_error(pred, u_exact_flat),
            "steep_mae": abs_err[steep_mask].mean().item(),
            "background_mae": abs_err[bg_mask].mean().item(),
        }


def pretrain_base_model(
    moe_model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    *,
    cfg: StagedBurgers2DConfig,
    nu: float,
    save_dir: str,
    xyt_test: torch.Tensor,
    u_exact_flat: torch.Tensor,
    batch_refresh_fn=None,
) -> Dict[str, List[float]]:
    if not hasattr(moe_model, "base_model"):
        return {}

    print("\n" + "-" * 60)
    print("Stage 0: Pretraining base model")
    print("-" * 60)

    loss_fn = PhysicsLoss2D(LossConfig2D(nu=nu, w_res=1.0, w_ic=5.0, w_bc=2.0))

    def eval_l2() -> float:
        with torch.no_grad():
            return l2_relative_error(moe_model.base_model(xyt_test), u_exact_flat)

    trainer = Trainer(
        moe_model.base_model,
        loss_fn,
        lr=cfg.base_lr,
        n_steps=cfg.base_steps,
        device=xyt_test.device,
        save_dir=save_dir,
    )
    if batch_refresh_fn is not None and cfg.base_batch_refresh_freq > 0:
        history = trainer.train(
            batch,
            eval_fn=eval_l2,
            eval_freq=max(50, cfg.base_steps // 5),
            batch_refresh_fn=batch_refresh_fn,
            batch_refresh_freq=cfg.base_batch_refresh_freq,
        )
    else:
        history = trainer.train(batch, eval_fn=eval_l2, eval_freq=max(50, cfg.base_steps // 5))
    torch.save(
        {"model_state": moe_model.base_model.state_dict(), "history": history},
        os.path.join(save_dir, "burgers2d_base_pretrain.pt"),
    )
    return history


def pretrain_experts(
    moe_model: torch.nn.Module,
    expert_batches: List[Dict[str, torch.Tensor]],
    *,
    cfg: StagedBurgers2DConfig,
    nu: float,
    save_dir: str,
    xyt_test: torch.Tensor,
    u_exact_flat: torch.Tensor,
    steep_mask_flat: torch.Tensor,
) -> Dict[str, Dict[str, List[float]]]:
    histories: Dict[str, Dict[str, List[float]]] = {}

    for idx, (expert, batch) in enumerate(zip(moe_model.experts, expert_batches)):
        print("\n" + "-" * 60)
        print(f"Stage A.{idx + 1}: Pretraining expert {idx} ({batch['name']})")
        print("-" * 60)

        local_steps = _expert_steps(cfg, batch["name"])
        eval_freq = max(50, local_steps // 5)
        sup_weight = _expert_sup_weight(cfg, batch["name"])
        optimizer = torch.optim.Adam(expert.parameters(), lr=cfg.expert_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=local_steps,
            eta_min=cfg.expert_lr * 0.1,
        )

        history: Dict[str, List[float]] = {
            "total": [],
            "res": [],
            "ic": [],
            "bc": [],
            "sup": [],
            "l2_error": [],
            "steep_mae": [],
            "background_mae": [],
        }

        pbar = tqdm(range(1, local_steps + 1), desc=f"Expert{idx}", ncols=100)
        for step in pbar:
            optimizer.zero_grad()
            total_loss, loss_dict = _expert_physics_loss(
                expert,
                batch,
                base_model=getattr(moe_model, "base_model", None),
                correction_scale=float(getattr(moe_model, "correction_scale", 1.0)),
                nu=nu,
                sup_weight=sup_weight,
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(expert.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            history["total"].append(total_loss.item())
            for key in ["res", "ic", "bc", "sup"]:
                history[key].append(loss_dict[key].item())

            if step % eval_freq == 0 or step == local_steps:
                metrics = _expert_eval_metrics(
                    expert,
                    xyt_test,
                    u_exact_flat,
                    steep_mask_flat,
                    base_model=getattr(moe_model, "base_model", None),
                    correction_scale=float(getattr(moe_model, "correction_scale", 1.0)),
                )
                history["l2_error"].append(metrics["l2"])
                history["steep_mae"].append(metrics["steep_mae"])
                history["background_mae"].append(metrics["background_mae"])

            if step % cfg.expert_log_freq == 0 or step == local_steps:
                latest_l2 = history["l2_error"][-1] if history["l2_error"] else float("nan")
                pbar.set_postfix(
                    {
                        "L": f"{total_loss.item():.3e}",
                        "res": f"{loss_dict['res'].item():.3e}",
                        "sup": f"{loss_dict['sup'].item():.3e}",
                        "l2": f"{latest_l2:.3e}",
                    }
                )

        histories[batch["name"]] = history
        torch.save(
            {"model_state": expert.state_dict(), "history": history},
            os.path.join(save_dir, f"burgers2d_expert_{idx}_{batch['name']}.pt"),
        )

    return histories


def pretrain_rotation_layer(
    moe_model: torch.nn.Module,
    rotation_batch: Dict[str, torch.Tensor],
    *,
    cfg: StagedBurgers2DConfig,
    save_dir: str,
) -> Dict[str, List[float]]:
    if not hasattr(moe_model, "rotation_layer") or moe_model.rotation_layer is None:
        return {}
    if not rotation_batch:
        return {}
    if cfg.rotation_steps <= 0:
        return {}

    print("\n" + "-" * 60)
    print("Stage B0: Pretraining rotation layer")
    print("-" * 60)

    _freeze_module(getattr(moe_model, "base_model", None))
    for expert in getattr(moe_model, "experts", []):
        _freeze_module(expert)
    _freeze_module(getattr(moe_model, "gating", None))
    _freeze_module(getattr(moe_model, "rotation_route_adapter", None))
    _unfreeze_module(moe_model.rotation_layer)

    optimizer = torch.optim.Adam(moe_model.rotation_layer.parameters(), lr=cfg.rotation_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.rotation_steps,
        eta_min=cfg.rotation_lr * 0.1,
    )
    history: Dict[str, List[float]] = {
        "total": [],
        "angle": [],
        "concentration": [],
        "activation": [],
        "axis_error_deg": [],
        "activation_mean": [],
        "concentration_mean": [],
    }

    xt_rotation = rotation_batch["xt_rotation"]
    target_angle = rotation_batch["rotation_target_angle"].reshape(-1)
    target_weights = rotation_batch["rotation_target_weights"].reshape(-1)
    target_concentration = rotation_batch["rotation_target_concentration"].reshape(-1)
    target_activation = rotation_batch["rotation_target_activation"].reshape(-1)

    pbar = tqdm(range(1, cfg.rotation_steps + 1), desc="Rotation", ncols=100)
    for step in pbar:
        optimizer.zero_grad()
        rot_state = moe_model.get_rotation_state(xt_rotation)
        pred_angle = rot_state["rotation_angle"]
        angle_loss_sample = 1.0 - torch.cos(2.0 * (pred_angle - target_angle))
        angle_loss = (angle_loss_sample * target_weights).sum() / target_weights.sum().clamp_min(1e-8)

        concentration_loss_sample = (rot_state["concentration"] - target_concentration).square()
        concentration_loss = (
            concentration_loss_sample * target_weights
        ).sum() / target_weights.sum().clamp_min(1e-8)

        activation_loss_sample = (rot_state["activation"] - target_activation).square()
        activation_loss = (
            activation_loss_sample * target_weights
        ).sum() / target_weights.sum().clamp_min(1e-8)

        total_loss = (
            cfg.joint_rotation_supervision_weight * angle_loss
            + cfg.joint_rotation_concentration_weight * concentration_loss
            + cfg.joint_rotation_activation_weight * activation_loss
        )
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(moe_model.rotation_layer.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        axis_delta = torch.atan2(
            torch.sin(2.0 * (pred_angle - target_angle)),
            torch.cos(2.0 * (pred_angle - target_angle)),
        )
        axis_error_deg = torch.rad2deg(axis_delta.abs() * 0.5)
        weighted_axis_error = (axis_error_deg * target_weights).sum() / target_weights.sum().clamp_min(1e-8)

        history["total"].append(float(total_loss.item()))
        history["angle"].append(float(angle_loss.item()))
        history["concentration"].append(float(concentration_loss.item()))
        history["activation"].append(float(activation_loss.item()))
        history["axis_error_deg"].append(float(weighted_axis_error.item()))
        history["activation_mean"].append(float(rot_state["activation"].mean().item()))
        history["concentration_mean"].append(float(rot_state["concentration"].mean().item()))

        if step % max(20, cfg.rotation_steps // 5) == 0 or step == cfg.rotation_steps:
            pbar.set_postfix(
                {
                    "L": f"{total_loss.item():.3e}",
                    "ang": f"{angle_loss.item():.3e}",
                    "conc": f"{concentration_loss.item():.3e}",
                    "act": f"{rot_state['activation'].mean().item():.2f}",
                    "axis": f"{weighted_axis_error.item():.1f}",
                }
            )

    torch.save(
        {"model_state": moe_model.rotation_layer.state_dict(), "history": history},
        os.path.join(save_dir, "burgers2d_rotation_pretrain.pt"),
    )
    _unfreeze_module(getattr(moe_model, "base_model", None))
    for expert in getattr(moe_model, "experts", []):
        _unfreeze_module(expert)
    _unfreeze_module(getattr(moe_model, "gating", None))
    _unfreeze_module(getattr(moe_model, "rotation_route_adapter", None))
    return history


def build_gate_targets(
    moe_model: torch.nn.Module,
    xyt_gate: torch.Tensor,
    u_exact_gate: torch.Tensor,
    region_scores: Dict[str, np.ndarray],
    *,
    temperature: float,
    use_region_prior: bool = False,
    region_prior_power: float = 1.15,
    branch_superiority_bonus: float = 0.0,
    branch_superiority_power: float = 1.25,
    branch_alignment_power: float = 1.0,
    preferred_expert_name: str | None = None,
    preferred_expert_bias: float = 0.0,
    preferred_expert_power: float = 1.25,
    directional_gate_bias: float = 0.0,
    directional_gate_bias_power: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    with torch.no_grad():
        expert_preds = moe_model.get_expert_predictions(xyt_gate)
        point_errors = (expert_preds - u_exact_gate.unsqueeze(1)).abs().squeeze(-1)
        best_errors = point_errors.min(dim=1).values
        best_two_errors = torch.topk(
            point_errors,
            k=min(2, point_errors.shape[1]),
            dim=1,
            largest=False,
        ).values
        rel_errors = point_errors / (point_errors.mean(dim=1, keepdim=True) + 1e-8)
        perf_probs = torch.softmax(-temperature * rel_errors, dim=1)
        top2_probs = torch.topk(perf_probs, k=min(2, perf_probs.shape[1]), dim=1).values
        max_prob = top2_probs[:, 0]
        if perf_probs.shape[1] > 1:
            margin = top2_probs[:, 0] - top2_probs[:, 1]
            second_best_error = best_two_errors[:, 1]
            superiority = ((second_best_error - best_errors) / second_best_error.clamp_min(1e-8)).clamp(0.0, 1.0)
        else:
            margin = torch.ones_like(max_prob)
            superiority = torch.ones_like(max_prob)
        uniform_prob = 1.0 / perf_probs.shape[1]
        dominance = ((max_prob - uniform_prob) / max(1.0 - uniform_prob, 1e-8)).clamp(0.0, 1.0)
        confidence = (0.45 * margin + 0.25 * dominance + 0.30 * superiority).clamp(0.0, 1.0)

        targets = perf_probs
        winner_alignment = torch.ones_like(confidence)
        preferred_alignment = torch.zeros_like(confidence)
        if use_region_prior:
            directional_idx = None
            preferred_idx = None
            directional_name = getattr(moe_model, "directional_branch_name", "directional_shock")
            prior = _build_region_prior_tensor(
                moe_model.expert_names,
                region_scores,
                device=xyt_gate.device,
                dtype=xyt_gate.dtype,
            ).clamp_min(1e-4)
            for expert_name in moe_model.expert_names:
                if expert_name == directional_name:
                    directional_idx = moe_model.expert_names.index(expert_name)
                if preferred_expert_name is not None and expert_name == preferred_expert_name:
                    preferred_idx = moe_model.expert_names.index(expert_name)
            targets = perf_probs * prior.pow(region_prior_power)
            if directional_idx is not None and directional_gate_bias > 0:
                directional_score = prior[:, directional_idx]
                directional_score = directional_score / directional_score.max().clamp_min(1e-6)
                directional_boost = 1.0 + directional_gate_bias * directional_score.pow(
                    directional_gate_bias_power
                )
                targets[:, directional_idx] = targets[:, directional_idx] * directional_boost
            if preferred_idx is not None and preferred_expert_bias > 0:
                preferred_alignment = prior[:, preferred_idx] / prior.max(dim=1).values.clamp_min(1e-8)
                preferred_boost = 1.0 + preferred_expert_bias * preferred_alignment.pow(preferred_expert_power)
                targets[:, preferred_idx] = targets[:, preferred_idx] * preferred_boost
            targets = targets / targets.sum(dim=1, keepdim=True).clamp_min(1e-8)
            winner_idx = targets.argmax(dim=1, keepdim=True)
            winner_alignment = prior.gather(1, winner_idx).squeeze(1)
            winner_alignment = winner_alignment / prior.max(dim=1).values.clamp_min(1e-8)

        if branch_superiority_bonus > 0:
            bonus = superiority.pow(branch_superiority_power)
            if use_region_prior:
                bonus = bonus * winner_alignment.pow(branch_alignment_power)
            bonus = (branch_superiority_bonus * bonus).clamp(0.0, 1.0)
            confidence = 1.0 - (1.0 - confidence) * (1.0 - bonus)

        stats = {
            "confidence_mean": float(confidence.mean().item()),
            "confidence_std": float(confidence.std(unbiased=False).item()),
            "confidence_min": float(confidence.min().item()),
            "confidence_max": float(confidence.max().item()),
            "target_max_mean": float(targets.max(dim=1).values.mean().item()),
            "target_superiority_mean": float(superiority.mean().item()),
            "target_alignment_mean": float(winner_alignment.mean().item()),
            "target_preferred_alignment_mean": float(preferred_alignment.mean().item()),
            "target_directional_mean": float(
                targets[:, moe_model.expert_names.index(getattr(moe_model, "directional_branch_name", "directional_shock"))].mean().item()
            )
            if getattr(moe_model, "directional_branch_name", "directional_shock") in moe_model.expert_names
            else 0.0,
        }
        return targets, confidence, stats


def train_gate(
    moe_model: torch.nn.Module,
    xyt_gate: torch.Tensor,
    target_probs: torch.Tensor,
    target_confidence: torch.Tensor,
    shock_priority: torch.Tensor | None = None,
    preferred_priority: torch.Tensor | None = None,
    *,
    cfg: StagedBurgers2DConfig,
) -> tuple[Dict[str, List[float]], Dict[str, float]]:
    print("\n" + "-" * 60)
    print("Stage B: Training gating network")
    print("-" * 60)

    for expert in moe_model.experts:
        _freeze_module(expert)
    _freeze_module(getattr(moe_model, "base_model", None))
    _unfreeze_module(moe_model.gating)
    _unfreeze_module(getattr(moe_model, "rotation_route_adapter", None))

    gate_params = list(moe_model.gating.parameters())
    route_adapter = getattr(moe_model, "rotation_route_adapter", None)
    if route_adapter is not None:
        gate_params.extend(list(route_adapter.parameters()))
    optimizer = torch.optim.Adam(gate_params, lr=cfg.gate_lr)
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
        "confidence_mean": [],
        "keep_ratio": [],
    }

    n_samples = xyt_gate.shape[0]
    conf_threshold = max(
        float(torch.quantile(target_confidence, cfg.gate_confidence_quantile).item()),
        cfg.gate_min_confidence,
    )
    sample_weights = torch.clamp(target_confidence - conf_threshold, min=0.0)
    if sample_weights.max().item() > 0:
        sample_weights = sample_weights / sample_weights.max().clamp_min(1e-8)
    sample_weights = sample_weights.pow(cfg.gate_confidence_power)

    shock_mask = torch.zeros_like(target_confidence, dtype=torch.bool)
    if shock_priority is not None:
        shock_priority = shock_priority.to(device=xyt_gate.device, dtype=target_confidence.dtype)
        shock_priority = shock_priority / shock_priority.max().clamp_min(1e-8)
        shock_threshold = float(torch.quantile(shock_priority, cfg.gate_shock_quantile).item())
        shock_mask = shock_priority >= shock_threshold
        shock_boost = 1.0 + cfg.gate_shock_bias * shock_priority.pow(cfg.gate_shock_power)
        sample_weights = sample_weights * shock_boost
        min_shock_weight = cfg.gate_min_shock_weight * shock_priority.clamp_min(0.05)
        sample_weights = torch.where(shock_mask, torch.maximum(sample_weights, min_shock_weight), sample_weights)

    if preferred_priority is not None and cfg.gate_preferred_sample_boost > 0:
        preferred_priority = preferred_priority.to(device=xyt_gate.device, dtype=target_confidence.dtype)
        preferred_priority = preferred_priority / preferred_priority.max().clamp_min(1e-8)
        preferred_boost = 1.0 + cfg.gate_preferred_sample_boost * preferred_priority.pow(
            cfg.gate_preferred_sample_power
        )
        sample_weights = sample_weights * preferred_boost
        preferred_floor = 0.18 * cfg.gate_preferred_sample_boost * preferred_priority
        sample_weights = torch.maximum(sample_weights, preferred_floor)

    positive_mask = sample_weights > 0
    positive_mask = positive_mask | shock_mask
    keep_ratio = float(positive_mask.float().mean().item())
    min_keep = max(1, int(n_samples * cfg.gate_min_keep_ratio))
    if int(positive_mask.sum().item()) < min_keep:
        topk = min(n_samples, min_keep)
        top_idx = torch.topk(target_confidence, k=topk).indices
        sample_weights = torch.zeros_like(target_confidence)
        sample_weights[top_idx] = 1.0
        positive_mask = sample_weights > 0
        conf_threshold = float(target_confidence[top_idx[-1]].item())
        keep_ratio = float(positive_mask.float().mean().item())

    candidate_idx = positive_mask.nonzero(as_tuple=False).squeeze(-1)
    batch_size = min(cfg.gate_batch_size, candidate_idx.shape[0])
    gate_stats = {
        "confidence_threshold": conf_threshold,
        "selected_ratio": keep_ratio,
        "selected_count": float(candidate_idx.shape[0]),
        "confidence_mean_selected": float(target_confidence[positive_mask].mean().item()),
        "shock_selected_ratio": float(shock_mask.float().mean().item()),
    }

    pbar = tqdm(range(1, cfg.gate_steps + 1), desc="Gate", ncols=100)
    for step in pbar:
        perm = torch.randperm(candidate_idx.shape[0], device=xyt_gate.device)[:batch_size]
        idx = candidate_idx[perm]
        x_batch = xyt_gate[idx]
        y_batch = target_probs[idx]
        w_batch = sample_weights[idx]

        optimizer.zero_grad()
        probs = moe_model.compute_gate_weights(x_batch)
        sample_kl = F.kl_div(
            probs.clamp_min(1e-8).log(),
            y_batch,
            reduction="none",
        ).sum(dim=1)
        match_loss = (w_batch * sample_kl).sum() / w_batch.sum().clamp_min(1e-8)
        if cfg.gate_balance_weight > 0:
            balance_loss = moe_model.load_balance_loss(x_batch)
        else:
            balance_loss = torch.zeros((), dtype=match_loss.dtype, device=match_loss.device)
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
        history["confidence_mean"].append(float(target_confidence[idx].mean().item()))
        history["keep_ratio"].append(keep_ratio)

        if step % max(40, cfg.gate_steps // 6) == 0 or step == cfg.gate_steps:
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

    _unfreeze_module(getattr(moe_model, "base_model", None))
    for expert in moe_model.experts:
        _unfreeze_module(expert)
    _unfreeze_module(getattr(moe_model, "rotation_route_adapter", None))
    return history, gate_stats


def build_joint_gate_supervision_batch(
    xyt_gate: torch.Tensor,
    gate_targets: torch.Tensor,
    gate_confidence: torch.Tensor,
    shock_priority: torch.Tensor | None,
    preferred_priority: torch.Tensor | None,
    *,
    cfg: StagedBurgers2DConfig,
) -> Dict[str, torch.Tensor]:
    conf_threshold = max(
        float(torch.quantile(gate_confidence, cfg.gate_confidence_quantile).item()),
        cfg.gate_min_confidence,
    )
    sample_weights = torch.clamp(gate_confidence - conf_threshold, min=0.0)
    if sample_weights.max().item() > 0:
        sample_weights = sample_weights / sample_weights.max().clamp_min(1e-8)
    sample_weights = sample_weights.pow(cfg.gate_confidence_power)

    shock_mask = torch.zeros_like(gate_confidence, dtype=torch.bool)
    if shock_priority is not None:
        shock_priority = shock_priority.to(device=xyt_gate.device, dtype=gate_confidence.dtype)
        shock_priority = shock_priority / shock_priority.max().clamp_min(1e-8)
        shock_threshold = float(torch.quantile(shock_priority, cfg.gate_shock_quantile).item())
        shock_mask = shock_priority >= shock_threshold
        shock_boost = 1.0 + cfg.gate_shock_bias * shock_priority.pow(cfg.gate_shock_power)
        sample_weights = sample_weights * shock_boost
        min_shock_weight = cfg.gate_min_shock_weight * shock_priority.clamp_min(0.05)
        sample_weights = torch.where(shock_mask, torch.maximum(sample_weights, min_shock_weight), sample_weights)

    if preferred_priority is not None and cfg.gate_preferred_sample_boost > 0:
        preferred_priority = preferred_priority.to(device=xyt_gate.device, dtype=gate_confidence.dtype)
        preferred_priority = preferred_priority / preferred_priority.max().clamp_min(1e-8)
        preferred_boost = 1.0 + cfg.gate_preferred_sample_boost * preferred_priority.pow(
            cfg.gate_preferred_sample_power
        )
        sample_weights = sample_weights * preferred_boost
        preferred_floor = 0.18 * cfg.gate_preferred_sample_boost * preferred_priority
        sample_weights = torch.maximum(sample_weights, preferred_floor)

    positive_mask = (sample_weights > 0) | shock_mask
    min_keep = max(1, int(xyt_gate.shape[0] * cfg.gate_min_keep_ratio))
    if int(positive_mask.sum().item()) < min_keep:
        topk = min(xyt_gate.shape[0], min_keep)
        top_idx = torch.topk(gate_confidence, k=topk).indices
        sample_weights = torch.zeros_like(gate_confidence)
        sample_weights[top_idx] = 1.0
        positive_mask = sample_weights > 0
    candidate_idx = positive_mask.nonzero(as_tuple=False).squeeze(-1)
    num_points = min(cfg.joint_gate_supervision_points, candidate_idx.shape[0])
    probs = sample_weights[candidate_idx]
    if probs.sum().item() <= 0:
        probs = torch.ones_like(probs) / max(1, probs.numel())
    else:
        probs = probs / probs.sum()
    chosen = candidate_idx[torch.multinomial(probs, num_samples=num_points, replacement=False)]
    return {
        "xt_gate": xyt_gate[chosen],
        "gate_target_probs": gate_targets[chosen],
        "gate_target_weights": sample_weights[chosen],
    }


def build_joint_branch_consistency_batch(
    moe_model: torch.nn.Module,
    xyt_reference: torch.Tensor,
    u_exact_reference: torch.Tensor,
    region_scores: Dict[str, np.ndarray] | None,
    *,
    cfg: StagedBurgers2DConfig,
) -> Dict[str, torch.Tensor]:
    if cfg.joint_branch_consistency_weight <= 0 or cfg.joint_branch_consistency_points <= 0:
        return {}
    if not hasattr(moe_model, "get_expert_predictions"):
        return {}

    device = xyt_reference.device
    dtype = xyt_reference.dtype
    expert_names = list(getattr(moe_model, "expert_names", []))
    if not expert_names:
        return {}

    with torch.no_grad():
        expert_preds = moe_model.get_expert_predictions(xyt_reference).squeeze(-1)
        mixture_pred = _batched_predict(moe_model, xyt_reference).squeeze(-1)

    u_exact = u_exact_reference.squeeze(-1)
    point_errors = (expert_preds - u_exact.unsqueeze(1)).abs()
    mixture_error = (mixture_pred - u_exact).abs()
    num_points, num_experts = point_errors.shape

    prior = None
    if region_scores:
        prior = _build_region_prior_tensor(
            expert_names,
            region_scores,
            device=device,
            dtype=dtype,
        ).clamp_min(1e-6)

    focus_expert = cfg.joint_branch_consistency_focus_expert
    if focus_expert and focus_expert in expert_names:
        focus_idx = expert_names.index(focus_expert)
        target_idx = torch.full((num_points,), focus_idx, dtype=torch.long, device=device)
        branch_error = point_errors[:, focus_idx]
        if num_experts > 1:
            competitor_errors = point_errors.clone()
            competitor_errors[:, focus_idx] = torch.inf
            competitor_best = competitor_errors.min(dim=1).values
            superiority = ((competitor_best - branch_error) / competitor_best.clamp_min(1e-8)).clamp(0.0, 1.0)
        else:
            superiority = torch.ones_like(branch_error)
        if prior is not None:
            alignment = prior[:, focus_idx] / prior.max(dim=1).values.clamp_min(1e-8)
        else:
            alignment = torch.ones_like(branch_error)
        candidate_mask = alignment >= cfg.joint_branch_consistency_min_align
    else:
        best_two = torch.topk(point_errors, k=min(2, num_experts), dim=1, largest=False)
        target_idx = best_two.indices[:, 0]
        branch_error = best_two.values[:, 0]
        if num_experts > 1:
            superiority = ((best_two.values[:, 1] - branch_error) / best_two.values[:, 1].clamp_min(1e-8)).clamp(
                0.0,
                1.0,
            )
        else:
            superiority = torch.ones_like(branch_error)
        if prior is not None:
            alignment = prior.gather(1, target_idx[:, None]).squeeze(1)
            alignment = alignment / prior.max(dim=1).values.clamp_min(1e-8)
        else:
            alignment = torch.ones_like(branch_error)
        candidate_mask = torch.ones_like(branch_error, dtype=torch.bool)

    mixture_gap = ((mixture_error - branch_error) / mixture_error.clamp_min(1e-8)).clamp(0.0, 1.0)
    candidate_mask = candidate_mask & (mixture_gap >= cfg.joint_branch_consistency_margin)
    sample_weights = mixture_gap.pow(cfg.joint_branch_consistency_power)
    sample_weights = sample_weights * (0.35 + 0.65 * superiority)
    sample_weights = sample_weights * alignment.clamp_min(cfg.joint_branch_consistency_min_align).pow(
        cfg.joint_branch_consistency_region_power
    )
    sample_weights = torch.where(candidate_mask, sample_weights, torch.zeros_like(sample_weights))

    if sample_weights.sum().item() <= 0:
        fallback_weights = mixture_gap * alignment
        topk = min(num_points, max(1, cfg.joint_branch_consistency_points))
        chosen = torch.topk(fallback_weights, k=topk).indices
        fallback_sample_weights = fallback_weights[chosen]
        if fallback_sample_weights.sum().item() <= 0:
            fallback_sample_weights = torch.ones_like(fallback_sample_weights)
        return {
            "xt_branch_consistency": xyt_reference[chosen],
            "branch_target_idx": target_idx[chosen],
            "branch_target_weights": fallback_sample_weights,
        }

    candidate_idx = torch.nonzero(sample_weights > 0, as_tuple=False).squeeze(-1)
    chosen_count = min(cfg.joint_branch_consistency_points, candidate_idx.shape[0])
    probs = sample_weights[candidate_idx]
    probs = probs / probs.sum().clamp_min(1e-8)
    chosen = candidate_idx[torch.multinomial(probs, num_samples=chosen_count, replacement=False)]
    return {
        "xt_branch_consistency": xyt_reference[chosen],
        "branch_target_idx": target_idx[chosen],
        "branch_target_weights": sample_weights[chosen],
    }


def build_joint_base_consistency_batch(
    moe_model: torch.nn.Module,
    xyt_reference: torch.Tensor,
    u_exact_reference: torch.Tensor,
    region_scores: Dict[str, np.ndarray] | None,
    *,
    cfg: StagedBurgers2DConfig,
    layout_variant: str = "categorical",
) -> Dict[str, torch.Tensor]:
    if cfg.joint_base_consistency_weight <= 0 or cfg.joint_base_consistency_points <= 0:
        return {}
    if not region_scores:
        return {}

    device = xyt_reference.device
    dtype = xyt_reference.dtype
    smooth_key = "smooth"
    if smooth_key not in region_scores:
        return {}

    smooth = torch.tensor(region_scores[smooth_key].reshape(-1), dtype=dtype, device=device)
    shock = torch.tensor(
        shock_priority_from_region_scores(
            region_scores,
            directional_bonus=cfg.joint_directional_bonus,
            layout_variant=layout_variant,
        ).reshape(-1),
        dtype=dtype,
        device=device,
    )
    shock = shock / shock.max().clamp_min(1e-8)
    smooth = smooth / smooth.max().clamp_min(1e-8)

    candidate_mask = smooth >= cfg.joint_base_consistency_min_smooth
    sample_weights = smooth.pow(cfg.joint_base_consistency_smooth_power) * (
        1.0 - shock
    ).clamp_min(0.0).pow(cfg.joint_base_consistency_shock_power)

    if cfg.joint_base_consistency_guard_advantage:
        with torch.no_grad():
            base_pred = moe_model.base_model(xyt_reference).squeeze(-1)
            mixture_pred = _batched_predict(moe_model, xyt_reference).squeeze(-1)
            if hasattr(moe_model, "get_expert_predictions"):
                expert_preds = moe_model.get_expert_predictions(xyt_reference).squeeze(-1)
            else:
                expert_preds = None
            if cfg.joint_base_consistency_use_uncertainty and hasattr(moe_model, "compute_gate_weights"):
                gate_probs = _batched_gate_weights(moe_model, xyt_reference)
            else:
                gate_probs = None

        u_exact = u_exact_reference.squeeze(-1)
        base_error = (base_pred - u_exact).abs()
        mixture_error = (mixture_pred - u_exact).abs()

        base_compete = (mixture_error - base_error) / mixture_error.clamp_min(1e-8)
        candidate_mask = candidate_mask & (base_compete >= -cfg.joint_base_consistency_base_margin)
        sample_weights = sample_weights * (
            0.25 + 0.75 * (base_compete + cfg.joint_base_consistency_base_margin).clamp_min(0.0)
        )

        if expert_preds is not None:
            best_expert_error = (expert_preds - u_exact[:, None]).abs().min(dim=1).values
            expert_advantage = ((base_error - best_expert_error) / base_error.clamp_min(1e-8)).clamp(0.0, 1.0)
            candidate_mask = candidate_mask & (expert_advantage <= cfg.joint_base_consistency_expert_adv_margin)
            sample_weights = sample_weights * (
                1.0 - expert_advantage
            ).clamp_min(0.0).pow(cfg.joint_base_consistency_advantage_power)

        advantage_region = torch.zeros_like(sample_weights)
        if layout_variant == "categorical":
            if "directional_shock" in region_scores:
                directional = torch.tensor(
                    region_scores["directional_shock"].reshape(-1),
                    dtype=dtype,
                    device=device,
                )
                directional = directional / directional.max().clamp_min(1e-8)
                advantage_region = torch.maximum(advantage_region, directional)
            preferred_name = cfg.gate_preferred_expert_name
            if preferred_name and preferred_name in region_scores:
                preferred = torch.tensor(
                    region_scores[preferred_name].reshape(-1),
                    dtype=dtype,
                    device=device,
                )
                preferred = preferred / preferred.max().clamp_min(1e-8)
                advantage_region = torch.maximum(advantage_region, preferred)
        candidate_mask = candidate_mask & (advantage_region <= cfg.joint_base_consistency_advantage_region_max)
        sample_weights = sample_weights * (
            1.0 - advantage_region
        ).clamp_min(0.0).pow(cfg.joint_base_consistency_advantage_power)

        if gate_probs is not None:
            entropy = -(gate_probs * gate_probs.clamp_min(1e-8).log()).sum(dim=1)
            entropy = entropy / np.log(max(2, gate_probs.shape[1]))
            uncertainty_threshold = max(
                float(torch.quantile(entropy, cfg.joint_base_consistency_uncertainty_quantile).item()),
                cfg.joint_base_consistency_min_uncertainty,
            )
            candidate_mask = candidate_mask & (entropy >= uncertainty_threshold)
            sample_weights = sample_weights * entropy.clamp_min(0.0).pow(
                cfg.joint_base_consistency_uncertainty_power
            )

    sample_weights = torch.where(candidate_mask, sample_weights, torch.zeros_like(sample_weights))

    if sample_weights.sum().item() <= 0:
        fallback_weights = smooth * (1.0 - shock).clamp_min(0.0)
        topk = min(xyt_reference.shape[0], max(1, cfg.joint_base_consistency_points))
        chosen = torch.topk(fallback_weights, k=topk).indices
        chosen_weights = fallback_weights[chosen]
        if chosen_weights.sum().item() <= 0:
            chosen_weights = torch.ones_like(chosen_weights)
        return {
            "xt_base_consistency": xyt_reference[chosen],
            "base_target_weights": chosen_weights,
        }

    candidate_idx = torch.nonzero(sample_weights > 0, as_tuple=False).squeeze(-1)
    chosen_count = min(cfg.joint_base_consistency_points, candidate_idx.shape[0])
    probs = sample_weights[candidate_idx]
    probs = probs / probs.sum().clamp_min(1e-8)
    chosen = candidate_idx[torch.multinomial(probs, num_samples=chosen_count, replacement=False)]
    return {
        "xt_base_consistency": xyt_reference[chosen],
        "base_target_weights": sample_weights[chosen],
    }


def build_joint_rotation_supervision_batch(
    reference: ReferenceSolution2D,
    xyt_reference: torch.Tensor,
    region_scores: Dict[str, np.ndarray] | None,
    *,
    cfg: StagedBurgers2DConfig,
    layout_variant: str = "categorical",
) -> tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    if cfg.joint_rotation_supervision_weight <= 0 or cfg.joint_rotation_supervision_points <= 0:
        return {}, {}
    if region_scores is None:
        return {}, {}

    rotation_maps = compute_rotation_reference_maps(reference)
    device = xyt_reference.device
    dtype = xyt_reference.dtype

    grad_norm = torch.tensor(rotation_maps["grad_norm"].reshape(-1), dtype=dtype, device=device)
    grad_angle = torch.tensor(rotation_maps["grad_angle"].reshape(-1), dtype=dtype, device=device)

    if layout_variant == "categorical":
        directional_key = "directional_shock"
        wave_key = "wave"
    else:
        directional_key = "anisotropy_directional"
        wave_key = "curvature_wave"

    directional = torch.tensor(region_scores[directional_key].reshape(-1), dtype=dtype, device=device)
    wave = torch.tensor(region_scores[wave_key].reshape(-1), dtype=dtype, device=device)
    directional = directional / directional.max().clamp_min(1e-8)
    wave = wave / wave.max().clamp_min(1e-8)
    wave_focus = ((1.0 + cfg.joint_rotation_wave_bonus) * wave).clamp(0.0, 1.0)
    region_focus = torch.maximum(directional, wave_focus)
    region_focus = region_focus.clamp(0.0, 1.0)

    grad_threshold = float(torch.quantile(grad_norm, cfg.joint_rotation_grad_quantile).item())
    candidate_mask = (grad_norm >= grad_threshold) & (region_focus >= cfg.joint_rotation_min_region)
    sample_weights = grad_norm.clamp_min(0.0).pow(cfg.joint_rotation_grad_power)
    sample_weights = sample_weights * (
        0.30 + 0.70 * region_focus.pow(cfg.joint_rotation_region_power)
    )
    sample_weights = torch.where(candidate_mask, sample_weights, torch.zeros_like(sample_weights))

    total_points = max(1, cfg.joint_rotation_supervision_points)
    negative_ratio = float(np.clip(cfg.joint_rotation_negative_ratio, 0.0, 0.95))
    positive_budget = max(1, int(round(total_points * (1.0 - negative_ratio))))
    negative_budget = max(0, total_points - positive_budget)

    activation_floor = float(np.clip(cfg.joint_rotation_activation_floor_target, 0.0, 1.0))
    activation_cap = float(np.clip(cfg.joint_rotation_activation_cap, activation_floor, 1.0))
    concentration_floor = float(np.clip(cfg.joint_rotation_concentration_floor_target, 0.0, 1.0))
    concentration_cap = float(np.clip(cfg.joint_rotation_concentration_cap, concentration_floor, 1.0))

    def _positive_targets(chosen_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        positive_signal = torch.clamp(0.45 * grad_norm[chosen_idx] + 0.55 * region_focus[chosen_idx], 0.0, 1.0)
        target_concentration = concentration_floor + (concentration_cap - concentration_floor) * torch.clamp(
            0.5 * grad_norm[chosen_idx] + 0.5 * region_focus[chosen_idx],
            0.0,
            1.0,
        )
        target_activation = activation_floor + (activation_cap - activation_floor) * positive_signal
        return target_concentration, target_activation

    smooth = None
    if "smooth" in region_scores:
        smooth = torch.tensor(region_scores["smooth"].reshape(-1), dtype=dtype, device=device)
        smooth = smooth / smooth.max().clamp_min(1e-8)

    negative_grad_threshold = float(torch.quantile(grad_norm, cfg.joint_rotation_negative_grad_quantile).item())
    negative_mask = (grad_norm <= negative_grad_threshold) & (region_focus <= cfg.joint_rotation_negative_max_region)
    negative_weights = (1.0 - region_focus).clamp_min(0.0)
    negative_weights = negative_weights * (1.0 - torch.clamp(grad_norm / grad_threshold, 0.0, 1.0))
    if smooth is not None:
        negative_weights = negative_weights * (0.35 + 0.65 * smooth)
    negative_weights = negative_weights * cfg.joint_rotation_negative_weight_scale
    negative_weights = torch.where(negative_mask, negative_weights, torch.zeros_like(negative_weights))

    if sample_weights.sum().item() <= 0:
        fallback_weights = grad_norm * region_focus
        topk = min(xyt_reference.shape[0], max(1, positive_budget))
        chosen = torch.topk(fallback_weights, k=topk).indices
        chosen_weights = fallback_weights[chosen]
        if chosen_weights.sum().item() <= 0:
            chosen_weights = torch.ones_like(chosen_weights)
        target_concentration, target_activation = _positive_targets(chosen)
        stats = {
            "rotation_selected_count": float(chosen.shape[0]),
            "rotation_grad_threshold": grad_threshold,
            "rotation_weight_mean": float(chosen_weights.mean().item()),
            "rotation_focus_mean": float(region_focus[chosen].mean().item()),
            "rotation_grad_mean": float(grad_norm[chosen].mean().item()),
            "rotation_negative_selected_count": 0.0,
        }
        return {
            "xt_rotation": xyt_reference[chosen],
            "rotation_target_angle": grad_angle[chosen],
            "rotation_target_weights": chosen_weights,
            "rotation_target_concentration": target_concentration,
            "rotation_target_activation": target_activation,
        }, stats

    candidate_idx = torch.nonzero(sample_weights > 0, as_tuple=False).squeeze(-1)
    chosen_count = min(positive_budget, candidate_idx.shape[0])
    probs = sample_weights[candidate_idx]
    probs = probs / probs.sum().clamp_min(1e-8)
    chosen = candidate_idx[torch.multinomial(probs, num_samples=chosen_count, replacement=False)]
    target_concentration, target_activation = _positive_targets(chosen)

    chosen_all = [chosen]
    chosen_weights_all = [sample_weights[chosen]]
    target_angles_all = [grad_angle[chosen]]
    target_concentration_all = [target_concentration]
    target_activation_all = [target_activation]

    negative_selected = 0
    if negative_budget > 0 and negative_weights.sum().item() > 0:
        neg_candidate_idx = torch.nonzero(negative_weights > 0, as_tuple=False).squeeze(-1)
        neg_count = min(negative_budget, neg_candidate_idx.shape[0])
        neg_probs = negative_weights[neg_candidate_idx]
        neg_probs = neg_probs / neg_probs.sum().clamp_min(1e-8)
        chosen_neg = neg_candidate_idx[torch.multinomial(neg_probs, num_samples=neg_count, replacement=False)]
        negative_selected = int(neg_count)
        chosen_all.append(chosen_neg)
        chosen_weights_all.append(negative_weights[chosen_neg])
        target_angles_all.append(grad_angle[chosen_neg])
        target_concentration_all.append(
            torch.full_like(negative_weights[chosen_neg], cfg.joint_rotation_negative_concentration)
        )
        target_activation_all.append(
            torch.full_like(negative_weights[chosen_neg], cfg.joint_rotation_negative_activation)
        )

    chosen = torch.cat(chosen_all, dim=0)
    chosen_weights = torch.cat(chosen_weights_all, dim=0)
    target_angles = torch.cat(target_angles_all, dim=0)
    target_concentration = torch.cat(target_concentration_all, dim=0)
    target_activation = torch.cat(target_activation_all, dim=0)
    stats = {
        "rotation_selected_count": float(chosen.shape[0]),
        "rotation_grad_threshold": grad_threshold,
        "rotation_weight_mean": float(chosen_weights.mean().item()),
        "rotation_focus_mean": float(region_focus[chosen].mean().item()),
        "rotation_grad_mean": float(grad_norm[chosen].mean().item()),
        "rotation_negative_selected_count": float(negative_selected),
    }
    return {
        "xt_rotation": xyt_reference[chosen],
        "rotation_target_angle": target_angles,
        "rotation_target_weights": chosen_weights,
        "rotation_target_concentration": target_concentration,
        "rotation_target_activation": target_activation,
    }, stats


def joint_finetune(
    moe_model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    *,
    cfg: StagedBurgers2DConfig,
    nu: float,
    save_dir: str,
    xyt_test: torch.Tensor,
    u_exact_flat: torch.Tensor,
    batch_refresh_fn=None,
) -> Dict[str, List[float]]:
    print("\n" + "-" * 60)
    print("Stage C: Joint fine-tuning")
    print("-" * 60)

    if cfg.joint_steps <= 0 or cfg.joint_train_mode == "skip":
        print("[OK] Stage C skipped.")
        return {"total": [], "res": [], "ic": [], "bc": [], "l2_error": []}

    if cfg.joint_train_mode == "gate_only":
        _freeze_module(getattr(moe_model, "base_model", None))
        for expert in moe_model.experts:
            _freeze_module(expert)
        _unfreeze_module(moe_model.gating)
        _unfreeze_module(getattr(moe_model, "rotation_layer", None))
        _unfreeze_module(getattr(moe_model, "rotation_route_adapter", None))
    elif cfg.joint_train_mode == "base_gate":
        _unfreeze_module(getattr(moe_model, "base_model", None))
        for expert in moe_model.experts:
            _freeze_module(expert)
        _unfreeze_module(moe_model.gating)
        _unfreeze_module(getattr(moe_model, "rotation_layer", None))
        _unfreeze_module(getattr(moe_model, "rotation_route_adapter", None))
    elif cfg.joint_train_mode == "experts_gate":
        _freeze_module(getattr(moe_model, "base_model", None))
        for expert in moe_model.experts:
            _unfreeze_module(expert)
        _unfreeze_module(moe_model.gating)
        _unfreeze_module(getattr(moe_model, "rotation_layer", None))
        _unfreeze_module(getattr(moe_model, "rotation_route_adapter", None))
    else:
        _unfreeze_module(getattr(moe_model, "base_model", None))
        for expert in moe_model.experts:
            _unfreeze_module(expert)
        _unfreeze_module(moe_model.gating)
        _unfreeze_module(getattr(moe_model, "rotation_layer", None))
        _unfreeze_module(getattr(moe_model, "rotation_route_adapter", None))

    loss_fn = PhysicsLoss2D(
        LossConfig2D(
            nu=nu,
            w_res=1.0,
            w_ic=5.0,
            w_bc=2.0,
            w_sparse=cfg.joint_sparse_weight,
            w_balance=cfg.joint_balance_weight,
            w_gate_supervised=cfg.joint_gate_supervised_weight,
            w_gate_misroute=cfg.joint_gate_misroute_weight,
            gate_misroute_power=cfg.joint_gate_misroute_power,
            w_branch_consistency=cfg.joint_branch_consistency_weight,
            w_base_consistency=cfg.joint_base_consistency_weight,
            w_rotation_supervised=cfg.joint_rotation_supervision_weight,
            w_rotation_concentration=cfg.joint_rotation_concentration_weight,
            w_rotation_activation=cfg.joint_rotation_activation_weight,
        )
    )

    def eval_l2() -> float:
        with torch.no_grad():
            return l2_relative_error(_batched_predict(moe_model, xyt_test), u_exact_flat)

    trainer = Trainer(
        moe_model,
        loss_fn,
        lr=cfg.joint_lr,
        n_steps=cfg.joint_steps,
        device=xyt_test.device,
        save_dir=save_dir,
    )
    history = trainer.train(
        batch,
        eval_fn=eval_l2,
        eval_freq=max(50, cfg.joint_steps // 5),
        batch_refresh_fn=batch_refresh_fn,
        batch_refresh_freq=cfg.joint_batch_refresh_freq,
    )
    trainer.save_checkpoint("burgers2d_moe_staged.pt")
    _unfreeze_module(getattr(moe_model, "base_model", None))
    for expert in moe_model.experts:
        _unfreeze_module(expert)
    _unfreeze_module(moe_model.gating)
    _unfreeze_module(getattr(moe_model, "rotation_layer", None))
    _unfreeze_module(getattr(moe_model, "rotation_route_adapter", None))
    return history


def steep_mask_from_reference(
    reference: ReferenceSolution2D,
    *,
    device: torch.device,
) -> torch.Tensor:
    mask, _ = steep_region_mask(reference.u, x=reference.x, y=reference.y, quantile=0.9)
    return torch.tensor(mask.reshape(-1), dtype=torch.bool, device=device)
