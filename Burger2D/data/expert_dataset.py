"""Curated dataset generation and region classification for Burger2D."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import torch

from Burger2D.equations.burgers2d import Burgers2DProblem, ReferenceSolution2D


REGION_NAMES = ["smooth", "iso_shock", "directional_shock", "wave", "ambiguous"]
EXPERT_REGION_NAMES = ["smooth", "iso_shock", "directional_shock", "wave"]


@dataclass(frozen=True)
class RegionRuleConfig:
    smooth_grad_max: float = 0.58
    smooth_lap_max: float = 0.66
    smooth_aniso_max: float = 0.88
    steep_grad_min: float = 0.95
    iso_aniso_max: float = 0.78
    iso_mixed_max: float = 0.88
    iso_lap_max: float = 1.20
    directional_grad_min: float = 0.90
    directional_aniso_min: float = 0.95
    directional_mixed_min: float = 0.90
    directional_time_min: float = 0.20
    wave_lap_min: float = 1.05
    wave_grad_max: float = 1.30
    wave_time_min: float = 0.15
    wave_aniso_max: float = 1.20
    smooth_bonus: float = 0.28
    iso_bonus: float = 0.40
    directional_bonus: float = 0.45
    wave_bonus: float = 0.35
    smooth_neighbor_penalty: float = 0.26
    smooth_core_grad_scale: float = 0.82
    smooth_core_lap_scale: float = 0.82
    smooth_core_nonsmooth_max: float = 0.10
    smooth_core_prob_min: float = 0.22
    patch_radius: int = 1
    patch_prob_blend: float = 0.38
    patch_confidence_blend: float = 0.30
    patch_valid_min: float = 0.15
    ambiguous_margin_max: float = 0.08
    ambiguous_confidence_max: float = 0.18


DEFAULT_REGION_RULES = RegionRuleConfig()


@dataclass(frozen=True)
class RandomProfileSpec:
    amp_diag_plus: float
    amp_diag_minus: float
    amp_xy: float
    amp_alt: float
    freq_x: int
    freq_y: int
    phase_diag_plus: float
    phase_diag_minus: float
    phase_x: float
    phase_y: float
    skew_x: float
    skew_y: float


def sample_random_profile_spec(rng: np.random.Generator) -> RandomProfileSpec:
    return RandomProfileSpec(
        amp_diag_plus=float(rng.uniform(0.35, 0.95)),
        amp_diag_minus=float(rng.uniform(-0.75, 0.45)),
        amp_xy=float(rng.uniform(-0.45, 0.45)),
        amp_alt=float(rng.uniform(-0.35, 0.35)),
        freq_x=int(rng.integers(1, 4)),
        freq_y=int(rng.integers(1, 4)),
        phase_diag_plus=float(rng.uniform(-np.pi, np.pi)),
        phase_diag_minus=float(rng.uniform(-np.pi, np.pi)),
        phase_x=float(rng.uniform(-np.pi, np.pi)),
        phase_y=float(rng.uniform(-np.pi, np.pi)),
        skew_x=float(rng.uniform(-0.35, 0.35)),
        skew_y=float(rng.uniform(-0.35, 0.35)),
    )


def make_profile_functions(
    spec: RandomProfileSpec,
) -> tuple[Callable[[np.ndarray, np.ndarray], np.ndarray], Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
    def profile_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x_tilt = x + spec.skew_x * y
        y_tilt = y + spec.skew_y * x
        envelope = (1.0 - x**2) * (1.0 - y**2)
        carrier = (
            spec.amp_diag_plus * np.sin(np.pi * (x_tilt + y_tilt) + spec.phase_diag_plus)
            + spec.amp_diag_minus * np.sin(np.pi * (x_tilt - y_tilt) + spec.phase_diag_minus)
            + spec.amp_xy
            * np.sin(spec.freq_x * np.pi * x_tilt + spec.phase_x)
            * np.sin(spec.freq_y * np.pi * y_tilt + spec.phase_y)
            + spec.amp_alt
            * np.sin((spec.freq_x + 1) * np.pi * x_tilt - 0.5 * spec.phase_x)
            * np.sin(np.pi * (x_tilt + 0.5 * y_tilt))
        )
        return -envelope * carrier

    def profile_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pi = torch.tensor(np.pi, dtype=x.dtype, device=x.device)
        x_tilt = x + spec.skew_x * y
        y_tilt = y + spec.skew_y * x
        envelope = (1.0 - x.square()) * (1.0 - y.square())
        carrier = (
            spec.amp_diag_plus * torch.sin(pi * (x_tilt + y_tilt) + spec.phase_diag_plus)
            + spec.amp_diag_minus * torch.sin(pi * (x_tilt - y_tilt) + spec.phase_diag_minus)
            + spec.amp_xy
            * torch.sin(spec.freq_x * pi * x_tilt + spec.phase_x)
            * torch.sin(spec.freq_y * pi * y_tilt + spec.phase_y)
            + spec.amp_alt
            * torch.sin((spec.freq_x + 1) * pi * x_tilt - 0.5 * spec.phase_x)
            * torch.sin(pi * (x_tilt + 0.5 * y_tilt))
        )
        return -envelope * carrier

    return profile_np, profile_torch


def compute_region_feature_maps(reference: ReferenceSolution2D) -> dict[str, np.ndarray]:
    u_ref = reference.u.astype(np.float64)
    x = reference.x.astype(np.float64)
    y = reference.y.astype(np.float64)
    t = reference.t.astype(np.float64)
    grad_mag = np.zeros_like(u_ref, dtype=np.float32)
    lap_abs = np.zeros_like(u_ref, dtype=np.float32)
    mixed_abs = np.zeros_like(u_ref, dtype=np.float32)

    for i in range(u_ref.shape[0]):
        u_slice = u_ref[i]
        u_y, u_x = np.gradient(u_slice, y, x, edge_order=1)
        u_xx = np.gradient(u_x, x, axis=1, edge_order=1)
        u_yy = np.gradient(u_y, y, axis=0, edge_order=1)
        u_xy = np.gradient(u_x, y, axis=0, edge_order=1)
        grad_mag[i] = np.sqrt(u_x**2 + u_y**2).astype(np.float32)
        lap_abs[i] = np.abs(u_xx + u_yy).astype(np.float32)
        mixed_abs[i] = np.abs(u_xy).astype(np.float32)

    grad_scale = float(np.quantile(grad_mag, 0.90) + 1e-8)
    lap_scale = float(np.quantile(lap_abs, 0.90) + 1e-8)
    mixed_scale = float(np.quantile(mixed_abs, 0.90) + 1e-8)
    grad_norm = grad_mag / grad_scale
    lap_norm = lap_abs / lap_scale
    mixed_norm = mixed_abs / mixed_scale
    time_frac = (t / max(float(t.max()), 1e-8)).astype(np.float32)[:, None, None]
    # Stabilize the directional proxy in nearly flat zones where lap_abs can be tiny.
    anisotropy_raw = mixed_abs / np.maximum(lap_abs, 0.08 * lap_scale)
    anisotropy_cap = float(np.quantile(anisotropy_raw, 0.98) + 1e-6)
    anisotropy = np.clip(anisotropy_raw, 0.0, anisotropy_cap)

    return {
        "grad_mag": grad_mag,
        "lap_abs": lap_abs,
        "mixed_abs": mixed_abs,
        "grad_norm": grad_norm.astype(np.float32),
        "lap_norm": lap_norm.astype(np.float32),
        "mixed_norm": mixed_norm.astype(np.float32),
        "anisotropy": anisotropy.astype(np.float32),
        "anisotropy_cap": np.array(anisotropy_cap, dtype=np.float32),
        "time_frac": np.broadcast_to(time_frac, grad_mag.shape).astype(np.float32),
        "grad_scale": np.array(grad_scale, dtype=np.float32),
        "lap_scale": np.array(lap_scale, dtype=np.float32),
        "mixed_scale": np.array(mixed_scale, dtype=np.float32),
    }


def _positive_margin(value: np.ndarray, threshold: float, *, direction: str) -> np.ndarray:
    if direction == "ge":
        return np.clip(value - threshold, 0.0, None)
    if direction == "le":
        return np.clip(threshold - value, 0.0, None)
    raise ValueError(f"Unsupported direction: {direction}")


def _box_filter_spatial(value: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return value.astype(np.float32, copy=False)
    kernel = 2 * radius + 1
    value_t = torch.from_numpy(value.astype(np.float32))
    if value_t.ndim == 3:
        value_t = value_t.unsqueeze(-1)
    value_t = value_t.permute(0, 3, 1, 2).contiguous()
    filtered = torch.nn.functional.avg_pool2d(
        value_t,
        kernel_size=kernel,
        stride=1,
        padding=radius,
        count_include_pad=False,
    )
    return filtered.permute(0, 2, 3, 1).cpu().numpy()


def classify_region_maps(
    feature_maps: dict[str, np.ndarray],
    rules: RegionRuleConfig = DEFAULT_REGION_RULES,
) -> dict[str, np.ndarray]:
    grad_norm = feature_maps["grad_norm"]
    lap_norm = feature_maps["lap_norm"]
    mixed_norm = feature_maps["mixed_norm"]
    anisotropy = feature_maps["anisotropy"]
    time_frac = feature_maps["time_frac"]

    smooth_mask = (
        (grad_norm <= rules.smooth_grad_max)
        & (lap_norm <= rules.smooth_lap_max)
        & (anisotropy <= rules.smooth_aniso_max)
    )
    iso_mask = (
        (grad_norm >= rules.steep_grad_min)
        & (anisotropy <= rules.iso_aniso_max)
        & (mixed_norm <= rules.iso_mixed_max)
        & (lap_norm <= rules.iso_lap_max)
    )
    directional_mask = (
        (grad_norm >= rules.directional_grad_min)
        & (anisotropy >= rules.directional_aniso_min)
        & (mixed_norm >= rules.directional_mixed_min)
        & (time_frac >= rules.directional_time_min)
    )
    wave_mask = (
        (lap_norm >= rules.wave_lap_min)
        & (grad_norm <= rules.wave_grad_max)
        & (time_frac >= rules.wave_time_min)
        & (anisotropy <= rules.wave_aniso_max)
    )

    smooth_score = (
        1.10 * _positive_margin(grad_norm, rules.smooth_grad_max, direction="le")
        + 0.90 * _positive_margin(lap_norm, rules.smooth_lap_max, direction="le")
        + 0.60 * _positive_margin(anisotropy, rules.smooth_aniso_max, direction="le")
    )
    iso_score = (
        1.20 * _positive_margin(grad_norm, rules.steep_grad_min, direction="ge")
        + 0.80 * _positive_margin(anisotropy, rules.iso_aniso_max, direction="le")
        + 0.80 * _positive_margin(mixed_norm, rules.iso_mixed_max, direction="le")
        + 0.50 * _positive_margin(lap_norm, rules.iso_lap_max, direction="le")
    )
    directional_score = (
        1.10 * _positive_margin(grad_norm, rules.directional_grad_min, direction="ge")
        + 1.15 * _positive_margin(anisotropy, rules.directional_aniso_min, direction="ge")
        + 0.90 * _positive_margin(mixed_norm, rules.directional_mixed_min, direction="ge")
        + 0.35 * _positive_margin(time_frac, rules.directional_time_min, direction="ge")
    )
    wave_score = (
        1.15 * _positive_margin(lap_norm, rules.wave_lap_min, direction="ge")
        + 0.60 * _positive_margin(grad_norm, rules.wave_grad_max, direction="le")
        + 0.50 * _positive_margin(time_frac, rules.wave_time_min, direction="ge")
        + 0.40 * _positive_margin(anisotropy, rules.wave_aniso_max, direction="le")
    )

    nonsmooth_seed = np.maximum.reduce(
        [
            iso_mask.astype(np.float32),
            directional_mask.astype(np.float32),
            wave_mask.astype(np.float32),
            (grad_norm >= 0.85 * rules.steep_grad_min).astype(np.float32),
        ]
    )
    local_nonsmooth = _box_filter_spatial(nonsmooth_seed, rules.patch_radius)[..., 0]
    smooth_core_mask = (
        (grad_norm <= rules.smooth_core_grad_scale * rules.smooth_grad_max)
        & (lap_norm <= rules.smooth_core_lap_scale * rules.smooth_lap_max)
        & (anisotropy <= rules.smooth_aniso_max)
        & (local_nonsmooth <= rules.smooth_core_nonsmooth_max)
    )
    smooth_score = smooth_score - rules.smooth_neighbor_penalty * np.clip(local_nonsmooth - 0.15, 0.0, None)
    smooth_score = np.where(smooth_core_mask, smooth_score + 0.32, smooth_score)

    smooth_score = np.where(smooth_mask, smooth_score + rules.smooth_bonus, smooth_score)
    iso_score = np.where(iso_mask, iso_score + rules.iso_bonus, iso_score)
    directional_score = np.where(directional_mask, directional_score + rules.directional_bonus, directional_score)
    wave_score = np.where(wave_mask, wave_score + rules.wave_bonus, wave_score)

    primary_scores = np.stack([smooth_score, iso_score, directional_score, wave_score], axis=-1)
    primary_scores = np.clip(primary_scores, 0.0, None).astype(np.float32)
    valid_mask = np.stack([smooth_mask, iso_mask, directional_mask, wave_mask], axis=-1)
    top2_scores = np.sort(np.partition(primary_scores, -2, axis=-1)[..., -2:], axis=-1)
    score_margin = top2_scores[..., 1] - top2_scores[..., 0]
    ambiguous_score = (
        0.40
        + 0.65 * np.clip(1.0 - primary_scores.max(axis=-1), 0.0, 1.0)
        + 0.35 * np.clip(rules.ambiguous_margin_max - score_margin, 0.0, None)
        + 0.25 * (valid_mask.sum(axis=-1) == 0).astype(np.float32)
    )
    logits = np.concatenate([primary_scores, ambiguous_score[..., None]], axis=-1)
    logits = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(1.8 * logits)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    local_probs = _box_filter_spatial(probs, rules.patch_radius)
    probs = (1.0 - rules.patch_prob_blend) * probs + rules.patch_prob_blend * local_probs
    probs = probs / np.clip(probs.sum(axis=-1, keepdims=True), 1e-8, None)
    labels = probs.argmax(axis=-1).astype(np.int16)
    top2_probs = np.sort(np.partition(probs, -2, axis=-1)[..., -2:], axis=-1)
    point_confidence = top2_probs[..., 1] - top2_probs[..., 0]
    local_confidence = _box_filter_spatial(point_confidence, rules.patch_radius)[..., 0]
    confidence = (
        (1.0 - rules.patch_confidence_blend) * point_confidence
        + rules.patch_confidence_blend * local_confidence
    ).astype(np.float32)
    local_valid_ratio = _box_filter_spatial(valid_mask.astype(np.float32), rules.patch_radius)

    ambiguous_idx = REGION_NAMES.index("ambiguous")
    labels = np.where(
        (score_margin <= rules.ambiguous_margin_max)
        | (confidence <= rules.ambiguous_confidence_max)
        | (valid_mask.sum(axis=-1) == 0),
        ambiguous_idx,
        labels,
    ).astype(np.int16)
    labels = np.where(
        local_valid_ratio.max(axis=-1) < rules.patch_valid_min,
        ambiguous_idx,
        labels,
    ).astype(np.int16)
    smooth_idx = REGION_NAMES.index("smooth")
    labels = np.where(
        smooth_core_mask & (probs[..., smooth_idx] >= rules.smooth_core_prob_min),
        smooth_idx,
        labels,
    ).astype(np.int16)

    thresholds = {
        "smooth_grad_max": np.array(rules.smooth_grad_max, dtype=np.float32),
        "smooth_lap_max": np.array(rules.smooth_lap_max, dtype=np.float32),
        "smooth_aniso_max": np.array(rules.smooth_aniso_max, dtype=np.float32),
        "steep_grad_min": np.array(rules.steep_grad_min, dtype=np.float32),
        "iso_aniso_max": np.array(rules.iso_aniso_max, dtype=np.float32),
        "iso_mixed_max": np.array(rules.iso_mixed_max, dtype=np.float32),
        "iso_lap_max": np.array(rules.iso_lap_max, dtype=np.float32),
        "directional_grad_min": np.array(rules.directional_grad_min, dtype=np.float32),
        "directional_aniso_min": np.array(rules.directional_aniso_min, dtype=np.float32),
        "directional_mixed_min": np.array(rules.directional_mixed_min, dtype=np.float32),
        "directional_time_min": np.array(rules.directional_time_min, dtype=np.float32),
        "wave_lap_min": np.array(rules.wave_lap_min, dtype=np.float32),
        "wave_grad_max": np.array(rules.wave_grad_max, dtype=np.float32),
        "wave_time_min": np.array(rules.wave_time_min, dtype=np.float32),
        "wave_aniso_max": np.array(rules.wave_aniso_max, dtype=np.float32),
        "smooth_bonus": np.array(rules.smooth_bonus, dtype=np.float32),
        "iso_bonus": np.array(rules.iso_bonus, dtype=np.float32),
        "directional_bonus": np.array(rules.directional_bonus, dtype=np.float32),
        "wave_bonus": np.array(rules.wave_bonus, dtype=np.float32),
        "smooth_neighbor_penalty": np.array(rules.smooth_neighbor_penalty, dtype=np.float32),
        "smooth_core_grad_scale": np.array(rules.smooth_core_grad_scale, dtype=np.float32),
        "smooth_core_lap_scale": np.array(rules.smooth_core_lap_scale, dtype=np.float32),
        "smooth_core_nonsmooth_max": np.array(rules.smooth_core_nonsmooth_max, dtype=np.float32),
        "smooth_core_prob_min": np.array(rules.smooth_core_prob_min, dtype=np.float32),
        "patch_radius": np.array(rules.patch_radius, dtype=np.int16),
        "patch_prob_blend": np.array(rules.patch_prob_blend, dtype=np.float32),
        "patch_confidence_blend": np.array(rules.patch_confidence_blend, dtype=np.float32),
        "patch_valid_min": np.array(rules.patch_valid_min, dtype=np.float32),
        "ambiguous_margin_max": np.array(rules.ambiguous_margin_max, dtype=np.float32),
        "ambiguous_confidence_max": np.array(rules.ambiguous_confidence_max, dtype=np.float32),
    }
    return {
        "scores": logits.astype(np.float32),
        "probs": probs.astype(np.float32),
        "labels": labels,
        "confidence": confidence,
        "score_margin": score_margin.astype(np.float32),
        "hard_masks": valid_mask.astype(np.int8),
        "thresholds": thresholds,
    }


def _flatten_reference(reference: ReferenceSolution2D) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xx, yy = np.meshgrid(reference.x, reference.y, indexing="xy")
    coords = []
    time_index = []
    for idx, t_value in enumerate(reference.t):
        tt = np.full_like(xx, t_value)
        coords.append(np.stack([xx, yy, tt], axis=-1).reshape(-1, 3))
        time_index.append(np.full(xx.size, idx, dtype=np.int16))
    return (
        np.concatenate(coords, axis=0).astype(np.float32),
        reference.u.reshape(-1).astype(np.float32),
        np.concatenate(time_index, axis=0),
    )


def _sample_case_catalog(
    *,
    case_id: int,
    reference: ReferenceSolution2D,
    feature_maps: dict[str, np.ndarray],
    class_maps: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    coords, u_flat, time_index = _flatten_reference(reference)
    probs = class_maps["probs"].reshape(-1, len(REGION_NAMES))
    labels = class_maps["labels"].reshape(-1)
    confidence = class_maps["confidence"].reshape(-1)

    return {
        "coords": coords,
        "u": u_flat,
        "time_index": time_index,
        "case_id": np.full(coords.shape[0], case_id, dtype=np.int16),
        "label": labels.astype(np.int16),
        "label_confidence": confidence.astype(np.float32),
        "grad_mag": feature_maps["grad_mag"].reshape(-1).astype(np.float32),
        "lap_abs": feature_maps["lap_abs"].reshape(-1).astype(np.float32),
        "mixed_abs": feature_maps["mixed_abs"].reshape(-1).astype(np.float32),
        "anisotropy": feature_maps["anisotropy"].reshape(-1).astype(np.float32),
        "time_frac": feature_maps["time_frac"].reshape(-1).astype(np.float32),
        "probs": probs.astype(np.float32),
    }


def build_curated_expert_dataset(
    *,
    output_dir: str,
    num_cases: int,
    nx: int,
    ny: int,
    nt: int,
    seed: int = 42,
    nu: float = 0.01 / np.pi,
    expert_keep_quantile: float = 0.72,
    expert_min_weight: float = 0.22,
    target_points_per_expert: int | None = None,
    region_rules: RegionRuleConfig = DEFAULT_REGION_RULES,
    expert_confidence_quantile: float = 0.55,
    expert_min_confidence: float = 0.16,
) -> dict[str, object]:
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    cases_dir = os.path.join(output_dir, "cases")
    os.makedirs(cases_dir, exist_ok=True)

    catalog_parts: dict[str, list[np.ndarray]] = {
        "coords": [],
        "u": [],
        "time_index": [],
        "case_id": [],
        "label": [],
        "label_confidence": [],
        "grad_mag": [],
        "lap_abs": [],
        "mixed_abs": [],
        "anisotropy": [],
        "time_frac": [],
        "probs": [],
    }
    expert_parts = {
        name: {
            "coords": [],
            "u": [],
            "weight": [],
            "case_id": [],
            "time_index": [],
            "label_confidence": [],
        }
        for name in EXPERT_REGION_NAMES
    }
    manifest_cases = []
    label_hist = np.zeros(len(REGION_NAMES), dtype=np.int64)

    for case_id in range(num_cases):
        spec = sample_random_profile_spec(rng)
        profile_np, profile_torch = make_profile_functions(spec)
        problem = Burgers2DProblem(
            nu=nu,
            seed=seed + case_id,
            initial_profile_np_fn=profile_np,
            initial_profile_torch_fn=profile_torch,
        )
        reference = problem.generate_reference_solution(nx=nx, ny=ny, nt=nt)
        feature_maps = compute_region_feature_maps(reference)
        class_maps = classify_region_maps(feature_maps, rules=region_rules)
        case_catalog = _sample_case_catalog(
            case_id=case_id,
            reference=reference,
            feature_maps=feature_maps,
            class_maps=class_maps,
        )

        for key, value in case_catalog.items():
            catalog_parts[key].append(value)
        label_hist += np.bincount(case_catalog["label"], minlength=len(REGION_NAMES))

        np.savez_compressed(
            os.path.join(cases_dir, f"case_{case_id:03d}.npz"),
            x=reference.x,
            y=reference.y,
            t=reference.t,
            u=reference.u,
            labels=class_maps["labels"].astype(np.int16),
            label_confidence=class_maps["confidence"].astype(np.float32),
            probs=class_maps["probs"].astype(np.float32),
            score_margin=class_maps["score_margin"].astype(np.float32),
            hard_masks=class_maps["hard_masks"].astype(np.int8),
            grad_mag=feature_maps["grad_mag"].astype(np.float32),
            lap_abs=feature_maps["lap_abs"].astype(np.float32),
            mixed_abs=feature_maps["mixed_abs"].astype(np.float32),
            anisotropy=feature_maps["anisotropy"].astype(np.float32),
        )

        case_probs = case_catalog["probs"]
        case_labels = case_catalog["label"]
        case_confidence = case_catalog["label_confidence"]
        ambiguous_idx = REGION_NAMES.index("ambiguous")
        for expert_idx, expert_name in enumerate(EXPERT_REGION_NAMES):
            expert_prob = case_probs[:, expert_idx]
            threshold = max(float(np.quantile(expert_prob, expert_keep_quantile)), expert_min_weight)
            label_match = case_labels == expert_idx
            if np.any(label_match):
                conf_threshold = max(
                    float(np.quantile(case_confidence[label_match], expert_confidence_quantile)),
                    expert_min_confidence,
                )
            else:
                conf_threshold = expert_min_confidence

            primary_mask = label_match & (case_confidence >= conf_threshold)
            secondary_mask = (
                (expert_prob >= threshold)
                & (case_confidence >= 0.5 * conf_threshold)
                & (case_labels != ambiguous_idx)
            )
            select_mask = primary_mask | secondary_mask
            if not np.any(select_mask):
                top_idx = np.argsort(expert_prob)[-max(1, expert_prob.size // 20):]
                select_mask = np.zeros_like(expert_prob, dtype=bool)
                select_mask[top_idx] = True
            expert_parts[expert_name]["coords"].append(case_catalog["coords"][select_mask])
            expert_parts[expert_name]["u"].append(case_catalog["u"][select_mask])
            sample_weight = expert_prob * (0.35 + np.clip(case_confidence, 0.0, 1.0))
            expert_parts[expert_name]["weight"].append(sample_weight[select_mask].astype(np.float32))
            expert_parts[expert_name]["case_id"].append(case_catalog["case_id"][select_mask])
            expert_parts[expert_name]["time_index"].append(case_catalog["time_index"][select_mask])
            expert_parts[expert_name]["label_confidence"].append(
                case_catalog["label_confidence"][select_mask].astype(np.float32)
            )

        manifest_cases.append(
            {
                "case_id": case_id,
                "spec": asdict(spec),
                "grid": {"nx": nx, "ny": ny, "nt": nt},
                "label_histogram": np.bincount(
                    case_catalog["label"],
                    minlength=len(REGION_NAMES),
                ).tolist(),
            }
        )

    catalog_arrays = {
        key: np.concatenate(parts, axis=0) if key != "probs" else np.concatenate(parts, axis=0)
        for key, parts in catalog_parts.items()
    }
    np.savez_compressed(os.path.join(output_dir, "catalog.npz"), **catalog_arrays)

    expert_summary = {}
    for expert_name, parts in expert_parts.items():
        export = {key: np.concatenate(value, axis=0) for key, value in parts.items()}
        if target_points_per_expert is not None and export["coords"].shape[0] > 0:
            prob = export["weight"].astype(np.float64)
            prob = prob / np.clip(prob.sum(), 1e-12, None)
            sample_idx = rng.choice(
                export["coords"].shape[0],
                size=target_points_per_expert,
                replace=export["coords"].shape[0] < target_points_per_expert,
                p=prob,
            )
            export = {
                key: value[sample_idx]
                for key, value in export.items()
            }
        np.savez_compressed(os.path.join(output_dir, f"expert_{expert_name}.npz"), **export)
        expert_summary[expert_name] = {
            "num_points": int(export["coords"].shape[0]),
            "mean_weight": float(export["weight"].mean()),
            "mean_confidence": float(export["label_confidence"].mean()),
        }

    manifest = {
        "num_cases": num_cases,
        "seed": seed,
        "grid": {"nx": nx, "ny": ny, "nt": nt},
        "nu": nu,
        "target_points_per_expert": target_points_per_expert,
        "region_rules": asdict(region_rules),
        "expert_export_rules": {
            "expert_keep_quantile": expert_keep_quantile,
            "expert_min_weight": expert_min_weight,
            "expert_confidence_quantile": expert_confidence_quantile,
            "expert_min_confidence": expert_min_confidence,
        },
        "region_names": REGION_NAMES,
        "expert_region_names": EXPERT_REGION_NAMES,
        "catalog_num_points": int(catalog_arrays["coords"].shape[0]),
        "label_histogram": {name: int(label_hist[idx]) for idx, name in enumerate(REGION_NAMES)},
        "expert_summary": expert_summary,
        "cases": manifest_cases,
    }
    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest
