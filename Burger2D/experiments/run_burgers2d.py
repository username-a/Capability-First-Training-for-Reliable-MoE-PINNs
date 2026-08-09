"""
2D Burgers experiments: Vanilla PINN, end-to-end MoE, and staged MoE.
"""

from __future__ import annotations

import json
import os
import random
import sys
from typing import Any, Optional

import numpy as np
import torch

from Burger2D.data import REGION_NAMES, classify_region_maps, compute_region_feature_maps

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Burger2D.core.models import VanillaPINN
from Burger2D.core.moe_pinn import build_burgers2d_moe
from Burger2D.equations.burgers2d import Burgers2DProblem, ReferenceSolution2D
from Burger2D.training.losses import (
    LossConfig2D,
    PhysicsLoss2D,
    l2_relative_error,
    max_absolute_error,
    steep_region_metrics,
)
from Burger2D.training.staged_burgers2d import (
    StagedBurgers2DConfig,
    build_joint_base_consistency_batch,
    build_joint_branch_consistency_batch,
    build_gate_targets,
    build_joint_gate_supervision_batch,
    build_joint_rotation_supervision_batch,
    build_joint_training_batch,
    build_specialist_batches,
    build_specialist_batches_from_curated_dataset,
    compute_region_scores,
    flatten_reference_solution,
    joint_finetune,
    pretrain_base_model,
    pretrain_experts,
    pretrain_rotation_layer,
    shock_priority_from_region_scores,
    steep_mask_from_reference,
    train_gate,
)
from Burger2D.training.trainer import Trainer
from Burger2D.visualization.plots import (
    plot_centerline_slices,
    plot_directional_diagnostics,
    plot_expert_signed_error_maps,
    plot_gating_maps,
    plot_model_metric_comparison,
    plot_rotation_gate_diagnostics,
    plot_snapshot_comparison,
    plot_training_curves,
)


DEFAULT_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
NU = 0.01 / np.pi
RESULTS_ROOT = os.path.join(PACKAGE_ROOT, "results")


def _set_global_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


_set_global_seed(DEFAULT_SEED)


def _predict_in_batches(
    fn,
    xyt: torch.Tensor,
    *,
    batch_size: int = 65536,
) -> torch.Tensor:
    owner = getattr(fn, "__self__", None)
    if owner is not None:
        batch_size = min(batch_size, int(getattr(owner, "inference_batch_size", batch_size)))
    else:
        batch_size = min(batch_size, int(getattr(fn, "inference_batch_size", batch_size)))
    outputs = []
    for start in range(0, xyt.shape[0], batch_size):
        outputs.append(fn(xyt[start:start + batch_size]))
    return torch.cat(outputs, dim=0)


def _predict_grid(
    model: torch.nn.Module,
    reference: ReferenceSolution2D,
    *,
    xyt_test: torch.Tensor,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred = _predict_in_batches(model, xyt_test).cpu().numpy()
    return pred.reshape(reference.u.shape)


def _predict_gates(
    model: torch.nn.Module,
    reference: ReferenceSolution2D,
    *,
    xyt_test: torch.Tensor,
) -> np.ndarray:
    with torch.no_grad():
        gates = _predict_in_batches(model.get_gate_weights, xyt_test).cpu().numpy()
    return gates.reshape(reference.u.shape[0], reference.u.shape[1], reference.u.shape[2], -1)


def _reference_gradient_features(reference: ReferenceSolution2D) -> dict[str, np.ndarray]:
    grad_mag = np.zeros_like(reference.u, dtype=np.float32)
    grad_angle = np.zeros_like(reference.u, dtype=np.float32)
    for i in range(reference.u.shape[0]):
        du_dy, du_dx = np.gradient(reference.u[i], reference.y, reference.x, edge_order=1)
        grad_mag[i] = np.sqrt(du_dx**2 + du_dy**2)
        grad_angle[i] = np.arctan2(du_dy, du_dx)
    threshold = float(np.quantile(grad_mag, 0.90))
    steep_mask = grad_mag >= threshold
    return {
        "grad_mag": grad_mag,
        "grad_angle": grad_angle,
        "steep_mask": steep_mask,
        "steep_threshold": np.array(threshold, dtype=np.float32),
    }


def _compute_directional_gate_diagnostics(
    model: torch.nn.Module,
    reference: ReferenceSolution2D,
    gradient_features: dict[str, np.ndarray],
    *,
    xyt_test: torch.Tensor,
) -> dict[str, Any]:
    if not hasattr(model, "gating"):
        return {}

    gates = _predict_gates(model, reference, xyt_test=xyt_test)
    steep_mask = gradient_features["steep_mask"].reshape(-1)
    grad_angle = gradient_features["grad_angle"].reshape(-1)
    gate_flat = gates.reshape(-1, gates.shape[-1])

    steep_angles = grad_angle[steep_mask]
    steep_gates = gate_flat[steep_mask]
    num_bins = 12
    edges = np.linspace(-np.pi, np.pi, num_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    steep_density = np.histogram(steep_angles, bins=edges, density=True)[0]
    mean_gate_by_bin = np.zeros((num_bins, gate_flat.shape[-1]), dtype=np.float32)
    for idx in range(num_bins):
        if idx == num_bins - 1:
            mask = (steep_angles >= edges[idx]) & (steep_angles <= edges[idx + 1])
        else:
            mask = (steep_angles >= edges[idx]) & (steep_angles < edges[idx + 1])
        if np.any(mask):
            mean_gate_by_bin[idx] = steep_gates[mask].mean(axis=0)

    dominant_expert_idx = mean_gate_by_bin.argmax(axis=1)
    expert_names = getattr(model, "expert_names", [f"expert_{i}" for i in range(gate_flat.shape[-1])])
    return {
        "bin_centers_deg": np.degrees(centers).tolist(),
        "steep_density": steep_density.tolist(),
        "mean_gate_by_bin": mean_gate_by_bin.tolist(),
        "dominant_expert_by_bin": [expert_names[i] for i in dominant_expert_idx],
    }


def _compute_rotation_alignment_metrics(
    model: torch.nn.Module,
    reference: ReferenceSolution2D,
    gradient_features: dict[str, np.ndarray],
    *,
    xyt_test: torch.Tensor,
) -> dict[str, Any]:
    if not hasattr(model, "get_rotation_state"):
        return {}

    layout_variant = getattr(model, "expert_layout_variant", "categorical")
    region_scores = compute_region_scores(reference, layout_variant=layout_variant)
    if layout_variant == "categorical":
        focus = np.maximum(region_scores["directional_shock"], region_scores["wave"])
    else:
        focus = np.maximum(region_scores["anisotropy_directional"], region_scores["curvature_wave"])
    focus = focus.reshape(-1)
    grad_norm = gradient_features["grad_mag"].reshape(-1)
    grad_norm = grad_norm / max(float(np.quantile(grad_norm, 0.90)), 1e-8)
    focus = focus / max(float(focus.max()), 1e-8)
    focus_mask = (grad_norm >= np.quantile(grad_norm, 0.78)) & (focus >= 0.10)
    if not np.any(focus_mask):
        return {}

    batch_size = int(getattr(model, "inference_batch_size", xyt_test.shape[0]))
    pred_angles = []
    pred_concentration = []
    for start in range(0, xyt_test.shape[0], batch_size):
        chunk = xyt_test[start:start + batch_size]
        rot_state = model.get_rotation_state(chunk)
        if rot_state is None:
            return {}
        pred_angles.append(rot_state["rotation_angle"].detach().cpu().numpy())
        pred_concentration.append(rot_state["concentration"].detach().cpu().numpy())

    pred_angle = np.concatenate(pred_angles, axis=0)
    pred_concentration = np.concatenate(pred_concentration, axis=0)
    target_angle = gradient_features["grad_angle"].reshape(-1)

    axial_delta = np.angle(np.exp(1j * 2.0 * (pred_angle - target_angle)))
    axial_error_deg = np.degrees(np.abs(axial_delta) * 0.5)
    focus_weights = 0.5 * grad_norm + 0.5 * focus
    focus_weights = np.clip(focus_weights, 1e-8, None)
    focus_weights = focus_weights[focus_mask]
    focus_weights = focus_weights / focus_weights.sum()

    return {
        "rotation_focus_axis_error_deg": float(np.sum(axial_error_deg[focus_mask] * focus_weights)),
        "rotation_focus_concentration_mean": float(pred_concentration[focus_mask].mean()),
        "rotation_focus_coverage": float(focus_mask.mean()),
    }


def _compute_rotation_visual_diagnostics(
    model: torch.nn.Module,
    reference: ReferenceSolution2D,
    gradient_features: dict[str, np.ndarray],
    *,
    xyt_test: torch.Tensor,
    gates: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    if not hasattr(model, "get_rotation_state"):
        return {}

    layout_variant = getattr(model, "expert_layout_variant", "categorical")
    region_scores = compute_region_scores(reference, layout_variant=layout_variant)
    if layout_variant == "categorical":
        focus_raw = np.maximum(region_scores["directional_shock"], region_scores["wave"])
        oriented_names = {"directional_shock", "wave"}
    else:
        focus_raw = np.maximum(region_scores["anisotropy_directional"], region_scores["curvature_wave"])
        oriented_names = {"anisotropy_directional", "curvature_wave"}

    batch_size = int(getattr(model, "inference_batch_size", xyt_test.shape[0]))
    rotation_angle = []
    activation = []
    confidence = []
    concentration = []
    context_focus_chunks = []
    for start in range(0, xyt_test.shape[0], batch_size):
        chunk = xyt_test[start:start + batch_size]
        rot_state = model.get_rotation_state(chunk)
        if rot_state is None:
            return {}
        rotation_angle.append(rot_state["rotation_angle"].detach().cpu().numpy())
        activation.append(rot_state["activation"].detach().cpu().numpy())
        confidence.append(rot_state["max_prob"].detach().cpu().numpy())
        concentration.append(rot_state["concentration"].detach().cpu().numpy())
        if "context_focus" in rot_state:
            context_focus_chunks.append(rot_state["context_focus"].detach().cpu().numpy())

    grid_shape = reference.u.shape
    rotation_angle = np.concatenate(rotation_angle, axis=0).reshape(grid_shape)
    rotation_activation = np.concatenate(activation, axis=0).reshape(grid_shape)
    rotation_confidence = np.concatenate(confidence, axis=0).reshape(grid_shape)
    rotation_concentration = np.concatenate(concentration, axis=0).reshape(grid_shape)
    if context_focus_chunks:
        context_focus = np.concatenate(context_focus_chunks, axis=0).reshape(grid_shape)
    else:
        context_focus = np.ones_like(rotation_activation, dtype=np.float32)

    if gates is None and hasattr(model, "gating"):
        gates = _predict_gates(model, reference, xyt_test=xyt_test)

    expert_names = getattr(model, "expert_names", [])
    oriented_indices = [idx for idx, name in enumerate(expert_names) if name in oriented_names]
    if gates is not None and oriented_indices:
        oriented_gate_mass = gates[..., oriented_indices].sum(axis=-1)
        dominant_expert_idx = gates.argmax(axis=-1).astype(np.int32)
        gate_entropy = -(gates * np.log(np.clip(gates, 1e-8, 1.0))).sum(axis=-1)
    else:
        oriented_gate_mass = np.zeros_like(reference.u, dtype=np.float32)
        dominant_expert_idx = np.zeros_like(reference.u, dtype=np.int32)
        gate_entropy = np.zeros_like(reference.u, dtype=np.float32)

    grad_mag = gradient_features["grad_mag"]
    grad_angle = gradient_features["grad_angle"]
    grad_mag_norm = grad_mag / max(float(np.quantile(grad_mag, 0.90)), 1e-8)
    focus_norm = focus_raw / max(float(focus_raw.max()), 1e-8)
    focus_map = np.clip(0.55 * np.clip(focus_norm, 0.0, 1.0) + 0.45 * np.clip(grad_mag_norm, 0.0, 1.0), 0.0, 1.0)

    focus_mask = (np.clip(grad_mag_norm, 0.0, 1.0) >= float(np.quantile(np.clip(grad_mag_norm, 0.0, 1.0), 0.78))) & (
        focus_norm >= 0.10
    )
    if not np.any(focus_mask):
        focus_mask = focus_map >= float(np.quantile(focus_map.reshape(-1), 0.85))

    axial_delta = np.angle(np.exp(1j * 2.0 * (rotation_angle - grad_angle)))
    axis_error_deg = np.degrees(np.abs(axial_delta) * 0.5).astype(np.float32)
    angle_display_deg = ((np.degrees(rotation_angle) + 90.0) % 180.0) - 90.0

    focus_weights = focus_map[focus_mask]
    focus_weights = np.clip(focus_weights, 1e-8, None)
    focus_weights = focus_weights / focus_weights.sum()

    high_conf_threshold = float(np.quantile(rotation_confidence[focus_mask], 0.75))
    high_act_threshold = max(0.10, float(np.quantile(rotation_activation[focus_mask], 0.70)))
    high_conf_mask = focus_mask & (rotation_confidence >= high_conf_threshold)
    high_act_mask = focus_mask & (rotation_activation >= high_act_threshold)
    low_gate_mask = oriented_gate_mass < 0.35

    summary = {
        "focus_fraction": float(focus_mask.mean()),
        "rotation_focus_axis_error_deg_weighted": float(np.sum(axis_error_deg[focus_mask] * focus_weights)),
        "focus_mean_activation": float(rotation_activation[focus_mask].mean()),
        "background_mean_activation": float(rotation_activation[~focus_mask].mean()),
        "focus_mean_concentration": float(rotation_concentration[focus_mask].mean()),
        "background_mean_concentration": float(rotation_concentration[~focus_mask].mean()),
        "focus_mean_confidence": float(rotation_confidence[focus_mask].mean()),
        "focus_mean_context_focus": float(context_focus[focus_mask].mean()),
        "background_mean_context_focus": float(context_focus[~focus_mask].mean()),
        "focus_mean_oriented_gate_mass": float(oriented_gate_mass[focus_mask].mean()),
        "background_mean_oriented_gate_mass": float(oriented_gate_mass[~focus_mask].mean()),
        "focus_mean_gate_entropy": float(gate_entropy[focus_mask].mean()),
        "background_mean_gate_entropy": float(gate_entropy[~focus_mask].mean()),
        "high_conf_low_gate_fraction": float(low_gate_mask[high_conf_mask].mean()) if np.any(high_conf_mask) else 0.0,
        "high_activation_low_gate_fraction": float(low_gate_mask[high_act_mask].mean()) if np.any(high_act_mask) else 0.0,
        "high_confidence_threshold": high_conf_threshold,
        "high_activation_threshold": high_act_threshold,
        "oriented_expert_names": [expert_names[idx] for idx in oriented_indices],
    }
    return {
        "focus_map": focus_map.astype(np.float32),
        "rotation_angle_deg": angle_display_deg.astype(np.float32),
        "rotation_activation": rotation_activation.astype(np.float32),
        "rotation_confidence": rotation_confidence.astype(np.float32),
        "rotation_concentration": rotation_concentration.astype(np.float32),
        "context_focus": context_focus.astype(np.float32),
        "oriented_gate_mass": oriented_gate_mass.astype(np.float32),
        "gate_entropy": gate_entropy.astype(np.float32),
        "dominant_expert_idx": dominant_expert_idx.astype(np.int32),
        "axis_error_deg": axis_error_deg.astype(np.float32),
        "focus_mask": focus_mask.astype(np.bool_),
        "summary": summary,
    }


def _write_rotation_diagnostics_report(
    save_dir: str,
    diagnostics: dict[str, Any],
) -> None:
    summary = diagnostics.get("summary", {})
    if not summary:
        return

    lines = [
        "# Rotation Diagnostics Report",
        "",
        "This report explains the joint rotation-gate visualization for the rotation side branch.",
        "",
        "## Terms",
        "",
        "- `focus map`: a normalized map that combines directional/wave region score and reference gradient magnitude. Brighter means the location is more likely to need direction-sensitive handling.",
        "- `rotation axis (deg)`: the axial orientation predicted by the rotation layer. Because it is axial, `theta` and `theta + 180 deg` are equivalent.",
        "- `rotation activation`: how strongly the high-threshold rotation layer decides to intervene before direction-sensitive experts receive rotated coordinates.",
        "- `oriented gate mass`: total gate weight assigned to direction-sensitive experts, currently the directional-shock and wave experts.",
        "- `dominant expert`: the top-1 expert selected by the gate at each point.",
        "",
        "## Summary",
        "",
        f"- Focus-region coverage: `{summary['focus_fraction']:.4f}`",
        f"- Weighted axis error in focus region: `{summary['rotation_focus_axis_error_deg_weighted']:.2f} deg`",
        f"- Focus activation vs background: `{summary['focus_mean_activation']:.4f}` vs `{summary['background_mean_activation']:.4f}`",
        f"- Focus concentration vs background: `{summary['focus_mean_concentration']:.6f}` vs `{summary['background_mean_concentration']:.6f}`",
        f"- Focus context-focus vs background: `{summary['focus_mean_context_focus']:.4f}` vs `{summary['background_mean_context_focus']:.4f}`",
        f"- Focus oriented gate mass vs background: `{summary['focus_mean_oriented_gate_mass']:.4f}` vs `{summary['background_mean_oriented_gate_mass']:.4f}`",
        f"- Focus gate entropy vs background: `{summary['focus_mean_gate_entropy']:.4f}` vs `{summary['background_mean_gate_entropy']:.4f}`",
        f"- High-confidence but low oriented-gate fraction: `{summary['high_conf_low_gate_fraction']:.4f}`",
        f"- High-activation but low oriented-gate fraction: `{summary['high_activation_low_gate_fraction']:.4f}`",
        "",
        "## Interpretation",
        "",
    ]

    if summary["focus_mean_activation"] <= summary["background_mean_activation"] * 1.1:
        lines.append("- Rotation activation is only weakly higher in focus regions, so the layer still intervenes too conservatively.")
    else:
        lines.append("- Rotation activation is visibly concentrated in focus regions, so the main remaining question is whether routing converts that signal into expert usage.")

    if summary["high_conf_low_gate_fraction"] >= 0.35:
        lines.append("- A large fraction of high-confidence rotation points still receive low oriented-expert gate mass, which means routing is not fully trusting the learned direction signal yet.")
    else:
        lines.append("- Once the rotation layer becomes confident, routing usually follows; the bottleneck is now more about producing enough confident activation.")

    if summary["focus_mean_oriented_gate_mass"] <= summary["background_mean_oriented_gate_mass"] * 1.05:
        lines.append("- Direction-sensitive experts are not receiving much more total gate mass in focus regions than in background regions, so the current gate remains cautious.")
    else:
        lines.append("- Direction-sensitive experts already receive more route mass in focus regions, so the next improvement should target stronger activation rather than basic routing awareness.")

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `rotation_gate_diagnostics.png`: joint visualization of focus, rotation angle, activation, oriented-expert gate mass, and dominant expert.",
            "- `rotation_diagnostics_maps.npz`: raw diagnostic arrays for follow-up analysis or paper figures.",
            "- `rotation_diagnostics.json`: scalar summary for experiment tracking.",
            "",
        ]
    )

    with open(os.path.join(save_dir, "rotation_diagnostics_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _compute_expert_diagnostics(
    model: torch.nn.Module,
    reference: ReferenceSolution2D,
    gradient_features: dict[str, np.ndarray],
    *,
    xyt_test: torch.Tensor,
    u_exact_flat: torch.Tensor,
) -> tuple[dict[str, Any], Optional[np.ndarray]]:
    if not hasattr(model, "experts"):
        return {}, None

    with torch.no_grad():
        expert_pred_flat = model.get_expert_predictions(xyt_test).cpu().numpy()
    expert_pred = expert_pred_flat.reshape(reference.u.shape[0], reference.u.shape[1], reference.u.shape[2], -1)
    steep_mask = gradient_features["steep_mask"]
    metrics: dict[str, Any] = {}

    for idx, name in enumerate(getattr(model, "expert_names", [f"expert_{i}" for i in range(expert_pred.shape[-1])])):
        pred = expert_pred[:, :, :, idx]
        abs_err = np.abs(pred - reference.u)
        metrics[name] = {
            "l2_relative_error": float(
                np.linalg.norm(pred - reference.u) / (np.linalg.norm(reference.u) + 1e-10)
            ),
            "max_absolute_error": float(abs_err.max()),
            "steep_mae": float(abs_err[steep_mask].mean()),
            "background_mae": float(abs_err[~steep_mask].mean()),
        }

    return metrics, expert_pred


def _compute_directional_stress_spec(reference: ReferenceSolution2D) -> dict[str, Any]:
    feature_maps = compute_region_feature_maps(reference)
    class_maps = classify_region_maps(feature_maps)
    directional_idx = REGION_NAMES.index("directional_shock")

    labels = class_maps["labels"]
    confidence = class_maps["confidence"]
    probs = class_maps["probs"][..., directional_idx]
    grad_norm = feature_maps["grad_norm"]
    anisotropy = feature_maps["anisotropy"]
    time_frac = feature_maps["time_frac"]

    directional_mask = labels == directional_idx
    if np.any(directional_mask):
        conf_thresh = float(np.quantile(confidence[directional_mask], 0.70))
        prob_thresh = float(np.quantile(probs[directional_mask], 0.70))
        grad_thresh = float(np.quantile(grad_norm[directional_mask], 0.65))
        aniso_thresh = float(np.quantile(anisotropy[directional_mask], 0.65))
    else:
        conf_thresh = float(np.quantile(confidence.reshape(-1), 0.85))
        prob_thresh = float(np.quantile(probs.reshape(-1), 0.85))
        grad_thresh = float(np.quantile(grad_norm.reshape(-1), 0.90))
        aniso_thresh = float(np.quantile(anisotropy.reshape(-1), 0.90))

    stress_mask = (
        directional_mask
        & (confidence >= conf_thresh)
        & (probs >= prob_thresh)
        & (grad_norm >= grad_thresh)
        & (anisotropy >= aniso_thresh)
        & (time_frac >= 0.30)
    )
    if int(stress_mask.sum()) < 64:
        fallback_mask = (
            directional_mask
            & (confidence >= conf_thresh)
            & (grad_norm >= grad_thresh)
            & (time_frac >= 0.25)
        )
        if int(fallback_mask.sum()) >= 64:
            stress_mask = fallback_mask

    return {
        "feature_maps": feature_maps,
        "class_maps": class_maps,
        "directional_idx": directional_idx,
        "stress_mask": stress_mask,
        "stress_thresholds": {
            "confidence_min": conf_thresh,
            "directional_prob_min": prob_thresh,
            "grad_norm_min": grad_thresh,
            "anisotropy_min": aniso_thresh,
            "time_frac_min": 0.30,
        },
    }


def _subset_error_metrics(
    pred: np.ndarray,
    ref: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    abs_err = np.abs(pred - ref)
    masked_err = abs_err[mask]
    ref_masked = ref[mask]
    pred_masked = pred[mask]
    return {
        "count": int(mask.sum()),
        "fraction": float(mask.mean()),
        "mae": float(masked_err.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(masked_err)))),
        "max_error": float(masked_err.max()),
        "l2_relative_error": float(
            np.linalg.norm(pred_masked - ref_masked) / (np.linalg.norm(ref_masked) + 1e-10)
        ),
    }


def _compute_directional_stress_metrics(
    model: torch.nn.Module,
    reference: ReferenceSolution2D,
    *,
    u_pred: np.ndarray,
    xyt_test: torch.Tensor,
) -> dict[str, Any]:
    spec = _compute_directional_stress_spec(reference)
    stress_mask = spec["stress_mask"]
    if int(stress_mask.sum()) == 0:
        return {"stress_mask_count": 0}

    metrics: dict[str, Any] = {
        "stress_mask_count": int(stress_mask.sum()),
        "stress_mask_fraction": float(stress_mask.mean()),
        "stress_thresholds": spec["stress_thresholds"],
        "mixture": _subset_error_metrics(u_pred, reference.u, stress_mask),
        "directional_branch_name": getattr(model, "directional_branch_name", None),
    }

    if hasattr(model, "experts"):
        with torch.no_grad():
            expert_pred_flat = model.get_expert_predictions(xyt_test).cpu().numpy()
        expert_pred = expert_pred_flat.reshape(reference.u.shape[0], reference.u.shape[1], reference.u.shape[2], -1)
        expert_metrics: dict[str, Any] = {}
        for idx, name in enumerate(getattr(model, "expert_names", [f"expert_{i}" for i in range(expert_pred.shape[-1])])):
            expert_metrics[name] = _subset_error_metrics(expert_pred[:, :, :, idx], reference.u, stress_mask)
        metrics["experts"] = expert_metrics

    return metrics


def _evaluate_metrics(
    model: torch.nn.Module,
    reference: ReferenceSolution2D,
    *,
    xyt_test: torch.Tensor,
    u_exact_flat: torch.Tensor,
) -> tuple[dict[str, Any], np.ndarray]:
    model.eval()
    with torch.no_grad():
        pred_flat = _predict_in_batches(model, xyt_test)
    u_pred = pred_flat.cpu().numpy().reshape(reference.u.shape)

    metrics: dict[str, Any] = {
        "l2_relative_error": l2_relative_error(pred_flat, u_exact_flat),
        "max_absolute_error": max_absolute_error(pred_flat, u_exact_flat),
    }
    metrics.update(
        steep_region_metrics(
            u_pred=u_pred,
            u_ref=reference.u,
            x=reference.x,
            y=reference.y,
            quantile=0.90,
        )
    )
    gradient_features = _reference_gradient_features(reference)

    if hasattr(model, "gating"):
        stats = model.load_balance_stats(xyt_test)
        metrics["route_entropy"] = float(stats["mean_entropy"])
        metrics["route_max_weight"] = float(stats["max_gate_weight"])
        metrics["expert_load_frac"] = [float(v) for v in stats["expert_load_frac"]]
    if hasattr(model, "get_rotation_state"):
        batch_size = int(getattr(model, "inference_batch_size", xyt_test.shape[0]))
        act_sum = 0.0
        raw_act_sum = 0.0
        conf_sum = 0.0
        ent_sum = 0.0
        conc_sum = 0.0
        focus_sum = 0.0
        count = 0
        for start in range(0, xyt_test.shape[0], batch_size):
            chunk = xyt_test[start:start + batch_size]
            rot_state = model.get_rotation_state(chunk)
            if rot_state is None:
                break
            act_sum += float(rot_state["activation"].sum().item())
            if "raw_activation" in rot_state:
                raw_act_sum += float(rot_state["raw_activation"].sum().item())
            conf_sum += float(rot_state["max_prob"].sum().item())
            ent_sum += float(rot_state["entropy"].sum().item())
            if "concentration" in rot_state:
                conc_sum += float(rot_state["concentration"].sum().item())
            if "context_focus" in rot_state:
                focus_sum += float(rot_state["context_focus"].sum().item())
            count += int(chunk.shape[0])
        if count > 0:
            metrics["rotation_activation_mean"] = act_sum / count
            if raw_act_sum > 0:
                metrics["rotation_raw_activation_mean"] = raw_act_sum / count
            metrics["rotation_confidence_mean"] = conf_sum / count
            metrics["rotation_entropy_mean"] = ent_sum / count
            if conc_sum > 0:
                metrics["rotation_concentration_mean"] = conc_sum / count
            if focus_sum > 0:
                metrics["rotation_context_focus_mean"] = focus_sum / count
        metrics.update(
            _compute_rotation_alignment_metrics(
                model,
                reference,
                gradient_features,
                xyt_test=xyt_test,
            )
        )

    return metrics, u_pred


def _save_metrics(save_dir: str, metrics: dict[str, Any]) -> None:
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(os.path.join(save_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        for key, value in metrics.items():
            if isinstance(value, list):
                f.write(f"{key}: {value}\n")
            else:
                f.write(f"{key}: {float(value):.8e}\n")


def _save_artifacts(
    *,
    save_dir: str,
    reference: ReferenceSolution2D,
    model: torch.nn.Module,
    history: dict[str, list[float]],
    metrics: dict[str, Any],
    stress_metrics: dict[str, Any],
    u_pred: np.ndarray,
    xyt_test: torch.Tensor,
    u_exact_flat: torch.Tensor,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(save_dir, "reference_and_prediction.npz"),
        x=reference.x,
        y=reference.y,
        t=reference.t,
        u_ref=reference.u,
        u_pred=u_pred,
    )
    _save_metrics(save_dir, metrics)
    with open(os.path.join(save_dir, "directional_stress_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(stress_metrics, f, ensure_ascii=False, indent=2)
    plot_training_curves(history, os.path.join(save_dir, "training_curves.png"))
    plot_snapshot_comparison(
        reference.x,
        reference.y,
        reference.t,
        reference.u,
        u_pred,
        os.path.join(save_dir, "snapshot_comparison.png"),
    )
    plot_centerline_slices(
        reference.x,
        reference.y,
        reference.t,
        reference.u,
        u_pred,
        os.path.join(save_dir, "centerline_slices.png"),
    )

    gradient_features = _reference_gradient_features(reference)

    if hasattr(model, "gating"):
        gates = _predict_gates(model, reference, xyt_test=xyt_test)
        plot_gating_maps(
            reference.x,
            reference.y,
            reference.t,
            gates,
            expert_names=getattr(model, "expert_names", [f"expert_{i}" for i in range(gates.shape[-1])]),
            save_path=os.path.join(save_dir, "gating_maps.png"),
        )
        directional = _compute_directional_gate_diagnostics(
            model,
            reference,
            gradient_features,
            xyt_test=xyt_test,
        )
        with open(os.path.join(save_dir, "directional_diagnostics.json"), "w", encoding="utf-8") as f:
            json.dump(directional, f, ensure_ascii=False, indent=2)
        plot_directional_diagnostics(
            np.asarray(directional["bin_centers_deg"]),
            np.asarray(directional["steep_density"]),
            np.asarray(directional["mean_gate_by_bin"]),
            expert_names=getattr(model, "expert_names", [f"expert_{i}" for i in range(gates.shape[-1])]),
            save_path=os.path.join(save_dir, "directional_diagnostics.png"),
        )
        if hasattr(model, "get_rotation_state"):
            rotation_diag = _compute_rotation_visual_diagnostics(
                model,
                reference,
                gradient_features,
                xyt_test=xyt_test,
                gates=gates,
            )
            if rotation_diag:
                with open(os.path.join(save_dir, "rotation_diagnostics.json"), "w", encoding="utf-8") as f:
                    json.dump(rotation_diag["summary"], f, ensure_ascii=False, indent=2)
                np.savez_compressed(
                    os.path.join(save_dir, "rotation_diagnostics_maps.npz"),
                    focus_map=rotation_diag["focus_map"],
                    rotation_angle_deg=rotation_diag["rotation_angle_deg"],
                    rotation_activation=rotation_diag["rotation_activation"],
                    rotation_confidence=rotation_diag["rotation_confidence"],
                    rotation_concentration=rotation_diag["rotation_concentration"],
                    context_focus=rotation_diag["context_focus"],
                    oriented_gate_mass=rotation_diag["oriented_gate_mass"],
                    gate_entropy=rotation_diag["gate_entropy"],
                    dominant_expert_idx=rotation_diag["dominant_expert_idx"],
                    axis_error_deg=rotation_diag["axis_error_deg"],
                    focus_mask=rotation_diag["focus_mask"],
                )
                plot_rotation_gate_diagnostics(
                    reference.x,
                    reference.y,
                    reference.t,
                    rotation_diag["focus_map"],
                    rotation_diag["rotation_angle_deg"],
                    rotation_diag["rotation_activation"],
                    rotation_diag["oriented_gate_mass"],
                    rotation_diag["dominant_expert_idx"],
                    expert_names=getattr(model, "expert_names", [f"expert_{i}" for i in range(gates.shape[-1])]),
                    save_path=os.path.join(save_dir, "rotation_gate_diagnostics.png"),
                )
                _write_rotation_diagnostics_report(save_dir, rotation_diag)

    expert_metrics, expert_pred = _compute_expert_diagnostics(
        model,
        reference,
        gradient_features,
        xyt_test=xyt_test,
        u_exact_flat=u_exact_flat,
    )
    if expert_metrics and expert_pred is not None:
        with open(os.path.join(save_dir, "expert_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(expert_metrics, f, ensure_ascii=False, indent=2)
        plot_expert_signed_error_maps(
            reference.x,
            reference.y,
            reference.t,
            reference.u,
            expert_pred,
            expert_names=getattr(model, "expert_names", [f"expert_{i}" for i in range(expert_pred.shape[-1])]),
            save_path=os.path.join(save_dir, "expert_signed_error_maps.png"),
        )


def _train_vanilla(
    batch: dict[str, torch.Tensor],
    xyt_test: torch.Tensor,
    u_exact_flat: torch.Tensor,
    *,
    n_steps: int,
    device: torch.device,
    save_dir: str,
) -> tuple[torch.nn.Module, dict[str, list[float]]]:
    print("\n" + "=" * 60)
    print("Training 2D Vanilla PINN")
    print("=" * 60)

    model = VanillaPINN(
        in_dim=3,
        out_dim=1,
        hidden=96,
        depth=5,
        activation="tanh",
        output_transform="burgers2d_hard_icbc",
    ).to(device).to(DTYPE)
    print(f"Vanilla parameters: {model.count_parameters():,}")

    loss_fn = PhysicsLoss2D(LossConfig2D(nu=NU, w_res=1.0, w_ic=5.0, w_bc=2.0))

    def eval_l2() -> float:
        with torch.no_grad():
            pred = _predict_in_batches(model, xyt_test)
        return l2_relative_error(pred, u_exact_flat)

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        lr=1e-3,
        n_steps=n_steps,
        device=device,
        save_dir=save_dir,
    )
    history = trainer.train(batch, eval_fn=eval_l2, eval_freq=max(50, n_steps // 10), log_freq=max(25, n_steps // 20))
    trainer.save_checkpoint("burgers2d_vanilla.pt")
    return model, history


def _train_moe_end_to_end(
    batch: dict[str, torch.Tensor],
    xyt_test: torch.Tensor,
    u_exact_flat: torch.Tensor,
    *,
    n_steps: int,
    device: torch.device,
    save_dir: str,
    directional_expert_variant: str = "hybrid",
    wave_expert_variant: str = "base",
    expert_layout_variant: str = "categorical",
    attribute_expert_variant: str = "base",
    gate_variant: str = "pointwise",
    rotation_variant: str = "none",
    exclude_experts: tuple[str, ...] = (),
    extra_experts: tuple[str, ...] = (),
    balance_weight_override: float | None = None,
) -> tuple[torch.nn.Module, dict[str, list[float]]]:
    print("\n" + "=" * 60)
    print("Training 2D MoE-PINN end-to-end")
    print("=" * 60)

    model = build_burgers2d_moe(
        directional_expert_variant=directional_expert_variant,
        wave_expert_variant=wave_expert_variant,
        expert_layout_variant=expert_layout_variant,
        attribute_expert_variant=attribute_expert_variant,
        gate_variant=gate_variant,
        rotation_variant=rotation_variant,
        exclude_experts=exclude_experts,
        extra_experts=extra_experts,
    ).to(device).to(DTYPE)
    print(f"MoE parameters: {model.count_parameters()['total']:,}")

    balance_w = balance_weight_override if balance_weight_override is not None else model.balance_weight
    loss_fn = PhysicsLoss2D(
        LossConfig2D(
            nu=NU,
            w_res=1.0,
            w_ic=5.0,
            w_bc=2.0,
            w_sparse=model.sparsity_weight,
            w_balance=balance_w,
        )
    )

    def eval_l2() -> float:
        with torch.no_grad():
            pred = _predict_in_batches(model, xyt_test)
        return l2_relative_error(pred, u_exact_flat)

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        lr=1e-3,
        n_steps=n_steps,
        device=device,
        save_dir=save_dir,
    )
    history = trainer.train(batch, eval_fn=eval_l2, eval_freq=max(50, n_steps // 10), log_freq=max(25, n_steps // 20))
    trainer.save_checkpoint("burgers2d_moe_end_to_end.pt")
    return model, history


def _train_moe_staged(
    problem: Burgers2DProblem,
    batch: dict[str, torch.Tensor],
    reference: ReferenceSolution2D,
    xyt_test: torch.Tensor,
    u_exact_flat: torch.Tensor,
    *,
    n_steps: int,
    device: torch.device,
    save_dir: str,
    staged_variant: str = "gate_only_joint",
    expert_dataset_dir: Optional[str] = None,
    directional_expert_variant: str = "hybrid",
    wave_expert_variant: str = "base",
    expert_layout_variant: str = "categorical",
    attribute_expert_variant: str = "base",
    gate_variant: str = "pointwise",
    rotation_variant: str = "none",
    exclude_experts: tuple[str, ...] = (),
    extra_experts: tuple[str, ...] = (),
    expert_pretrain_fraction: float = 1.0,
    joint_steps_override: int | None = None,
) -> tuple[torch.nn.Module, dict[str, list[float]], dict[str, Any]]:
    print("\n" + "=" * 60)
    print("Training 2D MoE-PINN with staged specialists")
    print("=" * 60)

    model = build_burgers2d_moe(
        directional_expert_variant=directional_expert_variant,
        wave_expert_variant=wave_expert_variant,
        expert_layout_variant=expert_layout_variant,
        attribute_expert_variant=attribute_expert_variant,
        gate_variant=gate_variant,
        rotation_variant=rotation_variant,
        exclude_experts=exclude_experts,
        extra_experts=extra_experts,
    ).to(device).to(DTYPE)
    print(f"MoE parameters: {model.count_parameters()['total']:,}")

    cfg = StagedBurgers2DConfig.from_total_steps(n_steps)
    cfg.expert_dataset_dir = expert_dataset_dir
    if expert_dataset_dir and expert_layout_variant != "categorical":
        raise ValueError("Curated expert datasets currently support only the categorical expert layout.")
    if staged_variant == "no_joint":
        cfg.joint_train_mode = "skip"
        cfg.joint_steps = 0
    elif staged_variant == "gate_only_joint":
        cfg.joint_train_mode = "gate_only"
    elif staged_variant == "stronger_expert_calibration":
        cfg.joint_train_mode = "all"
        cfg.use_gate_region_prior = True
        cfg.gate_region_prior_power = 1.35
        cfg.gate_confidence_quantile = 0.70
        cfg.gate_min_keep_ratio = 0.30
        cfg.gate_branch_superiority_bonus = 0.95
        cfg.gate_branch_superiority_power = 1.5
        cfg.gate_branch_alignment_power = 1.2
        cfg.gate_preferred_expert_name = "wave" if expert_layout_variant == "categorical" else "curvature_wave"
        cfg.gate_preferred_expert_bias = 0.90
        cfg.gate_preferred_expert_power = 1.30
        cfg.gate_preferred_sample_boost = 0.60
        cfg.gate_preferred_sample_power = 1.10
        cfg.joint_gate_supervised_weight = 0.45
        cfg.joint_branch_consistency_weight = 0.30
        cfg.joint_branch_consistency_points = 12288
        cfg.joint_branch_consistency_margin = 0.08
        cfg.joint_branch_consistency_power = 1.75
        cfg.joint_branch_consistency_region_power = 1.35
        cfg.joint_branch_consistency_min_align = 0.20
        cfg.expert_step_scale["wave" if expert_layout_variant == "categorical" else "curvature_wave"] = 1.40
        cfg.expert_sup_weight_scale["wave" if expert_layout_variant == "categorical" else "curvature_wave"] = 1.30
        cfg.expert_focus_ratio_scale["wave" if expert_layout_variant == "categorical" else "curvature_wave"] = 1.15
        cfg.expert_res_focus_scale["wave" if expert_layout_variant == "categorical" else "curvature_wave"] = 1.28
        cfg.expert_sup_points_scale["wave" if expert_layout_variant == "categorical" else "curvature_wave"] = 1.20
        if expert_layout_variant == "categorical":
            cfg.joint_branch_consistency_focus_expert = "wave"
        elif expert_layout_variant == "attribute":
            cfg.joint_branch_consistency_focus_expert = "curvature_wave"
    elif staged_variant == "stronger_expert_route_only":
        cfg.joint_train_mode = "gate_only"
        cfg.use_gate_region_prior = True
        cfg.gate_region_prior_power = 1.30
        cfg.gate_confidence_quantile = 0.68
        cfg.gate_min_keep_ratio = 0.32
        cfg.gate_branch_superiority_bonus = 0.85
        cfg.gate_branch_superiority_power = 1.45
        cfg.gate_branch_alignment_power = 1.15
        cfg.gate_preferred_expert_name = "wave" if expert_layout_variant == "categorical" else "curvature_wave"
        cfg.gate_preferred_expert_bias = 1.20
        cfg.gate_preferred_expert_power = 1.35
        cfg.gate_preferred_sample_boost = 0.95
        cfg.gate_preferred_sample_power = 1.15
        cfg.joint_gate_supervised_weight = 0.50
        cfg.joint_gate_misroute_weight = 0.20
        cfg.joint_gate_misroute_power = 2.10
        cfg.joint_branch_consistency_weight = 0.22
        cfg.joint_branch_consistency_points = 12288
        cfg.joint_branch_consistency_margin = 0.07
        cfg.joint_branch_consistency_power = 1.60
        cfg.joint_branch_consistency_region_power = 1.25
        cfg.joint_branch_consistency_min_align = 0.20
        cfg.expert_step_scale["wave" if expert_layout_variant == "categorical" else "curvature_wave"] = 1.55
        cfg.expert_sup_weight_scale["wave" if expert_layout_variant == "categorical" else "curvature_wave"] = 1.40
        cfg.expert_focus_ratio_scale["wave" if expert_layout_variant == "categorical" else "curvature_wave"] = 1.20
        cfg.expert_res_focus_scale["wave" if expert_layout_variant == "categorical" else "curvature_wave"] = 1.35
        cfg.expert_sup_points_scale["wave" if expert_layout_variant == "categorical" else "curvature_wave"] = 1.25
        if expert_layout_variant == "categorical":
            cfg.joint_branch_consistency_focus_expert = "wave"
        elif expert_layout_variant == "attribute":
            cfg.joint_branch_consistency_focus_expert = "curvature_wave"
    elif staged_variant == "stronger_expert_oracle_consistency":
        cfg.joint_train_mode = "all"
        cfg.use_gate_region_prior = True
        cfg.gate_region_prior_power = 1.28
        cfg.gate_confidence_quantile = 0.68
        cfg.gate_min_keep_ratio = 0.34
        cfg.gate_branch_superiority_bonus = 0.88
        cfg.gate_branch_superiority_power = 1.45
        cfg.gate_branch_alignment_power = 1.12
        cfg.gate_preferred_expert_name = "wave" if expert_layout_variant == "categorical" else "curvature_wave"
        cfg.gate_preferred_expert_bias = 1.05
        cfg.gate_preferred_expert_power = 1.30
        cfg.gate_preferred_sample_boost = 0.82
        cfg.gate_preferred_sample_power = 1.12
        cfg.directional_gate_bias = 0.18
        cfg.directional_gate_bias_power = 1.05
        cfg.gate_shock_bias = 0.95
        cfg.gate_shock_power = 1.20
        cfg.gate_min_shock_weight = 0.24
        cfg.joint_gate_supervised_weight = 0.22
        cfg.joint_gate_misroute_weight = 0.14
        cfg.joint_gate_misroute_power = 1.85
        cfg.joint_branch_consistency_weight = 0.16
        cfg.joint_branch_consistency_points = 8192
        cfg.joint_branch_consistency_margin = 0.05
        cfg.joint_branch_consistency_power = 1.35
        cfg.joint_branch_consistency_region_power = 1.10
        cfg.joint_branch_consistency_min_align = 0.0
        cfg.joint_branch_consistency_focus_expert = None
        cfg.joint_focus_ratio = 0.70
        cfg.joint_focus_quantile = 0.80
        cfg.joint_focus_power = 1.55
        cfg.joint_res_focus = 2.30
        cfg.expert_step_scale["wave" if expert_layout_variant == "categorical" else "curvature_wave"] = 1.50
        cfg.expert_sup_weight_scale["wave" if expert_layout_variant == "categorical" else "curvature_wave"] = 1.35
        cfg.expert_focus_ratio_scale["wave" if expert_layout_variant == "categorical" else "curvature_wave"] = 1.18
        cfg.expert_res_focus_scale["wave" if expert_layout_variant == "categorical" else "curvature_wave"] = 1.30
        cfg.expert_sup_points_scale["wave" if expert_layout_variant == "categorical" else "curvature_wave"] = 1.22
    elif staged_variant in {
        "stronger_expert_route_sharp",
        "stronger_expert_route_sharp_anchor_light",
        "stronger_expert_route_sharp_anchor_focus",
        "stronger_expert_route_sharp_anchor_guarded",
        "stronger_expert_route_sharp_anchor_uncertainty",
    }:
        cfg.joint_train_mode = "gate_only"
        cfg.use_gate_region_prior = True
        cfg.gate_target_temperature = 10.5
        cfg.gate_confidence_power = 2.25
        cfg.gate_region_prior_power = 1.30
        cfg.gate_confidence_quantile = 0.66
        cfg.gate_min_keep_ratio = 0.34
        cfg.gate_branch_superiority_bonus = 0.92
        cfg.gate_branch_superiority_power = 1.55
        cfg.gate_branch_alignment_power = 1.18
        cfg.gate_preferred_expert_name = "wave" if expert_layout_variant == "categorical" else "curvature_wave"
        cfg.gate_preferred_expert_bias = 1.10
        cfg.gate_preferred_expert_power = 1.30
        cfg.gate_preferred_sample_boost = 0.80
        cfg.gate_preferred_sample_power = 1.12
        cfg.directional_gate_bias = 0.12
        cfg.directional_gate_bias_power = 1.05
        cfg.gate_shock_bias = 1.18
        cfg.gate_shock_power = 1.35
        cfg.gate_shock_quantile = 0.78
        cfg.gate_min_shock_weight = 0.28
        cfg.joint_gate_supervised_weight = 0.50
        cfg.joint_gate_misroute_weight = 0.34
        cfg.joint_gate_misroute_power = 2.35
        cfg.joint_base_consistency_weight = 0.14
        cfg.joint_base_consistency_points = 8192
        cfg.joint_base_consistency_smooth_power = 1.65
        cfg.joint_base_consistency_shock_power = 1.35
        cfg.joint_base_consistency_min_smooth = 0.30
        if staged_variant == "stronger_expert_route_sharp_anchor_light":
            cfg.joint_base_consistency_weight = 0.10
            cfg.joint_base_consistency_points = 6144
            cfg.joint_base_consistency_smooth_power = 1.45
            cfg.joint_base_consistency_shock_power = 1.18
            cfg.joint_base_consistency_min_smooth = 0.22
        elif staged_variant == "stronger_expert_route_sharp_anchor_focus":
            cfg.joint_base_consistency_weight = 0.18
            cfg.joint_base_consistency_points = 9216
            cfg.joint_base_consistency_smooth_power = 1.95
            cfg.joint_base_consistency_shock_power = 1.55
            cfg.joint_base_consistency_min_smooth = 0.38
        elif staged_variant == "stronger_expert_route_sharp_anchor_guarded":
            cfg.joint_base_consistency_weight = 0.14
            cfg.joint_base_consistency_points = 8192
            cfg.joint_base_consistency_smooth_power = 1.65
            cfg.joint_base_consistency_shock_power = 1.35
            cfg.joint_base_consistency_min_smooth = 0.30
            cfg.joint_base_consistency_guard_advantage = True
            cfg.joint_base_consistency_base_margin = 0.04
            cfg.joint_base_consistency_expert_adv_margin = 0.08
            cfg.joint_base_consistency_advantage_region_max = 0.40
            cfg.joint_base_consistency_advantage_power = 1.75
        elif staged_variant == "stronger_expert_route_sharp_anchor_uncertainty":
            cfg.joint_base_consistency_weight = 0.15
            cfg.joint_base_consistency_points = 8192
            cfg.joint_base_consistency_smooth_power = 1.65
            cfg.joint_base_consistency_shock_power = 1.35
            cfg.joint_base_consistency_min_smooth = 0.30
            cfg.joint_base_consistency_guard_advantage = True
            cfg.joint_base_consistency_base_margin = 0.04
            cfg.joint_base_consistency_expert_adv_margin = 0.08
            cfg.joint_base_consistency_advantage_region_max = 0.40
            cfg.joint_base_consistency_advantage_power = 1.75
            cfg.joint_base_consistency_use_uncertainty = True
            cfg.joint_base_consistency_uncertainty_quantile = 0.62
            cfg.joint_base_consistency_min_uncertainty = 0.34
            cfg.joint_base_consistency_uncertainty_power = 1.45
        cfg.expert_step_scale["wave" if expert_layout_variant == "categorical" else "curvature_wave"] = 1.55
        cfg.expert_sup_weight_scale["wave" if expert_layout_variant == "categorical" else "curvature_wave"] = 1.40
        cfg.expert_focus_ratio_scale["wave" if expert_layout_variant == "categorical" else "curvature_wave"] = 1.20
        cfg.expert_res_focus_scale["wave" if expert_layout_variant == "categorical" else "curvature_wave"] = 1.35
        cfg.expert_sup_points_scale["wave" if expert_layout_variant == "categorical" else "curvature_wave"] = 1.25
        if expert_layout_variant == "categorical":
            cfg.joint_branch_consistency_focus_expert = "wave"
        elif expert_layout_variant == "attribute":
            cfg.joint_branch_consistency_focus_expert = "curvature_wave"
    elif staged_variant == "stronger_expert_base_gate":
        cfg.joint_train_mode = "base_gate"
        cfg.use_gate_region_prior = True
        cfg.gate_region_prior_power = 1.30
        cfg.gate_confidence_quantile = 0.68
        cfg.gate_min_keep_ratio = 0.32
        cfg.gate_branch_superiority_bonus = 0.90
        cfg.gate_branch_superiority_power = 1.45
        cfg.gate_branch_alignment_power = 1.15
        cfg.joint_gate_supervised_weight = 0.48
        cfg.joint_branch_consistency_weight = 0.24
        cfg.joint_branch_consistency_points = 12288
        cfg.joint_branch_consistency_margin = 0.07
        cfg.joint_branch_consistency_power = 1.60
        cfg.joint_branch_consistency_region_power = 1.25
        cfg.joint_branch_consistency_min_align = 0.20
        if expert_layout_variant == "categorical":
            cfg.joint_branch_consistency_focus_expert = "wave"
        elif expert_layout_variant == "attribute":
            cfg.joint_branch_consistency_focus_expert = "curvature_wave"
    else:
        cfg.joint_train_mode = "all"

    if rotation_variant != "none":
        cfg.rotation_steps = max(cfg.rotation_steps, max(24, int(cfg.joint_steps * 0.45)))
        cfg.rotation_lr = 7e-4
        cfg.joint_rotation_supervision_weight = 0.38
        cfg.joint_rotation_concentration_weight = 0.16
        cfg.joint_rotation_activation_weight = 0.14
        cfg.joint_rotation_supervision_points = 8192
        cfg.joint_rotation_grad_quantile = 0.72
        cfg.joint_rotation_grad_power = 1.55
        cfg.joint_rotation_region_power = 1.45
        cfg.joint_rotation_min_region = 0.12
        cfg.joint_rotation_wave_bonus = 0.30
        if rotation_variant == "complex_low_threshold_focus":
            cfg.rotation_steps = max(cfg.rotation_steps, max(30, int(cfg.joint_steps * 0.55)))
            cfg.rotation_lr = 8e-4
            cfg.joint_rotation_supervision_weight = 0.42
            cfg.joint_rotation_concentration_weight = 0.22
            cfg.joint_rotation_activation_weight = 0.24
            cfg.joint_rotation_supervision_points = 12288
            cfg.joint_rotation_grad_quantile = 0.70
            cfg.joint_rotation_grad_power = 1.65
            cfg.joint_rotation_region_power = 1.85
            cfg.joint_rotation_min_region = 0.24
            cfg.joint_rotation_wave_bonus = 0.38
        elif rotation_variant == "complex_low_threshold_sparse":
            cfg.rotation_steps = max(cfg.rotation_steps, max(36, int(cfg.joint_steps * 0.45)))
            cfg.rotation_lr = 6e-4
            cfg.joint_rotation_supervision_weight = 0.44
            cfg.joint_rotation_concentration_weight = 0.18
            cfg.joint_rotation_activation_weight = 0.18
            cfg.joint_rotation_supervision_points = 12288
            cfg.joint_rotation_grad_quantile = 0.74
            cfg.joint_rotation_grad_power = 1.55
            cfg.joint_rotation_region_power = 1.95
            cfg.joint_rotation_min_region = 0.28
            cfg.joint_rotation_wave_bonus = 0.40
            cfg.joint_rotation_activation_floor_target = 0.02
            cfg.joint_rotation_activation_cap = 0.68
            cfg.joint_rotation_concentration_floor_target = 0.18
            cfg.joint_rotation_concentration_cap = 0.82
            cfg.joint_rotation_negative_ratio = 0.34
            cfg.joint_rotation_negative_max_region = 0.06
            cfg.joint_rotation_negative_grad_quantile = 0.52
            cfg.joint_rotation_negative_activation = 0.01
            cfg.joint_rotation_negative_concentration = 0.03
            cfg.joint_rotation_negative_weight_scale = 0.75
    print(
        f"Stage config | base={cfg.base_steps} | expert={cfg.expert_steps} each | "
        f"gate={cfg.gate_steps} | joint={cfg.joint_steps} | joint_mode={cfg.joint_train_mode} | "
        f"directional={directional_expert_variant} | wave={wave_expert_variant} | layout={expert_layout_variant} | attr={attribute_expert_variant} | gate={gate_variant} | rotation={rotation_variant}"
    )

    if expert_pretrain_fraction < 1.0:
        cfg.expert_steps = int(cfg.expert_steps * expert_pretrain_fraction)
        cfg.joint_train_mode = "experts_gate"
        cfg.gate_steps = 0  # gate is introduced cold at the joint stage
        cfg.joint_gate_supervised_weight = 0.0
        cfg.joint_gate_misroute_weight = 0.0
        cfg.joint_branch_consistency_weight = 0.0
        cfg.joint_base_consistency_weight = 0.0
        print(
            f"[gate-intro ablation] expert_pretrain_fraction={expert_pretrain_fraction} -> "
            f"expert_steps={cfg.expert_steps} | gate_steps=0 (cold gate) | "
            f"joint_mode={cfg.joint_train_mode} | joint_steps={cfg.joint_steps}"
        )
    if joint_steps_override is not None:
        cfg.joint_steps = joint_steps_override
        print(f"[joint-length ablation] joint_steps_override={joint_steps_override}")

    n_col = int(batch["xt_col"].shape[0])
    n_ic = int(batch["xt_ic"].shape[0])
    n_bc_per_face = int(batch["xt_bc"].shape[0] // 4)

    def sample_uniform_batch() -> dict[str, torch.Tensor]:
        return problem.training_batch(
            n_col=n_col,
            n_ic=n_ic,
            n_bc_per_face=n_bc_per_face,
        )

    base_history = pretrain_base_model(
        model,
        batch,
        cfg=cfg,
        nu=NU,
        save_dir=save_dir,
        xyt_test=xyt_test,
        u_exact_flat=u_exact_flat,
        batch_refresh_fn=sample_uniform_batch,
    )
    if expert_dataset_dir:
        specialist_batches = build_specialist_batches_from_curated_dataset(
            expert_dataset_dir,
            batch,
            model.expert_names,
            cfg=cfg,
        )
        region_scores = compute_region_scores(reference, layout_variant=expert_layout_variant)
    else:
        specialist_batches, region_scores, _, _ = build_specialist_batches(
            reference,
            batch,
            model.expert_names,
            cfg=cfg,
            layout_variant=expert_layout_variant,
        )
    joint_batch = build_joint_training_batch(reference, batch, cfg=cfg, layout_variant=expert_layout_variant)
    shock_priority = (
        torch.tensor(
            shock_priority_from_region_scores(
                region_scores,
                directional_bonus=cfg.joint_directional_bonus,
                layout_variant=expert_layout_variant,
            ),
            dtype=xyt_test.dtype,
            device=xyt_test.device,
        )
        if region_scores
        else None
    )
    preferred_priority = None
    if region_scores and cfg.gate_preferred_expert_name and cfg.gate_preferred_expert_name in region_scores:
        preferred_priority = torch.tensor(
            region_scores[cfg.gate_preferred_expert_name].reshape(-1),
            dtype=xyt_test.dtype,
            device=xyt_test.device,
        )
    steep_mask = steep_mask_from_reference(reference, device=device)
    expert_histories: dict[str, dict[str, list[float]]] = {}
    if cfg.expert_steps > 0:
        expert_histories = pretrain_experts(
            model,
            specialist_batches,
            cfg=cfg,
            nu=NU,
            save_dir=save_dir,
            xyt_test=xyt_test,
            u_exact_flat=u_exact_flat,
            steep_mask_flat=steep_mask,
        )
    rotation_history = {}
    rotation_batch_stats = {}
    if rotation_variant != "none":
        rotation_batch, rotation_batch_stats = build_joint_rotation_supervision_batch(
            reference,
            xyt_test,
            region_scores,
            cfg=cfg,
            layout_variant=expert_layout_variant,
        )
        rotation_history = pretrain_rotation_layer(
            model,
            rotation_batch,
            cfg=cfg,
            save_dir=save_dir,
        )
    gate_history: dict[str, list[float]] = {}
    gate_target_stats: dict[str, float] = {}
    gate_training_stats: dict[str, float] = {}
    if cfg.gate_steps > 0:
        gate_targets, gate_confidence, gate_target_stats = build_gate_targets(
            model,
            xyt_test,
            u_exact_flat,
            region_scores,
            temperature=cfg.gate_target_temperature,
            use_region_prior=cfg.use_gate_region_prior and bool(region_scores),
            region_prior_power=cfg.gate_region_prior_power,
            branch_superiority_bonus=cfg.gate_branch_superiority_bonus,
            branch_superiority_power=cfg.gate_branch_superiority_power,
            branch_alignment_power=cfg.gate_branch_alignment_power,
            preferred_expert_name=cfg.gate_preferred_expert_name,
            preferred_expert_bias=cfg.gate_preferred_expert_bias,
            preferred_expert_power=cfg.gate_preferred_expert_power,
            directional_gate_bias=cfg.directional_gate_bias,
            directional_gate_bias_power=cfg.directional_gate_bias_power,
        )
        gate_history, gate_training_stats = train_gate(
            model,
            xyt_test,
            gate_targets,
            gate_confidence,
            shock_priority,
            preferred_priority,
            cfg=cfg,
        )
        torch.save(
            {
                "model_state": model.state_dict(),
                "stage": "pre_joint",
            },
            os.path.join(save_dir, "burgers2d_pre_joint.pt"),
        )
    if cfg.joint_gate_supervised_weight > 0 and cfg.joint_steps > 0 and cfg.joint_train_mode != "skip":
        joint_batch.update(
            build_joint_gate_supervision_batch(
                xyt_test,
                gate_targets,
                gate_confidence,
                shock_priority,
                preferred_priority,
                cfg=cfg,
            )
        )
    if cfg.joint_branch_consistency_weight > 0 and cfg.joint_steps > 0 and cfg.joint_train_mode != "skip":
        joint_batch.update(
            build_joint_branch_consistency_batch(
                model,
                xyt_test,
                u_exact_flat,
                region_scores,
                cfg=cfg,
            )
        )
    if cfg.joint_base_consistency_weight > 0 and cfg.joint_steps > 0 and cfg.joint_train_mode != "skip":
        joint_batch.update(
            build_joint_base_consistency_batch(
                model,
                xyt_test,
                u_exact_flat,
                region_scores,
                cfg=cfg,
                layout_variant=expert_layout_variant,
            )
        )
    if cfg.joint_rotation_supervision_weight > 0 and cfg.joint_steps > 0 and cfg.joint_train_mode != "skip":
        rotation_batch, rotation_batch_stats = build_joint_rotation_supervision_batch(
            reference,
            xyt_test,
            region_scores,
            cfg=cfg,
            layout_variant=expert_layout_variant,
        )
        joint_batch.update(rotation_batch)

    def sample_joint_batch() -> dict[str, torch.Tensor]:
        fresh_base = sample_uniform_batch()
        refreshed = build_joint_training_batch(
            reference,
            fresh_base,
            cfg=cfg,
            global_xt_col=fresh_base["xt_col"],
            layout_variant=expert_layout_variant,
        )
        if "xt_gate" in joint_batch:
            refreshed["xt_gate"] = joint_batch["xt_gate"]
            refreshed["gate_target_probs"] = joint_batch["gate_target_probs"]
            refreshed["gate_target_weights"] = joint_batch["gate_target_weights"]
        if "xt_branch_consistency" in joint_batch:
            refreshed["xt_branch_consistency"] = joint_batch["xt_branch_consistency"]
            refreshed["branch_target_idx"] = joint_batch["branch_target_idx"]
            refreshed["branch_target_weights"] = joint_batch["branch_target_weights"]
        if "xt_base_consistency" in joint_batch:
            refreshed["xt_base_consistency"] = joint_batch["xt_base_consistency"]
            refreshed["base_target_weights"] = joint_batch["base_target_weights"]
        if "xt_rotation" in joint_batch:
            refreshed["xt_rotation"] = joint_batch["xt_rotation"]
            refreshed["rotation_target_angle"] = joint_batch["rotation_target_angle"]
            refreshed["rotation_target_weights"] = joint_batch["rotation_target_weights"]
            refreshed["rotation_target_concentration"] = joint_batch["rotation_target_concentration"]
            refreshed["rotation_target_activation"] = joint_batch["rotation_target_activation"]
        return refreshed

    joint_history = joint_finetune(
        model,
        joint_batch,
        cfg=cfg,
        nu=NU,
        save_dir=save_dir,
        xyt_test=xyt_test,
        u_exact_flat=u_exact_flat,
        batch_refresh_fn=sample_joint_batch,
    )

    stage_info: dict[str, Any] = {
        "base_history": base_history,
        "expert_histories": expert_histories,
        "rotation_history": rotation_history,
        "gate_history": gate_history,
        "gate_target_stats": gate_target_stats,
        "gate_training_stats": gate_training_stats,
        "rotation_batch_stats": rotation_batch_stats,
        "expert_dataset_dir": expert_dataset_dir,
        "directional_expert_variant": directional_expert_variant,
        "wave_expert_variant": wave_expert_variant,
        "expert_layout_variant": expert_layout_variant,
        "attribute_expert_variant": attribute_expert_variant,
        "gate_variant": gate_variant,
        "rotation_variant": rotation_variant,
        "exclude_experts": list(exclude_experts),
        "stage_config": cfg.__dict__,
    }
    torch.save(stage_info, os.path.join(save_dir, "burgers2d_staged_training.pt"))
    return model, joint_history, stage_info


def run_experiment(
    *,
    train_mode: str = "vanilla",
    n_steps: int = 1500,
    n_col: int = 6000,
    n_ic: int = 1200,
    n_bc_per_face: int = 400,
    nx: int = 65,
    ny: int = 65,
    nt: int = 21,
    device_override: Optional[str] = None,
    results_root: Optional[str] = None,
    staged_variant: str = "gate_only_joint",
    expert_dataset_dir: Optional[str] = None,
    directional_expert_variant: str = "hybrid",
    wave_expert_variant: str = "base",
    expert_layout_variant: str = "categorical",
    attribute_expert_variant: str = "base",
    gate_variant: str = "pointwise",
    rotation_variant: str = "none",
    exclude_experts: str = "",
    extra_experts: str = "",
    expert_pretrain_fraction: float = 1.0,
    joint_steps_override: int | None = None,
    balance_weight_override: float | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, dict[str, Any]]:
    _set_global_seed(seed)
    device = torch.device(device_override) if device_override else DEVICE
    active_results_root = results_root or RESULTS_ROOT
    os.makedirs(active_results_root, exist_ok=True)

    print(f"Device: {device}")
    parsed_exclude = tuple(n.strip() for n in exclude_experts.split(",") if n.strip())
    if parsed_exclude:
        print(f"Excluding experts: {parsed_exclude}")
    parsed_extra = tuple(n.strip() for n in extra_experts.split(",") if n.strip())
    if parsed_extra:
        print(f"Extra experts: {parsed_extra}")
    if expert_pretrain_fraction < 1.0:
        print(f"Gate-introduction ablation: expert_pretrain_fraction={expert_pretrain_fraction}")
    print("Building 2D Burgers problem...")
    problem = Burgers2DProblem(nu=NU, device=device, dtype=DTYPE, seed=seed)
    batch = problem.training_batch(n_col=n_col, n_ic=n_ic, n_bc_per_face=n_bc_per_face)

    print("Generating reference solution...")
    reference = problem.generate_reference_solution(nx=nx, ny=ny, nt=nt)
    xyt_test, u_exact_flat = flatten_reference_solution(reference, device=device, dtype=DTYPE)

    modes = [train_mode] if train_mode != "all" else ["vanilla", "end_to_end", "staged"]
    metrics_summary: dict[str, dict[str, Any]] = {}

    if "vanilla" in modes:
        save_dir = os.path.join(active_results_root, "burgers2d_vanilla")
        model, history = _train_vanilla(batch, xyt_test, u_exact_flat, n_steps=n_steps, device=device, save_dir=save_dir)
        metrics, u_pred = _evaluate_metrics(model, reference, xyt_test=xyt_test, u_exact_flat=u_exact_flat)
        stress_metrics = _compute_directional_stress_metrics(model, reference, u_pred=u_pred, xyt_test=xyt_test)
        _save_artifacts(save_dir=save_dir, reference=reference, model=model, history=history, metrics=metrics, stress_metrics=stress_metrics, u_pred=u_pred, xyt_test=xyt_test, u_exact_flat=u_exact_flat)
        metrics_summary["vanilla"] = metrics

    if "end_to_end" in modes:
        save_dir = os.path.join(active_results_root, "burgers2d_moe_end_to_end")
        model, history = _train_moe_end_to_end(
            batch,
            xyt_test,
            u_exact_flat,
            n_steps=n_steps,
            device=device,
            save_dir=save_dir,
            directional_expert_variant=directional_expert_variant,
            wave_expert_variant=wave_expert_variant,
            expert_layout_variant=expert_layout_variant,
            attribute_expert_variant=attribute_expert_variant,
            gate_variant=gate_variant,
            rotation_variant=rotation_variant,
            exclude_experts=parsed_exclude,
            extra_experts=parsed_extra,
            balance_weight_override=balance_weight_override,
        )
        metrics, u_pred = _evaluate_metrics(model, reference, xyt_test=xyt_test, u_exact_flat=u_exact_flat)
        stress_metrics = _compute_directional_stress_metrics(model, reference, u_pred=u_pred, xyt_test=xyt_test)
        _save_artifacts(save_dir=save_dir, reference=reference, model=model, history=history, metrics=metrics, stress_metrics=stress_metrics, u_pred=u_pred, xyt_test=xyt_test, u_exact_flat=u_exact_flat)
        metrics_summary["end_to_end"] = metrics

    if "staged" in modes:
        save_dir = os.path.join(active_results_root, "burgers2d_moe_staged")
        model, history, stage_info = _train_moe_staged(
            problem,
            batch,
            reference,
            xyt_test,
            u_exact_flat,
            n_steps=n_steps,
            device=device,
            save_dir=save_dir,
            staged_variant=staged_variant,
            expert_dataset_dir=expert_dataset_dir,
            directional_expert_variant=directional_expert_variant,
            wave_expert_variant=wave_expert_variant,
            expert_layout_variant=expert_layout_variant,
            attribute_expert_variant=attribute_expert_variant,
            gate_variant=gate_variant,
            rotation_variant=rotation_variant,
            exclude_experts=parsed_exclude,
            extra_experts=parsed_extra,
            expert_pretrain_fraction=expert_pretrain_fraction,
            joint_steps_override=joint_steps_override,
        )
        metrics, u_pred = _evaluate_metrics(model, reference, xyt_test=xyt_test, u_exact_flat=u_exact_flat)
        stress_metrics = _compute_directional_stress_metrics(model, reference, u_pred=u_pred, xyt_test=xyt_test)
        _save_artifacts(save_dir=save_dir, reference=reference, model=model, history=history, metrics=metrics, stress_metrics=stress_metrics, u_pred=u_pred, xyt_test=xyt_test, u_exact_flat=u_exact_flat)
        with open(os.path.join(save_dir, "stage_config.json"), "w", encoding="utf-8") as f:
            json.dump(stage_info["stage_config"], f, ensure_ascii=False, indent=2)
        metrics_summary["staged"] = metrics

    if len(metrics_summary) > 1:
        summary_path = os.path.join(active_results_root, "burgers2d_metrics_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(metrics_summary, f, ensure_ascii=False, indent=2)
        plot_model_metric_comparison(
            metrics_summary,
            os.path.join(active_results_root, "burgers2d_metrics_summary.png"),
        )
        print(f"[OK] Saved summary: {summary_path}")

    print("Evaluation summary:")
    for name, metrics in metrics_summary.items():
        print(f"  [{name}]")
        for key, value in metrics.items():
            if isinstance(value, list):
                print(f"    {key}: {value}")
            else:
                print(f"    {key}: {float(value):.6e}")
    return metrics_summary


def run_vanilla_experiment(**kwargs) -> dict[str, dict[str, Any]]:
    return run_experiment(train_mode="vanilla", **kwargs)
