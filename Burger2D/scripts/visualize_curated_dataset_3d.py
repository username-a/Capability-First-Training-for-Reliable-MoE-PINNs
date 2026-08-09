"""Create annotated 3D scene figures for the curated Burger2D dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Burger2D.data import REGION_NAMES
from Burger2D.visualization.plots import plot_curated_case_3d_scene


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a curated Burger2D dataset case in 3D")
    parser.add_argument("--dataset-dir", required=True, help="Curated dataset directory with manifest.json and cases/")
    parser.add_argument("--output-dir", required=True, help="Directory for generated figures and reports")
    parser.add_argument("--case-id", default="auto", help="Case id like 0, 1, ... or auto")
    parser.add_argument("--max-points", type=int, default=12000, help="Maximum points to scatter in the 3D volume")
    return parser.parse_args()


def _load_case(dataset_dir: str, case_id: int) -> dict[str, np.ndarray]:
    case_path = os.path.join(dataset_dir, "cases", f"case_{case_id:03d}.npz")
    if not os.path.exists(case_path):
        raise FileNotFoundError(f"Missing case file: {case_path}")
    with np.load(case_path) as npz:
        return {key: np.asarray(npz[key]) for key in npz.files}


def _auto_select_case(dataset_dir: str) -> tuple[int, dict[str, float]]:
    cases_dir = os.path.join(dataset_dir, "cases")
    best_case = 0
    best_score = -1.0
    best_stats: dict[str, float] = {}
    wave_idx = REGION_NAMES.index("wave")
    for file_name in sorted(os.listdir(cases_dir)):
        if not file_name.endswith(".npz"):
            continue
        case_id = int(file_name.split("_")[1].split(".")[0])
        case = _load_case(dataset_dir, case_id)
        wave_prob = case["probs"][..., wave_idx]
        wave_label_frac = float((case["labels"] == wave_idx).mean())
        wave_prob_mean = float(wave_prob.mean())
        wave_prob_peak = float(wave_prob.max())
        score = 0.55 * wave_prob_mean + 0.30 * wave_label_frac + 0.15 * wave_prob_peak
        if score > best_score:
            best_score = score
            best_case = case_id
            best_stats = {
                "wave_prob_mean": wave_prob_mean,
                "wave_label_frac": wave_label_frac,
                "wave_prob_peak": wave_prob_peak,
                "selection_score": score,
            }
    return best_case, best_stats


def _choose_anchor(
    score: np.ndarray,
    used_indices: set[int],
) -> int:
    flat = score.reshape(-1)
    order = np.argsort(flat)
    for idx in order[::-1]:
        idx_int = int(idx)
        if idx_int not in used_indices:
            used_indices.add(idx_int)
            return idx_int
    idx_int = int(order[-1])
    used_indices.add(idx_int)
    return idx_int


def _build_anchor_records(case: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    probs = case["probs"]
    labels = case["labels"]
    confidence = case["label_confidence"]
    grad_mag = case["grad_mag"]
    lap_abs = case["lap_abs"]
    anisotropy = case["anisotropy"]
    x = case["x"]
    y = case["y"]
    t = case["t"]
    u = case["u"]
    xx, yy = np.meshgrid(x, y, indexing="xy")
    coords_x = np.broadcast_to(xx[None, :, :], labels.shape)
    coords_y = np.broadcast_to(yy[None, :, :], labels.shape)
    coords_t = np.broadcast_to(t[:, None, None], labels.shape)
    interior_mask = np.ones_like(labels, dtype=bool)
    interior_mask[:, 0, :] = False
    interior_mask[:, -1, :] = False
    interior_mask[:, :, 0] = False
    interior_mask[:, :, -1] = False

    smooth_idx = REGION_NAMES.index("smooth")
    directional_idx = REGION_NAMES.index("directional_shock")
    wave_idx = REGION_NAMES.index("wave")
    used: set[int] = set()
    aniso_clipped = np.clip(anisotropy, 0.0, np.quantile(anisotropy, 0.98))
    smooth_stability = 1.0 / np.clip(1.0 + grad_mag + lap_abs + 0.5 * aniso_clipped, 1e-6, None)
    anchor_specs = [
        ("Wave Peak", probs[..., wave_idx] * interior_mask),
        ("Wave High-Confidence Core", probs[..., wave_idx] * confidence * interior_mask),
        ("Wave Curvature Ridge", probs[..., wave_idx] * lap_abs * interior_mask),
        (
            "Directional Transition",
            probs[..., directional_idx] * aniso_clipped * np.clip(grad_mag, 0.0, np.quantile(grad_mag, 0.97)) * interior_mask,
        ),
        ("Smooth Buffer", probs[..., smooth_idx] * confidence * smooth_stability * interior_mask),
    ]

    anchors: list[dict[str, Any]] = []
    for name, score in anchor_specs:
        idx_flat = _choose_anchor(score, used)
        idx_t, idx_y, idx_x = np.unravel_index(idx_flat, labels.shape)
        label_idx = int(labels[idx_t, idx_y, idx_x])
        anchors.append(
            {
                "name": name,
                "time_index": int(idx_t),
                "x": float(coords_x[idx_t, idx_y, idx_x]),
                "y": float(coords_y[idx_t, idx_y, idx_x]),
                "t": float(coords_t[idx_t, idx_y, idx_x]),
                "u": float(u[idx_t, idx_y, idx_x]),
                "label_idx": label_idx,
                "label_name": REGION_NAMES[label_idx],
                "wave_prob": float(probs[idx_t, idx_y, idx_x, wave_idx]),
                "confidence": float(confidence[idx_t, idx_y, idx_x]),
                "grad_mag": float(grad_mag[idx_t, idx_y, idx_x]),
                "lap_abs": float(lap_abs[idx_t, idx_y, idx_x]),
                "anisotropy": float(anisotropy[idx_t, idx_y, idx_x]),
            }
        )
    return anchors


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.dataset_dir, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if args.case_id == "auto":
        case_id, selection_stats = _auto_select_case(args.dataset_dir)
    else:
        case_id = int(args.case_id)
        selection_stats = {"selection_mode": "manual"}

    case = _load_case(args.dataset_dir, case_id)
    wave_idx = REGION_NAMES.index("wave")
    wave_prob = case["probs"][..., wave_idx]
    key_time_index = int(np.argmax(wave_prob.mean(axis=(1, 2))))
    anchors = _build_anchor_records(case)

    figure_path = os.path.join(args.output_dir, f"curated_case_{case_id:03d}_3d_scene.png")
    plot_curated_case_3d_scene(
        case["x"],
        case["y"],
        case["t"],
        case["u"],
        case["labels"],
        wave_prob,
        case["label_confidence"],
        case["grad_mag"],
        case["lap_abs"],
        case["anisotropy"],
        REGION_NAMES,
        anchors,
        key_time_index,
        figure_path,
        title=f"Curated Burger2D Case {case_id:03d}",
        max_points=args.max_points,
    )

    report = {
        "dataset_dir": args.dataset_dir,
        "case_id": case_id,
        "selection_stats": selection_stats,
        "key_time_index": key_time_index,
        "key_time_value": float(case["t"][key_time_index]),
        "case_wave_prob_mean": float(wave_prob.mean()),
        "case_wave_label_fraction": float((case["labels"] == wave_idx).mean()),
        "anchors": anchors,
        "figure_path": figure_path,
        "dataset_manifest_summary": {
            "num_cases": manifest.get("num_cases"),
            "grid": manifest.get("grid"),
            "label_histogram": manifest.get("label_histogram"),
            "expert_summary": manifest.get("expert_summary"),
        },
    }
    report_json = os.path.join(args.output_dir, f"curated_case_{case_id:03d}_scene_report.json")
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    md_lines = [
        "# Curated Dataset 3D Scene",
        "",
        f"- Dataset: `{args.dataset_dir}`",
        f"- Selected case: `{case_id:03d}`",
        f"- Key wave-rich time slice: `t={case['t'][key_time_index]:.3f}`",
        f"- Mean wave probability: `{wave_prob.mean():.6f}`",
        f"- Wave label fraction: `{(case['labels'] == wave_idx).mean():.6f}`",
        f"- Figure: `{figure_path}`",
        "",
        "## Anchor Notes",
        "",
    ]
    for idx, anchor in enumerate(anchors, start=1):
        md_lines.append(
            f"{idx}. `{anchor['name']}` at `(x={anchor['x']:.3f}, y={anchor['y']:.3f}, t={anchor['t']:.3f})`, "
            f"label=`{anchor['label_name']}`, wave=`{anchor['wave_prob']:.3f}`, conf=`{anchor['confidence']:.3f}`, "
            f"grad=`{anchor['grad_mag']:.3f}`, lap=`{anchor['lap_abs']:.3f}`, aniso=`{anchor['anisotropy']:.3f}`."
        )
    report_md = os.path.join(args.output_dir, f"curated_case_{case_id:03d}_scene_report.md")
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"[OK] Saved 3D curated-dataset scene to: {figure_path}")


if __name__ == "__main__":
    main()
