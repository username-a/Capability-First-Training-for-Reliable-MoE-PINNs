"""Quantify how well gating uses the wave expert on oracle-best regions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np
import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Burger2D.core.moe_pinn import build_burgers2d_moe
from Burger2D.equations.burgers2d import ReferenceSolution2D
from Burger2D.training.staged_burgers2d import compute_region_scores, flatten_reference_solution
from Burger2D.visualization.plots import plot_wave_routing_comparison, plot_wave_routing_panels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze wave routing on oracle-best regions")
    parser.add_argument("--result-dir", action="append", required=True, help="Staged result directory to analyze")
    parser.add_argument("--name", action="append", default=None, help="Optional display name for each result dir")
    parser.add_argument("--output-dir", required=True, help="Output directory for reports and figures")
    parser.add_argument("--device", default=None, help="Optional device override, e.g. cpu or cuda")
    return parser.parse_args()


def _predict_in_batches(fn, coords: torch.Tensor, batch_size: int) -> torch.Tensor:
    outputs = []
    for start in range(0, coords.shape[0], batch_size):
        outputs.append(fn(coords[start:start + batch_size]))
    return torch.cat(outputs, dim=0)


def _load_stage_info(result_dir: str) -> dict[str, Any]:
    stage_path = os.path.join(result_dir, "burgers2d_staged_training.pt")
    if not os.path.exists(stage_path):
        raise FileNotFoundError(f"Missing staged training artifact: {stage_path}")
    return torch.load(stage_path, map_location="cpu")


def _load_model(result_dir: str, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    stage_info = _load_stage_info(result_dir)
    model = build_burgers2d_moe(
        directional_expert_variant=stage_info.get("directional_expert_variant", "hybrid"),
        wave_expert_variant=stage_info.get("wave_expert_variant", "base"),
        expert_layout_variant=stage_info.get("expert_layout_variant", "categorical"),
        attribute_expert_variant=stage_info.get("attribute_expert_variant", "base"),
        gate_variant=stage_info.get("gate_variant", "pointwise"),
    ).to(device).to(torch.float32)
    checkpoint = torch.load(os.path.join(result_dir, "burgers2d_moe_staged.pt"), map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, stage_info


def _load_reference(result_dir: str, device: torch.device) -> tuple[ReferenceSolution2D, torch.Tensor]:
    npz = np.load(os.path.join(result_dir, "reference_and_prediction.npz"))
    reference = ReferenceSolution2D(
        x=np.asarray(npz["x"]),
        y=np.asarray(npz["y"]),
        t=np.asarray(npz["t"]),
        u=np.asarray(npz["u_ref"]),
    )
    coords, _ = flatten_reference_solution(reference, device=device, dtype=torch.float32)
    return reference, coords


def _wave_metrics(
    *,
    reference: ReferenceSolution2D,
    model: torch.nn.Module,
    coords: torch.Tensor,
) -> tuple[dict[str, float | list[float]], np.ndarray, np.ndarray, np.ndarray]:
    batch_size = int(getattr(model, "inference_batch_size", 4096))
    with torch.no_grad():
        gates = _predict_in_batches(model.get_gate_weights, coords, batch_size).cpu().numpy()
        expert_preds = _predict_in_batches(model.get_expert_predictions, coords, batch_size).cpu().numpy()
        mixture_pred = _predict_in_batches(model, coords, batch_size).cpu().numpy()

    shape = reference.u.shape
    gates = gates.reshape(shape[0], shape[1], shape[2], -1)
    expert_preds = expert_preds.reshape(shape[0], shape[1], shape[2], -1)
    mixture_pred = mixture_pred.reshape(shape)
    ref = reference.u

    expert_names = list(getattr(model, "expert_names", []))
    wave_name = "wave" if "wave" in expert_names else "curvature_wave"
    wave_idx = expert_names.index(wave_name)
    expert_abs_err = np.abs(expert_preds - ref[..., None])
    mixture_abs_err = np.abs(mixture_pred - ref)
    best_idx = expert_abs_err.argmin(axis=-1)
    wave_err = expert_abs_err[..., wave_idx]
    other_err = np.delete(expert_abs_err, wave_idx, axis=-1)
    next_best_err = other_err.min(axis=-1)
    wave_margin = np.clip(next_best_err - wave_err, 0.0, None) / np.clip(next_best_err, 1e-8, None)
    wave_oracle_mask = best_idx == wave_idx
    if np.any(wave_oracle_mask):
        strong_thresh = max(0.05, float(np.quantile(wave_margin[wave_oracle_mask], 0.60)))
    else:
        strong_thresh = 0.05
    strong_wave_oracle_mask = wave_oracle_mask & (wave_margin >= strong_thresh)
    wave_gate = gates[..., wave_idx]
    route_idx = gates.argmax(axis=-1)
    wave_top1_mask = route_idx == wave_idx
    wave_top2_threshold = np.partition(gates, kth=max(gates.shape[-1] - 2, 0), axis=-1)[..., -2]
    wave_top2_mask = wave_gate >= wave_top2_threshold
    wave_regret = np.where(wave_oracle_mask, np.clip(mixture_abs_err - wave_err, 0.0, None), 0.0)

    region_scores = compute_region_scores(reference, layout_variant=getattr(model, "expert_layout_variant", "categorical"))
    wave_region = region_scores.get(wave_name, region_scores.get("wave")).reshape(shape)
    wave_region_thresh = float(np.quantile(wave_region, 0.82))
    wave_region_mask = wave_region >= wave_region_thresh
    oracle_wave_region_mask = wave_oracle_mask & wave_region_mask

    def _safe_mean(values: np.ndarray, mask: np.ndarray) -> float:
        return float(values[mask].mean()) if np.any(mask) else 0.0

    metrics: dict[str, float | list[float]] = {
        "wave_oracle_count": int(wave_oracle_mask.sum()),
        "wave_oracle_frac": float(wave_oracle_mask.mean()),
        "strong_wave_oracle_count": int(strong_wave_oracle_mask.sum()),
        "strong_wave_oracle_frac": float(strong_wave_oracle_mask.mean()),
        "wave_region_frac": float(wave_region_mask.mean()),
        "wave_oracle_in_wave_region_frac": float(oracle_wave_region_mask.mean()),
        "wave_mean_gate_on_oracle": _safe_mean(wave_gate, wave_oracle_mask),
        "wave_mean_gate_on_strong_oracle": _safe_mean(wave_gate, strong_wave_oracle_mask),
        "wave_mean_gate_on_wave_region": _safe_mean(wave_gate, wave_region_mask),
        "wave_top1_on_oracle": float(wave_top1_mask[wave_oracle_mask].mean()) if np.any(wave_oracle_mask) else 0.0,
        "wave_top2_on_oracle": float(wave_top2_mask[wave_oracle_mask].mean()) if np.any(wave_oracle_mask) else 0.0,
        "wave_top1_on_strong_oracle": float(wave_top1_mask[strong_wave_oracle_mask].mean()) if np.any(strong_wave_oracle_mask) else 0.0,
        "wave_regret_mean": _safe_mean(wave_regret, wave_oracle_mask),
        "wave_regret_mean_strong": _safe_mean(wave_regret, strong_wave_oracle_mask),
        "wave_margin_mean_on_oracle": _safe_mean(wave_margin, wave_oracle_mask),
        "wave_margin_mean_on_strong_oracle": _safe_mean(wave_margin, strong_wave_oracle_mask),
        "wave_load_frac_global": float(getattr(model, "load_balance_stats")(coords)["expert_load_frac"][wave_idx]),
        "timewise_wave_oracle_frac": wave_oracle_mask.mean(axis=(1, 2)).astype(float).tolist(),
        "timewise_wave_mean_gate": wave_gate.mean(axis=(1, 2)).astype(float).tolist(),
        "timewise_wave_top1": wave_top1_mask.mean(axis=(1, 2)).astype(float).tolist(),
    }
    return metrics, wave_oracle_mask, wave_gate, wave_regret


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    names = args.name or []
    comparison_rows: list[dict[str, float | str]] = []
    report_lines = [
        "# Wave Routing Quantification",
        "",
        f"Device: `{device}`",
        "",
    ]

    for idx, result_dir in enumerate(args.result_dir):
        name = names[idx] if idx < len(names) else os.path.basename(os.path.dirname(result_dir.rstrip("\\/"))) or os.path.basename(result_dir)
        model, stage_info = _load_model(result_dir, device)
        reference, coords = _load_reference(result_dir, device)
        metrics, wave_oracle_mask, wave_gate, wave_regret = _wave_metrics(reference=reference, model=model, coords=coords)

        output_json = os.path.join(args.output_dir, f"{name}_wave_routing_metrics.json")
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "name": name,
                    "result_dir": result_dir,
                    "stage_info": {
                        "wave_expert_variant": stage_info.get("wave_expert_variant"),
                        "expert_layout_variant": stage_info.get("expert_layout_variant"),
                        "gate_variant": stage_info.get("gate_variant"),
                    },
                    "metrics": metrics,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        figure_path = os.path.join(args.output_dir, f"{name}_wave_routing_panels.png")
        plot_wave_routing_panels(
            reference.x,
            reference.y,
            reference.t,
            wave_oracle_mask=wave_oracle_mask,
            wave_gate=wave_gate,
            wave_regret=wave_regret,
            save_path=figure_path,
            title=f"{name} wave-routing diagnostics",
        )

        comparison_rows.append(
            {
                "name": name,
                "wave_oracle_frac": float(metrics["wave_oracle_frac"]),
                "wave_mean_gate_on_oracle": float(metrics["wave_mean_gate_on_oracle"]),
                "wave_top1_on_oracle": float(metrics["wave_top1_on_oracle"]),
                "wave_regret_mean": float(metrics["wave_regret_mean"]),
            }
        )
        report_lines.extend(
            [
                f"## {name}",
                "",
                f"- Result dir: `{result_dir}`",
                f"- Wave oracle fraction: `{float(metrics['wave_oracle_frac']):.6f}`",
                f"- Mean wave gate on oracle points: `{float(metrics['wave_mean_gate_on_oracle']):.6f}`",
                f"- Wave top-1 routing on oracle points: `{float(metrics['wave_top1_on_oracle']):.6f}`",
                f"- Wave top-2 routing on oracle points: `{float(metrics['wave_top2_on_oracle']):.6f}`",
                f"- Mean wave regret on oracle points: `{float(metrics['wave_regret_mean']):.6f}`",
                f"- Global wave load fraction: `{float(metrics['wave_load_frac_global']):.6f}`",
                f"- Panel figure: `{figure_path}`",
                "",
            ]
        )

    comparison_figure = os.path.join(args.output_dir, "wave_routing_comparison.png")
    plot_wave_routing_comparison(comparison_rows, comparison_figure)
    report_lines.extend(
        [
            "## Comparison Figure",
            "",
            f"- `{comparison_figure}`",
            "",
        ]
    )
    report_path = os.path.join(args.output_dir, "wave_routing_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"[OK] Saved wave routing report to: {report_path}")


if __name__ == "__main__":
    main()
