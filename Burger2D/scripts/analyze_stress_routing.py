"""Quantify oracle routing behavior on the directional-stress subset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Burger2D.core.moe_pinn import build_burgers2d_moe
from Burger2D.equations.burgers2d import ReferenceSolution2D
from Burger2D.experiments.run_burgers2d import _compute_directional_stress_spec
from Burger2D.training.staged_burgers2d import flatten_reference_solution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze expert routing on the directional-stress subset")
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


def _safe_mean(values: np.ndarray, mask: np.ndarray) -> float:
    return float(values[mask].mean()) if np.any(mask) else 0.0


def _analyze_stress_subset(
    *,
    reference: ReferenceSolution2D,
    model: torch.nn.Module,
    coords: torch.Tensor,
) -> dict[str, Any]:
    batch_size = int(getattr(model, "inference_batch_size", 4096))
    with torch.no_grad():
        gates = _predict_in_batches(model.get_gate_weights, coords, batch_size).cpu().numpy()
        expert_preds = _predict_in_batches(model.get_expert_predictions, coords, batch_size).cpu().numpy()
        mixture_pred = _predict_in_batches(model, coords, batch_size).cpu().numpy()

    shape = reference.u.shape
    ref = reference.u.reshape(-1)
    gates = gates.reshape(-1, gates.shape[-1])
    expert_preds = np.squeeze(expert_preds, axis=-1).reshape(-1, expert_preds.shape[1])
    mixture_pred = mixture_pred.reshape(-1)

    stress_mask = _compute_directional_stress_spec(reference)["stress_mask"].reshape(-1)
    expert_abs_err = np.abs(expert_preds - ref[:, None])
    mixture_abs_err = np.abs(mixture_pred - ref)
    oracle_idx = expert_abs_err.argmin(axis=1)
    route_idx = gates.argmax(axis=1)
    oracle_err = expert_abs_err.min(axis=1)
    oracle_match = route_idx == oracle_idx

    expert_names = list(getattr(model, "expert_names", []))
    expert_rows: list[dict[str, float | str]] = []
    for idx, expert_name in enumerate(expert_names):
        oracle_frac = float((oracle_idx[stress_mask] == idx).mean()) if np.any(stress_mask) else 0.0
        route_frac = float((route_idx[stress_mask] == idx).mean()) if np.any(stress_mask) else 0.0
        mean_gate = _safe_mean(gates[:, idx], stress_mask)
        expert_rows.append(
            {
                "expert": expert_name,
                "oracle_frac": oracle_frac,
                "route_frac": route_frac,
                "mean_gate": mean_gate,
            }
        )

    metrics = {
        "stress_count": int(stress_mask.sum()),
        "stress_fraction": float(stress_mask.mean()),
        "oracle_match_top1": float(oracle_match[stress_mask].mean()) if np.any(stress_mask) else 0.0,
        "mean_regret_vs_oracle": _safe_mean(np.clip(mixture_abs_err - oracle_err, 0.0, None), stress_mask),
        "mixture_mae_on_stress": _safe_mean(mixture_abs_err, stress_mask),
        "oracle_mae_on_stress": _safe_mean(oracle_err, stress_mask),
        "mean_top1_gate_on_stress": _safe_mean(gates.max(axis=1), stress_mask),
    }
    return {
        "metrics": metrics,
        "experts": expert_rows,
    }


def _plot_breakdown(name: str, expert_rows: list[dict[str, float | str]], save_path: str) -> None:
    expert_names = [str(row["expert"]) for row in expert_rows]
    oracle_vals = [float(row["oracle_frac"]) for row in expert_rows]
    route_vals = [float(row["route_frac"]) for row in expert_rows]
    gate_vals = [float(row["mean_gate"]) for row in expert_rows]

    x = np.arange(len(expert_names))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - width, oracle_vals, width=width, label="oracle frac", color="#4c78a8")
    ax.bar(x, route_vals, width=width, label="top1 route frac", color="#f58518")
    ax.bar(x + width, gate_vals, width=width, label="mean gate", color="#54a24b")
    ax.set_xticks(x)
    ax.set_xticklabels(expert_names, rotation=18, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("fraction / mean weight")
    ax.set_title(f"{name} stress-subset expert breakdown")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_comparison(comparison_rows: list[dict[str, float | str]], save_path: str) -> None:
    model_names = [str(row["name"]) for row in comparison_rows]
    metrics = [
        ("oracle_match_top1", "Oracle Match Top1"),
        ("mean_regret_vs_oracle", "Mean Regret vs Oracle"),
        ("mixture_mae_on_stress", "Mixture MAE on Stress"),
        ("mean_top1_gate_on_stress", "Mean Top1 Gate on Stress"),
    ]
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2"]
    x = np.arange(len(model_names))
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4.2))
    fig.suptitle(
        "Stress Routing Comparison\nHigher is better for oracle match and top1 gate; lower is better for regret and stress MAE.",
        fontsize=12,
        y=0.995,
    )
    for ax, (metric, title) in zip(axes, metrics):
        values = [float(row[metric]) for row in comparison_rows]
        ax.bar(x, values, color=colors[: len(model_names)])
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=18, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    names = args.name or []
    comparison_rows: list[dict[str, float | str]] = []
    report_lines = [
        "# Stress Routing Quantification",
        "",
        f"Device: `{device}`",
        "",
    ]

    for idx, result_dir in enumerate(args.result_dir):
        name = names[idx] if idx < len(names) else os.path.basename(os.path.dirname(result_dir.rstrip("\\/"))) or os.path.basename(result_dir)
        model, stage_info = _load_model(result_dir, device)
        reference, coords = _load_reference(result_dir, device)
        analysis = _analyze_stress_subset(reference=reference, model=model, coords=coords)
        metrics = analysis["metrics"]
        expert_rows = analysis["experts"]

        output_json = os.path.join(args.output_dir, f"{name}_stress_routing_metrics.json")
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
                    "analysis": analysis,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        breakdown_figure = os.path.join(args.output_dir, f"{name}_stress_expert_breakdown.png")
        _plot_breakdown(name, expert_rows, breakdown_figure)

        comparison_rows.append({"name": name, **metrics})
        report_lines.extend(
            [
                f"## {name}",
                "",
                f"- Result dir: `{result_dir}`",
                f"- Stress count: `{int(metrics['stress_count'])}`",
                f"- Oracle-match top1: `{float(metrics['oracle_match_top1']):.6f}`",
                f"- Mean regret vs oracle: `{float(metrics['mean_regret_vs_oracle']):.6f}`",
                f"- Mixture MAE on stress: `{float(metrics['mixture_mae_on_stress']):.6f}`",
                f"- Oracle MAE on stress: `{float(metrics['oracle_mae_on_stress']):.6f}`",
                f"- Breakdown figure: `{breakdown_figure}`",
                "",
                "### Expert Breakdown",
                "",
            ]
        )
        for row in expert_rows:
            report_lines.append(
                f"- `{row['expert']}`: oracle=`{float(row['oracle_frac']):.6f}`, route=`{float(row['route_frac']):.6f}`, mean_gate=`{float(row['mean_gate']):.6f}`"
            )
        report_lines.append("")

    comparison_figure = os.path.join(args.output_dir, "stress_routing_comparison.png")
    _plot_comparison(comparison_rows, comparison_figure)
    report_lines.extend(
        [
            "## Comparison Figure",
            "",
            f"- `{comparison_figure}`",
            "",
        ]
    )

    report_path = os.path.join(args.output_dir, "stress_routing_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"[OK] Saved stress routing report to: {report_path}")


if __name__ == "__main__":
    main()
