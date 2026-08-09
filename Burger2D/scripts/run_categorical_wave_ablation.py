"""
Run a controlled ablation for categorical wave expert variants.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Burger2D.experiments.run_burgers2d import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Categorical wave expert ablation")
    parser.add_argument("--steps", type=int, default=300, help="Training steps per run")
    parser.add_argument("--seed", type=int, default=42, help="Shared seed for all runs")
    parser.add_argument(
        "--staged-variant",
        default="gate_only_joint",
        choices=["gate_only_joint", "stronger_expert_calibration", "stronger_expert_route_only", "stronger_expert_route_sharp", "stronger_expert_oracle_consistency", "stronger_expert_base_gate", "full_joint", "no_joint"],
        help="Staged training preset to use for all variants",
    )
    parser.add_argument(
        "--gate-variant",
        default="pointwise",
        choices=["pointwise", "local_knn", "local_conv"],
        help="Gate architecture to use for all variants",
    )
    parser.add_argument("--results-root", default=None, help="Optional ablation root directory")
    return parser.parse_args()


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _run_variant(root: str, wave_expert_variant: str, steps: int, seed: int, staged_variant: str, gate_variant: str) -> dict:
    run_experiment(
        train_mode="staged",
        n_steps=steps,
        results_root=root,
        staged_variant=staged_variant,
        expert_dataset_dir=None,
        directional_expert_variant="hybrid",
        wave_expert_variant=wave_expert_variant,
        expert_layout_variant="categorical",
        attribute_expert_variant="base",
        gate_variant=gate_variant,
        seed=seed,
    )
    staged_dir = os.path.join(root, "burgers2d_moe_staged")
    return {
        "metrics": _load_json(os.path.join(staged_dir, "metrics.json")),
        "stress_metrics": _load_json(os.path.join(staged_dir, "directional_stress_metrics.json")),
        "expert_metrics": _load_json(os.path.join(staged_dir, "expert_metrics.json")),
        "stage_config": _load_json(os.path.join(staged_dir, "stage_config.json")),
    }


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = args.results_root or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results",
        f"categorical_wave_ablation_{stamp}",
    )
    os.makedirs(root, exist_ok=True)

    variants = ["base", "mixed", "mixed_lite"]
    summary: dict[str, dict] = {}
    for variant in variants:
        summary[variant] = _run_variant(
            os.path.join(root, variant),
            variant,
            args.steps,
            args.seed,
            args.staged_variant,
            args.gate_variant,
        )

    base = summary["base"]
    branch_name = base["stress_metrics"].get("directional_branch_name", "directional_shock")
    comparison: dict[str, dict] = {}
    for variant in variants:
        if variant == "base":
            continue
        current = summary[variant]
        comparison[variant] = {
            "overall": {
                "l2_relative_error_delta": current["metrics"]["l2_relative_error"] - base["metrics"]["l2_relative_error"],
                "max_absolute_error_delta": current["metrics"]["max_absolute_error"] - base["metrics"]["max_absolute_error"],
                "steep_mae_delta": current["metrics"]["steep_mae"] - base["metrics"]["steep_mae"],
                "background_mae_delta": current["metrics"]["background_mae"] - base["metrics"]["background_mae"],
            },
            "directional_stress_mixture": {
                "mae_delta": current["stress_metrics"]["mixture"]["mae"] - base["stress_metrics"]["mixture"]["mae"],
                "rmse_delta": current["stress_metrics"]["mixture"]["rmse"] - base["stress_metrics"]["mixture"]["rmse"],
                "max_error_delta": current["stress_metrics"]["mixture"]["max_error"] - base["stress_metrics"]["mixture"]["max_error"],
                "l2_relative_error_delta": current["stress_metrics"]["mixture"]["l2_relative_error"] - base["stress_metrics"]["mixture"]["l2_relative_error"],
            },
            "directional_branch": {
                "branch_name": branch_name,
                "base": base["stress_metrics"]["experts"][branch_name],
                "current": current["stress_metrics"]["experts"][branch_name],
            },
            "wave_branch": {
                "base": base["expert_metrics"]["wave"],
                "current": current["expert_metrics"]["wave"],
            },
        }

    with open(os.path.join(root, "ablation_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"runs": summary, "comparison": comparison}, f, ensure_ascii=False, indent=2)

    report_lines = [
        "# Categorical Wave Expert Ablation",
        "",
        f"Root: `{root}`",
        f"Staged variant: `{args.staged_variant}`",
        f"Gate variant: `{args.gate_variant}`",
        "",
        "## Overall Metrics",
        "",
        "| Variant | L2 | MaxErr | Steep MAE | Background MAE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for variant in variants:
        metrics = summary[variant]["metrics"]
        report_lines.append(
            f"| `{variant}` | `{metrics['l2_relative_error']:.6f}` | `{metrics['max_absolute_error']:.6f}` | `{metrics['steep_mae']:.6f}` | `{metrics['background_mae']:.6f}` |"
        )

    report_lines.extend(
        [
            "",
            "## Directional Stress Mixture Metrics",
            "",
            "| Variant | Stress MAE | Stress MaxErr | Stress L2 rel |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for variant in variants:
        stress = summary[variant]["stress_metrics"]["mixture"]
        report_lines.append(
            f"| `{variant}` | `{stress['mae']:.6f}` | `{stress['max_error']:.6f}` | `{stress['l2_relative_error']:.6f}` |"
        )

    report_lines.extend(
        [
            "",
            "## Branch Diagnostics",
            "",
            f"- Directional branch: `{branch_name}`",
            "",
            "| Variant | Directional stress MAE | Wave L2 | Wave Steep MAE |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for variant in variants:
        directional = summary[variant]["stress_metrics"]["experts"][branch_name]
        wave = summary[variant]["expert_metrics"]["wave"]
        report_lines.append(
            f"| `{variant}` | `{directional['mae']:.6f}` | `{wave['l2_relative_error']:.6f}` | `{wave['steep_mae']:.6f}` |"
        )

    with open(os.path.join(root, "ablation_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"[OK] Saved categorical wave ablation to: {root}")


if __name__ == "__main__":
    main()
