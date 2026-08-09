"""
Run a focused ablation for the complex-inspired directional expert.
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
    parser = argparse.ArgumentParser(description="Complex directional expert ablation")
    parser.add_argument("--steps", type=int, default=600, help="Training steps per run")
    parser.add_argument("--seed", type=int, default=42, help="Shared seed for both runs")
    parser.add_argument(
        "--results-root",
        default=None,
        help="Optional root directory for the ablation outputs",
    )
    return parser.parse_args()


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _run_variant(root: str, variant: str, steps: int, seed: int) -> dict:
    run_experiment(
        train_mode="staged",
        n_steps=steps,
        results_root=root,
        staged_variant="gate_only_joint",
        expert_dataset_dir=None,
        directional_expert_variant=variant,
        seed=seed,
    )
    staged_dir = os.path.join(root, "burgers2d_moe_staged")
    return {
        "metrics": _load_json(os.path.join(staged_dir, "metrics.json")),
        "stress_metrics": _load_json(os.path.join(staged_dir, "directional_stress_metrics.json")),
    }


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = args.results_root or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results",
        f"complex_directional_ablation_{stamp}",
    )
    os.makedirs(root, exist_ok=True)

    variants = ["hybrid", "complex_frame"]
    summary: dict[str, dict] = {}
    for variant in variants:
        summary[variant] = _run_variant(
            os.path.join(root, variant),
            variant=variant,
            steps=args.steps,
            seed=args.seed,
        )

    hybrid = summary["hybrid"]
    complex_frame = summary["complex_frame"]
    comparison = {
        "overall": {
            "l2_relative_error_delta": complex_frame["metrics"]["l2_relative_error"] - hybrid["metrics"]["l2_relative_error"],
            "max_absolute_error_delta": complex_frame["metrics"]["max_absolute_error"] - hybrid["metrics"]["max_absolute_error"],
            "steep_mae_delta": complex_frame["metrics"]["steep_mae"] - hybrid["metrics"]["steep_mae"],
            "background_mae_delta": complex_frame["metrics"]["background_mae"] - hybrid["metrics"]["background_mae"],
        },
        "directional_stress_mixture": {
            "mae_delta": complex_frame["stress_metrics"]["mixture"]["mae"] - hybrid["stress_metrics"]["mixture"]["mae"],
            "rmse_delta": complex_frame["stress_metrics"]["mixture"]["rmse"] - hybrid["stress_metrics"]["mixture"]["rmse"],
            "max_error_delta": complex_frame["stress_metrics"]["mixture"]["max_error"] - hybrid["stress_metrics"]["mixture"]["max_error"],
            "l2_relative_error_delta": complex_frame["stress_metrics"]["mixture"]["l2_relative_error"] - hybrid["stress_metrics"]["mixture"]["l2_relative_error"],
        },
        "directional_expert_branch": {
            "hybrid_branch": hybrid["stress_metrics"]["experts"]["directional_shock"],
            "complex_frame_branch": complex_frame["stress_metrics"]["experts"]["directional_shock"],
        },
    }

    with open(os.path.join(root, "ablation_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"runs": summary, "comparison": comparison}, f, ensure_ascii=False, indent=2)

    report_lines = [
        "# Complex Directional Expert Ablation",
        "",
        f"Root: `{root}`",
        "",
        "## Overall Metrics",
        "",
        f"- Hybrid L2: `{hybrid['metrics']['l2_relative_error']:.6f}`",
        f"- Complex-frame L2: `{complex_frame['metrics']['l2_relative_error']:.6f}`",
        f"- Hybrid MaxErr: `{hybrid['metrics']['max_absolute_error']:.6f}`",
        f"- Complex-frame MaxErr: `{complex_frame['metrics']['max_absolute_error']:.6f}`",
        f"- Hybrid Steep MAE: `{hybrid['metrics']['steep_mae']:.6f}`",
        f"- Complex-frame Steep MAE: `{complex_frame['metrics']['steep_mae']:.6f}`",
        f"- Hybrid Background MAE: `{hybrid['metrics']['background_mae']:.6f}`",
        f"- Complex-frame Background MAE: `{complex_frame['metrics']['background_mae']:.6f}`",
        "",
        "## Directional Stress Mixture Metrics",
        "",
        f"- Hybrid stress MAE: `{hybrid['stress_metrics']['mixture']['mae']:.6f}`",
        f"- Complex-frame stress MAE: `{complex_frame['stress_metrics']['mixture']['mae']:.6f}`",
        f"- Hybrid stress MaxErr: `{hybrid['stress_metrics']['mixture']['max_error']:.6f}`",
        f"- Complex-frame stress MaxErr: `{complex_frame['stress_metrics']['mixture']['max_error']:.6f}`",
        f"- Stress mask count: `{hybrid['stress_metrics']['stress_mask_count']}`",
        "",
        "## Directional Expert Branch",
        "",
        f"- Hybrid branch stress MAE: `{hybrid['stress_metrics']['experts']['directional_shock']['mae']:.6f}`",
        f"- Complex-frame branch stress MAE: `{complex_frame['stress_metrics']['experts']['directional_shock']['mae']:.6f}`",
        f"- Hybrid branch stress MaxErr: `{hybrid['stress_metrics']['experts']['directional_shock']['max_error']:.6f}`",
        f"- Complex-frame branch stress MaxErr: `{complex_frame['stress_metrics']['experts']['directional_shock']['max_error']:.6f}`",
    ]
    with open(os.path.join(root, "ablation_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"[OK] Saved complex directional ablation to: {root}")


if __name__ == "__main__":
    main()
