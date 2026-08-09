"""
Run a controlled A/B ablation for the directional shock expert.
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
    parser = argparse.ArgumentParser(description="Directional shock expert stress ablation")
    parser.add_argument("--steps", type=int, default=1500, help="Training steps per run")
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


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = args.results_root or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results",
        f"directional_stress_ablation_{stamp}",
    )
    os.makedirs(root, exist_ok=True)

    variants = ["legacy", "hybrid"]
    summary: dict[str, dict] = {}

    for variant in variants:
        variant_root = os.path.join(root, variant)
        run_experiment(
            train_mode="staged",
            n_steps=args.steps,
            results_root=variant_root,
            staged_variant="gate_only_joint",
            expert_dataset_dir=None,
            directional_expert_variant=variant,
            seed=args.seed,
        )
        staged_dir = os.path.join(variant_root, "burgers2d_moe_staged")
        summary[variant] = {
            "metrics": _load_json(os.path.join(staged_dir, "metrics.json")),
            "stress_metrics": _load_json(os.path.join(staged_dir, "directional_stress_metrics.json")),
        }

    legacy = summary["legacy"]
    hybrid = summary["hybrid"]
    comparison = {
        "overall": {
            "l2_relative_error_delta": hybrid["metrics"]["l2_relative_error"] - legacy["metrics"]["l2_relative_error"],
            "max_absolute_error_delta": hybrid["metrics"]["max_absolute_error"] - legacy["metrics"]["max_absolute_error"],
            "steep_mae_delta": hybrid["metrics"]["steep_mae"] - legacy["metrics"]["steep_mae"],
            "background_mae_delta": hybrid["metrics"]["background_mae"] - legacy["metrics"]["background_mae"],
        },
        "directional_stress_mixture": {
            "mae_delta": hybrid["stress_metrics"]["mixture"]["mae"] - legacy["stress_metrics"]["mixture"]["mae"],
            "rmse_delta": hybrid["stress_metrics"]["mixture"]["rmse"] - legacy["stress_metrics"]["mixture"]["rmse"],
            "max_error_delta": hybrid["stress_metrics"]["mixture"]["max_error"] - legacy["stress_metrics"]["mixture"]["max_error"],
            "l2_relative_error_delta": hybrid["stress_metrics"]["mixture"]["l2_relative_error"] - legacy["stress_metrics"]["mixture"]["l2_relative_error"],
        },
        "directional_expert_branch": {
            "hybrid_branch": hybrid["stress_metrics"]["experts"]["directional_shock"],
            "legacy_branch": legacy["stress_metrics"]["experts"]["directional_shock"],
        },
    }

    with open(os.path.join(root, "ablation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "runs": summary,
                "comparison": comparison,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    report_lines = [
        "# Directional Stress Ablation",
        "",
        f"Root: `{root}`",
        "",
        "## Overall Metrics",
        "",
        f"- Legacy L2: `{legacy['metrics']['l2_relative_error']:.6f}`",
        f"- Hybrid L2: `{hybrid['metrics']['l2_relative_error']:.6f}`",
        f"- Legacy MaxErr: `{legacy['metrics']['max_absolute_error']:.6f}`",
        f"- Hybrid MaxErr: `{hybrid['metrics']['max_absolute_error']:.6f}`",
        f"- Legacy Steep MAE: `{legacy['metrics']['steep_mae']:.6f}`",
        f"- Hybrid Steep MAE: `{hybrid['metrics']['steep_mae']:.6f}`",
        f"- Legacy Background MAE: `{legacy['metrics']['background_mae']:.6f}`",
        f"- Hybrid Background MAE: `{hybrid['metrics']['background_mae']:.6f}`",
        "",
        "## Directional Stress Mixture Metrics",
        "",
        f"- Legacy stress MAE: `{legacy['stress_metrics']['mixture']['mae']:.6f}`",
        f"- Hybrid stress MAE: `{hybrid['stress_metrics']['mixture']['mae']:.6f}`",
        f"- Legacy stress MaxErr: `{legacy['stress_metrics']['mixture']['max_error']:.6f}`",
        f"- Hybrid stress MaxErr: `{hybrid['stress_metrics']['mixture']['max_error']:.6f}`",
        f"- Stress mask count: `{hybrid['stress_metrics']['stress_mask_count']}`",
        "",
        "## Directional Expert Branch",
        "",
        f"- Legacy branch stress MAE: `{legacy['stress_metrics']['experts']['directional_shock']['mae']:.6f}`",
        f"- Hybrid branch stress MAE: `{hybrid['stress_metrics']['experts']['directional_shock']['mae']:.6f}`",
        f"- Legacy branch stress MaxErr: `{legacy['stress_metrics']['experts']['directional_shock']['max_error']:.6f}`",
        f"- Hybrid branch stress MaxErr: `{hybrid['stress_metrics']['experts']['directional_shock']['max_error']:.6f}`",
    ]
    with open(os.path.join(root, "ablation_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"[OK] Saved ablation summary to: {root}")


if __name__ == "__main__":
    main()
