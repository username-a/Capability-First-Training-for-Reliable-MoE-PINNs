"""
Run a controlled A/B ablation for the local-context gate on the attribute taxonomy.
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
    parser = argparse.ArgumentParser(description="Local gate ablation")
    parser.add_argument("--steps", type=int, default=300, help="Training steps per run")
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


def _run_variant(root: str, gate_variant: str, steps: int, seed: int) -> dict:
    run_experiment(
        train_mode="staged",
        n_steps=steps,
        results_root=root,
        staged_variant="gate_only_joint",
        expert_dataset_dir=None,
        directional_expert_variant="hybrid",
        expert_layout_variant="attribute",
        gate_variant=gate_variant,
        seed=seed,
    )
    staged_dir = os.path.join(root, "burgers2d_moe_staged")
    return {
        "metrics": _load_json(os.path.join(staged_dir, "metrics.json")),
        "stress_metrics": _load_json(os.path.join(staged_dir, "directional_stress_metrics.json")),
        "stage_config": _load_json(os.path.join(staged_dir, "stage_config.json")),
    }


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = args.results_root or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results",
        f"local_gate_ablation_{stamp}",
    )
    os.makedirs(root, exist_ok=True)

    variants = ["pointwise", "local_knn"]
    summary: dict[str, dict] = {}
    for gate_variant in variants:
        summary[gate_variant] = _run_variant(
            os.path.join(root, gate_variant),
            gate_variant=gate_variant,
            steps=args.steps,
            seed=args.seed,
        )

    pointwise = summary["pointwise"]
    local_knn = summary["local_knn"]
    branch_name = local_knn["stress_metrics"].get("directional_branch_name", "anisotropy_directional")

    comparison = {
        "overall": {
            "l2_relative_error_delta": local_knn["metrics"]["l2_relative_error"] - pointwise["metrics"]["l2_relative_error"],
            "max_absolute_error_delta": local_knn["metrics"]["max_absolute_error"] - pointwise["metrics"]["max_absolute_error"],
            "steep_mae_delta": local_knn["metrics"]["steep_mae"] - pointwise["metrics"]["steep_mae"],
            "background_mae_delta": local_knn["metrics"]["background_mae"] - pointwise["metrics"]["background_mae"],
        },
        "directional_stress_mixture": {
            "mae_delta": local_knn["stress_metrics"]["mixture"]["mae"] - pointwise["stress_metrics"]["mixture"]["mae"],
            "rmse_delta": local_knn["stress_metrics"]["mixture"]["rmse"] - pointwise["stress_metrics"]["mixture"]["rmse"],
            "max_error_delta": local_knn["stress_metrics"]["mixture"]["max_error"] - pointwise["stress_metrics"]["mixture"]["max_error"],
            "l2_relative_error_delta": local_knn["stress_metrics"]["mixture"]["l2_relative_error"] - pointwise["stress_metrics"]["mixture"]["l2_relative_error"],
        },
        "directional_branch": {
            "branch_name": branch_name,
            "pointwise": pointwise["stress_metrics"]["experts"][branch_name],
            "local_knn": local_knn["stress_metrics"]["experts"][branch_name],
        },
    }

    with open(os.path.join(root, "ablation_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"runs": summary, "comparison": comparison}, f, ensure_ascii=False, indent=2)

    report_lines = [
        "# Local Gate Ablation",
        "",
        f"Root: `{root}`",
        "",
        "## Overall Metrics",
        "",
        f"- Pointwise L2: `{pointwise['metrics']['l2_relative_error']:.6f}`",
        f"- Local-kNN L2: `{local_knn['metrics']['l2_relative_error']:.6f}`",
        f"- Pointwise MaxErr: `{pointwise['metrics']['max_absolute_error']:.6f}`",
        f"- Local-kNN MaxErr: `{local_knn['metrics']['max_absolute_error']:.6f}`",
        f"- Pointwise Steep MAE: `{pointwise['metrics']['steep_mae']:.6f}`",
        f"- Local-kNN Steep MAE: `{local_knn['metrics']['steep_mae']:.6f}`",
        f"- Pointwise Background MAE: `{pointwise['metrics']['background_mae']:.6f}`",
        f"- Local-kNN Background MAE: `{local_knn['metrics']['background_mae']:.6f}`",
        "",
        "## Directional Stress Mixture Metrics",
        "",
        f"- Pointwise stress MAE: `{pointwise['stress_metrics']['mixture']['mae']:.6f}`",
        f"- Local-kNN stress MAE: `{local_knn['stress_metrics']['mixture']['mae']:.6f}`",
        f"- Pointwise stress MaxErr: `{pointwise['stress_metrics']['mixture']['max_error']:.6f}`",
        f"- Local-kNN stress MaxErr: `{local_knn['stress_metrics']['mixture']['max_error']:.6f}`",
        "",
        "## Directional Branch",
        "",
        f"- Branch: `{branch_name}`",
        f"- Pointwise branch stress MAE: `{pointwise['stress_metrics']['experts'][branch_name]['mae']:.6f}`",
        f"- Local-kNN branch stress MAE: `{local_knn['stress_metrics']['experts'][branch_name]['mae']:.6f}`",
        f"- Pointwise branch stress MaxErr: `{pointwise['stress_metrics']['experts'][branch_name]['max_error']:.6f}`",
        f"- Local-kNN branch stress MaxErr: `{local_knn['stress_metrics']['experts'][branch_name]['max_error']:.6f}`",
    ]
    with open(os.path.join(root, "ablation_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"[OK] Saved local gate ablation to: {root}")


if __name__ == "__main__":
    main()
