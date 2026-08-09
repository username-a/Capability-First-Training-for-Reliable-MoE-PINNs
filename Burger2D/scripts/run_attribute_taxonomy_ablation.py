"""
Run a controlled A/B ablation for categorical vs attribute expert taxonomies.
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
    parser = argparse.ArgumentParser(description="Attribute-taxonomy ablation")
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


def _run_variant(root: str, layout_variant: str, steps: int, seed: int) -> dict:
    run_experiment(
        train_mode="staged",
        n_steps=steps,
        results_root=root,
        staged_variant="gate_only_joint",
        expert_dataset_dir=None,
        directional_expert_variant="hybrid",
        expert_layout_variant=layout_variant,
        seed=seed,
    )
    staged_dir = os.path.join(root, "burgers2d_moe_staged")
    return {
        "metrics": _load_json(os.path.join(staged_dir, "metrics.json")),
        "stress_metrics": _load_json(os.path.join(staged_dir, "directional_stress_metrics.json")),
        "stage_config": _load_json(os.path.join(staged_dir, "stage_config.json")),
    }


def _directional_branch_metrics(summary: dict) -> tuple[str | None, dict | None]:
    branch_name = summary["stress_metrics"].get("directional_branch_name")
    experts = summary["stress_metrics"].get("experts", {})
    if branch_name is None or branch_name not in experts:
        return branch_name, None
    return branch_name, experts[branch_name]


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = args.results_root or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results",
        f"attribute_taxonomy_ablation_{stamp}",
    )
    os.makedirs(root, exist_ok=True)

    layouts = ["categorical", "attribute"]
    summary: dict[str, dict] = {}
    for layout in layouts:
        summary[layout] = _run_variant(
            os.path.join(root, layout),
            layout_variant=layout,
            steps=args.steps,
            seed=args.seed,
        )

    categorical = summary["categorical"]
    attribute = summary["attribute"]
    cat_branch_name, cat_branch = _directional_branch_metrics(categorical)
    attr_branch_name, attr_branch = _directional_branch_metrics(attribute)

    comparison = {
        "overall": {
            "l2_relative_error_delta": attribute["metrics"]["l2_relative_error"] - categorical["metrics"]["l2_relative_error"],
            "max_absolute_error_delta": attribute["metrics"]["max_absolute_error"] - categorical["metrics"]["max_absolute_error"],
            "steep_mae_delta": attribute["metrics"]["steep_mae"] - categorical["metrics"]["steep_mae"],
            "background_mae_delta": attribute["metrics"]["background_mae"] - categorical["metrics"]["background_mae"],
        },
        "directional_stress_mixture": {
            "mae_delta": attribute["stress_metrics"]["mixture"]["mae"] - categorical["stress_metrics"]["mixture"]["mae"],
            "rmse_delta": attribute["stress_metrics"]["mixture"]["rmse"] - categorical["stress_metrics"]["mixture"]["rmse"],
            "max_error_delta": attribute["stress_metrics"]["mixture"]["max_error"] - categorical["stress_metrics"]["mixture"]["max_error"],
            "l2_relative_error_delta": attribute["stress_metrics"]["mixture"]["l2_relative_error"] - categorical["stress_metrics"]["mixture"]["l2_relative_error"],
        },
        "directional_branch": {
            "categorical_name": cat_branch_name,
            "attribute_name": attr_branch_name,
            "categorical": cat_branch,
            "attribute": attr_branch,
        },
    }

    with open(os.path.join(root, "ablation_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"runs": summary, "comparison": comparison}, f, ensure_ascii=False, indent=2)

    report_lines = [
        "# Attribute Taxonomy Ablation",
        "",
        f"Root: `{root}`",
        "",
        "## Overall Metrics",
        "",
        f"- Categorical L2: `{categorical['metrics']['l2_relative_error']:.6f}`",
        f"- Attribute L2: `{attribute['metrics']['l2_relative_error']:.6f}`",
        f"- Categorical MaxErr: `{categorical['metrics']['max_absolute_error']:.6f}`",
        f"- Attribute MaxErr: `{attribute['metrics']['max_absolute_error']:.6f}`",
        f"- Categorical Steep MAE: `{categorical['metrics']['steep_mae']:.6f}`",
        f"- Attribute Steep MAE: `{attribute['metrics']['steep_mae']:.6f}`",
        f"- Categorical Background MAE: `{categorical['metrics']['background_mae']:.6f}`",
        f"- Attribute Background MAE: `{attribute['metrics']['background_mae']:.6f}`",
        "",
        "## Directional Stress Mixture Metrics",
        "",
        f"- Categorical stress MAE: `{categorical['stress_metrics']['mixture']['mae']:.6f}`",
        f"- Attribute stress MAE: `{attribute['stress_metrics']['mixture']['mae']:.6f}`",
        f"- Categorical stress MaxErr: `{categorical['stress_metrics']['mixture']['max_error']:.6f}`",
        f"- Attribute stress MaxErr: `{attribute['stress_metrics']['mixture']['max_error']:.6f}`",
        f"- Stress mask count: `{attribute['stress_metrics']['stress_mask_count']}`",
        "",
        "## Directional Branch",
        "",
        f"- Categorical branch: `{cat_branch_name}`",
        f"- Attribute branch: `{attr_branch_name}`",
    ]
    if cat_branch is not None and attr_branch is not None:
        report_lines.extend(
            [
                f"- Categorical branch stress MAE: `{cat_branch['mae']:.6f}`",
                f"- Attribute branch stress MAE: `{attr_branch['mae']:.6f}`",
                f"- Categorical branch stress MaxErr: `{cat_branch['max_error']:.6f}`",
                f"- Attribute branch stress MaxErr: `{attr_branch['max_error']:.6f}`",
            ]
        )
    with open(os.path.join(root, "ablation_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"[OK] Saved attribute taxonomy ablation to: {root}")


if __name__ == "__main__":
    main()
