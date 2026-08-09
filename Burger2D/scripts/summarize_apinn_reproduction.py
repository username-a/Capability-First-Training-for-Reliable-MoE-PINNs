"""Aggregate multi-seed APINN metrics and counterfactual diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--run-template", default="seed{seed}_spatial_matched")
    parser.add_argument("--output-prefix", default="apinn_multiseed_summary")
    parser.add_argument("--protocol", default="APINN matched-capacity, spatial gate prior, 1000 gate-pretraining + 8000 joint physics steps")
    parser.add_argument("--claim-profile", choices=("matched", "official2", "matched2"), default="matched")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    rows = []
    for seed in args.seeds:
        run_dir = root / args.run_template.format(seed=seed)
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        counter = json.loads((run_dir / "counterfactual_metrics.json").read_text(encoding="utf-8"))
        history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()]
        joint = [row for row in history if row.get("phase") == "joint_physics_training"]
        best = min(joint, key=lambda row: row["mixture_l2"])
        final = metrics["metrics"]
        drop_ratios = [value["l2_ratio_vs_soft"] for value in counter["drop_one_subnet"].values()]
        routed = [
            value for value in counter["routed_region_diagnostics"].values()
            if value.get("points", 0) > 0 and value.get("branch_rmse") is not None
        ]
        routed_points = sum(value["points"] for value in routed)
        routed_branch_mse = sum(value["points"] * value["branch_rmse"] ** 2 for value in routed) / routed_points
        routed_mixture_mse = sum(value["points"] * value["mixture_rmse"] ** 2 for value in routed) / routed_points
        routed_branch_win = sum(
            value["points"] * value["branch_pointwise_win_rate"] for value in routed
        ) / routed_points
        row = {
            "seed": seed,
            "final_mixture_l2": final["mixture_l2"],
            "final_worst_subnet_l2": final["worst_subnet_l2"],
            "worst_to_mixture_ratio": final["worst_subnet_l2"] / final["mixture_l2"],
            "final_cancellation_ratio": final["cancellation_ratio"],
            "minimum_soft_load": min(final["soft_load"]),
            "near_starved_subnets": sum(value < 0.05 for value in final["soft_load"]),
            "top1_to_soft_l2_ratio": counter["scenarios"]["top1_hard_gate"]["l2_ratio_vs_soft"],
            "uniform_to_soft_l2_ratio": counter["scenarios"]["uniform_average"]["l2_ratio_vs_soft"],
            "oracle_l2": counter["scenarios"]["oracle_hard_gate"]["relative_l2"],
            "responsibility_branch_to_mixture_rmse_ratio": math.sqrt(routed_branch_mse / routed_mixture_mse),
            "responsibility_branch_pointwise_win_rate": routed_branch_win,
            "redundant_by_deletion_count": sum(value < 1.0 for value in drop_ratios),
            "best_mixture_l2": best["mixture_l2"],
            "best_mixture_step": best["step"],
            "worst_subnet_l2_at_best_mixture": best["worst_subnet_l2"],
            "cancellation_at_best_mixture": best["cancellation_ratio"],
        }
        rows.append(row)

    numeric_keys = [key for key in rows[0] if key != "seed"]
    aggregate = {key: _mean_std([float(row[key]) for row in rows]) for key in numeric_keys}
    if args.claim_profile == "matched2":
        supported = [
            "Low mixture error does not identify healthy subnetworks under a capacity-matched comparison.",
            "Soft convex mixing can substantially outperform even pointwise oracle hard routing.",
            "Balanced global load can coexist with high-error branches and cancellation-dominated accuracy.",
            "The phenomenon persists across the tested capacity-matched two-subnetwork seeds.",
        ]
        unsupported = [
            "The observed rates generalise beyond this PDE and training protocol.",
            "Every APINN configuration is cancellation-dominated.",
        ]
    elif args.claim_profile == "official2":
        supported = [
            "Low mixture error does not identify healthy subnetworks.",
            "Soft convex mixing can substantially outperform even pointwise oracle hard routing.",
            "Balanced global load can coexist with high-error branches and cancellation-dominated accuracy.",
            "The phenomenon persists across the tested official-architecture APINN adaptations.",
        ]
        unsupported = [
            "The official-size model is capacity-matched to the proposed MoE-PINN.",
            "The observed rates generalise beyond this PDE and training protocol.",
            "Every APINN configuration is cancellation-dominated.",
        ]
    else:
        supported = [
            "Low mixture error does not identify globally healthy subnetworks.",
            "Useful routing and near-starved or degraded subnetworks can coexist.",
            "The phenomenon persists across matched-capacity APINN seeds.",
        ]
        unsupported = [
            "All soft-mixture accuracy is pathological error cancellation.",
            "APINN lacks meaningful local routing.",
            "The observed rates generalise beyond this PDE and training protocol.",
        ]
    summary = {
        "protocol": args.protocol,
        "seeds": args.seeds,
        "runs": rows,
        "aggregate": aggregate,
        "claims_supported": supported,
        "claims_not_established_by_this_experiment_alone": unsupported,
    }
    (root / f"{args.output_prefix}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (root / f"{args.output_prefix}.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
