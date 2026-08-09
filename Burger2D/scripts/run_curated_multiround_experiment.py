"""Run multi-round full-size staged experiments with a curated expert dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from statistics import mean, pstdev


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Burger2D.data import build_curated_expert_dataset
from Burger2D.experiments.run_burgers2d import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-round full Burger2D curated experiments")
    parser.add_argument("--rounds", type=int, default=3, help="Number of full staged rounds")
    parser.add_argument("--base-seed", type=int, default=42, help="Base seed for dataset and runs")
    parser.add_argument("--dataset-cases", type=int, default=24, help="Number of random reference cases")
    parser.add_argument("--dataset-nx", type=int, default=49, help="Curated dataset grid size in x")
    parser.add_argument("--dataset-ny", type=int, default=49, help="Curated dataset grid size in y")
    parser.add_argument("--dataset-nt", type=int, default=21, help="Curated dataset grid size in t")
    parser.add_argument(
        "--dataset-target-points-per-expert",
        type=int,
        default=120000,
        help="Balanced number of exported points per expert",
    )
    parser.add_argument("--steps", type=int, default=2500, help="Training steps per round")
    parser.add_argument("--n-col", type=int, default=12000, help="Collocation points")
    parser.add_argument("--n-ic", type=int, default=3000, help="Initial-condition points")
    parser.add_argument("--n-bc", type=int, default=800, help="Boundary points per face")
    parser.add_argument("--nx", type=int, default=81, help="Reference grid size in x")
    parser.add_argument("--ny", type=int, default=81, help="Reference grid size in y")
    parser.add_argument("--nt", type=int, default=31, help="Reference grid size in t")
    parser.add_argument("--device", default=None, help="Force device: cpu or cuda")
    parser.add_argument(
        "--results-root",
        default=None,
        help="Optional root directory for this multi-round run",
    )
    return parser.parse_args()


def _summary_stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = args.results_root or os.path.join(PACKAGE_ROOT, "results", f"curated_multiround_{timestamp}")
    os.makedirs(run_root, exist_ok=True)

    dataset_dir = os.path.join(run_root, "curated_dataset_v2")
    print("=" * 72)
    print("Burger2D curated multi-round full experiment")
    print("=" * 72)
    print(f"run_root   : {run_root}")
    print(f"dataset    : {dataset_dir}")
    print(f"rounds     : {args.rounds}")
    print(f"base_seed  : {args.base_seed}")
    print(f"train grid : ({args.nx}, {args.ny}, {args.nt})")
    print(f"data  grid : ({args.dataset_nx}, {args.dataset_ny}, {args.dataset_nt})")

    manifest = build_curated_expert_dataset(
        output_dir=dataset_dir,
        num_cases=args.dataset_cases,
        nx=args.dataset_nx,
        ny=args.dataset_ny,
        nt=args.dataset_nt,
        seed=args.base_seed,
        target_points_per_expert=args.dataset_target_points_per_expert,
    )
    with open(os.path.join(run_root, "dataset_manifest_snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    round_metrics = []
    for round_idx in range(args.rounds):
        seed = args.base_seed + round_idx
        round_dir = os.path.join(run_root, f"round_{round_idx + 1:02d}_seed_{seed}")
        print("-" * 72)
        print(f"Round {round_idx + 1}/{args.rounds} | seed={seed}")
        print(f"Output: {round_dir}")
        results = run_experiment(
            train_mode="staged",
            n_steps=args.steps,
            n_col=args.n_col,
            n_ic=args.n_ic,
            n_bc_per_face=args.n_bc,
            nx=args.nx,
            ny=args.ny,
            nt=args.nt,
            device_override=args.device,
            results_root=round_dir,
            staged_variant="gate_only_joint",
            expert_dataset_dir=dataset_dir,
            seed=seed,
        )
        staged_metrics = results["staged"]
        round_metrics.append(
            {
                "round": round_idx + 1,
                "seed": seed,
                "result_dir": os.path.join(round_dir, "burgers2d_moe_staged"),
                **staged_metrics,
            }
        )

    summary = {
        "run_root": run_root,
        "dataset_dir": dataset_dir,
        "rounds": round_metrics,
        "aggregate": {
            key: _summary_stats([float(item[key]) for item in round_metrics])
            for key in [
                "l2_relative_error",
                "max_absolute_error",
                "steep_mae",
                "background_mae",
                "route_entropy",
                "route_max_weight",
            ]
        },
    }
    with open(os.path.join(run_root, "multiround_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("[OK] Multi-round curated experiment finished")
    print("=" * 72)
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
