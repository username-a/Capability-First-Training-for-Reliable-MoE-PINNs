"""Build a curated Burger2D dataset for specialist expert training."""

from __future__ import annotations

import argparse
import json
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Burger2D.data import build_curated_expert_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build curated Burger2D expert dataset")
    parser.add_argument("--output-dir", required=True, help="Dataset output directory")
    parser.add_argument("--num-cases", type=int, default=12, help="Number of random reference cases")
    parser.add_argument("--nx", type=int, default=41, help="Reference grid size in x")
    parser.add_argument("--ny", type=int, default=41, help="Reference grid size in y")
    parser.add_argument("--nt", type=int, default=17, help="Reference grid size in t")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--expert-keep-quantile",
        type=float,
        default=0.72,
        help="Keep points above this per-expert score quantile",
    )
    parser.add_argument(
        "--expert-min-weight",
        type=float,
        default=0.22,
        help="Minimum per-expert weight threshold",
    )
    parser.add_argument(
        "--target-points-per-expert",
        type=int,
        default=None,
        help="Optional balanced export size for each expert dataset",
    )
    parser.add_argument(
        "--expert-confidence-quantile",
        type=float,
        default=0.55,
        help="Confidence quantile used when filtering label-matched expert samples",
    )
    parser.add_argument(
        "--expert-min-confidence",
        type=float,
        default=0.16,
        help="Minimum confidence used for expert-sample filtering",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_curated_expert_dataset(
        output_dir=args.output_dir,
        num_cases=args.num_cases,
        nx=args.nx,
        ny=args.ny,
        nt=args.nt,
        seed=args.seed,
        expert_keep_quantile=args.expert_keep_quantile,
        expert_min_weight=args.expert_min_weight,
        target_points_per_expert=args.target_points_per_expert,
        expert_confidence_quantile=args.expert_confidence_quantile,
        expert_min_confidence=args.expert_min_confidence,
    )
    print("=" * 72)
    print("Curated Burger2D expert dataset built")
    print("=" * 72)
    print(json.dumps(
        {
            "output_dir": args.output_dir,
            "num_cases": manifest["num_cases"],
            "catalog_num_points": manifest["catalog_num_points"],
            "label_histogram": manifest["label_histogram"],
            "expert_summary": manifest["expert_summary"],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
