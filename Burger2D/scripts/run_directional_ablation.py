"""
Directional-shock ablation for 2D Burgers staged MoE-PINN.

Runs the mainline staged recipe (stronger_expert_route_sharp + local_conv gate
+ mixed_lite wave + hybrid directional) with and without the
`directional_shock` expert, for several seeds, and aggregates the metrics.

Usage (smoke):
    python Burger2D/scripts/run_directional_ablation.py --smoke

Usage (full):
    python Burger2D/scripts/run_directional_ablation.py --steps 1500 \
        --seeds 42 43 44 --nx 81 --ny 81 --nt 31
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Burger2D.experiments.run_burgers2d import run_experiment  # noqa: E402


STAGED_VARIANT = "stronger_expert_route_sharp"
GATE_VARIANT = "local_conv"
WAVE_VARIANT = "mixed_lite"
DIRECTIONAL_VARIANT = "hybrid"
LAYOUT_VARIANT = "categorical"


def run_one(
    *,
    results_root: str,
    exclude: str,
    seed: int,
    steps: int,
    n_col: int,
    n_ic: int,
    n_bc_per_face: int,
    nx: int,
    ny: int,
    nt: int,
) -> dict:
    tag = "no_directional" if exclude else "full4"
    run_dir = os.path.join(results_root, f"{tag}_seed{seed}")
    t0 = time.time()
    print(f"\n{'='*72}\n[{tag} seed={seed}] steps={steps} grid={nx}x{ny}x{nt}\n{'='*72}")
    metrics = run_experiment(
        train_mode="staged",
        n_steps=steps,
        n_col=n_col,
        n_ic=n_ic,
        n_bc_per_face=n_bc_per_face,
        nx=nx,
        ny=ny,
        nt=nt,
        results_root=run_dir,
        staged_variant=STAGED_VARIANT,
        gate_variant=GATE_VARIANT,
        wave_expert_variant=WAVE_VARIANT,
        directional_expert_variant=DIRECTIONAL_VARIANT,
        expert_layout_variant=LAYOUT_VARIANT,
        exclude_experts=exclude,
        seed=seed,
    )
    elapsed = time.time() - t0
    entry = {
        "exclude": exclude,
        "seed": seed,
        "elapsed_sec": elapsed,
        "metrics": metrics.get("staged", {}),
    }
    with open(os.path.join(run_dir, "ablation_entry.json"), "w", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False, indent=2)
    print(f"[{tag} seed={seed}] done in {elapsed:.0f}s, L2={metrics.get('staged', {}).get('l2_relative_error', float('nan')):.4f}")
    return entry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="tiny budget sanity check")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--nx", type=int, default=81)
    parser.add_argument("--ny", type=int, default=81)
    parser.add_argument("--nt", type=int, default=31)
    args = parser.parse_args()

    if args.smoke:
        args.steps = 300
        args.nx, args.ny, args.nt = 33, 33, 11
        args.seeds = [42]

    results_root = os.path.join(
        PACKAGE_ROOT,
        "results",
        f"directional_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(results_root, exist_ok=True)

    entries = []
    for seed in args.seeds:
        for exclude in ("", "directional_shock"):
            entries.append(
                run_one(
                    results_root=results_root,
                    exclude=exclude,
                    seed=seed,
                    steps=args.steps,
                    n_col=3000 if args.smoke else 12000,
                    n_ic=800 if args.smoke else 3000,
                    n_bc_per_face=200 if args.smoke else 800,
                    nx=args.nx,
                    ny=args.ny,
                    nt=args.nt,
                )
            )

    summary_path = os.path.join(results_root, "ablation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 100)
    print("ABLATION SUMMARY")
    print("=" * 100)
    header = f"{'variant':<16}{'seed':<6}{'L2':>9}{'MaxErr':>10}{'SteepMAE':>10}{'BgMAE':>10}{'entropy':>9}{'loads':>40}"
    print(header)
    for e in entries:
        m = e["metrics"]
        loads = ",".join(f"{x:.2f}" for x in m.get("expert_load_frac", []))
        print(
            f"{('no_directional' if e['exclude'] else 'full4'):<16}{e['seed']:<6}"
            f"{m.get('l2_relative_error', float('nan')):>9.4f}"
            f"{m.get('max_absolute_error', float('nan')):>10.4f}"
            f"{m.get('steep_mae', float('nan')):>10.4f}"
            f"{m.get('background_mae', float('nan')):>10.4f}"
            f"{m.get('route_entropy', float('nan')):>9.3f}"
            f"{loads:>40}"
        )
    print(f"\n[OK] Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
