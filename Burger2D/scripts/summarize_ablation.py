"""Aggregate a directional ablation summary JSON into mean +/- std tables."""

from __future__ import annotations

import json
import statistics as st
import sys


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else (
        "Burger2D/results/directional_ablation_20260802_234846/ablation_summary.json"
    )
    data = json.load(open(path, encoding="utf-8"))
    groups: dict[str, list[dict]] = {"full4": [], "no_directional": []}
    for e in data:
        m = e["metrics"]
        key = "full4" if not e["exclude"] else "no_directional"
        groups[key].append(
            {
                "seed": e["seed"],
                "L2": m["l2_relative_error"],
                "MaxErr": m["max_absolute_error"],
                "Steep": m["steep_mae"],
                "Bg": m["background_mae"],
                "entropy": m["route_entropy"],
                "loads": m["expert_load_frac"],
            }
        )

    def fmt(vals: list[float]) -> str:
        return f"{st.mean(vals):.4f}±{st.stdev(vals):.4f}" if len(vals) > 1 else f"{vals[0]:.4f}"

    print(f"{'variant':<15}{'L2':>16}{'MaxErr':>14}{'Steep':>12}{'Bg':>12}{'entropy':>12}")
    for name, rows in groups.items():
        print(
            f"{name:<15}"
            f"{fmt([r['L2'] for r in rows]):>16}"
            f"{fmt([r['MaxErr'] for r in rows]):>14}"
            f"{fmt([r['Steep'] for r in rows]):>12}"
            f"{fmt([r['Bg'] for r in rows]):>12}"
            f"{fmt([r['entropy'] for r in rows]):>12}"
        )
        loads = [r["loads"] for r in rows]
        avg = [sum(l[i] for l in loads) / len(loads) for i in range(len(loads[0]))]
        print(f"{'':<15}avg loads: {[round(x, 3) for x in avg]}")

    print("\nper-seed deltas (no_directional - full4):")
    seeds = sorted({r["seed"] for g in groups.values() for r in g})
    for seed in seeds:
        f4 = next(r for r in groups["full4"] if r["seed"] == seed)
        nd = next(r for r in groups["no_directional"] if r["seed"] == seed)
        print(
            f"  seed {seed}: dL2={nd['L2'] - f4['L2']:+.4f}  "
            f"dMaxErr={nd['MaxErr'] - f4['MaxErr']:+.4f}  "
            f"dSteep={nd['Steep'] - f4['Steep']:+.4f}  "
            f"dBg={nd['Bg'] - f4['Bg']:+.4f}"
        )


if __name__ == "__main__":
    main()
