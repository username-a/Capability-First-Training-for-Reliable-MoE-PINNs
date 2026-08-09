"""Print a compact table from expert_convergence_summary.json."""

from __future__ import annotations

import json
import sys


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else (
        "Burger2D/results/expert_convergence_summary.json"
    )
    data = json.load(open(path, encoding="utf-8"))
    for run, models in data.items():
        print("=" * 100)
        print("RUN:", run)
        for mdir, mods in models.get("models", {}).items():
            for pt, m in mods.items():
                if "error" in m:
                    print("  ", mdir, pt, "-> ERROR", m["error"])
                    continue
                print(
                    "--- %s/%s [gate=%s dir=%s wave=%s]"
                    % (
                        mdir,
                        pt,
                        m["gate_variant_detected"],
                        m["directional_variant_detected"],
                        m["wave_variant_detected"],
                    )
                )
                print(
                    "   mixture L2=%.4f MaxErr=%.4f eff_experts=%.2f entropy=%.3f max_w=%.3f"
                    % (
                        m["mixture_l2_relative_error"],
                        m["mixture_max_absolute_error"],
                        m["effective_experts"],
                        m["route_entropy"],
                        m["route_max_weight"],
                    )
                )
                print(
                    "   per-expert L2: %s"
                    % {
                        k: round(v, 3)
                        for k, v in m["per_expert_l2_relative_error"].items()
                    }
                )
                print(
                    "   load frac: %s"
                    % {k: round(v, 3) for k, v in m["expert_load_frac"].items()}
                )
                cc = m["correction_correlation"]
                cs = m["correction_cosine"]
                print(
                    "   corr-corr mean=%.3f max=%.3f | corr-cosine mean=%.3f max=%.3f"
                    % (cc["mean_off_diag"], cc["max_off_diag"], cs["mean_off_diag"], cs["max_off_diag"])
                )
                print(
                    "   branch-corr mean=%.3f | branch-cosine mean=%.3f"
                    % (m["branch_correlation"]["mean_off_diag"], m["branch_cosine"]["mean_off_diag"])
                )


if __name__ == "__main__":
    main()
