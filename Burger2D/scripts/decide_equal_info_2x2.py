"""Return success only for a strong, directionally consistent 2x2 result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    audits_ok = bool(data["pair_checks"]) and all(row["valid"] for row in data["pair_checks"])
    r_l2 = data["paired_differences_B_minus_I"]["R"]["l2_relative_error"]["raw"]
    r_regret = data["paired_differences_B_minus_I"]["R"]["soft_routing_regret"]["raw"]
    # B-I < 0 means blocked is better.  "Strong" requires at least four of five
    # paired seeds in the predicted direction for both primary routing outcomes.
    direction_ok = len(r_l2) >= 5 and sum(value < 0 for value in r_l2) >= 4
    regret_ok = len(r_regret) >= 5 and sum(value < 0 for value in r_regret) >= 4
    group = data["groups"]
    rb = group["R-B"]["l2_relative_error"]["mean"]
    ri = group["R-I"]["l2_relative_error"]["mean"]
    practical = ri > 0 and (ri - rb) / ri >= 0.05
    perfect = audits_ok and direction_ok and regret_ok and practical and not data["missing"]
    decision = {
        "perfect": perfect,
        "audits_ok": audits_ok,
        "r_l2_blocked_better_seeds": sum(value < 0 for value in r_l2),
        "r_regret_blocked_better_seeds": sum(value < 0 for value in r_regret),
        "relative_l2_improvement": (ri - rb) / ri if ri else None,
        "action": "probe_expert" if perfect else "extend_2x2_then_probe",
    }
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    raise SystemExit(0 if perfect else 1)


if __name__ == "__main__":
    main()
