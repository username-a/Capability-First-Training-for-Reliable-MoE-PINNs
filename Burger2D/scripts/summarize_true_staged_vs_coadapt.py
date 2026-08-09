from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


METRICS = ["l2_relative_error", "soft_routing_regret", "worst_expert_l2", "effective_experts", "min_load"]


def stats(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0, "raw": a.tolist()}


def paired(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    mean = float(a.mean())
    if len(a) > 1:
        # Normal interval is used for monitoring; the raw paired values are
        # retained for the final bootstrap/permutation analysis.
        half = 1.96 * float(a.std(ddof=1)) / math.sqrt(len(a))
    else:
        half = float("nan")
    return {"mean": mean, "ci95_normal": [mean - half, mean + half], "raw": a.tolist()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    out: dict = {"missing": [], "audits": [], "stages": {}}
    cache: dict[tuple[int, str, str], dict] = {}
    for seed in args.seeds:
        for mode in ("staged", "coadapt"):
            run = root / f"seed{seed}_{mode}"
            audit_path = run / "audit.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {"valid": False}
            out["audits"].append({"seed": seed, "mode": mode, "valid": bool(audit.get("valid", False))})
            for stage, filename in (("pre", "pre_metrics.json"), ("post", "test_metrics.json")):
                path = run / filename
                if not path.exists():
                    out["missing"].append(str(path))
                else:
                    cache[(seed, mode, stage)] = json.loads(path.read_text(encoding="utf-8"))
    for stage in ("pre", "post"):
        stage_out = {"groups": {}, "paired_staged_minus_coadapt": {}}
        for mode in ("staged", "coadapt"):
            stage_out["groups"][mode] = {
                metric: stats([cache[(s, mode, stage)][metric] for s in args.seeds if (s, mode, stage) in cache])
                for metric in METRICS
            }
        for metric in METRICS:
            diffs = [
                cache[(s, "staged", stage)][metric] - cache[(s, "coadapt", stage)][metric]
                for s in args.seeds
                if (s, "staged", stage) in cache and (s, "coadapt", stage) in cache
            ]
            stage_out["paired_staged_minus_coadapt"][metric] = paired(diffs)
        out["stages"][stage] = stage_out
    (root / "confirmatory_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
