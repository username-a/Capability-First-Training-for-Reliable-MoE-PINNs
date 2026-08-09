"""Audit and summarize completed equal-information 2x2 runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


GROUPS = ("P-B", "P-I", "R-B", "R-I")
METRICS = (
    "l2_relative_error",
    "worst_expert_l2",
    "routing_regret",
    "soft_routing_regret",
    "oracle_hit",
    "effective_experts",
    "min_load",
)


def _load(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _bootstrap_ci(values: np.ndarray, seed: int = 20260806) -> list[float]:
    if values.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(10000, values.size), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = parser.parse_args()
    root = Path(args.root)
    records = []
    audits = {}
    missing = []
    for seed in args.seeds:
        for group in GROUPS:
            run = root / f"seed{seed}_{group}"
            if not (run / "audit.json").exists() or not (run / "test_metrics.json").exists():
                missing.append(str(run))
                continue
            audit = _load(run / "audit.json")
            metrics = _load(run / "test_metrics.json")
            audits[(seed, group)] = audit
            records.append({"seed": seed, "group": group, **{name: metrics[name] for name in METRICS}})

    pair_checks = []
    for seed in args.seeds:
        for prefix in ("P", "R"):
            key_b, key_i = (seed, f"{prefix}-B"), (seed, f"{prefix}-I")
            if key_b not in audits or key_i not in audits:
                continue
            b, i = audits[key_b], audits[key_i]
            exact_fields = (
                "initial_hash",
                "base_hash",
                "base_batch_hash",
                "expert_batch_hashes",
                "gate_pool_coordinate_hash",
                "final_expert_hashes",
                "counters",
            )
            checks = {field: b[field] == i[field] for field in exact_fields}
            checks["test_not_loaded"] = not b["reference_access"]["test_values"] and not i["reference_access"]["test_values"]
            checks["physics_has_no_reference"] = True
            if prefix == "P":
                checks["physics_has_no_reference"] = not any(
                    b["reference_access"][name] or i["reference_access"][name]
                    for name in ("teacher_values", "reference_region_maps")
                )
            pair_checks.append({"seed": seed, "pair": prefix, "valid": all(checks.values()), "checks": checks})

    summary = {"missing": missing, "pair_checks": pair_checks, "groups": {}, "paired_differences_B_minus_I": {}}
    for group in GROUPS:
        rows = [row for row in records if row["group"] == group]
        summary["groups"][group] = {
            metric: {
                "mean": float(np.mean([row[metric] for row in rows])) if rows else float("nan"),
                "std": float(np.std([row[metric] for row in rows], ddof=1)) if len(rows) > 1 else 0.0,
                "raw": [float(row[metric]) for row in rows],
            }
            for metric in METRICS
        }
    for prefix in ("P", "R"):
        summary["paired_differences_B_minus_I"][prefix] = {}
        for metric in METRICS:
            diffs = []
            for seed in args.seeds:
                b = next((row for row in records if row["seed"] == seed and row["group"] == f"{prefix}-B"), None)
                i = next((row for row in records if row["seed"] == seed and row["group"] == f"{prefix}-I"), None)
                if b is not None and i is not None:
                    diffs.append(float(b[metric] - i[metric]))
            values = np.asarray(diffs, dtype=float)
            summary["paired_differences_B_minus_I"][prefix][metric] = {
                "mean": float(values.mean()) if values.size else float("nan"),
                "ci95": _bootstrap_ci(values),
                "raw": diffs,
            }

    root.mkdir(parents=True, exist_ok=True)
    with open(root / "confirmatory_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with open(root / "confirmatory_raw.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "group", *METRICS])
        writer.writeheader()
        writer.writerows(records)
    lines = ["# Equal-information 2x2 confirmatory summary", ""]
    lines.append(f"- completed runs: {len(records)}/{len(args.seeds) * len(GROUPS)}")
    lines.append(f"- missing runs: {len(missing)}")
    lines.append(f"- valid paired audits: {sum(row['valid'] for row in pair_checks)}/{len(pair_checks)}")
    lines.extend(["", "| group | L2 | worst expert L2 | routing regret | oracle hit |", "|---|---:|---:|---:|---:|"])
    for group in GROUPS:
        item = summary["groups"][group]
        lines.append(
            f"| {group} | {item['l2_relative_error']['mean']:.6f} | "
            f"{item['worst_expert_l2']['mean']:.6f} | {item['routing_regret']['mean']:.6e} | "
            f"{item['oracle_hit']['mean']:.4f} |"
        )
    lines.extend(["", "## Paired blocked-minus-interleaved effects", ""])
    for prefix in ("P", "R"):
        lines.append(f"- {prefix} L2: {summary['paired_differences_B_minus_I'][prefix]['l2_relative_error']}")
        lines.append(f"- {prefix} routing regret: {summary['paired_differences_B_minus_I'][prefix]['routing_regret']}")
    (root / "confirmatory_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"completed": len(records), "missing": missing, "pair_checks": pair_checks}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
