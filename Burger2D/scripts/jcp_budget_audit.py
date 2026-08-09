"""Build an auditable information/update/compute ledger from existing runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {"n": int(arr.size), "mean": float(arr.mean()), "sample_std": float(arr.std(ddof=1)), "min": float(arr.min()), "max": float(arr.max())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="Burger2D/results/true_staged_vs_coadapt_20260806")
    parser.add_argument("--output-dir", default="Burger2D/results/jcp_validation_20260808")
    args = parser.parse_args()
    root, out = Path(args.results_root), Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in range(42, 52):
        for mode in ("staged", "coadapt"):
            run = root / f"seed{seed}_{mode}"
            audit = json.loads((run / "audit.json").read_text(encoding="utf-8"))
            checkpoint = __import__("torch").load(run / "pre_calibration_checkpoint.pt", map_location="cpu", weights_only=False)
            cfg = checkpoint["config"]
            records.append({"seed": seed, "mode": mode, "train_seconds": audit["train_seconds"],
                            "unique_reference_grid_points": cfg["teacher_nx"] * cfg["teacher_ny"] * cfg["teacher_nt"],
                            "base_steps": cfg["base_steps"], "updates_per_expert": cfg["expert_updates"],
                            "gate_main_updates": cfg["gate_updates"], "gate_calibration_updates": cfg["refinement_updates"],
                            "unique_expert_supervision_points": 4 * cfg["expert_sup_points"],
                            "branch_supervised_point_exposures": 4 * cfg["expert_sup_points"] * cfg["expert_updates"],
                            "pde_collocation_points_per_update": cfg["n_col"],
                            "ic_points_per_update": cfg["n_ic"], "bc_points_per_update": 4 * cfg["n_bc_per_face"]})
    config = records[0]
    ledger = {
        "shared": {k: config[k] for k in config if k not in {"seed", "mode", "train_seconds"}},
        "wall_time_seconds": {mode: _stats([r["train_seconds"] for r in records if r["mode"] == mode]) for mode in ("staged", "coadapt")},
        "interpretation": {
            "matched": ["same unique reference grid", "same initialization per paired seed", "same base steps", "same updates per expert", "same branch-level supervised point exposures", "same post-main gate-only calibration"],
            "not_identical": ["staged uses four sequential branch objectives and a static gate teacher", "coadapt uses joint mixture objectives and refreshes the moving gate teacher", "forward/backward graph structure is therefore not identical"],
        },
    }
    with (out / "budget_audit.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    with (out / "budget_audit.json").open("w", encoding="utf-8") as f:
        json.dump({"ledger": ledger, "runs": records}, f, indent=2)
    print(json.dumps(ledger, indent=2))


if __name__ == "__main__":
    main()
