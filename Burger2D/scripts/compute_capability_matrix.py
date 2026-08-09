"""Compute ex-ante capability-region error matrices from saved checkpoints.

This is an evaluation-only audit.  The four regions are defined by the same
reference-derived capability scores used to construct the specialist pools,
but are evaluated on the disjoint test grid.  No gate output is used to define
the regions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from run_equal_information_2x2 import (
    DTYPE,
    NU,
    EqualInfoConfig,
    _batched_model_outputs,
    _build_model,
    _coordinate_rows,
)
from Burger2D.equations.burgers2d import Burgers2DProblem
from Burger2D.training.staged_burgers2d import compute_region_scores, flatten_reference_solution


def evaluate_run(
    run_dir: Path,
    device: torch.device,
    *,
    checkpoint_name: str,
    test_nx: int,
    test_ny: int,
    test_nt: int,
) -> dict:
    checkpoint = torch.load(run_dir / checkpoint_name, map_location=device, weights_only=False)
    cfg = EqualInfoConfig(**checkpoint["config"])
    model = _build_model(cfg, device)
    model.load_state_dict(checkpoint["model_state"])

    problem = Burgers2DProblem(nu=NU, device=device, dtype=DTYPE, seed=cfg.seed + 90000)
    reference = problem.generate_reference_solution(test_nx, test_ny, test_nt)
    test_coords, test_values = flatten_reference_solution(reference, device=device, dtype=DTYPE)

    training_coords = np.load(run_dir / "training_pool_coords.npy")
    train_rows = _coordinate_rows(training_coords)
    test_np = test_coords.detach().cpu().numpy()
    overlap = np.array([tuple(row) in train_rows for row in np.round(test_np, 7)], dtype=bool)
    keep_np = ~overlap
    keep = torch.tensor(keep_np, dtype=torch.bool)
    if not bool(keep.any()):
        raise RuntimeError(f"No disjoint test points remain for {run_dir}")

    _, branches, _ = _batched_model_outputs(model, test_coords)
    truth = test_values.detach().cpu().reshape(-1)
    branches = branches[keep]
    truth = truth[keep]

    scores = compute_region_scores(reference, layout_variant=cfg.expert_layout_variant)
    region_names = list(model.expert_names)
    score_stack = np.stack([scores[name].reshape(-1) for name in region_names], axis=1)
    labels = torch.from_numpy(score_stack.argmax(axis=1)[keep_np]).long()

    sq = (branches - truth[:, None]).square()
    abs_err = (branches - truth[:, None]).abs()
    rows = []
    counts = []
    maes = []
    winner_rates = []
    pointwise_best = sq.argmin(dim=1)
    for region_idx in range(len(region_names)):
        mask = labels == region_idx
        count = int(mask.sum().item())
        if count == 0:
            raise RuntimeError(f"Empty capability region {region_names[region_idx]} in {run_dir}")
        denom = torch.linalg.vector_norm(truth[mask]).clamp_min(1e-10)
        rows.append([
            float(torch.linalg.vector_norm(branches[mask, k] - truth[mask]) / denom)
            for k in range(branches.shape[1])
        ])
        maes.append([float(abs_err[mask, k].mean().item()) for k in range(branches.shape[1])])
        winner_rates.append([
            float((pointwise_best[mask] == k).float().mean().item())
            for k in range(branches.shape[1])
        ])
        counts.append(count)

    return {
        "seed": cfg.seed,
        "mode": checkpoint["mode"],
        "region_names": region_names,
        "expert_names": list(model.expert_names),
        "region_definition": "argmax of ex-ante capability scores; independent of learned gate",
        "region_counts": counts,
        "region_fractions": [count / int(keep.sum().item()) for count in counts],
        "local_relative_l2": rows,
        "local_mae": maes,
        "pointwise_best_expert_rate": winner_rates,
        "test_grid": [test_nx, test_ny, test_nt],
        "test_disjoint_points": int(keep.sum().item()),
        "test_overlap_excluded": int(overlap.sum()),
    }


def aggregate(records: list[dict]) -> dict:
    result = {"runs": records, "groups": {}}
    for mode in ("staged", "coadapt"):
        group = [record for record in records if record["mode"] == mode]
        if not group:
            continue
        result["groups"][mode] = {
            "n_seeds": len(group),
            "seeds": [record["seed"] for record in group],
            "region_names": group[0]["region_names"],
            "expert_names": group[0]["expert_names"],
            "region_fraction_mean": np.asarray(
                [record["region_fractions"] for record in group], dtype=np.float64
            ).mean(axis=0).tolist(),
        }
        for key in ("local_relative_l2", "local_mae", "pointwise_best_expert_rate"):
            values = np.asarray([record[key] for record in group], dtype=np.float64)
            result["groups"][mode][f"{key}_mean"] = values.mean(axis=0).tolist()
            result["groups"][mode][f"{key}_std"] = values.std(axis=0, ddof=1).tolist()
    return result


def markdown_table(summary: dict) -> str:
    lines = [
        "# Burger2D capability-region 4x4 audit",
        "",
        "Regions are fixed by the argmax of the four pre-defined capability scores on the",
        "disjoint test grid; the learned gate is not used to define them. Values are local",
        "relative L2 errors (mean +/- sample std over seeds).",
        "",
    ]
    for mode, group in summary["groups"].items():
        names = group["expert_names"]
        lines.extend([
            f"## {mode} (n={group['n_seeds']})",
            "",
            "| capability region | " + " | ".join(names) + " |",
            "|---|" + "---:|" * len(names),
        ])
        mean = np.asarray(group["local_relative_l2_mean"])
        std = np.asarray(group["local_relative_l2_std"])
        for i, region in enumerate(group["region_names"]):
            cells = [f"{mean[i, j]:.3f} +/- {std[i, j]:.3f}" for j in range(len(names))]
            lines.append("| " + region + " | " + " | ".join(cells) + " |")
        fractions = ", ".join(
            f"{name}={100.0 * frac:.1f}%"
            for name, frac in zip(group["region_names"], group["region_fraction_mean"])
        )
        lines.extend(["", f"Region fractions: {fractions}.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("Burger2D/results/true_staged_vs_coadapt_20260806"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-name", default="train_checkpoint.pt")
    parser.add_argument("--test-nx", type=int, default=82)
    parser.add_argument("--test-ny", type=int, default=83)
    parser.add_argument("--test-nt", type=int, default=32)
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    run_dirs = sorted(path for path in args.root.glob("seed*_*" ) if path.is_dir())
    records = []
    for index, run_dir in enumerate(run_dirs, start=1):
        print(f"[{index}/{len(run_dirs)}] {run_dir.name}", flush=True)
        records.append(evaluate_run(
            run_dir,
            device,
            checkpoint_name=args.checkpoint_name,
            test_nx=args.test_nx,
            test_ny=args.test_ny,
            test_nt=args.test_nt,
        ))
    summary = aggregate(records)
    (args.root / "capability_matrix_4x4.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.root / "capability_matrix_4x4.md").write_text(markdown_table(summary), encoding="utf-8")
    print(markdown_table(summary))


if __name__ == "__main__":
    main()
