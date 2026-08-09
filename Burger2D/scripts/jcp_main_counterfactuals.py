"""Counterfactual robustness audit for staged and co-adaptive checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def _score(pred: torch.Tensor, truth: torch.Tensor) -> dict[str, float]:
    import torch

    err = pred - truth
    return {
        "relative_l2": float(torch.linalg.vector_norm(err) / torch.linalg.vector_norm(truth).clamp_min(1e-12)),
        "rmse": float(err.square().mean().sqrt()),
        "max_abs": float(err.abs().max()),
    }


def _mix(branches: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
    return (branches * gates).sum(dim=1)


def _renorm(g: torch.Tensor) -> torch.Tensor:
    return g / g.sum(dim=1, keepdim=True).clamp_min(1e-12)


def _evaluate_run(run_dir: Path, seed: int, mode: str, device: torch.device, noise_draws: int) -> list[dict]:
    import torch
    from Burger2D.equations.burgers2d import Burgers2DProblem, ReferenceSolution2D
    from Burger2D.training.staged_burgers2d import flatten_reference_solution
    from run_equal_information_2x2 import (
        DTYPE, NU, EqualInfoConfig, _batched_model_outputs, _build_model,
        _coordinate_rows,
    )

    checkpoint = torch.load(run_dir / "pre_calibration_checkpoint.pt", map_location=device, weights_only=False)
    cfg = EqualInfoConfig(**checkpoint["config"])
    model = _build_model(cfg, device)
    model.load_state_dict(checkpoint["model_state"])
    fine = Burgers2DProblem(nu=NU, device=device, dtype=DTYPE, seed=seed + 90000).generate_reference_solution(257, 257, 41)
    # Offset nested nodes exclude all spatial nodes of the 65x65 training grid,
    # while retaining the fine-grid reference values without interpolation.
    reference = ReferenceSolution2D(
        x=fine.x[1::2], y=fine.y[1::2], t=fine.t[::2],
        u=fine.u[::2, 1::2, 1::2],
    )
    coords, values = flatten_reference_solution(reference, device=device, dtype=DTYPE)
    train_rows = _coordinate_rows(np.load(run_dir / "training_pool_coords.npy"))
    coords_np = np.round(coords.detach().cpu().numpy(), 7)
    keep = torch.tensor([tuple(row) not in train_rows for row in coords_np], dtype=torch.bool)
    _, branches, gates = _batched_model_outputs(model, coords)
    truth = values.detach().cpu().reshape(-1)[keep]
    branches, gates = branches[keep], gates[keep]
    original = _score(_mix(branches, gates), truth)
    rows: list[dict] = []

    def add(name: str, pred: torch.Tensor) -> dict:
        metrics = _score(pred, truth)
        row = {"seed": seed, "mode": mode, "scenario": name, **metrics,
               "l2_ratio_vs_soft": metrics["relative_l2"] / original["relative_l2"]}
        rows.append(row)
        return row

    soft_row = add("soft", _mix(branches, gates))
    sqerr = (branches - truth[:, None]).square()
    denom = torch.linalg.vector_norm(truth).clamp_min(1e-12)
    expert_l2 = [float(torch.linalg.vector_norm(branches[:, k] - truth) / denom)
                 for k in range(branches.shape[1])]
    weighted_mse = float((gates * sqerr).sum(dim=1).mean())
    mixture_mse = float((_mix(branches, gates) - truth).square().mean())
    oracle_pred = branches[torch.arange(branches.shape[0]), sqerr.argmin(dim=1)]
    mean_load = gates.mean(dim=0)
    soft_row.update({
        **{f"expert_{k}_l2": value for k, value in enumerate(expert_l2)},
        "worst_expert_l2": max(expert_l2),
        "oracle_l2": _score(oracle_pred, truth)["relative_l2"],
        "aggregation_gain": 1.0 - mixture_mse / max(weighted_mse, 1e-14),
        "soft_routing_regret": float(((gates * sqerr).sum(dim=1) - sqerr.min(dim=1).values).mean()),
        "effective_experts": float(torch.exp(-(mean_load.clamp_min(1e-12) * mean_load.clamp_min(1e-12).log()).sum())),
        **{f"soft_load_{k}": float(mean_load[k]) for k in range(branches.shape[1])},
    })
    top = gates.argmax(dim=1)
    add("top1_hard", branches[torch.arange(branches.shape[0]), top])
    for temp in (0.5, 2.0):
        adjusted = _renorm(gates.clamp_min(1e-12).pow(1.0 / temp))
        add(f"temperature_{temp:g}", _mix(branches, adjusted))

    soft_load = gates.mean(dim=0)
    most_loaded = int(soft_load.argmax())
    deletion_scores = []
    for k in range(branches.shape[1]):
        adjusted = gates.clone(); adjusted[:, k] = 0.0; adjusted = _renorm(adjusted)
        score = _score(_mix(branches, adjusted), truth)
        deletion_scores.append((score["relative_l2"], k, adjusted))
        if k == most_loaded:
            add("delete_most_loaded", _mix(branches, adjusted))
    _, worst_k, worst_g = max(deletion_scores, key=lambda x: x[0])
    add("delete_worst_case", _mix(branches, worst_g))
    rows[-1]["deleted_expert"] = int(worst_k)

    logg = gates.clamp_min(1e-12).log()
    for sigma in (0.05, 0.10):
        preds = []
        gen = torch.Generator(device="cpu"); gen.manual_seed(100000 + seed + int(sigma * 1000))
        for _ in range(noise_draws):
            noise = torch.randn(logg.shape, generator=gen) * sigma
            perturbed = torch.softmax(logg + noise, dim=1)
            preds.append(_mix(branches, perturbed))
        stacked = torch.stack(preds)
        metrics = [_score(p, truth) for p in stacked]
        mean_l2 = float(np.mean([m["relative_l2"] for m in metrics]))
        rows.append({"seed": seed, "mode": mode, "scenario": f"logit_noise_{sigma:.2f}",
                     "relative_l2": mean_l2,
                     "rmse": float(np.mean([m["rmse"] for m in metrics])),
                     "max_abs": float(np.mean([m["max_abs"] for m in metrics])),
                     "l2_ratio_vs_soft": mean_l2 / original["relative_l2"]})
    return rows


def _summary(rows: list[dict]) -> dict:
    out: dict = {}
    for mode in ("staged", "coadapt"):
        out[mode] = {}
        scenarios = sorted({r["scenario"] for r in rows if r["mode"] == mode})
        for scenario in scenarios:
            selected = [r for r in rows if r["mode"] == mode and r["scenario"] == scenario]
            out[mode][scenario] = {}
            for key in ("relative_l2", "rmse", "max_abs", "l2_ratio_vs_soft"):
                vals = np.asarray([r[key] for r in selected], dtype=float)
                out[mode][scenario][key] = {"mean": float(vals.mean()), "sample_std": float(vals.std(ddof=1)), "raw": vals.tolist()}
            if scenario == "soft":
                extra = [k for k in selected[0] if k not in {"seed", "mode", "scenario", "relative_l2", "rmse", "max_abs", "l2_ratio_vs_soft"}]
                for key in extra:
                    vals = np.asarray([r[key] for r in selected], dtype=float)
                    out[mode][scenario][key] = {"mean": float(vals.mean()), "sample_std": float(vals.std(ddof=1)), "raw": vals.tolist()}
    return out


def _plot(summary: dict, path: Path) -> None:
    scenarios = ["top1_hard", "temperature_2", "delete_most_loaded"]
    labels = ["Top-1", "T=2", "Delete top-load"]
    x = np.arange(len(scenarios)); width = 0.36
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 7.2,
        "axes.titlesize": 7.2,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.7,
        "ytick.labelsize": 6.7,
        "legend.fontsize": 6.7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.spines.right": False,
        "axes.spines.top": False,
    })
    fig, ax = plt.subplots(figsize=(5.0, 2.55), constrained_layout=True)
    for offset, mode, color, hatch, label in (
        (-width/2, "staged", "#2F6B8A", "", "Staged"),
        (width/2, "coadapt", "#D07A5F", "///", "Co-adaptation"),
    ):
        means = [summary[mode][s]["l2_ratio_vs_soft"]["mean"] for s in scenarios]
        stds = [summary[mode][s]["l2_ratio_vs_soft"]["sample_std"] for s in scenarios]
        ax.bar(x + offset, means, width, yerr=stds, capsize=2.5, label=label,
               color=color, hatch=hatch, edgecolor="#4D4D4D", linewidth=0.45,
               error_kw={"elinewidth": 0.8, "capthick": 0.8})
    ax.axhline(1.0, color="#333333", lw=0.8, ls="--")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Counterfactual / soft-mixture relative L2")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.grid(axis="y", alpha=0.55, color="#D9DEE3", linewidth=0.5)
    base = path.with_suffix("")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="Burger2D/results/true_staged_vs_coadapt_20260806")
    parser.add_argument("--output-dir", default="Burger2D/results/jcp_validation_20260808")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--noise-draws", type=int, default=10)
    args = parser.parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    root, out = Path(args.results_root), Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for seed in range(42, 52):
        for mode in ("staged", "coadapt"):
            rows.extend(_evaluate_run(root / f"seed{seed}_{mode}", seed, mode, device, args.noise_draws))
    summary = _summary(rows)
    with (out / "main_counterfactuals.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader(); writer.writerows(rows)
    with (out / "main_counterfactuals.json").open("w", encoding="utf-8") as f:
        json.dump({"protocol": {"checkpoint": "pre_calibration", "reference_grid": [257, 257, 41],
                                 "evaluation_subset": "t[::2], y[1::2], x[1::2]",
                                 "evaluation_points": 21 * 128 * 128,
                                 "noise_draws": args.noise_draws}, "runs": rows, "summary": summary}, f, indent=2)
    _plot(summary, out / "main_counterfactuals.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
