"""Independent-grid audit and publication figures for staged vs co-adaptation.

The script evaluates the pre-calibration checkpoints on an 82 x 83 x 32 grid,
removes coordinates present in each run's reference-training pool, and exports
both machine-readable source data and publication-ready figures.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Burger2D.equations.burgers2d import Burgers2DProblem  # noqa: E402
from Burger2D.scripts.run_equal_information_2x2 import (  # noqa: E402
    DTYPE,
    NU,
    EqualInfoConfig,
    _build_model,
    _coordinate_rows,
)
from Burger2D.training.staged_burgers2d import flatten_reference_solution  # noqa: E402


ROOT = PACKAGE_ROOT / "results" / "true_staged_vs_coadapt_20260806"
FIGURE_DIR = PROJECT_ROOT / "docs" / "paper1_figures"
SEEDS = list(range(42, 52))
MODES = ("staged", "coadapt")
TEST_GRID = (82, 83, 32)
CHECKPOINT = "pre_calibration_checkpoint.pt"
# Match the formal evaluator exactly.  The local-context gate is evaluated in
# lexicographic grid order with this fixed chunk size for every method/seed.
BATCH_SIZE = 16384
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STAGED_COLOR = "#3775BA"
COADAPT_COLOR = "#B64342"
PAIR_COLOR = "#B8B8B8"
TEXT_COLOR = "#272727"


# Publication contract: editable SVG text, TrueType PDF text, >= 7 pt glyphs.
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
mpl.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42})
plt.rcParams["font.size"] = 7.5
plt.rcParams["axes.titlesize"] = 8.5
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["legend.frameon"] = False


def _load_model(checkpoint_path: Path) -> tuple[torch.nn.Module, EqualInfoConfig]:
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    cfg = EqualInfoConfig(**checkpoint["config"])
    model = _build_model(cfg, DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, cfg


def _evaluate_run(
    run_dir: Path,
    test_coords: torch.Tensor,
    test_truth: torch.Tensor,
    test_coords_np: np.ndarray,
) -> dict[str, Any]:
    model, cfg = _load_model(run_dir / CHECKPOINT)
    training_coords = np.load(run_dir / "training_pool_coords.npy")
    train_rows = _coordinate_rows(training_coords)
    overlap = np.fromiter(
        (tuple(row) in train_rows for row in np.round(test_coords_np, 7)),
        dtype=bool,
        count=test_coords_np.shape[0],
    )
    keep_np = ~overlap
    keep = torch.as_tensor(keep_np, device=DEVICE, dtype=torch.bool)
    coords = test_coords[keep]
    truth = test_truth[keep].reshape(-1)

    n_experts = len(model.expert_names)
    mix_sse = 0.0
    truth_sse = 0.0
    expert_sse = np.zeros(n_experts, dtype=np.float64)
    weighted_sse = 0.0
    soft_regret_sum = 0.0
    identity_max_abs = 0.0
    n_points = int(coords.shape[0])

    with torch.no_grad():
        for start in range(0, n_points, BATCH_SIZE):
            x = coords[start : start + BATCH_SIZE]
            y = truth[start : start + BATCH_SIZE]
            branches = model.get_expert_predictions(x).squeeze(-1)
            gates = model.compute_gate_weights(x, expert_preds=branches.unsqueeze(-1))
            pred = model(x).squeeze(-1)
            composed = (gates * branches).sum(dim=1)

            sq = (branches - y[:, None]).square()
            mix_sq = (pred - y).square()
            weighted_sq = (gates * sq).sum(dim=1)
            best_sq = sq.min(dim=1).values

            mix_sse += float(mix_sq.double().sum().item())
            truth_sse += float(y.square().double().sum().item())
            expert_sse += sq.double().sum(dim=0).cpu().numpy()
            weighted_sse += float(weighted_sq.double().sum().item())
            soft_regret_sum += float((weighted_sq - best_sq).double().sum().item())
            identity_max_abs = max(
                identity_max_abs,
                float((pred - composed).abs().max().item()),
            )

    mixture_mse = mix_sse / n_points
    weighted_expert_mse = weighted_sse / n_points
    aggregation_gain = (weighted_expert_mse - mixture_mse) / max(weighted_expert_mse, 1e-15)
    expert_l2 = np.sqrt(expert_sse / max(truth_sse, 1e-15))
    return {
        "seed": int(cfg.seed),
        "mode": str(run_dir.name.split("_", 1)[1]),
        "checkpoint": CHECKPOINT,
        "test_grid": list(TEST_GRID),
        "test_total_points": int(test_coords.shape[0]),
        "test_overlap_excluded": int(overlap.sum()),
        "test_disjoint_points": n_points,
        "mixture_l2": float(math.sqrt(mix_sse / max(truth_sse, 1e-15))),
        "per_expert_l2": [float(x) for x in expert_l2],
        "worst_expert_l2": float(expert_l2.max()),
        "mixture_mse": float(mixture_mse),
        "weighted_expert_mse": float(weighted_expert_mse),
        "aggregation_gain_ratio": float(aggregation_gain),
        "soft_routing_regret": float(soft_regret_sum / n_points),
        "mixture_identity_max_abs": float(identity_max_abs),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    metrics = (
        "mixture_l2",
        "worst_expert_l2",
        "soft_routing_regret",
        "mixture_mse",
        "weighted_expert_mse",
        "aggregation_gain_ratio",
        "mixture_identity_max_abs",
    )
    for mode in MODES:
        mode_rows = [row for row in rows if row["mode"] == mode]
        out[mode] = {"n_seeds": len(mode_rows), "metrics": {}}
        for metric in metrics:
            values = np.asarray([row[metric] for row in mode_rows], dtype=float)
            out[mode]["metrics"][metric] = {
                "mean": float(values.mean()),
                "sample_std": float(values.std(ddof=1)),
                "raw": [float(x) for x in values],
            }
    return out


def _write_source_data(rows: list[dict[str, Any]]) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = ROOT / "independent_grid_aggregation_source_data.csv"
    fields = [
        "seed",
        "mode",
        "mixture_l2",
        "worst_expert_l2",
        "soft_routing_regret",
        "mixture_mse",
        "weighted_expert_mse",
        "aggregation_gain_ratio",
        "mixture_identity_max_abs",
        "test_total_points",
        "test_overlap_excluded",
        "test_disjoint_points",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})

    payload = {
        "protocol": {
            "checkpoint": CHECKPOINT,
            "test_grid": list(TEST_GRID),
            "overlap_rule": "exclude coordinates present in each run's 65x65x21 reference-training pool after rounding to 7 decimals",
            "aggregation_gain_definition": "(gate-weighted expert MSE - mixture MSE) / gate-weighted expert MSE",
            "device": str(DEVICE),
        },
        "runs": rows,
        "groups": _aggregate(rows),
    }
    (ROOT / "independent_grid_aggregation_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _mean_ci95(values: np.ndarray) -> tuple[float, float]:
    mean = float(values.mean())
    # Student-t critical value for 9 degrees of freedom (n=10 paired seeds).
    half = 2.262157 * float(values.std(ddof=1)) / math.sqrt(values.size)
    return mean, half


def _paired_panel(
    ax: plt.Axes,
    staged: np.ndarray,
    coadapt: np.ndarray,
    ylabel: str,
    title: str,
    panel: str,
    *,
    log_scale: bool = False,
    ylim: tuple[float, float] | None = None,
) -> None:
    for left, right in zip(staged, coadapt):
        ax.plot([0, 1], [left, right], color=PAIR_COLOR, lw=0.8, alpha=0.72, zorder=1)
    ax.scatter(np.zeros_like(staged), staged, s=21, color=STAGED_COLOR, edgecolor="white", linewidth=0.45, zorder=3)
    ax.scatter(np.ones_like(coadapt), coadapt, s=21, color=COADAPT_COLOR, edgecolor="white", linewidth=0.45, zorder=3)
    for xpos, values, color in ((0, staged, STAGED_COLOR), (1, coadapt, COADAPT_COLOR)):
        mean, ci = _mean_ci95(values)
        ax.errorbar(
            xpos,
            mean,
            yerr=ci,
            fmt="D",
            ms=4.2,
            mfc="white",
            mec=color,
            mew=1.0,
            ecolor=color,
            elinewidth=1.2,
            capsize=3,
            capthick=1.2,
            zorder=4,
        )
    ax.set_xticks([0, 1], ["Staged", "Co-adaptation"])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=4)
    ax.text(-0.16, 1.05, panel, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
    if log_scale:
        if np.any(staged <= 0) or np.any(coadapt <= 0):
            raise ValueError(f"{title} contains non-positive values and cannot use a log axis.")
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#E5E5E5", lw=0.6, zorder=0)


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def _plot_paired_audit(rows: list[dict[str, Any]]) -> None:
    staged_rows = sorted((r for r in rows if r["mode"] == "staged"), key=lambda r: r["seed"])
    coadapt_rows = sorted((r for r in rows if r["mode"] == "coadapt"), key=lambda r: r["seed"])
    if [r["seed"] for r in staged_rows] != [r["seed"] for r in coadapt_rows]:
        raise RuntimeError("The staged and co-adaptation seed sets are not paired.")

    def values(metric: str, group: list[dict[str, Any]]) -> np.ndarray:
        return np.asarray([row[metric] for row in group], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.25))
    _paired_panel(
        axes[0, 0], values("mixture_l2", staged_rows), values("mixture_l2", coadapt_rows),
        "Relative L2", "Mixture error", "a",
    )
    _paired_panel(
        axes[0, 1], values("worst_expert_l2", staged_rows), values("worst_expert_l2", coadapt_rows),
        "Worst-expert relative L2", "Expert health", "b", log_scale=True,
    )
    _paired_panel(
        axes[1, 0], values("soft_routing_regret", staged_rows), values("soft_routing_regret", coadapt_rows),
        "Soft routing regret", "Routing cost", "c", log_scale=True,
    )
    _paired_panel(
        axes[1, 1], values("aggregation_gain_ratio", staged_rows), values("aggregation_gain_ratio", coadapt_rows),
        "Soft-aggregation gain Γagg", "Dependence on soft aggregation", "d", ylim=(0, 1.03),
    )
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.10, top=0.96, wspace=0.34, hspace=0.42)
    _save_figure(fig, FIGURE_DIR / "coadapt_independent_grid")


def _annotation_color(cmap: mpl.colors.Colormap, norm: mpl.colors.Normalize, value: float) -> str:
    r, g, b, _ = cmap(norm(value))
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 0.48 else TEXT_COLOR


def _plot_capability_heatmap() -> None:
    source = json.loads((ROOT / "capability_matrix_4x4.json").read_text(encoding="utf-8"))
    groups = source["groups"]
    matrices = [np.asarray(groups[m]["local_relative_l2_mean"], dtype=float) for m in MODES]
    spreads = [np.asarray(groups[m]["local_relative_l2_std"], dtype=float) for m in MODES]
    labels = ["Smooth", "Iso. shock", "Directional shock", "Wave"]
    positive = np.concatenate([m.ravel() for m in matrices])
    norm = LogNorm(vmin=float(positive.min()), vmax=float(positive.max()))
    cmap = mpl.colormaps["viridis"]

    fig = plt.figure(figsize=(7.15, 3.20))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.055], wspace=0.28)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    cax = fig.add_subplot(gs[0, 2])
    image = None
    for idx, (ax, matrix, spread, title) in enumerate(
        zip(axes, matrices, spreads, ("Staged", "Co-adaptation"))
    ):
        image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="equal")
        ax.set_xticks(range(4), labels, rotation=32, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(4), labels if idx == 0 else [])
        ax.set_xlabel("Expert evaluated independently")
        if idx == 0:
            ax.set_ylabel("Predefined capability region")
        ax.set_title(title, pad=5)
        ax.text(-0.18, 1.04, "ab"[idx], transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
        for i in range(4):
            for j in range(4):
                color = _annotation_color(cmap, norm, float(matrix[i, j]))
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.2f}\n±{spread[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=5.9,
                    linespacing=1.05,
                )
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
    assert image is not None
    cb = fig.colorbar(image, cax=cax)
    cb.set_label("Local relative L2 (log scale)", labelpad=5)
    cb.outline.set_linewidth(0.6)
    fig.subplots_adjust(left=0.13, right=0.93, bottom=0.22, top=0.92)
    _save_figure(fig, FIGURE_DIR / "capability_matrix_heatmap")

    csv_path = ROOT / "capability_matrix_source_data.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", "capability_region", "expert", "mean_local_relative_l2", "sample_std", "n_seeds"])
        for mode, matrix, spread in zip(MODES, matrices, spreads):
            for i, region in enumerate(labels):
                for j, expert in enumerate(labels):
                    writer.writerow([mode, region, expert, matrix[i, j], spread[i, j], groups[mode]["n_seeds"]])


def main() -> None:
    print(f"device={DEVICE}", flush=True)
    problem = Burgers2DProblem(nu=NU, device=DEVICE, dtype=DTYPE, seed=90042)
    reference = problem.generate_reference_solution(*TEST_GRID)
    test_coords, test_truth = flatten_reference_solution(reference, device=DEVICE, dtype=DTYPE)
    test_coords_np = test_coords.detach().cpu().numpy()

    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for mode in MODES:
            run_dir = ROOT / f"seed{seed}_{mode}"
            row = _evaluate_run(run_dir, test_coords, test_truth, test_coords_np)
            rows.append(row)
            print(
                f"seed={seed} mode={mode} mixL2={row['mixture_l2']:.6f} "
                f"worstL2={row['worst_expert_l2']:.6f} "
                f"Gamma={row['aggregation_gain_ratio']:.6f} "
                f"identity={row['mixture_identity_max_abs']:.3e}",
                flush=True,
            )

    _write_source_data(rows)
    _plot_paired_audit(rows)
    _plot_capability_heatmap()
    groups = _aggregate(rows)
    for mode in MODES:
        g = groups[mode]["metrics"]
        print(
            f"{mode}: Gamma={g['aggregation_gain_ratio']['mean']:.6f} "
            f"+/- {g['aggregation_gain_ratio']['sample_std']:.6f}; "
            f"mixL2={g['mixture_l2']['mean']:.6f}; "
            f"worstL2={g['worst_expert_l2']['mean']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
