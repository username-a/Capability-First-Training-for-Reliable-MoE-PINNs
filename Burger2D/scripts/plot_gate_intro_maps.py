"""
Generate the mechanism figures from the gate-introduction ablation checkpoints.

Figure A (gate routing map + responsibility evolution):
    five argmax-gate maps (f = 0/25/50/75/100%) at two time slices.
Figure B (expert specialization error maps):
    per-expert |u_i - u_ref| maps for f=100 (healthy) vs f=0 (degenerate).
Figure C (expert maturity):
    per-expert L2 during Stage A, gate entropy during Stage B, and
    expert load fractions during Stage C, from the f=100 staged run.

Outputs: docs/paper1_figures/gate_intro_*.png (English labels)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Burger2D.core.moe_pinn import build_burgers2d_moe  # noqa: E402


RESULTS = os.path.join(PACKAGE_ROOT, "results")
RUN_ROOT = os.path.join(RESULTS, "gate_intro_ablation_20260803_122825")
FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]
EXPERT_NAMES = ["smooth", "iso_shock", "directional_shock", "wave"]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

GATE_VARIANTS = ["pointwise", "local_conv", "local_knn"]
DIRECTIONAL_VARIANTS = ["hybrid", "legacy"]
WAVE_VARIANTS = ["base", "mixed_lite", "mixed"]


def load_model(pt_path: str) -> torch.nn.Module | None:
    for gate in GATE_VARIANTS:
        for directional in DIRECTIONAL_VARIANTS:
            for wave in WAVE_VARIANTS:
                model = build_burgers2d_moe(
                    directional_expert_variant=directional,
                    wave_expert_variant=wave,
                    expert_layout_variant="categorical",
                    attribute_expert_variant="base",
                    gate_variant=gate,
                    rotation_variant="none",
                )
                try:
                    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
                    model.load_state_dict(ckpt["model_state"])
                    model.eval()
                    return model
                except (RuntimeError, KeyError):
                    continue
    return None


def load_run(fraction: float, seed: int = 42) -> tuple[torch.nn.Module, dict, np.ndarray]:
    tag = f"f{int(fraction * 100)}"
    run_dir = os.path.join(RUN_ROOT, f"{tag}_seed{seed}")
    model = load_model(os.path.join(run_dir, "burgers2d_moe_staged", "burgers2d_moe_staged.pt"))
    if model is None:
        raise RuntimeError(f"could not load model for {tag} seed {seed}")
    data = np.load(os.path.join(run_dir, "burgers2d_moe_staged", "reference_and_prediction.npz"))
    return model, data, data["u_ref"]


def slice_predictions(
    model: torch.nn.Module,
    data: dict,
    t_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (argmax gate map, per-expert error map, reference slice)."""
    x = data["x"]
    y = data["y"]
    t = data["t"]
    xx, yy = np.meshgrid(x, y, indexing="xy")
    coords = np.stack(
        [xx.reshape(-1), yy.reshape(-1), np.full_like(xx, t[t_index]).reshape(-1)],
        axis=1,
    )
    coords_t = torch.tensor(coords, dtype=torch.float32)
    with torch.no_grad():
        gates = model.get_gate_weights(coords_t).cpu().numpy()  # (N, K)
        branch = model.get_expert_predictions(coords_t).cpu().numpy()[:, :, 0]  # (N, K)
    u_slice = data["u_ref"][t_index].reshape(-1)  # (ny*nx,)
    err = np.abs(branch - u_slice[:, None])  # (N, K)
    argmax = gates.argmax(axis=1)
    ny, nx = xx.shape
    return (
        argmax.reshape(ny, nx),
        err.reshape(ny, nx, len(EXPERT_NAMES)),
        data["u_ref"][t_index],
    )


def plot_figure_a() -> None:
    """Gate routing maps at two time slices for the five fractions."""
    t_indices = [5, 9]  # middle and late time (nt=11 in smoke? no, full nt=31)
    models_data = {}
    for f in FRACTIONS:
        model, data, _ = load_run(f)
        models_data[f] = (model, data)

    # nt depends on the run grid (81x81x31 -> nt=31)
    nt = models_data[1.0][1]["t"].shape[0]
    t_indices = [nt // 2, int(nt * 0.9)]
    t_labels = [f"t={models_data[1.0][1]['t'][i]:.2f}" for i in t_indices]

    fig, axes = plt.subplots(2, len(FRACTIONS), figsize=(16.5, 6.6))
    for row, ti in enumerate(t_indices):
        for col, f in enumerate(FRACTIONS):
            model, data = models_data[f]
            argmax, _, _ = slice_predictions(model, data, ti)
            ax = axes[row, col]
            im = ax.imshow(
                argmax,
                origin="lower",
                extent=[data["x"].min(), data["x"].max(), data["y"].min(), data["y"].max()],
                cmap="tab10",
                vmin=-0.5,
                vmax=3.5,
                aspect="auto",
            )
            ax.set_title(f"f={int(f * 100)}%" if row == 0 else "", fontsize=12)
            if row == 1:
                ax.set_xlabel("x")
            if col == 0:
                ax.set_ylabel(f"{t_labels[row]}\ny")
    cbar = fig.colorbar(im, ax=axes, ticks=[0, 1, 2, 3], shrink=0.8, pad=0.01)
    cbar.ax.set_yticklabels(EXPERT_NAMES)
    fig.suptitle(
        "Gate routing maps (argmax expert) vs gate introduction point f; "
        "rows: two time slices",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(FIG_DIR, "gate_intro_routing_evolution.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("[OK]", out)


def plot_figure_b() -> None:
    """Per-expert error maps for f=100 (healthy) vs f=0 (degenerate)."""
    fig, axes = plt.subplots(2, 4, figsize=(16.5, 7.0))
    for row, (f, row_title) in enumerate(
        [(1.0, "f=100% (healthy, experts frozen)"), (0.0, "f=0% (end-to-end / degenerate)")]
    ):
        model, data, u_ref = load_run(f)
        nt = data["t"].shape[0]
        ti = nt // 2
        _, err, u_slice = slice_predictions(model, data, ti)
        vmax = float(np.percentile(err, 99))
        for col in range(4):
            ax = axes[row, col]
            im = ax.imshow(
                err[:, :, col],
                origin="lower",
                extent=[data["x"].min(), data["x"].max(), data["y"].min(), data["y"].max()],
                cmap="viridis",
                vmin=0,
                vmax=vmax,
                aspect="auto",
            )
            ax.set_title(f"{row_title} | {EXPERT_NAMES[col]}", fontsize=11)
            if row == 1:
                ax.set_xlabel("x")
            if col == 0:
                ax.set_ylabel("y")
    fig.colorbar(im, ax=axes, shrink=0.8, pad=0.01)
    fig.suptitle(
        f"Expert specialization: |u_i - u_ref| at t={data['t'][ti]:.2f} "
        "(color scale normalized per row)",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(FIG_DIR, "gate_intro_expert_error_maps.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("[OK]", out)


def plot_figure_c() -> None:
    """Expert maturity curves from the f=100 staged run (Stage A/B/C histories)."""
    run_dir = os.path.join(RUN_ROOT, "f100_seed42", "burgers2d_moe_staged")
    info = torch.load(
        os.path.join(run_dir, "burgers2d_staged_training.pt"),
        map_location="cpu",
        weights_only=False,
    )
    expert_histories = info["expert_histories"]
    gate_history = info["gate_history"]
    final_ckpt = torch.load(
        os.path.join(run_dir, "burgers2d_moe_staged.pt"),
        map_location="cpu",
        weights_only=False,
    )
    joint_history = final_ckpt["history"]
    cfg = info["stage_config"]

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.8))

    ax = axes[0]
    for i, name in enumerate(EXPERT_NAMES):
        hist = expert_histories.get(name, {})
        l2 = hist.get("l2_error", [])
        steps = np.arange(1, len(l2) + 1) * max(1, cfg["expert_steps"] // max(len(l2), 1))
        ax.plot(steps, l2, marker="o", ms=3, color=COLORS[i], label=name)
    ax.set_title("(a) Stage A: per-expert L2 during expert pretraining", fontsize=11)
    ax.set_xlabel("Expert pretraining steps")
    ax.set_ylabel("Relative L2")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ent = np.asarray(gate_history["entropy"])
    mx = np.asarray(gate_history["max_gate"])
    steps_b = np.arange(1, len(ent) + 1)
    ax.plot(steps_b, ent, color="#7F7F7F", lw=1.8, label="Route entropy")
    ax.plot(steps_b, mx, color="#9467BD", lw=1.8, ls="--", label="Mean top-1 weight")
    ax.set_title("(b) Stage B: gate training convergence", fontsize=11)
    ax.set_xlabel("Stage B steps")
    ax.set_ylabel("Value")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[2]
    n = len(joint_history["total"])
    steps_c = (np.arange(len(joint_history["gate_entropy"])) + 1) * (
        n / len(joint_history["gate_entropy"])
    )
    for i, name in enumerate(EXPERT_NAMES):
        ax.plot(steps_c, joint_history[f"gate_load_{i}"], color=COLORS[i], lw=1.8, label=name)
    ax.set_title("(c) Stage C (gate-only joint): load fractions stay balanced", fontsize=11)
    ax.set_xlabel("Stage C steps")
    ax.set_ylabel("Top-1 load fraction")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Expert maturation (f=100% run, seed 42)", fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(FIG_DIR, "gate_intro_expert_maturity.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("[OK]", out)


def plot_trajectory() -> None:
    """Gate-introduction trajectory (accuracy / utilization) from the summary."""
    summary = json.load(
        open(os.path.join(RUN_ROOT, "gate_intro_ablation_summary.json"), encoding="utf-8")
    )
    rows = summary["rows"]
    fractions = summary["fractions"]
    xs = np.arange(len(fractions))
    labels = [f"{int(f * 100)}%" for f in fractions]

    def series(key: str) -> tuple[np.ndarray, np.ndarray]:
        mean = np.array([rows[str(f)][key]["mean"] for f in fractions])
        std = np.array([rows[str(f)][key]["std"] for f in fractions])
        return mean, std

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2))
    panels = [
        ("l2_relative_error", "Relative L2 (lower is better)"),
        ("max_absolute_error", "Max absolute error (lower is better)"),
        ("effective_experts", "Effective experts 1/sum(p^2) (higher is better)"),
        ("min_load_frac", "Min expert load (higher is more balanced)"),
        ("route_entropy", "Route entropy (lower is more decisive)"),
        ("max_expert_l2", "Worst per-expert L2 (lower is healthier)"),
    ]
    for ax, (key, title) in zip(axes.flat, panels):
        mean, std = series(key)
        ax.errorbar(xs, mean, yerr=std, marker="o", capsize=4, lw=1.8, color="#4C72B0")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Gate introduction point (expert training progress)", fontsize=9)
        ax.grid(alpha=0.3, linewidth=0.6)
        ax.tick_params(labelsize=8)
        for x, m, s in zip(xs, mean, std):
            ax.annotate(f"{m:.3f}±{s:.3f}", (x, m), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7)
    fig.suptitle(
        "Gate introduction timing vs accuracy and expert utilization "
        "(2D Burgers, 3 seeds, mean +/- std)",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(FIG_DIR, "gate_intro_trajectory.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("[OK]", out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        default=os.path.join(PROJECT_ROOT, "docs", "paper1_figures"),
    )
    args = parser.parse_args()
    global FIG_DIR
    FIG_DIR = args.outdir
    os.makedirs(FIG_DIR, exist_ok=True)
    plot_figure_a()
    plot_figure_b()
    plot_figure_c()
    plot_trajectory()


if __name__ == "__main__":
    main()
