"""
Regenerate the two theory-validation figures on the conservative WENO5
reference, using the final 10-seed staged/co-adaptation checkpoints:

  F11 capability matrix (4 experts x 4 capability regions, staged vs coadapt)
  F12 capability-gap distribution and oracle coverage (paired over seeds)

Outputs (PNG/PDF/SVG):
    docs/JCP_submission_20260808/figure_review/F11_capability_matrix_weno5.*
    docs/JCP_submission_20260808/figure_review/F12_capability_gap_weno5.*
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from jcp_figure_review import (  # noqa: E402
    RESULTS,
    REVIEW_DIR,
    _batch_eval,
    _build_burgers_model,
    _load_ckpt,
    _panel_label,
    _save_review,
)


RUN_ROOT = os.path.join(RESULTS, "true_staged_vs_coadapt_20260806")
OUT_DIR = os.path.join(RESULTS, "jcp_reference_rebuild_20260808")
LOG_PATH = os.path.join(OUT_DIR, "theory_figures_weno5_progress.log")
SEEDS = list(range(42, 52))


def log(msg: str) -> None:
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    print(msg, flush=True)


def load_eval(device: torch.device):
    import torch

    sub = np.load(os.path.join(OUT_DIR, "evaluation_subset.npz"), mmap_mode="r")
    x, y, t, u = sub["x"], sub["y"], sub["t"], sub["u"]
    tt, yy, xx = np.meshgrid(t, y, x, indexing="ij")
    coords = torch.tensor(
        np.stack([xx.ravel(), yy.ravel(), tt.ravel()], axis=1),
        dtype=torch.float32,
        device=device,
    )
    return x, y, t, u, coords


def load_model(mode: str, seed: int, device: torch.device):
    ck = _load_ckpt(os.path.join(RUN_ROOT, f"seed{seed}_{mode}", "pre_calibration_checkpoint.pt"))
    model = _build_burgers_model(ck["config"])
    model.load_state_dict(ck["model_state"])
    return model.to(device).eval(), ck


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    if args.plot_only:
        saved = json.load(open(os.path.join(OUT_DIR, "theory_validation_weno5.json"), encoding="utf-8"))
        names = saved["region_names"]
        matrices = {m: np.asarray(saved["capability_matrix_mean"][m]) for m in ("staged", "coadapt")}
        gap_stats = saved["gap_stats"]
    else:
        import torch
        from Burger2D.equations.burgers2d import ReferenceSolution2D
        from Burger2D.training.staged_burgers2d import compute_region_scores

        device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
        x, y, t, u, coords = load_eval(device)
        truth = np.asarray(u, dtype=np.float64).ravel()
        ref2d = ReferenceSolution2D(x=x, y=y, t=t, u=np.asarray(u, dtype=np.float32))

        dummy, _ = load_model("staged", SEEDS[0], device)
        names = list(dummy.expert_names)
        scores = compute_region_scores(ref2d, layout_variant="categorical")
        score_stack = np.stack([np.asarray(scores[n]).reshape(-1) for n in names], axis=1)
        labels = score_stack.argmax(axis=1)
        log("region fractions: " + ", ".join(
            f"{names[i]}={np.mean(labels == i):.3f}" for i in range(len(names))))

        matrices = {}
        gap_stats = {}
        for mode in ("staged", "coadapt"):
            matrix_rows = []
            meds, q10s, q90s, covs = [], [], [], []
            for s in SEEDS:
                model, _ = load_model(mode, s, device)
                branches = _batch_eval(model.get_expert_predictions, coords).numpy()[:, :, 0]
                gates = _batch_eval(model.get_gate_weights, coords).numpy()
                sq = (branches.astype(np.float64) - truth[:, None]) ** 2
                kstar = sq.argmin(axis=1)
                sorted_sq = np.sort(sq, axis=1)
                gap = sorted_sq[:, 1] - sorted_sq[:, 0]
                meds.append(float(np.median(gap)))
                q10s.append(float(np.percentile(gap, 10)))
                q90s.append(float(np.percentile(gap, 90)))
                covs.append(float(gates[np.arange(len(gates)), kstar].mean()))

                row = []
                for r in range(len(names)):
                    mask = labels == r
                    denom = np.linalg.norm(truth[mask]).clip(1e-10)
                    row.append([
                        float(np.linalg.norm(branches[mask, k].astype(np.float64) - truth[mask]) / denom)
                        for k in range(len(names))
                    ])
                matrix_rows.append(np.asarray(row))
                log(f"{mode} seed{s}: gap_med={meds[-1]:.2e} cov={covs[-1]:.3f}")
            matrices[mode] = np.mean(matrix_rows, axis=0)
            gap_stats[mode] = {"median": meds, "q10": q10s, "q90": q90s, "coverage": covs}

        with open(os.path.join(OUT_DIR, "theory_validation_weno5.json"), "w", encoding="utf-8") as fh:
            json.dump({
                "region_names": names,
                "reference": "conservative WENO5 513-grid subset (344064 points; 21 times, 128x128 spatial indices 1,5,...,509)",
                "capability_matrix_mean": {m: matrices[m].tolist() for m in matrices},
                "gap_stats": gap_stats,
            }, fh, ensure_ascii=False, indent=2)

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from jcp_figure_review import COADAPT_COLOR, GRID_COLOR, STAGED_COLOR, TEXT_GRAY

    # ---- F11 capability matrix ----
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0))
    display_names = ["Smooth", "Iso-shock", "Directional", "Wave"]
    vmax = max(float(matrices[m].max()) for m in matrices)
    vmin = min(float(matrices[m][matrices[m] > 0].min()) for m in matrices)
    for ax, mode, title in zip(axes, ("staged", "coadapt"), ("Staged", "Co-adaptation")):
        im = ax.imshow(matrices[mode], cmap="magma_r", norm=LogNorm(vmin=vmin, vmax=vmax), aspect="auto")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(display_names, rotation=25, ha="right")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(display_names)
        ax.set_xlabel("Complete branch")
        ax.set_title(title, color=TEXT_GRAY)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{matrices[mode][i, j]:.2f}",
                        ha="center", va="center", fontsize=6.7,
                        color="white" if matrices[mode][i, j] > 2.0 else "black")
    axes[0].set_ylabel("Ex-ante capability region")
    fig.colorbar(im, ax=axes, fraction=0.032, pad=0.025).set_label("Local relative L2 (log color scale)")
    for ax, lab in zip(axes, ["a", "b"]):
        _panel_label(ax, lab)
    fig.subplots_adjust(bottom=0.23, left=0.13, right=0.89, top=0.91, wspace=0.34)
    _save_review(fig, "F11_capability_matrix_weno5")

    # ---- F12 capability gap + oracle coverage ----
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65))
    ax = axes[0]
    for mode, color, lab in zip(("staged", "coadapt"), (STAGED_COLOR, COADAPT_COLOR), ("Staged", "Co-adaptation")):
        med = np.asarray(gap_stats[mode]["median"])
        lo = np.asarray(gap_stats[mode]["q10"])
        hi = np.asarray(gap_stats[mode]["q90"])
        ax.errorbar(SEEDS, med, yerr=[med - lo, hi - med], marker="o", ms=4,
                    capsize=3, lw=1.6, color=color, label=lab)
    ax.set_yscale("log")
    ax.set_xticks(SEEDS)
    ax.set_xlabel("seed")
    ax.set_ylabel("median capability gap $\\Delta(z)$ (log)")
    ax.set_title("Capability gap on final checkpoints", color=TEXT_GRAY)
    ax.grid(alpha=0.55, color=GRID_COLOR, linewidth=0.5, which="both")
    ax.legend()
    ax = axes[1]
    for mode, color, lab in zip(("staged", "coadapt"), (STAGED_COLOR, COADAPT_COLOR), ("Staged", "Co-adaptation")):
        ax.plot(SEEDS, gap_stats[mode]["coverage"], "o-", color=color, lw=1.6, ms=4, label=lab)
    ax.set_xticks(SEEDS)
    ax.set_xlabel("seed")
    ax.set_ylabel("oracle coverage $\\bar g_{k^\\star}$")
    ax.set_title("Oracle coverage on final checkpoints", color=TEXT_GRAY)
    ax.grid(alpha=0.55, color=GRID_COLOR, linewidth=0.5)
    ax.legend()
    for ax, lab in zip(axes, ["a", "b"]):
        _panel_label(ax, lab)
    _save_review(fig, "F12_capability_gap_weno5")

    log("ALL DONE")


if __name__ == "__main__":
    main()
