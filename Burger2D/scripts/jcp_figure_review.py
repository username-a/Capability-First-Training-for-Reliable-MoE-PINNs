"""
Generate the candidate figure set for the JCP manuscript into a review folder.

The paper is NOT modified by this script.  Outputs are written to
docs/JCP_submission_20260808/figure_review/ for human review before insertion.

Figures:
  F1 burgers_routing_partition.png   routing partition + directional expert
                                     error, staged vs co-adaptation (seed 43,
                                     pre-calibration checkpoints, WENO5 grid)
  F2 kdv_routing_partition.png       x-t routing partition, f=100% vs f=0%
  F3 per_seed_paired.png             per-seed paired lines (soft mixture L2,
                                     worst-branch L2)
  F4 burgers_gate_intro_trajectory.png  gate-introduction sweep (Burgers)
  F5 kdv_gate_intro_trajectory.png   gate-introduction sweep (KdV)
  F7 factorial_2x2_bars.png          2x2 factorial grouped bars
  F8 apinn_bars.png                  APINN grouped bars
  F9 allen_cahn_bars.png             Allen-Cahn grouped bars
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "burger1D")):
    if p not in sys.path:
        sys.path.insert(0, p)

RESULTS = os.path.join(PROJECT_ROOT, "Burger2D", "results")
REVIEW_DIR = os.path.join(
    PROJECT_ROOT, "docs", "JCP_submission_20260808", "figure_review"
)

STAGED_COLOR = "#2F6B8A"
COADAPT_COLOR = "#D07A5F"
REFERENCE_COLOR = "#3D8C77"
PHYSICS_COLOR = "#9AA0A6"
GRID_COLOR = "#D9DEE3"
TEXT_GRAY = "#333333"
def _desaturate(hex_color: str, factor: float = 0.9) -> str:
    """Uniformly lower saturation (keep hue and lightness) for a calmer look."""
    import colorsys

    r, g, b = matplotlib.colors.to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s * factor)
    return matplotlib.colors.to_hex((r2, g2, b2))


# Okabe--Ito-derived categorical scheme. Expert identity is kept consistent
# across routing panels and does not reuse the staged/co-adaptation colors.
CAT_BASE_4 = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
CAT_BASE_3 = ["#009E73", "#0072B2", "#8A8175"]  # dispersion, smooth, shock
EXPERT_COLORS_4 = [_desaturate(c) for c in CAT_BASE_4]
# KdV: dispersion = deep blue (structure accent), smooth = cool green,
# shock = neutral warm gray (it covers most of the flat background, so a loud
# red block would read as an artifact).
EXPERT_COLORS_3 = [_desaturate(c, 0.82) for c in CAT_BASE_3]
# Allen-Cahn: interior = sage green, exterior = deep blue,
# interface = warm clay accent (its near-disappearance under co-adaptation is
# the point of the figure).
AC_COLORS = [_desaturate(c, 0.82) for c in ["#009E73", "#0072B2", "#E69F00"]]
FIELD_CMAP = "viridis"  # perceptually uniform scientific gradient
FIELD_CMAP_EXACT = "magma"  # warm sequential for the KdV exact solution, so
# the continuous panel does not share the cool blue family of the routing maps
ERR_VMAX = 0.4  # fixed color-scale ceiling; extreme values are clipped, not stretched
BOUNDARY_COLOR = "#C8C8C8"  # very thin light-gray region outlines
BOUNDARY_LW = 0.5
BG_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "light_beige", ["#EFEAE0", "#C9C2B4"]
)  # very light beige -> light gray base layer

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7.5,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7.3,
    "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.2,
    "legend.fontsize": 7.2,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "axes.unicode_minus": False,
    "figure.dpi": 200,
})


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.09,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        ha="right",
        va="bottom",
    )


def _save_review(fig: plt.Figure, tag: str) -> tuple[str, str, str]:
    base = os.path.join(REVIEW_DIR, tag)
    png = base + ".png"
    pdf = base + ".pdf"
    svg = base + ".svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return png, pdf, svg


def _tinted_routing(
    ax: plt.Axes,
    argmax: np.ndarray,
    field: np.ndarray,
    palette: list[str],
    *,
    extent,
    xg: np.ndarray,
    yg: np.ndarray,
    alpha: float = 0.55,
) -> None:
    """Layered map: grayscale field underlay + translucent class tints + thin
    white region boundaries (cartographic, not color-block style)."""
    f = np.asarray(field)
    lo, hi = float(np.percentile(f, 5)), float(np.percentile(f, 95))
    if hi <= lo:
        hi = lo + 1e-9
    ax.imshow(
        f,
        cmap=BG_CMAP,
        vmin=lo,
        vmax=hi,
        origin="lower",
        extent=extent,
        aspect="auto",
        interpolation="bilinear",
    )
    h, w = argmax.shape
    rgb = np.array([matplotlib.colors.to_rgb(c) for c in palette])
    overlay = np.zeros((h, w, 4), dtype=float)
    overlay[..., :3] = rgb[argmax]
    overlay[..., 3] = alpha
    ax.imshow(
        overlay,
        origin="lower",
        extent=extent,
        aspect="auto",
        interpolation="nearest",
    )
    ax.contour(
        xg,
        yg,
        argmax,
        levels=[i + 0.5 for i in range(len(palette) - 1)],
        colors=BOUNDARY_COLOR,
        linewidths=BOUNDARY_LW,
    )


def _batch_eval(fn, x: torch.Tensor, batch: int = 65536) -> torch.Tensor:
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch):
            out.append(fn(x[i : i + batch]).cpu())
    return torch.cat(out, dim=0)


def _build_burgers_model(cfg: dict) -> torch.nn.Module:
    import torch
    from Burger2D.core.moe_pinn import build_burgers2d_moe

    return build_burgers2d_moe(
        balance_weight=cfg.get("gate_balance_weight", 0.01),
        directional_expert_variant=cfg.get("directional_expert_variant", "hybrid"),
        wave_expert_variant=cfg.get("wave_expert_variant", "mixed_lite"),
        expert_layout_variant=cfg.get("expert_layout_variant", "categorical"),
        gate_variant=cfg.get("gate_variant", "local_conv"),
        rotation_variant="none",
    ).to(torch.float32)


def _load_ckpt(path: str):
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def fig1_burgers_routing() -> str:
    import torch
    run_root = os.path.join(RESULTS, "true_staged_vs_coadapt_20260806")
    ref = np.load(
        os.path.join(RESULTS, "jcp_reference_rebuild_20260808", "burgers_weno5_reference_257.npz"),
        mmap_mode="r",
    )
    x, y, t, u_ref = ref["x"], ref["y"], ref["t"], ref["u"]
    ti = [20, 36]  # t = 0.5 and 0.9
    t_lab = [f"t={t[i]:.1f}" for i in ti]
    xx, yy = np.meshgrid(x, y, indexing="xy")

    reeval = json.load(
        open(os.path.join(RESULTS, "jcp_reference_rebuild_20260808", "main_checkpoint_reevaluation.json"), encoding="utf-8")
    )
    soft = {}
    for run in reeval["runs"]:
        if run["seed"] == 43:
            soft[run["mode"]] = run["soft"]["relative_l2"]

    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.15))
    panels = ["a", "b", "c", "d", "e", "f"]

    err_maps = {}
    for mode in ("staged", "coadapt"):
        ck = _load_ckpt(os.path.join(run_root, f"seed43_{mode}", "pre_calibration_checkpoint.pt"))
        model = _build_burgers_model(ck["config"])
        model.load_state_dict(ck["model_state"])
        model.eval()
        coords = torch.tensor(
            np.stack([xx.ravel(), yy.ravel(), np.full_like(xx, t[ti[0]]).ravel()], axis=1),
            dtype=torch.float32,
        )
        expert = _batch_eval(model.get_expert_predictions, coords).numpy()[:, :, 0]
        err = np.abs(expert[:, 2] - np.asarray(u_ref[ti[0]]).ravel())
        err_maps[mode] = err.reshape(y.size, x.size)

    for row, mode in enumerate(("staged", "coadapt")):
        ck = _load_ckpt(os.path.join(run_root, f"seed43_{mode}", "pre_calibration_checkpoint.pt"))
        model = _build_burgers_model(ck["config"])
        model.load_state_dict(ck["model_state"])
        model.eval()
        for col, ti_ in enumerate(ti):
            coords = torch.tensor(
                np.stack([xx.ravel(), yy.ravel(), np.full_like(xx, t[ti_]).ravel()], axis=1),
                dtype=torch.float32,
            )
            gates = _batch_eval(model.get_gate_weights, coords).numpy()
            argmax = gates.argmax(axis=1).reshape(y.size, x.size)
            ax = axes[row, col]
            _tinted_routing(
                ax,
                argmax,
                np.asarray(u_ref[ti_]),
                EXPERT_COLORS_4,
                extent=[x.min(), x.max(), y.min(), y.max()],
                xg=xx,
                yg=yy,
            )
            ax.set_title(t_lab[col], color=TEXT_GRAY)
            if row == 1:
                ax.set_xlabel("x")
        ax = axes[row, 2]
        im2 = ax.imshow(
            err_maps[mode],
            origin="lower",
            extent=[x.min(), x.max(), y.min(), y.max()],
            cmap=FIELD_CMAP,
            vmin=0.0,
            vmax=ERR_VMAX,
            aspect="auto",
        )
        ax.set_title(f"Directional expert |u-u*|, {t_lab[0]}", color=TEXT_GRAY)
        if row == 1:
            ax.set_xlabel("x")
        mode_label = f"{'Staged' if mode == 'staged' else 'Co-adaptation'}  ·  soft L2 = {soft[mode]:.4f}"
        axes[row, 0].text(
            -0.36,
            0.5,
            mode_label,
            rotation=90,
            rotation_mode="anchor",
            transform=axes[row, 0].transAxes,
            ha="center",
            va="center",
            fontsize=7.4,
            fontweight="bold",
        )

    for ax, lab in zip(axes.flat, panels):
        _panel_label(ax, lab)

    cax = fig.add_axes([0.915, 0.20, 0.014, 0.69])
    fig.colorbar(im2, cax=cax).set_label("|u - u*|")
    handles = [
        matplotlib.patches.Patch(facecolor=c, edgecolor="none", label=lab)
        for c, lab in zip(
            EXPERT_COLORS_4,
            ["Smooth", "Iso-shock", "Directional-shock", "Wave"],
        )
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.01),
        frameon=False,
        fontsize=6.7,
    )
    fig.subplots_adjust(bottom=0.16, left=0.10, right=0.89, top=0.94, wspace=0.30, hspace=0.30)
    png, pdf, svg = _save_review(fig, "F1_burgers_routing_partition")
    return png


def fig2_kdv_routing() -> str:
    import torch
    from burger1D.core.moe_pinn import build_kdv_moe
    from burger1D.equations.kdv import hirota_two_soliton
    root = os.path.join(PROJECT_ROOT, "burger1D", "results", "gate_intro_kdv_20260805_005901")
    # Interaction window [-10, 8] x [0, 6], rendered as a single row of three
    # panels with identical, moderate proportions (no banner, no split rows).
    nx, nt = 256, 240
    x = np.linspace(-10.0, 8.0, nx)
    t = np.linspace(0.0, 6.0, nt)
    X, T = np.meshgrid(x, t)
    U = hirota_two_soliton(X, T)
    coords = torch.tensor(np.stack([X.ravel(), T.ravel()], axis=1), dtype=torch.float32)
    extent = [x.min(), x.max(), 0.0, 6.0]

    models = {}
    for f in (1.0, 0.0):
        ck = _load_ckpt(os.path.join(root, f"f{int(f*100)}_seed42", "kdv_moe_staged.pt"))
        model = build_kdv_moe(num_experts=3, sparsity_weight=1e-3, balance_weight=5e-3, gate_temperature=0.9)
        state = ck["model_state"] if isinstance(ck, dict) and "model_state" in ck else ck
        model.load_state_dict(state)
        model.eval()
        models[f] = model

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.45))
    im0 = axes[0].imshow(U, origin="lower", extent=extent, aspect="auto", cmap=FIELD_CMAP_EXACT)
    axes[0].set_title("Exact two-soliton solution", color=TEXT_GRAY)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")

    for ax, f, label in [
        (axes[1], 1.0, "f=100% (gate after experts)"),
        (axes[2], 0.0, "f=0% (joint from scratch)"),
    ]:
        gates = _batch_eval(models[f].get_gate_weights, coords).numpy()
        _tinted_routing(
            ax,
            gates.argmax(axis=1).reshape(nt, nx),
            U,
            EXPERT_COLORS_3,
            extent=extent,
            xg=X,
            yg=T,
        )
        ax.set_title(f"Argmax routing, {label}", color=TEXT_GRAY)
        ax.set_xlabel("x")
        ax.set_ylabel("t")

    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02)
    handles = [
        matplotlib.patches.Patch(facecolor=c, edgecolor="none", label=lab)
        for c, lab in zip(EXPERT_COLORS_3, ["Dispersion", "Smooth", "Shock"])
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
        fontsize=6.7,
    )
    for ax, lab in zip(axes, ["a", "b", "c"]):
        _panel_label(ax, lab)
    fig.subplots_adjust(bottom=0.24, left=0.08, right=0.98, top=0.91, wspace=0.34)
    png, pdf, svg = _save_review(fig, "F2_kdv_routing_partition")
    return png


def fig3_per_seed_paired() -> str:
    reeval = json.load(
        open(os.path.join(RESULTS, "jcp_reference_rebuild_20260808", "main_checkpoint_reevaluation.json"), encoding="utf-8")
    )
    seeds = []
    soft_s, soft_c, worst_s, worst_c = [], [], [], []
    for run in reeval["runs"]:
        if run["seed"] not in seeds:
            seeds.append(run["seed"])
    seeds = sorted(seeds)
    by_seed = {s: {} for s in seeds}
    for run in reeval["runs"]:
        by_seed[run["seed"]][run["mode"]] = run
    for s in seeds:
        soft_s.append(by_seed[s]["staged"]["soft"]["relative_l2"])
        soft_c.append(by_seed[s]["coadapt"]["soft"]["relative_l2"])
        worst_s.append(by_seed[s]["staged"]["worst_branch_l2"])
        worst_c.append(by_seed[s]["coadapt"]["worst_branch_l2"])

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65))
    ax = axes[0]
    ax.plot(seeds, soft_s, "o-", color=STAGED_COLOR, lw=1.5, ms=3.5, label="Staged")
    ax.plot(seeds, soft_c, "s-", color=COADAPT_COLOR, lw=1.5, ms=3.5, label="Co-adaptation")
    for s in (42, 44, 49):
        ax.axvspan(s - 0.4, s + 0.4, color="#E8C9A8", alpha=0.55)
    ax.set_xticks(seeds)
    ax.set_xlabel("seed")
    ax.set_ylabel("soft-mixture relative L2")
    ax.set_title("Soft-mixture error", color=TEXT_GRAY)
    ax.grid(alpha=0.55, color=GRID_COLOR, linewidth=0.5)
    ax.legend()

    ax = axes[1]
    ax.plot(seeds, worst_s, "o-", color=STAGED_COLOR, lw=1.5, ms=3.5, label="Staged")
    ax.plot(seeds, worst_c, "s-", color=COADAPT_COLOR, lw=1.5, ms=3.5, label="Co-adaptation")
    ax.set_yscale("log")
    ax.set_xticks(seeds)
    ax.set_xlabel("seed")
    ax.set_ylabel("worst-branch relative L2 (log)")
    ax.set_title("Worst complete-branch error", color=TEXT_GRAY)
    ax.grid(alpha=0.55, color=GRID_COLOR, linewidth=0.5, which="both")
    ax.legend()
    _panel_label(axes[0], "a")
    _panel_label(axes[1], "b")
    png, pdf, svg = _save_review(fig, "F3_per_seed_paired")
    return png


def fig4_allen_cahn_routing() -> str:
    import torch
    from Burger2D.scripts.summarize_allen_cahn import load_run as ac_load_run
    staged_dir = os.path.join(RESULTS, "allen_cahn", "staged_seed42")
    coadapt_dir = os.path.join(RESULTS, "allen_cahn", "coadapt_seed42")
    st_m, st_model, st_data = ac_load_run(pathlib.Path(staged_dir))
    co_m, co_model, co_data = ac_load_run(pathlib.Path(coadapt_dir))
    coords_np = st_data[0]
    ref = st_data[3]
    nt, ny, nx = ref.u.shape
    t_idx = nt // 2
    mask = np.abs(coords_np[:, 2] - ref.t[t_idx]) < 1e-6
    idx = np.flatnonzero(mask)
    dev = next(st_model.parameters()).device
    chunk = torch.tensor(coords_np[idx], dtype=torch.float32, device=dev)
    with torch.no_grad():
        st_route = st_model.gate_weights(chunk).argmax(dim=-1).cpu().numpy().reshape(ny, nx)
        co_route = co_model.gate_weights(chunk).argmax(dim=-1).cpu().numpy().reshape(ny, nx)
    ref_slice = np.asarray(ref.u[t_idx])
    xx = coords_np[idx, 0].reshape(ny, nx)
    yy = coords_np[idx, 1].reshape(ny, nx)
    extent = [-1.0, 1.0, -1.0, 1.0]

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.55))
    _tinted_routing(axes[0], st_route, ref_slice, AC_COLORS, extent=extent, xg=xx, yg=yy)
    _tinted_routing(axes[1], co_route, ref_slice, AC_COLORS, extent=extent, xg=xx, yg=yy)
    axes[0].set_title(f"Staged routing, t={ref.t[t_idx]:.3f}", color=TEXT_GRAY)
    axes[1].set_title(f"Co-adaptation routing, t={ref.t[t_idx]:.3f}", color=TEXT_GRAY)
    im2 = axes[2].imshow(
        ref_slice,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        aspect="auto",
    )
    axes[2].set_title(f"Reference u, t={ref.t[t_idx]:.3f}", color=TEXT_GRAY)
    for ax in axes:
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[2].set_ylabel("y")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.02)
    handles = [
        matplotlib.patches.Patch(facecolor=c, edgecolor="none", label=lab)
        for c, lab in zip(AC_COLORS, ["Interior", "Exterior", "Interface"])
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.01),
        frameon=False,
        fontsize=6.7,
    )
    for ax, lab in zip(axes, ["a", "b", "c"]):
        _panel_label(ax, lab)
    fig.subplots_adjust(bottom=0.23, left=0.07, right=0.98, top=0.91, wspace=0.32)
    png, pdf, svg = _save_review(fig, "F4_allen_cahn_routing")
    return png


def _errorbar_panel(ax, mean, std, xlabels, title, xlabel, ylabel="Value", color=REFERENCE_COLOR):
    xs = np.arange(len(mean))
    ax.errorbar(xs, mean, yerr=std, marker="o", capsize=4, lw=1.8, color=color)
    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels)
    ax.set_title(title, color=TEXT_GRAY)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.55, color=GRID_COLOR, linewidth=0.5)


def fig4_burgers_gate_intro() -> str:
    summary = json.load(
        open(os.path.join(RESULTS, "jcp_reference_rebuild_20260808", "gate_intro_summary_weno5.json"), encoding="utf-8")
    )
    rows, fractions = summary["rows"], summary["fractions"]
    labels = [f"{int(f*100)}%" for f in fractions]

    def series(key):
        mean = np.array([rows[str(f)][key]["mean"] for f in fractions])
        std = np.array([rows[str(f)][key]["std"] for f in fractions])
        return mean, std

    panels = [
        ("l2_relative_error", "Mixture relative L2"),
        ("max_absolute_error", "Maximum absolute error"),
        ("effective_experts", "Effective experts"),
        ("min_load_frac", "Minimum expert load"),
        ("route_entropy", "Route entropy"),
        ("max_expert_l2", "Worst-expert relative L2"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.65))
    for ax, (key, title) in zip(axes.flat, panels):
        mean, std = series(key)
        _errorbar_panel(ax, mean, std, labels, title, "Independent expert pretraining, f")
    for ax, lab in zip(axes.flat, ["a", "b", "c", "d", "e", "f"]):
        _panel_label(ax, lab)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.11, top=0.94,
                        wspace=0.32, hspace=0.58)
    png, pdf, svg = _save_review(fig, "F4_burgers_gate_intro_trajectory")
    return png


def fig5_kdv_gate_intro() -> str:
    root = os.path.join(PROJECT_ROOT, "burger1D", "results", "gate_intro_kdv_20260805_005901")
    fracs = [0.0, 0.25, 0.5, 0.75, 1.0]
    seeds = [42, 43, 44]
    agg = {f: {k: [] for k in ("l2", "worst_expert", "entropy", "min_load")} for f in fracs}
    for f in fracs:
        for s in seeds:
            m = json.load(open(os.path.join(root, f"f{int(f*100)}_seed{s}", "metrics.json"), encoding="utf-8"))
            agg[f]["l2"].append(m["l2"])
            agg[f]["worst_expert"].append(max(m["per_expert_l2"]))
            agg[f]["entropy"].append(m["route_entropy"])
            agg[f]["min_load"].append(min(m["load_frac"]))
    labels = [f"{int(f*100)}%" for f in fracs]
    panels = [
        ("l2", "Mixture relative L2"),
        ("worst_expert", "Worst-expert relative L2"),
        ("entropy", "Route entropy"),
        ("min_load", "Minimum expert load"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.55))
    for ax, (key, title) in zip(axes.flat, panels):
        vals = np.array([agg[f][key] for f in fracs])
        mean, std = vals.mean(axis=1), vals.std(axis=1, ddof=1)
        _errorbar_panel(ax, mean, std, labels, title, "Independent expert pretraining, f")
    for ax, lab in zip(axes.flat, ["a", "b", "c", "d"]):
        _panel_label(ax, lab)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.11, top=0.94,
                        wspace=0.26, hspace=0.55)
    png, pdf, svg = _save_review(fig, "F5_kdv_gate_intro_trajectory")
    return png


def _grouped_bars(ax, groups, series, stds, titles, colors, xlabel):
    x = np.arange(len(groups))
    n = len(series)
    w = 0.8 / n
    for k, (name, vals) in enumerate(series.items()):
        ax.bar(x + (k - (n - 1) / 2) * w, vals, width=w * 0.9, yerr=stds[name],
               capsize=3, color=colors[k], label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_title(titles, color=TEXT_GRAY)
    ax.set_xlabel(xlabel)
    ax.grid(alpha=0.55, color=GRID_COLOR, linewidth=0.5, axis="y")


def fig7_2x2_bars() -> str:
    import csv
    rows = list(csv.DictReader(open(
        os.path.join(RESULTS, "equal_info_2x2_confirmatory_20260806", "confirmatory_raw.csv"),
        encoding="utf-8",
    )))
    groups = ["P-B", "P-I", "R-B", "R-I"]
    data = {g: {"l2": [], "worst": [], "eff": []} for g in groups}
    for r in rows:
        g = r["group"]
        data[g]["l2"].append(float(r["l2_relative_error"]))
        data[g]["worst"].append(float(r["worst_expert_l2"]))
        data[g]["eff"].append(float(r["effective_experts"]))

    def stat(key):
        mean = {g: float(np.mean(data[g][key])) for g in groups}
        std = {g: float(np.std(data[g][key], ddof=1)) for g in groups}
        return mean, std

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.55))
    for ax, key, title in zip(
        axes,
        ("l2", "worst", "eff"),
        ("Mixture relative L2", "Worst-branch relative L2", "Effective experts"),
    ):
        mean, std = stat(key)
        bars = ax.bar(
            np.arange(len(groups)),
            [mean[k] for k in groups],
            yerr=[std[k] for k in groups],
            color=[PHYSICS_COLOR, PHYSICS_COLOR, REFERENCE_COLOR, REFERENCE_COLOR],
            hatch=["", "///", "", "///"],
            edgecolor="#4D4D4D",
            linewidth=0.45,
            capsize=3,
            error_kw={"elinewidth": 1.2, "capthick": 1.2},
        )
        ax.set_xticks(np.arange(len(groups)))
        ax.set_xticklabels(groups)
        ax.set_title(title, color=TEXT_GRAY)
        ax.set_xlabel("Information x chronology")
        ax.grid(alpha=0.55, color=GRID_COLOR, linewidth=0.5, axis="y")
    for ax, lab in zip(axes, ["a", "b", "c"]):
        _panel_label(ax, lab)
    png, pdf, svg = _save_review(fig, "F7_factorial_2x2_bars")
    return png


def fig8_apinn_bars() -> str:
    s = json.load(
        open(os.path.join(RESULTS, "jcp_reference_rebuild_20260808", "apinn_checkpoint_reevaluation.json"), encoding="utf-8")
    )["summary"]
    groups = ["2-PM", "2-Official", "4-PM"]
    keys = ["two_subnet_parameter_matched", "two_subnet_official_size", "four_subnet_parameter_matched"]
    labels = ["Soft-mixture relative L2", "Worst-subnetwork L2", "Top-1 / soft ratio"]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.55))
    cfg_colors = ["#5B7FA3", "#91A6B8", "#3D8C77"]
    for ax, key, lab in zip(axes, ("soft_relative_l2", "worst_branch_l2", "top1_ratio"), labels):
        mean = [s[k][key]["mean"] for k in keys]
        std = [s[k][key]["sample_std"] for k in keys]
        x = np.arange(len(groups))
        ax.bar(x, mean, yerr=std, color=cfg_colors, capsize=2.5,
               edgecolor="#4D4D4D", linewidth=0.45,
               error_kw={"elinewidth": 0.8, "capthick": 0.8})
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.set_title(lab, color=TEXT_GRAY)
        ax.set_xlabel("APINN configuration")
        ax.grid(alpha=0.55, color=GRID_COLOR, linewidth=0.5, axis="y")
    for ax, lab in zip(axes, ["a", "b", "c"]):
        _panel_label(ax, lab)
    png, pdf, svg = _save_review(fig, "F8_apinn_bars")
    return png


def fig9_allen_cahn_bars() -> str:
    s = json.load(open(os.path.join(RESULTS, "allen_cahn", "summary_aggregate.json"), encoding="utf-8"))
    groups = ["staged", "coadapt"]
    labels = ["staged", "co-adaptation"]
    metrics = [
        ("l2_mixed", "Mixture relative L2"),
        ("region_l2_interface", "Interface-region L2"),
        ("load_interface", "Interface-expert load"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.55))
    for ax, (key, lab) in zip(axes, metrics):
        mean = [s[m][key]["mean"] for m in groups]
        std = [s[m][key]["std"] for m in groups]
        ax.bar(
            np.arange(len(groups)),
            mean,
            yerr=std,
            color=[STAGED_COLOR, COADAPT_COLOR],
            hatch=["", "///"],
            edgecolor="#4D4D4D",
            linewidth=0.45,
            capsize=3,
            error_kw={"elinewidth": 1.2, "capthick": 1.2},
        )
        ax.set_xticks(np.arange(len(groups)))
        ax.set_xticklabels(["Staged", "Co-adaptation"])
        ax.set_title(lab, color=TEXT_GRAY)
        ax.set_xlabel("Training mode")
        ax.grid(alpha=0.55, color=GRID_COLOR, linewidth=0.5, axis="y")
    for ax, lab in zip(axes, ["a", "b", "c"]):
        _panel_label(ax, lab)
    png, pdf, svg = _save_review(fig, "F9_allen_cahn_bars")
    return png


def copy_existing() -> list[str]:
    src_dir = os.path.join(PROJECT_ROOT, "docs", "JCP_submission_20260808", "chinese", "figures")
    names = [
        "theory_capability_gap.png",
        "theory_grad_starvation.png",
        "capability_matrix_heatmap.pdf",
        "coadapt_independent_grid.pdf",
        "kdv_gate_intro.png",
    ]
    copied = []
    for n in names:
        src = os.path.join(src_dir, n)
        if os.path.exists(src):
            dst = os.path.join(REVIEW_DIR, "P3_existing_" + n)
            shutil.copyfile(src, dst)
            copied.append(dst)
    return copied


def main() -> None:
    os.makedirs(REVIEW_DIR, exist_ok=True)
    outputs = [
        ("F1", fig1_burgers_routing()),
        ("F2", fig2_kdv_routing()),
        ("F3", fig3_per_seed_paired()),
        ("F4", fig4_burgers_gate_intro()),
        ("F5", fig5_kdv_gate_intro()),
        ("F7", fig7_2x2_bars()),
        ("F8", fig8_apinn_bars()),
        ("F9", fig9_allen_cahn_bars()),
    ]
    for tag, path in outputs:
        print(f"[{tag}] {path}")
    for path in copy_existing():
        print(f"[P3] {path}")
    print("ALL DONE ->", REVIEW_DIR)


if __name__ == "__main__":
    main()
