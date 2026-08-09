"""
Plot helpers for the Burgers and KdV experiments.
"""

import os
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, TwoSlopeNorm


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)

CMAP_SOLUTION = "RdBu_r"
CMAP_ERROR = "hot_r"
CMAP_GATE = "viridis"


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)


def _shock_center_and_mask(
    X: np.ndarray,
    T: np.ndarray,
    U_exact: np.ndarray,
    *,
    band_half_width: Optional[float] = None,
):
    x_vals = X[0, :]
    t_vals = T[:, 0]
    dx = float(np.mean(np.diff(x_vals)))
    grad_x = np.abs(np.gradient(U_exact, x_vals, axis=1))
    margin = max(6.0 * dx, 0.12 * (x_vals.max() - x_vals.min()))
    interior = (x_vals >= x_vals.min() + margin) & (x_vals <= x_vals.max() - margin)
    center_idx_inner = grad_x[:, interior].argmax(axis=1)
    center_idx = np.flatnonzero(interior)[center_idx_inner]
    center_x = x_vals[center_idx]
    if band_half_width is None:
        band_half_width = max(4.0 * dx, 0.08 * (x_vals.max() - x_vals.min()))
    shock_mask = np.abs(X - center_x[:, None]) <= band_half_width
    return x_vals, t_vals, center_x, float(band_half_width), shock_mask


def plot_solution_comparison(
    X: np.ndarray,
    T: np.ndarray,
    U_exact: np.ndarray,
    U_moe: np.ndarray,
    U_vanilla: np.ndarray,
    equation: str = "Burgers",
    save_path: str = "results/solution_comparison.png",
    t_slices: Optional[List[float]] = None,
):
    _ensure_dir(save_path)
    t_slices = t_slices or [0.25, 0.5, 0.75, 1.0]

    x_vals, t_vals, center_x, _, _ = _shock_center_and_mask(X, T, U_exact)
    zoom_half = max(0.22, 0.16 * (x_vals.max() - x_vals.min()))
    zoom_center = float(np.median(center_x))
    zoom_mask = (x_vals >= zoom_center - zoom_half) & (x_vals <= zoom_center + zoom_half)
    x_zoom = x_vals[zoom_mask]

    moe_signed_err = U_moe - U_exact
    van_signed_err = U_vanilla - U_exact
    signed_lim = max(
        float(np.abs(moe_signed_err).max()),
        float(np.abs(van_signed_err).max()),
        1e-8,
    )
    signed_norm = TwoSlopeNorm(vmin=-signed_lim, vcenter=0.0, vmax=signed_lim)

    fig = plt.figure(figsize=(17, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.35)

    vmin = U_exact.min()
    vmax = U_exact.max()
    titles = [f"Exact ({equation})", "MoE-PINN (Ours)", "Vanilla-PINN"]
    data_list = [U_exact, U_moe, U_vanilla]
    for col, (title, data) in enumerate(zip(titles, data_list)):
        ax = fig.add_subplot(gs[0, col])
        pcm = ax.pcolormesh(
            X,
            T,
            data,
            cmap=CMAP_SOLUTION,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
        )
        ax.plot(center_x, t_vals, "k--", lw=1.0, alpha=0.7)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_title(title)
        plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)

    ax_slice = fig.add_subplot(gs[1, 0])
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(t_slices)))
    for ti, color in zip(t_slices, colors):
        tidx = np.argmin(np.abs(t_vals - ti))
        ax_slice.plot(x_zoom, U_exact[tidx, zoom_mask], "-", color=color, lw=2.0, label=f"Exact t={t_vals[tidx]:.2f}")
        ax_slice.plot(x_zoom, U_moe[tidx, zoom_mask], "--", color=color, lw=1.8, label=f"MoE t={t_vals[tidx]:.2f}")
        ax_slice.plot(
            x_zoom,
            U_vanilla[tidx, zoom_mask],
            ":",
            color=color,
            lw=1.5,
            label=f"Vanilla t={t_vals[tidx]:.2f}",
        )
    ax_slice.set_xlabel("x")
    ax_slice.set_ylabel("u(x,t)")
    ax_slice.set_title("Zoomed Slices Near the Shock")
    ax_slice.legend(ncol=2, fontsize=8, loc="best")

    ax_err_moe = fig.add_subplot(gs[1, 1])
    pcm = ax_err_moe.pcolormesh(
        X,
        T,
        moe_signed_err,
        cmap="coolwarm",
        norm=signed_norm,
        shading="auto",
    )
    ax_err_moe.plot(center_x, t_vals, "k--", lw=1.0, alpha=0.7)
    ax_err_moe.set_xlabel("x")
    ax_err_moe.set_ylabel("t")
    ax_err_moe.set_title("MoE Signed Error")
    plt.colorbar(pcm, ax=ax_err_moe, fraction=0.046, pad=0.04, label="u_pred - u_exact")

    ax_err_van = fig.add_subplot(gs[1, 2])
    pcm = ax_err_van.pcolormesh(
        X,
        T,
        van_signed_err,
        cmap="coolwarm",
        norm=signed_norm,
        shading="auto",
    )
    ax_err_van.plot(center_x, t_vals, "k--", lw=1.0, alpha=0.7)
    ax_err_van.set_xlabel("x")
    ax_err_van.set_ylabel("t")
    ax_err_van.set_title("Vanilla Signed Error")
    plt.colorbar(pcm, ax=ax_err_van, fraction=0.046, pad=0.04, label="u_pred - u_exact")

    fig.suptitle(
        f"MoE-PINN Overview and Error Comparison: {equation} Equation",
        fontsize=14,
        fontweight="bold",
    )
    plt.savefig(save_path)
    plt.close(fig)
    print(f"[OK] Saved: {save_path}")


def plot_error_distribution(
    X: np.ndarray,
    T: np.ndarray,
    U_exact: np.ndarray,
    U_moe: np.ndarray,
    U_vanilla: np.ndarray,
    save_path: str = "results/error_distribution.png",
):
    _ensure_dir(save_path)
    err_moe = np.abs(U_moe - U_exact)
    err_van = np.abs(U_vanilla - U_exact)

    eps = 1e-8
    vmin = max(eps, min(err_moe.min(), err_van.min()))
    vmax = max(err_moe.max(), err_van.max())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, err, title in zip(
        axes,
        [err_moe, err_van],
        ["MoE-PINN Absolute Error", "Vanilla-PINN Absolute Error"],
    ):
        pcm = ax.pcolormesh(
            X,
            T,
            err,
            cmap=CMAP_ERROR,
            norm=LogNorm(vmin=vmin, vmax=vmax),
            shading="auto",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_title(title)
        plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04, label="|error|")

    fig.suptitle("Absolute Error Distribution (Log Scale)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"[OK] Saved: {save_path}")


def plot_loss_curves(
    history_moe: Dict[str, List[float]],
    history_vanilla: Dict[str, List[float]],
    save_path: str = "results/loss_curves.png",
):
    _ensure_dir(save_path)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.semilogy(np.arange(len(history_moe["total"])), history_moe["total"], label="MoE-PINN", lw=2)
    ax.semilogy(
        np.arange(len(history_vanilla["total"])),
        history_vanilla["total"],
        label="Vanilla PINN",
        lw=2,
        linestyle="--",
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Total Loss (log)")
    ax.set_title("Total Loss Convergence")
    ax.legend()

    ax = axes[1]
    for key, color in [("res", "C0"), ("ic", "C1"), ("bc", "C2")]:
        vals = history_moe.get(key, [])
        if vals:
            ax.semilogy(np.arange(len(vals)), vals, label=f"MoE {key}", color=color, lw=1.8)
        vals_v = history_vanilla.get(key, [])
        if vals_v:
            ax.semilogy(
                np.arange(len(vals_v)),
                vals_v,
                label=f"Vanilla {key}",
                color=color,
                lw=1.4,
                linestyle="--",
                alpha=0.7,
            )
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (log)")
    ax.set_title("PDE/IC/BC Components")
    ax.legend(fontsize=9)

    ax = axes[2]
    for key, color in [("ntk_w_res", "C0"), ("ntk_w_ic", "C1"), ("ntk_w_bc", "C2")]:
        vals = history_moe.get(key, [])
        if vals:
            ax.plot(np.arange(len(vals)), vals, label=key.replace("ntk_w_", "lambda_"), color=color, lw=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Adaptive Weight")
    ax.set_title("NTK Weight Evolution")
    ax.legend()

    fig.suptitle("Training Dynamics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"[OK] Saved: {save_path}")


def plot_gating_weights(
    X: np.ndarray,
    T: np.ndarray,
    gate_weights: np.ndarray,
    num_experts: int = 3,
    expert_names: Optional[List[str]] = None,
    save_path: str = "results/gating_weights.png",
):
    _ensure_dir(save_path)
    expert_names = expert_names or [f"Expert {i}" for i in range(num_experts)]

    fig, axes = plt.subplots(1, num_experts, figsize=(6 * num_experts, 5))
    if num_experts == 1:
        axes = [axes]

    for idx, (ax, name) in enumerate(zip(axes, expert_names)):
        weights = gate_weights[:, :, idx]
        pcm = ax.pcolormesh(X, T, weights, cmap=CMAP_GATE, vmin=0, vmax=1, shading="auto")
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_title(f"Gating Weight: {name}")
        plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04, label="g_i(x,t)")

    fig.suptitle("Expert Routing Map", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"[OK] Saved: {save_path}")


def plot_soliton_collision(
    X: np.ndarray,
    T: np.ndarray,
    U_exact: np.ndarray,
    U_moe: np.ndarray,
    t_snapshots: Optional[List[int]] = None,
    save_path: str = "results/soliton_collision.png",
):
    _ensure_dir(save_path)
    nt, _ = U_exact.shape
    if t_snapshots is None:
        t_snapshots = [nt // 5, 2 * nt // 5, 3 * nt // 5, 4 * nt // 5, nt - 1]

    fig, axes = plt.subplots(2, len(t_snapshots), figsize=(5 * len(t_snapshots), 8))
    t_vals = T[:, 0]
    x_vals = X[0, :]

    for col, tidx in enumerate(t_snapshots):
        ax0 = axes[0, col]
        ax0.plot(x_vals, U_exact[tidx], "b-", lw=2, label="Exact")
        ax0.set_title(f"t = {t_vals[tidx]:.2f}")
        ax0.set_xlabel("x")
        ax0.set_ylabel("u")

        ax1 = axes[1, col]
        ax1.plot(x_vals, U_exact[tidx], "b-", lw=2, label="Exact")
        ax1.plot(x_vals, U_moe[tidx], "r--", lw=1.8, label="MoE-PINN")
        err_max = np.abs(U_moe[tidx] - U_exact[tidx]).max()
        ax1.set_title(f"Max err: {err_max:.2e}")
        ax1.set_xlabel("x")
        ax1.legend(fontsize=9)

    fig.suptitle("KdV Two-Soliton Collision", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"[OK] Saved: {save_path}")


def plot_ntk_weight_evolution(
    history: Dict[str, List[float]],
    save_path: str = "results/ntk_evolution.png",
):
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(10, 5))
    for key, label in {
        "ntk_w_res": r"$\lambda_{res}$",
        "ntk_w_ic": r"$\lambda_{ic}$",
        "ntk_w_bc": r"$\lambda_{bc}$",
    }.items():
        vals = history.get(key, [])
        if vals:
            ax.plot(np.arange(len(vals)), vals, label=label, lw=2)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Adaptive Weight")
    ax.set_title("NTK Weight Evolution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"[OK] Saved: {save_path}")


def plot_shock_diagnostics(
    X: np.ndarray,
    T: np.ndarray,
    U_exact: np.ndarray,
    U_moe: np.ndarray,
    U_vanilla: np.ndarray,
    save_path: str = "results/shock_diagnostics.png",
    band_half_width: Optional[float] = None,
):
    _ensure_dir(save_path)

    x_vals, t_vals, center_x, band_half_width, shock_mask = _shock_center_and_mask(
        X,
        T,
        U_exact,
        band_half_width=band_half_width,
    )
    grad_x = np.abs(np.gradient(U_exact, x_vals, axis=1))

    err_moe = np.abs(U_moe - U_exact)
    err_van = np.abs(U_vanilla - U_exact)
    err_max = max(err_moe.max(), err_van.max(), 1e-8)

    mean_moe_shock = np.nanmean(np.where(shock_mask, err_moe, np.nan), axis=1)
    mean_van_shock = np.nanmean(np.where(shock_mask, err_van, np.nan), axis=1)
    mean_moe_bg = np.nanmean(np.where(~shock_mask, err_moe, np.nan), axis=1)
    mean_van_bg = np.nanmean(np.where(~shock_mask, err_van, np.nan), axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    ax = axes[0, 0]
    pcm = ax.pcolormesh(X, T, grad_x, cmap="magma", shading="auto")
    ax.plot(center_x, t_vals, "c--", lw=2, label="shock center")
    ax.plot(center_x - band_half_width, t_vals, "c:", lw=1)
    ax.plot(center_x + band_half_width, t_vals, "c:", lw=1)
    ax.set_title("Reference |u_x| and Shock Band")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.legend(loc="upper right")
    plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04, label="|u_x|")

    ax = axes[0, 1]
    pcm = ax.pcolormesh(
        X,
        T,
        err_moe,
        cmap=CMAP_ERROR,
        norm=LogNorm(vmin=max(err_moe.min(), 1e-8), vmax=err_max),
        shading="auto",
    )
    ax.plot(center_x, t_vals, "c--", lw=1.6)
    ax.set_title("MoE Error with Shock Overlay")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04, label="|error|")

    ax = axes[1, 0]
    pcm = ax.pcolormesh(
        X,
        T,
        err_van,
        cmap=CMAP_ERROR,
        norm=LogNorm(vmin=max(err_van.min(), 1e-8), vmax=err_max),
        shading="auto",
    )
    ax.plot(center_x, t_vals, "c--", lw=1.6)
    ax.set_title("Vanilla Error with Shock Overlay")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04, label="|error|")

    ax = axes[1, 1]
    ax.plot(t_vals, mean_moe_shock, label="MoE in shock band", lw=2.2, color="C0")
    ax.plot(t_vals, mean_van_shock, label="Vanilla in shock band", lw=2.2, color="C1")
    ax.plot(t_vals, mean_moe_bg, label="MoE outside shock", lw=1.8, color="C0", linestyle="--")
    ax.plot(t_vals, mean_van_bg, label="Vanilla outside shock", lw=1.8, color="C1", linestyle="--")
    ax.set_title("Mean Absolute Error: Shock Band vs Background")
    ax.set_xlabel("t")
    ax.set_ylabel("mean |error|")
    ax.legend(fontsize=9)

    fig.suptitle("Shock-Focused Diagnostics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"[OK] Saved: {save_path}")


def plot_metrics_bar(
    l2_moe: float,
    l2_vanilla: float,
    maxerr_moe: float,
    maxerr_vanilla: float,
    save_path: str = "results/metrics_bar.png",
):
    _ensure_dir(save_path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    panels = [
        ("Relative L2 Error", [l2_moe, l2_vanilla], l2_vanilla / max(l2_moe, 1e-12)),
        ("Max Absolute Error", [maxerr_moe, maxerr_vanilla], maxerr_vanilla / max(maxerr_moe, 1e-12)),
    ]
    labels = ["MoE-PINN", "Vanilla"]
    colors = ["#d1495b", "#4c78a8"]

    for ax, (title, values, gain) in zip(axes, panels):
        bars = ax.bar(labels, values, color=colors, width=0.62)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_ylabel("Error")
        ax.grid(True, axis="y", which="both", alpha=0.25)
        ax.text(
            0.5,
            0.96,
            f"MoE improvement: {gain:.2f}x",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value * 1.12,
                f"{value:.2e}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    fig.suptitle("Burgers Benchmark: Error Metrics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"[OK] Saved: {save_path}")


def plot_shock_zoom_comparison(
    X: np.ndarray,
    T: np.ndarray,
    U_exact: np.ndarray,
    U_moe: np.ndarray,
    U_vanilla: np.ndarray,
    save_path: str = "results/shock_zoom_comparison.png",
    x_window: Optional[float] = None,
    t_slices: Optional[List[float]] = None,
):
    _ensure_dir(save_path)

    x_vals, t_vals, center_x, _, shock_mask = _shock_center_and_mask(X, T, U_exact)
    x_window = x_window or max(0.24, 0.18 * (x_vals.max() - x_vals.min()))
    zoom_center = float(np.median(center_x))
    x_min = max(float(x_vals.min()), zoom_center - x_window)
    x_max = min(float(x_vals.max()), zoom_center + x_window)
    zoom_mask = (x_vals >= x_min) & (x_vals <= x_max)

    U_exact_zoom = U_exact[:, zoom_mask]
    U_moe_zoom = U_moe[:, zoom_mask]
    U_van_zoom = U_vanilla[:, zoom_mask]
    X_zoom = X[:, zoom_mask]
    T_zoom = T[:, zoom_mask]

    moe_signed_err = U_moe_zoom - U_exact_zoom
    van_signed_err = U_van_zoom - U_exact_zoom
    signed_lim = max(
        float(np.abs(moe_signed_err).max()),
        float(np.abs(van_signed_err).max()),
        1e-8,
    )
    signed_norm = TwoSlopeNorm(vmin=-signed_lim, vcenter=0.0, vmax=signed_lim)
    advantage = np.abs(U_van_zoom - U_exact_zoom) - np.abs(U_moe_zoom - U_exact_zoom)
    advantage_lim = max(float(np.abs(advantage).max()), 1e-8)
    advantage_norm = TwoSlopeNorm(vmin=-advantage_lim, vcenter=0.0, vmax=advantage_lim)
    t_slices = t_slices or [0.25, 0.5, 0.75, 1.0]

    fig = plt.figure(figsize=(17, 9.5))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.28)

    titles = ["Exact (zoomed shock zone)", "MoE signed error", "Vanilla signed error"]
    fields = [U_exact_zoom, moe_signed_err, van_signed_err]
    vmin = float(U_exact.min())
    vmax = float(U_exact.max())
    for idx, (title, field) in enumerate(zip(titles, fields)):
        ax = fig.add_subplot(gs[0, idx])
        if idx == 0:
            pcm = ax.pcolormesh(
                X_zoom,
                T_zoom,
                field,
                cmap=CMAP_SOLUTION,
                vmin=vmin,
                vmax=vmax,
                shading="auto",
            )
        else:
            pcm = ax.pcolormesh(
                X_zoom,
                T_zoom,
                field,
                cmap="coolwarm",
                norm=signed_norm,
                shading="auto",
            )
        ax.plot(center_x, t_vals, "k--", lw=1.0, alpha=0.7)
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_title(title)
        plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[1, 0])
    pcm = ax.pcolormesh(
        X_zoom,
        T_zoom,
        advantage,
        cmap="PiYG",
        norm=advantage_norm,
        shading="auto",
    )
    ax.plot(center_x, t_vals, "c--", lw=1.1)
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title("Error Advantage Map (positive = MoE better)")
    plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04, label="|err_van| - |err_moe|")

    ax = fig.add_subplot(gs[1, 1])
    slice_colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(t_slices)))
    x_zoom_vals = x_vals[zoom_mask]
    for ti, color in zip(t_slices, slice_colors):
        tidx = int(np.argmin(np.abs(t_vals - ti)))
        ax.plot(x_zoom_vals, U_exact[tidx, zoom_mask], color=color, lw=2.2, label=f"Exact t={t_vals[tidx]:.2f}")
        ax.plot(x_zoom_vals, U_moe[tidx, zoom_mask], color=color, lw=1.8, linestyle="--", label=f"MoE t={t_vals[tidx]:.2f}")
        ax.plot(x_zoom_vals, U_vanilla[tidx, zoom_mask], color=color, lw=1.5, linestyle=":", label=f"Vanilla t={t_vals[tidx]:.2f}")
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")
    ax.set_title("Zoomed solution slices near the shock")
    ax.legend(fontsize=8, ncol=2, loc="best")

    ax = fig.add_subplot(gs[1, 2])
    moe_band = np.nanmean(np.where(shock_mask, np.abs(U_moe - U_exact), np.nan), axis=1)
    van_band = np.nanmean(np.where(shock_mask, np.abs(U_vanilla - U_exact), np.nan), axis=1)
    ax.plot(t_vals, moe_band, lw=2.2, label="MoE shock-band MAE", color="C0")
    ax.plot(t_vals, van_band, lw=2.2, label="Vanilla shock-band MAE", color="C1")
    ax.fill_between(t_vals, 0.0, np.maximum(van_band - moe_band, 0.0), color="C0", alpha=0.15)
    ax.set_xlabel("t")
    ax.set_ylabel("mean |error|")
    ax.set_title("Shock-Band Error Over Time")
    ax.legend(fontsize=9)

    fig.suptitle("Shock-Zone Zoom Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"[OK] Saved: {save_path}")


def plot_expert_specialization(
    X: np.ndarray,
    T: np.ndarray,
    U_exact: np.ndarray,
    expert_predictions: List[np.ndarray],
    expert_names: Optional[List[str]] = None,
    save_path: str = "results/expert_specialization.png",
):
    _ensure_dir(save_path)
    num_experts = len(expert_predictions)
    expert_names = expert_names or [f"Expert {i}" for i in range(num_experts)]

    x_vals, t_vals, center_x, _, shock_band = _shock_center_and_mask(X, T, U_exact)
    zoom_half = max(0.22, 0.16 * (x_vals.max() - x_vals.min()))
    zoom_center = float(np.median(center_x))
    zoom_mask = (x_vals >= zoom_center - zoom_half) & (x_vals <= zoom_center + zoom_half)
    x_zoom_vals = x_vals[zoom_mask]

    signed_err_maps = [pred - U_exact for pred in expert_predictions]
    abs_err_maps = [np.abs(pred - U_exact) for pred in expert_predictions]
    err_lim = max(max(float(np.abs(err).max()) for err in signed_err_maps), 1e-8)
    err_norm = TwoSlopeNorm(vmin=-err_lim, vcenter=0.0, vmax=err_lim)

    fig, axes = plt.subplots(3, num_experts, figsize=(6 * num_experts, 12.5))
    if num_experts == 1:
        axes = np.array(axes).reshape(3, 1)

    for idx, (name, pred, signed_err, abs_err) in enumerate(
        zip(expert_names, expert_predictions, signed_err_maps, abs_err_maps)
    ):
        shock_mae_scalar = float(np.nanmean(np.where(shock_band, abs_err, np.nan)))
        bg_mae_scalar = float(np.nanmean(np.where(~shock_band, abs_err, np.nan)))

        ax = axes[0, idx]
        pcm = ax.pcolormesh(
            X,
            T,
            signed_err,
            cmap="coolwarm",
            norm=err_norm,
            shading="auto",
        )
        ax.plot(center_x, t_vals, "k--", lw=1.0, alpha=0.7)
        ax.set_title(f"{name}\nShockMAE={shock_mae_scalar:.2e}, BgMAE={bg_mae_scalar:.2e}")
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04, label="u_pred - u_exact")

        ax = axes[1, idx]
        for t_pick, style_exact, style_pred in [(0.25, "-", "--"), (0.75, "-", ":")]:
            tidx = int(np.argmin(np.abs(t_vals - t_pick)))
            color = "C0" if t_pick < 0.5 else "C3"
            ax.plot(x_zoom_vals, U_exact[tidx, zoom_mask], style_exact, color=color, lw=2.0, label=f"Exact t={t_vals[tidx]:.2f}")
            ax.plot(x_zoom_vals, pred[tidx, zoom_mask], style_pred, color=color, lw=1.8, label=f"Pred t={t_vals[tidx]:.2f}")
        ax.set_xlim(float(x_zoom_vals.min()), float(x_zoom_vals.max()))
        ax.set_xlabel("x")
        ax.set_ylabel("u(x,t)")
        ax.set_title(f"{name} shock-zone slices")
        ax.legend(fontsize=8, loc="best")

        shock_mae = np.nanmean(np.where(shock_band, abs_err, np.nan), axis=1)
        bg_mae = np.nanmean(np.where(~shock_band, abs_err, np.nan), axis=1)
        ax = axes[2, idx]
        ax.plot(t_vals, shock_mae, label="shock band", lw=2)
        ax.plot(t_vals, bg_mae, label="background", lw=2, linestyle="--")
        ax.set_title(f"{name} mean |error|")
        ax.set_xlabel("t")
        ax.set_ylabel("mean |error|")
        ax.legend(fontsize=9)

    fig.suptitle("Per-Expert Specialization Diagnostics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"[OK] Saved: {save_path}")
