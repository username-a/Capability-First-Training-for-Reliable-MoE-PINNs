"""
Plots for Burger2D.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def plot_training_curves(history: dict[str, list[float]], save_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].semilogy(history["total"], label="total")
    for key in ["res", "ic", "bc", "sparse", "balance"]:
        if key in history and history[key]:
            axes[0].semilogy(history[key], label=key)
    axes[0].set_title("Training Loss Curves")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss value")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    if history.get("l2_error"):
        axes[1].semilogy(history["l2_error"], color="tab:green")
        axes[1].set_title("Evaluation Relative L2")
        axes[1].set_xlabel("Eval Index")
        axes[1].set_ylabel("Relative L2")
        axes[1].grid(alpha=0.3)
    else:
        axes[1].axis("off")

    if history.get("gate_entropy"):
        axes[2].plot(history["gate_entropy"], label="entropy")
        if history.get("gate_max"):
            axes[2].plot(history["gate_max"], label="max_gate")
        axes[2].set_title("Routing Statistics")
        axes[2].set_xlabel("Log Index")
        axes[2].set_ylabel("Value")
        axes[2].grid(alpha=0.3)
        axes[2].legend()
    else:
        axes[2].axis("off")

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_snapshot_comparison(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    u_ref: np.ndarray,
    u_pred: np.ndarray,
    save_path: str,
    time_indices: Optional[Iterable[int]] = None,
) -> None:
    if time_indices is None:
        time_indices = [0, len(t) // 2, len(t) - 1]
    time_indices = list(dict.fromkeys(int(i) for i in time_indices))

    vmin = float(min(u_ref.min(), u_pred.min()))
    vmax = float(max(u_ref.max(), u_pred.max()))
    err_max = float(np.abs(u_pred - u_ref).max())

    fig, axes = plt.subplots(len(time_indices), 3, figsize=(12, 4 * len(time_indices)))
    if len(time_indices) == 1:
        axes = np.expand_dims(axes, axis=0)

    extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]
    fig.suptitle(
        "Field Comparison: reference solution, model prediction, and absolute error\n"
        "Brighter colors in the right column mean larger pointwise error.",
        fontsize=12,
        y=0.995,
    )
    for row, idx in enumerate(time_indices):
        ref_ax, pred_ax, err_ax = axes[row]
        ref_im = ref_ax.imshow(
            u_ref[idx],
            origin="lower",
            extent=extent,
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )
        pred_im = pred_ax.imshow(
            u_pred[idx],
            origin="lower",
            extent=extent,
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )
        err_im = err_ax.imshow(
            np.abs(u_pred[idx] - u_ref[idx]),
            origin="lower",
            extent=extent,
            cmap="magma",
            vmin=0.0,
            vmax=err_max,
            aspect="auto",
        )
        ref_ax.set_title(f"Reference Field | t={t[idx]:.2f}")
        pred_ax.set_title(f"Predicted Field | t={t[idx]:.2f}")
        err_ax.set_title(f"Absolute Error | t={t[idx]:.2f}")
        for ax in (ref_ax, pred_ax, err_ax):
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        cbar_ref = fig.colorbar(ref_im, ax=ref_ax, fraction=0.046)
        cbar_pred = fig.colorbar(pred_im, ax=pred_ax, fraction=0.046)
        cbar_err = fig.colorbar(err_im, ax=err_ax, fraction=0.046)
        cbar_ref.set_label("u value")
        cbar_pred.set_label("u value")
        cbar_err.set_label("|u_pred - u_ref|")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_gating_maps(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    gates: np.ndarray,
    expert_names: list[str],
    save_path: str,
    time_indices: Optional[Iterable[int]] = None,
) -> None:
    if time_indices is None:
        time_indices = [0, len(t) // 2, len(t) - 1]
    time_indices = list(dict.fromkeys(int(i) for i in time_indices))
    n_times = len(time_indices)
    n_experts = gates.shape[-1]

    fig, axes = plt.subplots(n_times, n_experts, figsize=(3.2 * n_experts, 3.1 * n_times))
    if n_times == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_experts == 1:
        axes = np.expand_dims(axes, axis=1)

    extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]
    fig.suptitle(
        "Gating Maps: each panel shows the gate weight assigned to one expert\n"
        "Color scale is gate weight in [0, 1]; brighter means that expert is used more strongly.",
        fontsize=12,
        y=0.995,
    )
    for row, t_idx in enumerate(time_indices):
        for col in range(n_experts):
            ax = axes[row, col]
            im = ax.imshow(
                gates[t_idx, :, :, col],
                origin="lower",
                extent=extent,
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                aspect="auto",
            )
            if row == 0:
                ax.set_title(f"Expert: {expert_names[col]}")
            ax.set_xlabel("x")
            ax.set_ylabel(f"y | t={t[t_idx]:.2f}")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046)
            cbar.set_label("gate weight")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_centerline_slices(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    u_ref: np.ndarray,
    u_pred: np.ndarray,
    save_path: str,
    time_indices: Optional[Iterable[int]] = None,
) -> None:
    if time_indices is None:
        time_indices = [0, len(t) // 2, len(t) - 1]
    y_idx = int(np.argmin(np.abs(y)))

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx in time_indices:
        ax.plot(x, u_ref[idx, y_idx], label=f"Reference t={t[idx]:.2f}")
        ax.plot(x, u_pred[idx, y_idx], "--", label=f"Prediction t={t[idx]:.2f}")
    ax.set_title(
        f"Centerline Slice Along y={y[y_idx]:.3f}\n"
        "Solid lines are reference values; dashed lines are model predictions."
    )
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_expert_signed_error_maps(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    u_ref: np.ndarray,
    expert_preds: np.ndarray,
    expert_names: list[str],
    save_path: str,
    time_index: Optional[int] = None,
) -> None:
    if time_index is None:
        time_index = len(t) - 1
    n_experts = expert_preds.shape[-1]
    ncols = min(3, n_experts)
    nrows = int(np.ceil(n_experts / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    ref = u_ref[time_index]
    signed_err = expert_preds[time_index] - ref[:, :, None]
    vmax = float(np.abs(signed_err).max())
    extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]
    fig.suptitle(
        "Single-Expert Signed Error Maps at the final time\n"
        "Red means the expert branch predicts values that are too large; blue means too small.\n"
        "These are individual expert branches, not the final MoE mixture output.",
        fontsize=12,
        y=0.995,
    )

    for idx in range(nrows * ncols):
        ax = axes.flat[idx]
        if idx >= n_experts:
            ax.axis("off")
            continue
        im = ax.imshow(
            signed_err[:, :, idx],
            origin="lower",
            extent=extent,
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            aspect="auto",
        )
        ax.set_title(f"{expert_names[idx]} | t={t[time_index]:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046)
        cbar.set_label("signed error")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_directional_diagnostics(
    bin_centers_deg: np.ndarray,
    steep_density: np.ndarray,
    mean_gate_by_bin: np.ndarray,
    expert_names: list[str],
    save_path: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(
        "Directional Diagnostics in steep-region only\n"
        "Top: gradient-direction density. Bottom: mean expert gate weight by direction bin.",
        fontsize=12,
        y=0.995,
    )
    axes[0].bar(bin_centers_deg, steep_density, width=(bin_centers_deg[1] - bin_centers_deg[0]) * 0.9)
    axes[0].set_title("Steep-Region Gradient Direction Density")
    axes[0].set_xlabel("Gradient direction (deg)")
    axes[0].set_ylabel("Density")
    axes[0].grid(alpha=0.3)

    for idx, name in enumerate(expert_names):
        axes[1].plot(bin_centers_deg, mean_gate_by_bin[:, idx], marker="o", label=name)
    axes[1].set_title("Mean Gate Weight by Direction Bin")
    axes[1].set_xlabel("Gradient direction (deg)")
    axes[1].set_ylabel("Mean gate weight")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_model_metric_comparison(
    metrics_summary: dict[str, dict[str, float]],
    save_path: str,
) -> None:
    model_names = list(metrics_summary.keys())
    metric_names = [
        "l2_relative_error",
        "max_absolute_error",
        "steep_mae",
        "background_mae",
    ]
    labels = ["L2", "MaxErr", "Steep MAE", "Background MAE"]

    fig, axes = plt.subplots(1, len(metric_names), figsize=(4 * len(metric_names), 4))
    x = np.arange(len(model_names))
    fig.suptitle(
        "Model Comparison Summary\nLower bars are better for all four metrics.",
        fontsize=12,
        y=0.995,
    )

    for ax, metric, label in zip(axes, metric_names, labels):
        values = [float(metrics_summary[name][metric]) for name in model_names]
        ax.bar(x, values, color=["#4c78a8", "#f58518", "#54a24b"][: len(model_names)])
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=15)
        ax.set_title(label)
        ax.set_ylabel("metric value")
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_rotation_gate_diagnostics(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    focus_map: np.ndarray,
    rotation_angle_deg: np.ndarray,
    rotation_activation: np.ndarray,
    oriented_gate_mass: np.ndarray,
    dominant_expert_idx: np.ndarray,
    expert_names: list[str],
    save_path: str,
    *,
    time_indices: Optional[Iterable[int]] = None,
) -> None:
    if time_indices is None:
        time_indices = [0, len(t) // 2, len(t) - 1]
    time_indices = list(dict.fromkeys(int(i) for i in time_indices))

    fig, axes = plt.subplots(len(time_indices), 5, figsize=(18, 3.8 * len(time_indices)))
    if len(time_indices) == 1:
        axes = np.expand_dims(axes, axis=0)

    extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]
    angle_cmap = plt.get_cmap("twilight_shifted").copy()
    angle_cmap.set_bad("#d9d9d9")
    dominant_cmap = ListedColormap(plt.get_cmap("tab10")(np.linspace(0.0, 1.0, max(len(expert_names), 3))))

    fig.suptitle(
        "Rotation-Gate Joint Diagnostics\n"
        "Columns show where directional structure exists, what angle the rotation layer predicts, "
        "whether rotation is activated, how much route mass goes to direction-sensitive experts, "
        "and which expert wins top-1 routing.",
        fontsize=12,
        y=0.995,
    )

    for row, idx in enumerate(time_indices):
        angle_panel = rotation_angle_deg[idx].copy()
        low_support = np.maximum(focus_map[idx], rotation_activation[idx]) < 0.12
        angle_panel[low_support] = np.nan
        panels = [
            (
                focus_map[idx],
                "Directional/Wave Focus Score",
                "magma",
                0.0,
                1.0,
                "focus",
                None,
            ),
            (
                angle_panel,
                "Rotation Axis (deg)",
                angle_cmap,
                -90.0,
                90.0,
                "deg",
                None,
            ),
            (
                rotation_activation[idx],
                "Rotation Activation",
                "viridis",
                0.0,
                1.0,
                "activation",
                None,
            ),
            (
                oriented_gate_mass[idx],
                "Directional/Wave Gate Mass",
                "viridis",
                0.0,
                1.0,
                "gate mass",
                None,
            ),
            (
                dominant_expert_idx[idx],
                "Dominant Expert (top-1)",
                dominant_cmap,
                -0.5,
                len(expert_names) - 0.5,
                "expert",
                np.arange(len(expert_names)),
            ),
        ]
        for col, (data, title, cmap, vmin, vmax, cbar_label, ticks) in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(
                data,
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
            )
            ax.set_title(f"{title} | t={t[idx]:.2f}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046)
            cbar.set_label(cbar_label)
            if ticks is not None:
                cbar.set_ticks(ticks)
                cbar.set_ticklabels(expert_names)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_wave_routing_panels(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    wave_oracle_mask: np.ndarray,
    wave_gate: np.ndarray,
    wave_regret: np.ndarray,
    save_path: str,
    *,
    title: str,
    time_indices: Optional[Iterable[int]] = None,
) -> None:
    if time_indices is None:
        time_indices = [0, len(t) // 2, len(t) - 1]
    time_indices = list(dict.fromkeys(int(i) for i in time_indices))

    fig, axes = plt.subplots(len(time_indices), 3, figsize=(12, 4 * len(time_indices)))
    if len(time_indices) == 1:
        axes = np.expand_dims(axes, axis=0)

    extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]
    regret_max = float(np.quantile(wave_regret, 0.99)) if np.any(wave_regret > 0) else 1.0
    fig.suptitle(
        title
        + "\nLeft: where wave is the oracle-best expert. Middle: actual wave gate weight. Right: missed gain when mixture underuses wave.",
        fontsize=12,
        y=0.995,
    )
    for row, idx in enumerate(time_indices):
        panels = [
            (wave_oracle_mask[idx].astype(np.float32), "Wave Oracle Mask", "magma", 0.0, 1.0, "oracle"),
            (wave_gate[idx], "Wave Gate Weight", "viridis", 0.0, 1.0, "gate"),
            (wave_regret[idx], "Wave Regret Map", "inferno", 0.0, regret_max, "regret"),
        ]
        for col, (data, panel_title, cmap, vmin, vmax, cbar_label) in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(
                data,
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
            )
            ax.set_title(f"{panel_title} | t={t[idx]:.2f}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046)
            cbar.set_label(cbar_label)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_wave_routing_comparison(
    comparison_rows: list[dict[str, float | str]],
    save_path: str,
) -> None:
    model_names = [str(row["name"]) for row in comparison_rows]
    metrics = [
        ("wave_oracle_frac", "Wave Oracle Frac"),
        ("wave_mean_gate_on_oracle", "Mean Wave Gate on Oracle"),
        ("wave_top1_on_oracle", "Wave Top1 Route on Oracle"),
        ("wave_regret_mean", "Mean Wave Regret"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    x_axis = np.arange(len(model_names))
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2"]
    fig.suptitle(
        "Wave Routing Comparison\nHigher is better for the first three metrics; lower is better for wave regret.",
        fontsize=12,
        y=0.995,
    )
    for ax, (metric, title) in zip(axes, metrics):
        values = [float(row[metric]) for row in comparison_rows]
        ax.bar(x_axis, values, color=colors[: len(model_names)])
        ax.set_xticks(x_axis)
        ax.set_xticklabels(model_names, rotation=18, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_curated_case_3d_scene(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    u: np.ndarray,
    labels: np.ndarray,
    wave_prob: np.ndarray,
    label_confidence: np.ndarray,
    grad_mag: np.ndarray,
    lap_abs: np.ndarray,
    anisotropy: np.ndarray,
    region_names: list[str],
    anchors: list[dict[str, object]],
    key_time_index: int,
    save_path: str,
    *,
    title: str,
    max_points: int = 12000,
    random_seed: int = 42,
) -> None:
    rng = np.random.default_rng(random_seed)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    tt = np.broadcast_to(t[:, None, None], labels.shape)
    flat_x = np.broadcast_to(xx[None, :, :], labels.shape).reshape(-1)
    flat_y = np.broadcast_to(yy[None, :, :], labels.shape).reshape(-1)
    flat_t = tt.reshape(-1)
    flat_label = labels.reshape(-1)
    flat_wave_prob = wave_prob.reshape(-1)
    flat_conf = label_confidence.reshape(-1)
    flat_grad = grad_mag.reshape(-1)
    flat_lap = lap_abs.reshape(-1)
    if flat_x.size > max_points:
        idx = rng.choice(flat_x.size, size=max_points, replace=False)
    else:
        idx = np.arange(flat_x.size)

    palette = np.array(["#7f8c8d", "#f1c40f", "#e74c3c", "#3498db", "#9b59b6"], dtype=object)
    label_colors = palette[np.clip(flat_label[idx], 0, len(palette) - 1)]

    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(131, projection="3d")
    ax2 = fig.add_subplot(132, projection="3d")
    ax3 = fig.add_subplot(133, projection="3d")

    fig.suptitle(
        title
        + "\nPanel A: x-y-t region cloud. Panel B: wave probability volume. Panel C: x-y-u surface at the most wave-rich time slice.",
        fontsize=12,
        y=0.98,
    )

    ax1.scatter(flat_x[idx], flat_y[idx], flat_t[idx], c=label_colors, s=8, alpha=0.35, linewidths=0)
    ax1.set_title("A. Region-Labeled Dataset Cloud")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("t")

    wave_scatter = ax2.scatter(
        flat_x[idx],
        flat_y[idx],
        flat_t[idx],
        c=flat_wave_prob[idx],
        s=10 + 30 * np.clip(flat_conf[idx], 0.0, 1.0),
        cmap="viridis",
        alpha=0.35,
        linewidths=0,
    )
    ax2.set_title("B. Wave Probability and Confidence")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("t")
    cbar2 = fig.colorbar(wave_scatter, ax=ax2, fraction=0.03, pad=0.08)
    cbar2.set_label("wave probability")

    surface = ax3.plot_surface(
        xx,
        yy,
        u[key_time_index],
        cmap="coolwarm",
        linewidth=0,
        antialiased=True,
        alpha=0.92,
    )
    ax3.set_title(f"C. Reference Field Surface at t={t[key_time_index]:.2f}")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("u")
    cbar3 = fig.colorbar(surface, ax=ax3, fraction=0.03, pad=0.08)
    cbar3.set_label("u")

    annotation_lines = []
    for idx_anchor, anchor in enumerate(anchors, start=1):
        ax1.scatter([anchor["x"]], [anchor["y"]], [anchor["t"]], c="black", s=70, marker="x")
        ax1.text(anchor["x"], anchor["y"], anchor["t"], str(idx_anchor), color="black", fontsize=9)
        ax2.scatter([anchor["x"]], [anchor["y"]], [anchor["t"]], c="white", s=80, marker="x")
        ax2.text(anchor["x"], anchor["y"], anchor["t"], str(idx_anchor), color="white", fontsize=9)
        if int(anchor["time_index"]) == int(key_time_index):
            ax3.scatter([anchor["x"]], [anchor["y"]], [anchor["u"]], c="black", s=70, marker="x")
            ax3.text(anchor["x"], anchor["y"], anchor["u"], str(idx_anchor), color="black", fontsize=9)
        annotation_lines.append(
            f"{idx_anchor}. {anchor['name']}: label={anchor['label_name']}, "
            f"(x={anchor['x']:.2f}, y={anchor['y']:.2f}, t={anchor['t']:.2f}), "
            f"wave={anchor['wave_prob']:.2f}, conf={anchor['confidence']:.2f}, "
            f"grad={anchor['grad_mag']:.2f}, lap={anchor['lap_abs']:.2f}, aniso={anchor['anisotropy']:.2f}"
        )

    legend_lines = [f"{i}. {name}" for i, name in enumerate(region_names)]
    fig.text(
        0.01,
        0.02,
        "Region Legend: " + " | ".join(legend_lines) + "\n" + "\n".join(annotation_lines),
        fontsize=9,
        va="bottom",
        ha="left",
        family="monospace",
    )

    fig.tight_layout(rect=[0, 0.10, 1, 0.94])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
