"""Aggregate Allen-Cahn staged-vs-coadapt runs and produce paper figures.

For every run directory under results/allen_cahn/*_seed*: rebuild the model
from its config, load the checkpoint, and evaluate on the full reference grid,
including per-expert error restricted to each expert's own region
(specialization health).  Writes summary.json/csv and, for the seed-42 pair,
paper figures: gate routing maps and expert error maps at a mid-time slice.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from Burger2D.scripts.run_allen_cahn_staged_vs_coadapt import (
    RESULTS_DIR,
    AllenCahnMoE,
    AllenConfig,
    EXPERT_NAMES,
    build_model,
    evaluate,
    flatten_reference,
    get_reference,
)


def region_expert_metrics(
    model: AllenCahnMoE,
    coords: torch.Tensor,
    values: torch.Tensor,
    region: torch.Tensor,
    batch: int = 65536,
) -> dict[str, float]:
    device = next(model.parameters()).device
    n = coords.shape[0]
    errs_list, weights_list = [], []
    with torch.no_grad():
        for start in range(0, n, batch):
            chunk = coords[start:start + batch].to(device)
            preds = model.expert_predictions(chunk).cpu().squeeze(-1)
            weights = model.gate_weights(chunk).cpu()
            errs_list.append((preds - values[start:start + batch]).abs())
            weights_list.append(weights)
    errs = torch.cat(errs_list, dim=0)
    weights = torch.cat(weights_list, dim=0)
    mix_pred_list = []
    with torch.no_grad():
        for start in range(0, n, batch):
            chunk = coords[start:start + batch].to(device)
            mix_pred_list.append(model(chunk).cpu())
    mix_pred = torch.cat(mix_pred_list, dim=0)
    out: dict[str, float] = {}
    for k, name in enumerate(EXPERT_NAMES):
        rmask = region == k
        rl2 = float(torch.sqrt((errs[rmask, k] ** 2).mean() / (values[rmask] ** 2).mean()))
        out[f"region_l2_{name}"] = rl2
    # mixture error restricted to the interface region
    rmask = region == 2
    out["interface_mixed_l2"] = float(
        torch.sqrt(((mix_pred[rmask] - values[rmask]) ** 2).mean() / (values[rmask] ** 2).mean())
    )
    return out


def load_run(run_dir: Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(run_dir / "checkpoint.pt", map_location=device)
    if (run_dir / "config.json").exists():
        cfg = AllenConfig(**json.loads((run_dir / "config.json").read_text()))
    else:
        cfg = AllenConfig(**ckpt["config"])
    model = build_model(cfg, device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    ref = get_reference(cfg)
    coords_np, values_np, region_np = flatten_reference(ref)
    coords = torch.tensor(coords_np, dtype=torch.float32)
    values = torch.tensor(values_np, dtype=torch.float32)
    region = torch.tensor(region_np, dtype=torch.long)
    metrics = evaluate(model, coords, values)
    metrics.update(region_expert_metrics(model, coords, values, region))
    metrics["mode"] = cfg.mode
    metrics["seed"] = cfg.seed
    metrics["run"] = run_dir.name
    return metrics, model, (coords_np, values_np, region_np, ref)


def plot_pair(staged_dir: Path, coadapt_dir: Path, out_dir: Path) -> None:
    st_m, st_model, st_data = load_run(staged_dir)
    co_m, co_model, co_data = load_run(coadapt_dir)
    _, _, _, ref = st_data
    coords_np = st_data[0]
    nt_, ny_, nx_ = ref.u.shape

    # choose a mid-time slice
    t_idx = nt_ // 2
    slice_mask = np.abs(coords_np[:, 2] - ref.t[t_idx]) < 1e-6
    idx = np.flatnonzero(slice_mask)
    xx = coords_np[idx, 0].reshape(ny_, nx_)
    yy = coords_np[idx, 1].reshape(ny_, nx_)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chunk = torch.tensor(coords_np[idx], dtype=torch.float32)
    with torch.no_grad():
        st_route = st_model.gate_weights(chunk.to(device)).argmax(dim=-1).cpu().numpy().reshape(ny_, nx_)
        co_route = co_model.gate_weights(chunk.to(device)).argmax(dim=-1).cpu().numpy().reshape(ny_, nx_)
        st_preds = st_model.expert_predictions(chunk.to(device)).cpu().numpy().reshape(ny_, nx_, -1)
        co_preds = co_model.expert_predictions(chunk.to(device)).cpu().numpy().reshape(ny_, nx_, -1)
    ref_slice = ref.u[t_idx]
    colors = ["#d62728", "#1f77b4", "#2ca02c"]

    route_colors = ["#d62728", "#1f77b4", "#2ca02c"]
    route_cmap = ListedColormap(route_colors)
    labels = ["Interior (+1)", "Exterior (-1)", "Interface"]

    # ---- figure 1: routing maps + reference ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for ax, route, title in [
        (axes[0], st_route, "Staged routing"),
        (axes[1], co_route, "Coadaptation routing"),
    ]:
        im = ax.imshow(route, origin="lower", extent=(-1, 1, -1, 1),
                       cmap=route_cmap, vmin=0, vmax=2)
        ax.set_title(f"{title} @ t={ref.t[t_idx]:.3f}")
        ax.set_xlabel("x"); ax.set_ylabel("y")
    axes[0].legend(
        handles=[Patch(color=c, label=l) for c, l in zip(route_colors, labels)],
        loc="lower right", fontsize=8, framealpha=0.9,
    )
    im2 = axes[2].imshow(ref_slice, origin="lower", extent=(-1, 1, -1, 1),
                         cmap="RdBu_r", vmin=-1, vmax=1)
    axes[2].set_title(f"Reference u @ t={ref.t[t_idx]:.3f}")
    axes[2].set_xlabel("x"); axes[2].set_ylabel("y")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)
    fig.savefig(out_dir / "allen_cahn_routing.png", dpi=150)
    plt.close(fig)

    # ---- figure 2: interface-expert error + mixture error, shared scales ----
    st_int_err = np.abs(st_preds[..., 2] - ref_slice)
    co_int_err = np.abs(co_preds[..., 2] - ref_slice)
    err_max = max(st_int_err.max(), co_int_err.max())
    with torch.no_grad():
        st_mix = np.abs(
            st_model(torch.tensor(coords_np[idx], dtype=torch.float32).to(device)).cpu().numpy().reshape(ny_, nx_)
            - ref_slice
        )
        co_mix = np.abs(
            co_model(torch.tensor(coords_np[idx], dtype=torch.float32).to(device)).cpu().numpy().reshape(ny_, nx_)
            - ref_slice
        )
    mix_max = max(st_mix.max(), co_mix.max())
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    panels = [
        (axes[0, 0], st_int_err, err_max, "Staged: interface-expert error"),
        (axes[0, 1], co_int_err, err_max, "Coadapt: interface-expert error"),
        (axes[1, 0], st_mix, mix_max, "Staged: mixture error"),
        (axes[1, 1], co_mix, mix_max, "Coadapt: mixture error"),
    ]
    for ax, data, vmax, title in panels:
        im = ax.imshow(data, origin="lower", extent=(-1, 1, -1, 1),
                       cmap="inferno", vmin=0, vmax=float(vmax))
        ax.set_title(title)
        ax.set_xlabel("x"); ax.set_ylabel("y")
    for ax in (axes[0, 0], axes[0, 1]):
        fig.colorbar(ax.images[0], ax=ax, fraction=0.046)
    fig.colorbar(axes[1, 0].images[0], ax=axes[1, 0], fraction=0.046)
    fig.colorbar(axes[1, 1].images[0], ax=axes[1, 1], fraction=0.046)
    fig.savefig(out_dir / "allen_cahn_expert_errors.png", dpi=150)
    plt.close(fig)

    # bar comparison of headline metrics
    keys = ["l2_mixed", "effective_experts", "error_cancellation", "region_l2_interface", "interface_mixed_l2"]
    labels = ["Mixed L2", "Eff. experts", "Cancellation", "Interface expert region L2", "Interface mixture L2"]
    st_vals = [st_m[k] for k in keys]
    co_vals = [co_m[k] for k in keys]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.2, st_vals, 0.4, label="staged", color="#2a9d8f")
    ax.bar(x + 0.2, co_vals, 0.4, label="coadapt", color="#e76f51")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("Allen-Cahn: staged vs coadaptation (seed 42)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "allen_cahn_metrics_bar.png", dpi=150)
    plt.close(fig)


def main() -> None:
    run_dirs = sorted(
        [
            p for p in RESULTS_DIR.glob("*_seed*")
            if (p / "checkpoint.pt").exists()
            and re.fullmatch(r"(staged|coadapt)_seed(4[2-9]|5[01])", p.name)
        ],
        key=lambda p: (
            p.name.split("_")[0],
            int(re.search(r"seed(\d+)", p.name).group(1)),
            p.name,
        ),
    )
    rows = []
    for run_dir in run_dirs:
        metrics, _, _ = load_run(run_dir)
        rows.append(metrics)
        print(f"{metrics['run']:32s} mixed={metrics['l2_mixed']:.4f} "
              f"worst={metrics['worst_expert_l2']:.4f} eff={metrics['effective_experts']:.3f} "
              f"canc={metrics['error_cancellation']:.3f} "
              f"intf_reg={metrics['region_l2_interface']:.4f}")
    (RESULTS_DIR / "summary.json").write_text(json.dumps(rows, indent=2))
    with open(RESULTS_DIR / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    keys = [
        "l2_mixed", "l2_oracle", "routing_regret", "worst_expert_l2",
        "effective_experts", "min_load", "error_cancellation",
        "region_l2_interface", "interface_mixed_l2",
        "load_interior", "load_exterior", "load_interface",
    ]
    agg = {}
    for mode in ("staged", "coadapt"):
        mode_rows = [r for r in rows if r["mode"] == mode]
        agg[mode] = {
            k: {
                "mean": float(np.mean([r[k] for r in mode_rows])),
                "std": float(np.std([r[k] for r in mode_rows])),
            }
            for k in keys
        }
    (RESULTS_DIR / "summary_aggregate.json").write_text(json.dumps(agg, indent=2))
    print("\naggregate (mean +/- std)")
    print(f"{'metric':26s} {'staged':>14s} {'coadapt':>14s}")
    for k in keys:
        print(f"{k:26s} {agg['staged'][k]['mean']:6.4f}+/-{agg['staged'][k]['std']:.4f}"
              f"  {agg['coadapt'][k]['mean']:6.4f}+/-{agg['coadapt'][k]['std']:.4f}")

    staged_dir = RESULTS_DIR / "staged_seed42"
    coadapt_dir = RESULTS_DIR / "coadapt_seed42"
    if staged_dir.exists() and coadapt_dir.exists():
        plot_pair(staged_dir, coadapt_dir, RESULTS_DIR)
        print("figures written to", RESULTS_DIR)


if __name__ == "__main__":
    main()
