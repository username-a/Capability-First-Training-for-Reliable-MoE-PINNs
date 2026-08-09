"""
Refresh figures for an existing Burger2D result directory after plot-style updates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Burger2D.core.models import VanillaPINN
from Burger2D.core.moe_pinn import build_burgers2d_moe
from Burger2D.visualization.plots import (
    plot_centerline_slices,
    plot_directional_diagnostics,
    plot_expert_signed_error_maps,
    plot_gating_maps,
    plot_model_metric_comparison,
    plot_snapshot_comparison,
    plot_training_curves,
)


def _predict_in_batches(fn, xyt: torch.Tensor, *, batch_size: int = 65536) -> torch.Tensor:
    owner = getattr(fn, "__self__", None)
    if owner is not None:
        batch_size = min(batch_size, int(getattr(owner, "inference_batch_size", batch_size)))
    else:
        batch_size = min(batch_size, int(getattr(fn, "inference_batch_size", batch_size)))
    outputs = []
    for start in range(0, xyt.shape[0], batch_size):
        outputs.append(fn(xyt[start:start + batch_size]))
    return torch.cat(outputs, dim=0)


def _load_reference(npz_path: str) -> dict[str, np.ndarray]:
    data = np.load(npz_path)
    return {key: data[key] for key in data.files}


def _build_eval_grid(x: np.ndarray, y: np.ndarray, t: np.ndarray, device: torch.device) -> torch.Tensor:
    xx, yy = np.meshgrid(x, y, indexing="xy")
    coords = []
    for t_value in t:
        tt = np.full_like(xx, t_value)
        coords.append(np.stack([xx, yy, tt], axis=-1).reshape(-1, 3))
    xyt = np.concatenate(coords, axis=0)
    return torch.tensor(xyt, dtype=torch.float32, device=device)


def _refresh_single_result_dir(result_dir: str, device: torch.device) -> None:
    npz_path = os.path.join(result_dir, "reference_and_prediction.npz")
    if not os.path.exists(npz_path):
        return

    arrays = _load_reference(npz_path)
    x = arrays["x"]
    y = arrays["y"]
    t = arrays["t"]
    u_ref = arrays["u_ref"]
    u_pred = arrays["u_pred"]

    checkpoint_candidates = [
        "burgers2d_vanilla.pt",
        "burgers2d_moe_end_to_end.pt",
        "burgers2d_moe_staged.pt",
    ]
    checkpoint_path = None
    for name in checkpoint_candidates:
        candidate = os.path.join(result_dir, name)
        if os.path.exists(candidate):
            checkpoint_path = candidate
            break

    if checkpoint_path is None:
        return

    if checkpoint_path.endswith("burgers2d_vanilla.pt"):
        model = VanillaPINN(
            in_dim=3,
            out_dim=1,
            hidden=96,
            depth=5,
            activation="tanh",
            output_transform="burgers2d_hard_icbc",
        )
    else:
        model = build_burgers2d_moe()
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).to(torch.float32)
    model.eval()

    history = ckpt.get("history", {})
    plot_training_curves(history, os.path.join(result_dir, "training_curves.png"))
    plot_snapshot_comparison(x, y, t, u_ref, u_pred, os.path.join(result_dir, "snapshot_comparison.png"))
    plot_centerline_slices(x, y, t, u_ref, u_pred, os.path.join(result_dir, "centerline_slices.png"))

    if hasattr(model, "gating"):
        xyt = _build_eval_grid(x, y, t, device)
        with torch.no_grad():
            gates = _predict_in_batches(model.get_gate_weights, xyt).cpu().numpy()
            expert_preds = _predict_in_batches(model.get_expert_predictions, xyt).cpu().numpy()
        gates = gates.reshape(u_ref.shape[0], u_ref.shape[1], u_ref.shape[2], -1)
        expert_preds = expert_preds.reshape(u_ref.shape[0], u_ref.shape[1], u_ref.shape[2], -1)

        plot_gating_maps(
            x,
            y,
            t,
            gates,
            expert_names=getattr(model, "expert_names", [f"expert_{i}" for i in range(gates.shape[-1])]),
            save_path=os.path.join(result_dir, "gating_maps.png"),
        )
        plot_expert_signed_error_maps(
            x,
            y,
            t,
            u_ref,
            expert_preds,
            expert_names=getattr(model, "expert_names", [f"expert_{i}" for i in range(expert_preds.shape[-1])]),
            save_path=os.path.join(result_dir, "expert_signed_error_maps.png"),
        )

        directional_json = os.path.join(result_dir, "directional_diagnostics.json")
        if os.path.exists(directional_json):
            with open(directional_json, "r", encoding="utf-8") as f:
                directional = json.load(f)
            plot_directional_diagnostics(
                np.asarray(directional["bin_centers_deg"]),
                np.asarray(directional["steep_density"]),
                np.asarray(directional["mean_gate_by_bin"]),
                expert_names=getattr(model, "expert_names", [f"expert_{i}" for i in range(gates.shape[-1])]),
                save_path=os.path.join(result_dir, "directional_diagnostics.png"),
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh figure titles/labels for an existing result directory")
    parser.add_argument(
        "--run-root",
        required=True,
        help="Path to a Burger2D result root, e.g. Burger2D/results/full_run_20260412_125627",
    )
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    run_root = os.path.abspath(args.run_root)
    device = torch.device(args.device)

    summary_json = os.path.join(run_root, "burgers2d_metrics_summary.json")
    if os.path.exists(summary_json):
        with open(summary_json, "r", encoding="utf-8") as f:
            metrics_summary = json.load(f)
        plot_model_metric_comparison(
            metrics_summary,
            os.path.join(run_root, "burgers2d_metrics_summary.png"),
        )

    for subdir in ["burgers2d_vanilla", "burgers2d_moe_end_to_end", "burgers2d_moe_staged"]:
        result_dir = os.path.join(run_root, subdir)
        if os.path.isdir(result_dir):
            _refresh_single_result_dir(result_dir, device=device)

    print(f"[OK] Refreshed figures under: {run_root}")


if __name__ == "__main__":
    main()
