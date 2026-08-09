"""
Check whether the current 2D Burgers MoE architecture has real division of
labor, using saved checkpoints:

  1. per-expert errors inside steep vs background regions;
  2. oracle-best expert -> gate weight alignment (does the gate route to the
     expert that is actually best at each point?);
  3. mixture vs best-single-expert / oracle errors.

Usage:
    python Burger2D/scripts/analyze_specialization.py \
        Burger2D/results/full_compare_local_conv_route_sharp_20260414_233000/burgers2d_moe_staged
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

from Burger2D.core.moe_pinn import build_burgers2d_moe  # noqa: E402


DTYPE = torch.float32
GATE_VARIANTS = ["pointwise", "local_conv", "local_knn"]
DIRECTIONAL_VARIANTS = ["hybrid", "legacy"]
WAVE_VARIANTS = ["base", "mixed_lite", "mixed"]


def _load_grid(npz_path: str) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    x, y, t, u_ref = data["x"], data["y"], data["t"], data["u_ref"]
    xx, yy = np.meshgrid(x, y, indexing="xy")
    coords = []
    for t_value in t:
        tt = np.full_like(xx, t_value)
        coords.append(np.stack([xx, yy, tt], axis=-1).reshape(-1, 3))
    xyt = np.concatenate(coords, axis=0)
    return (
        torch.tensor(xyt, dtype=DTYPE),
        torch.tensor(u_ref.reshape(-1, 1), dtype=DTYPE),
        x,
        y,
    )


def _build_and_load(model_path: str) -> torch.nn.Module | None:
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
                ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
                try:
                    model.load_state_dict(ckpt["model_state"])
                    model.eval()
                    return model
                except (RuntimeError, KeyError):
                    continue
    return None


def _steep_mask(u_ref: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    grad_mag = np.zeros_like(u_ref, dtype=np.float32)
    for i in range(u_ref.shape[0]):
        du_dy, du_dx = np.gradient(u_ref[i], y, x, edge_order=1)
        grad_mag[i] = np.sqrt(du_dx**2 + du_dy**2)
    threshold = float(np.quantile(grad_mag, 0.90))
    return grad_mag.reshape(-1) >= threshold


def analyze_dir(run_dir: str) -> dict:
    npz_path = os.path.join(run_dir, "reference_and_prediction.npz")
    pt_paths = [
        os.path.join(run_dir, f)
        for f in os.listdir(run_dir)
        if f.startswith("burgers2d_moe_") and f.endswith(".pt")
    ]
    if not os.path.exists(npz_path) or not pt_paths:
        return {"error": "missing npz or checkpoint"}
    xyt, u_exact, x, y = _load_grid(npz_path)
    steep = _steep_mask(np.load(npz_path)["u_ref"], x, y)
    bg = ~steep

    result = {}
    for pt_path in pt_paths:
        model = _build_and_load(pt_path)
        if model is None:
            result[os.path.basename(pt_path)] = {"error": "could not load"}
            continue
        names = model.expert_names
        with torch.no_grad():
            branch = model.get_expert_predictions(xyt).cpu().numpy()[:, :, 0]  # (N, K)
            pred = []
            for start in range(0, xyt.shape[0], 65536):
                pred.append(model(xyt[start:start + 65536]).cpu().numpy())
            pred = np.concatenate(pred, axis=0)[:, 0]
            gate = model.get_gate_weights(xyt).cpu().numpy()
        u = u_exact.numpy()[:, 0]

        abs_err = np.abs(branch - u[:, None])  # (N, K)
        oracle = np.argmin(abs_err, axis=1)
        gate_oracle = gate[np.arange(len(u)), oracle]
        top1 = gate.argmax(axis=1)
        oracle_top1_match = (top1 == oracle).mean()
        n_experts = len(names)
        random_oracle_weight = 1.0 / n_experts
        gate_non_oracle = (gate.sum(axis=1) - gate_oracle) / max(n_experts - 1, 1)

        entry = {
            "expert_names": names,
            "per_expert_steep_mae": {n: float(abs_err[steep, i].mean()) for i, n in enumerate(names)},
            "per_expert_bg_mae": {n: float(abs_err[bg, i].mean()) for i, n in enumerate(names)},
            "per_expert_steep_bg_gap": {
                n: float(abs_err[steep, i].mean() - abs_err[bg, i].mean())
                for i, n in enumerate(names)
            },
            "oracle_share": {n: float((oracle == i).mean()) for i, n in enumerate(names)},
            "mean_gate_weight_to_oracle": float(gate_oracle.mean()),
            "mean_gate_weight_to_oracle_steep": float(gate_oracle[steep].mean()),
            "mean_gate_weight_to_oracle_bg": float(gate_oracle[bg].mean()),
            "gate_weight_ratio_oracle_vs_random": float(gate_oracle.mean() / random_oracle_weight),
            "mean_gate_weight_to_non_oracle": float(gate_non_oracle.mean()),
            "mean_gate_weight_per_expert_steep": {
                n: float(gate[steep, i].mean()) for i, n in enumerate(names)
            },
            "mean_gate_weight_per_expert_bg": {
                n: float(gate[bg, i].mean()) for i, n in enumerate(names)
            },
            "oracle_top1_match_rate": float(oracle_top1_match),
            "mixture_mae": float(np.abs(pred - u).mean()),
            "oracle_mix_mae": float(abs_err.min(axis=1).mean()),
            "best_single_expert_mae": float(abs_err.mean(axis=0).min()),
            "load_frac": {n: float((top1 == i).mean()) for i, n in enumerate(names)},
        }
        result[os.path.basename(pt_path)] = entry
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+")
    args = parser.parse_args()
    for run_dir in args.run_dirs:
        print("=" * 96)
        print("RUN:", run_dir)
        print(json.dumps(analyze_dir(run_dir), ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
