"""
Compute the theory-section quantities from mechanism-training snapshots.

For every snapshot (full model state every 200 steps) of an e2e/joint 8000
run, evaluate on the snapshot grid and produce per-step:
    orc_mse (eps_orc^2), eta, zeta, cancel_ratio,
    capability-gap quantiles (overall + steep), soft-weight mean/RMS,
    oracle-label flip rate (adjacent snapshots and vs Stage-B endpoint),
    mixture MSE (to verify the global error bounds, Eq 29/31).

Usage:
    python Burger2D/scripts/compute_theory_metrics.py <run_dir> [--out path]
"""

from __future__ import annotations

import argparse
import glob
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
from Burger2D.equations.burgers2d import Burgers2DProblem  # noqa: E402
from Burger2D.training.staged_burgers2d import flatten_reference_solution  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
NU = 0.01 / np.pi


def _build_and_load(model_path: str) -> torch.nn.Module | None:
    for gate in ("pointwise", "local_conv", "local_knn"):
        for directional in ("hybrid", "legacy"):
            for wave in ("base", "mixed_lite", "mixed"):
                model = build_burgers2d_moe(
                    directional_expert_variant=directional,
                    wave_expert_variant=wave,
                    expert_layout_variant="categorical",
                    attribute_expert_variant="base",
                    gate_variant=gate,
                    rotation_variant="none",
                )
                try:
                    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
                    model.load_state_dict(ckpt["model_state"])
                    model.to(DEVICE).to(DTYPE)
                    model.eval()
                    return model
                except (RuntimeError, KeyError):
                    continue
    return None


def _batched(fn, xyt: torch.Tensor, batch_size: int = 65536) -> torch.Tensor:
    outs = []
    for start in range(0, xyt.shape[0], batch_size):
        outs.append(fn(xyt[start:start + batch_size]))
    return torch.cat(outs, dim=0)


def _steep_mask(reference) -> np.ndarray:
    u = reference.u
    grad_mag = np.zeros_like(u, dtype=np.float32)
    for i in range(u.shape[0]):
        du_dy, du_dx = np.gradient(u[i], reference.y, reference.x, edge_order=1)
        grad_mag[i] = np.sqrt(du_dx**2 + du_dy**2)
    return grad_mag.reshape(-1) >= float(np.quantile(grad_mag, 0.90))


def compute_snapshot_metrics(model, xyt, u_exact, steep) -> dict:
    with torch.no_grad():
        branch = _batched(model.get_expert_predictions, xyt).cpu().numpy()[:, :, 0]  # (N,K)
        gates = _batched(model.get_gate_weights, xyt).cpu().numpy()  # (N,K)
        mix = _batched(model, xyt).cpu().numpy()[:, 0]
    u = u_exact.cpu().numpy()[:, 0]
    l = (branch - u[:, None]) ** 2  # (N,K) pointwise expert losses
    k_star = np.argmin(l, axis=1)
    orc_mse = float(np.mean(l[np.arange(len(u)), k_star]))
    g_oracle = gates[np.arange(len(u)), k_star]
    eta = float(np.mean(1.0 - g_oracle))
    zeta = float(np.mean(np.sum(np.where(np.arange(4)[None, :] != k_star[:, None], gates * l, 0.0), axis=1)))
    e_weighted = float(np.mean(np.sum(gates * l, axis=1)))
    mix_mse = float(np.mean((mix - u) ** 2))
    cancel = (e_weighted - mix_mse) / max(e_weighted, 1e-12)
    gap = np.where(np.arange(4)[None, :] != k_star[:, None], l, np.inf).min(axis=1)
    gap_steep = gap[steep]
    m_bound = float(np.max(np.abs(branch - u[:, None])))
    bound_lo = orc_mse + zeta
    bound_hi = orc_mse + m_bound**2 * eta
    soft_mean = [float(np.mean(gates[:, k])) for k in range(4)]
    soft_rms = [float(np.sqrt(np.mean(gates[:, k] ** 2))) for k in range(4)]
    return {
        "orc_mse": orc_mse,
        "eta": eta,
        "zeta": zeta,
        "mix_mse": mix_mse,
        "cancel_ratio": float(cancel),
        "gap_q10": float(np.quantile(gap, 0.10)),
        "gap_q50": float(np.quantile(gap, 0.50)),
        "gap_q90": float(np.quantile(gap, 0.90)),
        "gap_steep_q50": float(np.quantile(gap_steep, 0.50)),
        "soft_mean": soft_mean,
        "soft_rms": soft_rms,
        "m_bound": float(m_bound),
        "bound_holds_lo": float(mix_mse <= bound_lo + 1e-8),
        "bound_holds_hi": float(mix_mse <= bound_hi + 1e-8),
        "oracle_index": k_star.astype(np.int16),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--out", default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    seed = int(os.path.basename(args.run_dir).split("_seed")[-1])
    problem = Burgers2DProblem(nu=NU, device=DEVICE, dtype=DTYPE, seed=seed)
    reference = problem.generate_reference_solution(nx=65, ny=65, nt=21)
    xyt, u_exact = flatten_reference_solution(reference, device=DEVICE, dtype=DTYPE)
    steep = _steep_mask(reference)

    snap_paths = sorted(
        glob.glob(os.path.join(args.run_dir, "snap_*.pt")),
        key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]),
    )
    if args.limit:
        snap_paths = snap_paths[: args.limit]
    rows = []
    prev_oracle = None
    endpoint_path = os.path.join(args.run_dir, "pretrain", "burgers2d_moe_staged", "burgers2d_pre_joint.pt")
    endpoint_model = _build_and_load(endpoint_path) if os.path.exists(endpoint_path) else None
    endpoint_oracle = None
    if endpoint_model is not None:
        with torch.no_grad():
            eb = _batched(endpoint_model.get_expert_predictions, xyt).cpu().numpy()[:, :, 0]
        endpoint_oracle = np.argmin((eb - u_exact.cpu().numpy()[:, 0][:, None]) ** 2, axis=1)

    for path in snap_paths:
        step = int(os.path.basename(path).split("_")[1].split(".")[0])
        model = _build_and_load(path)
        if model is None:
            print(f"[skip] step {step}: could not load", flush=True)
            continue
        m = compute_snapshot_metrics(model, xyt, u_exact, steep)
        m["step"] = step
        m["flip_adjacent"] = (
            float(np.mean(prev_oracle != m["oracle_index"]))
            if prev_oracle is not None else 0.0
        )
        m["flip_from_endpoint"] = (
            float(np.mean(endpoint_oracle != m["oracle_index"]))
            if endpoint_oracle is not None else 0.0
        )
        prev_oracle = m["oracle_index"]
        m.pop("oracle_index")
        rows.append(m)
        print(
            f"step={step}: orc_mse={m['orc_mse']:.4e} eta={m['eta']:.4f} "
            f"zeta={m['zeta']:.4e} cancel={m['cancel_ratio']:.3f} "
            f"flip_adj={m['flip_adjacent']:.3f} flip_ep={m['flip_from_endpoint']:.3f} "
            f"bound_lo={m['bound_holds_lo']} bound_hi={m['bound_holds_hi']}",
            flush=True,
        )

    out = args.out or os.path.join(args.run_dir, "theory_metrics.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] saved {out} ({len(rows)} snapshots)", flush=True)


if __name__ == "__main__":
    main()
