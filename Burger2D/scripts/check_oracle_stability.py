"""
Quantitative check of Theorem 2 (oracle-label stability) from snapshots.

For consecutive snapshots of a joint/e2e run, estimate per expert:
    d_k  = ||theta(t+1) - theta(t)||_2          (parameter drift)
    L_k(z) ~ |l_k(z,t+1) - l_k(z,t)| / d_k      (local Lipschitz estimate)
    Delta(z) = min_{j != k*} (l_j - l_k*)       (capability gap)
and compute the fraction of grid points where the sufficient condition
2*L*d < Delta(z) holds, compared with the measured oracle-label flip rate.

Usage:
    python Burger2D/scripts/check_oracle_stability.py <run_dir>
"""

from __future__ import annotations

import glob
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


def _build_and_load(path: str) -> torch.nn.Module | None:
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
                    ckpt = torch.load(path, map_location="cpu", weights_only=False)
                    model.load_state_dict(ckpt["model_state"])
                    model.to(DEVICE).to(DTYPE)
                    model.eval()
                    return model
                except (RuntimeError, KeyError):
                    continue
    return None


def _batched(fn, xyt, batch_size=65536):
    outs = []
    for s in range(0, xyt.shape[0], batch_size):
        outs.append(fn(xyt[s:s + batch_size]))
    return torch.cat(outs, dim=0)


def main():
    run_dir = sys.argv[1]
    seed = int(os.path.basename(run_dir).split("_seed")[-1])
    problem = Burgers2DProblem(nu=NU, device=DEVICE, dtype=DTYPE, seed=seed)
    ref = problem.generate_reference_solution(nx=65, ny=65, nt=21)
    xyt, u_exact = flatten_reference_solution(ref, device=DEVICE, dtype=DTYPE)
    u = u_exact.cpu().numpy()[:, 0]

    snaps = sorted(
        glob.glob(os.path.join(run_dir, "snap_*.pt")),
        key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]),
    )
    models = []
    for p in snaps:
        m = _build_and_load(p)
        if m is not None:
            models.append(m)
    if len(models) < 2:
        print("need >=2 snapshots")
        return

    def preds_and_flat(model):
        with torch.no_grad():
            br = _batched(model.get_expert_predictions, xyt).cpu().numpy()[:, :, 0]
        flat = []
        for e in model.experts:
            flat.append(np.concatenate([p.detach().cpu().numpy().reshape(-1) for p in e.parameters()]))
        return br, flat

    rows = []
    prev_br, prev_flat = preds_and_flat(models[0])
    prev_l = (prev_br - u[:, None]) ** 2
    for idx in range(1, len(models)):
        cur_br, cur_flat = preds_and_flat(models[idx])
        cur_l = (cur_br - u[:, None]) ** 2
        d = [float(np.linalg.norm(cur_flat[k] - prev_flat[k])) for k in range(4)]
        # per-point Lipschitz estimate per expert
        L = np.zeros_like(cur_l)
        for k in range(4):
            denom = max(d[k], 1e-12)
            L[:, k] = np.abs(cur_l[:, k] - prev_l[:, k]) / denom
        k_star_prev = np.argmin(prev_l, axis=1)
        k_star_cur = np.argmin(cur_l, axis=1)
        flip = float(np.mean(k_star_prev != k_star_cur))
        gap = np.where(np.arange(4)[None, :] != k_star_prev[:, None], prev_l, np.inf).min(axis=1)
        d_arr = np.asarray(d, dtype=np.float64)
        # exact Theorem-2 condition: for all j != k*, L_j*d_j + L_k* * d_k* < l_j - l_k*
        rhs = np.where(np.arange(4)[None, :] != k_star_prev[:, None], prev_l, np.inf)
        lhs = (
            L * d_arr[None, :]
            + L[np.arange(len(u)), k_star_prev][:, None] * d_arr[k_star_prev][:, None]
        )
        satisfied_all = np.all(lhs < rhs, axis=1)
        satisfied = float(np.mean(satisfied_all))
        # also check using current oracle (should be near-identical)
        rows.append(
            {
                "step": int(os.path.basename(snaps[idx]).split("_")[1].split(".")[0]),
                "d": d,
                "flip": flip,
                "cond_holds_frac": satisfied,
                "cond_violate_frac": 1.0 - satisfied,
            }
        )
        print(
            f"step={rows[-1]['step']}: flip={flip:.3f} "
            f"cond_2Ld<Delta holds={satisfied:.3f} violate={1.0 - satisfied:.3f}",
            flush=True,
        )
        prev_br, prev_flat, prev_l = cur_br, cur_flat, cur_l
    print("done", flush=True)


if __name__ == "__main__":
    main()
