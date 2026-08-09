"""
Quantify expert convergence / collapse from saved Burger2D checkpoints.

For each run directory containing a `burgers2d_moe_*.pt` checkpoint and a
`reference_and_prediction.npz`, this script:

  1. rebuilds the MoE-PINN architecture (auto-detects pointwise vs local_conv gate),
  2. evaluates expert branch predictions on the saved test grid,
  3. reports pairwise Pearson correlation / cosine similarity of expert outputs
     (both full branch predictions and residual corrections),
  4. reports per-expert errors, gate load fractions, route entropy, and the
     effective number of experts (participation ratio).

Usage:
    python Burger2D/scripts/analyze_expert_convergence.py \
        Burger2D/results/full_compare_local_conv_route_sharp_20260414_233000
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
from Burger2D.training.losses import l2_relative_error, max_absolute_error  # noqa: E402


DTYPE = torch.float32
GATE_VARIANTS = ["pointwise", "local_conv", "local_knn"]
DIRECTIONAL_VARIANTS = ["hybrid", "legacy"]
WAVE_VARIANTS = ["base", "mixed_lite", "mixed"]


def _load_grid(npz_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    data = np.load(npz_path)
    x = data["x"]
    y = data["y"]
    t = data["t"]
    u_ref = data["u_ref"]
    xx, yy = np.meshgrid(x, y, indexing="xy")
    coords = []
    for t_value in t:
        tt = np.full_like(xx, t_value)
        coords.append(np.stack([xx, yy, tt], axis=-1).reshape(-1, 3))
    xyt = np.concatenate(coords, axis=0)
    return (
        torch.tensor(xyt, dtype=DTYPE),
        torch.tensor(u_ref.reshape(-1, 1), dtype=DTYPE),
    )


def _build_and_load(
    model_path: str,
    gate_variant: str,
    directional_variant: str,
    wave_variant: str,
) -> torch.nn.Module | None:
    model = build_burgers2d_moe(
        directional_expert_variant=directional_variant,
        wave_expert_variant=wave_variant,
        expert_layout_variant="categorical",
        attribute_expert_variant="base",
        gate_variant=gate_variant,
        rotation_variant="none",
    )
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt)
    try:
        model.load_state_dict(state)
    except (RuntimeError, KeyError) as exc:
        print(f"    [gate={gate_variant}] state dict mismatch: {type(exc).__name__}")
        return None
    model.eval()
    return model


def _expert_outputs(
    model: torch.nn.Module,
    xyt: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (full branch predictions (N,K), residual corrections (N,K))."""
    with torch.no_grad():
        branch = model.get_expert_predictions(xyt).cpu().numpy()  # (N, K, 1)
        corr = model.get_expert_corrections(xyt).cpu().numpy()  # (N, K, 1)
    return branch[:, :, 0], corr[:, :, 0]


def _pairwise_correlation(outputs: np.ndarray) -> np.ndarray:
    n = outputs.shape[1]
    corr = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i, n):
            a, b = outputs[:, i], outputs[:, j]
            std_a, std_b = a.std(), b.std()
            if std_a < 1e-12 or std_b < 1e-12:
                corr[i, j] = 0.0
            else:
                corr[i, j] = float(np.corrcoef(a, b)[0, 1])
            corr[j, i] = corr[i, j]
    return corr


def _pairwise_cosine(outputs: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(outputs, axis=0, keepdims=True)
    safe = outputs / np.maximum(norm, 1e-12)
    return safe.T @ safe


def _parameter_cosine(model: torch.nn.Module) -> np.ndarray:
    """Pairwise cosine similarity of expert params, averaged over shape-compatible layers."""
    n = len(model.experts)
    sim = np.zeros((n, n))
    counts = np.zeros((n, n))
    for i in range(n):
        params_i = list(model.experts[i].named_parameters())
        for j in range(i, n):
            params_j = dict(model.experts[j].named_parameters())
            acc = 0.0
            cnt = 0
            for name, p in params_i:
                if name not in params_j:
                    continue
                q = params_j[name]
                if tuple(p.shape) != tuple(q.shape):
                    continue
                a = p.detach().cpu().numpy().reshape(-1)
                b = q.detach().cpu().numpy().reshape(-1)
                na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
                if na < 1e-12 or nb < 1e-12:
                    continue
                acc += float(a @ b) / (na * nb)
                cnt += 1
            sim[i, j] = acc / cnt if cnt else 0.0
            sim[j, i] = sim[i, j]
            counts[i, j] = cnt
    return sim


def _participation_ratio(outputs: np.ndarray) -> float:
    """Effective number of independent output dimensions (SVD participation ratio)."""
    centered = outputs - outputs.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    s2 = s**2
    total = s2.sum()
    if total < 1e-24:
        return 0.0
    return float(total**2 / (s2**2).sum())


def _summarize_matrix(mat: np.ndarray, names: list[str]) -> dict:
    n = mat.shape[0]
    off_diag = [mat[i, j] for i in range(n) for j in range(i + 1, n)]
    return {
        "matrix": [[float(mat[i, j]) for j in range(n)] for i in range(n)],
        "mean_off_diag": float(np.mean(off_diag)),
        "max_off_diag": float(np.max(off_diag)),
        "min_off_diag": float(np.min(off_diag)),
        "pairs": [f"{names[i]}-{names[j]}" for i in range(n) for j in range(i + 1, n)],
    }


def analyze_dir(run_dir: str) -> dict:
    result: dict = {"run_dir": run_dir, "models": {}}

    # Locate model directories: a dir that contains both a checkpoint and an npz.
    model_dirs: list[tuple[str, str]] = []  # (dir, npz_path)
    for root, dirs, files in os.walk(run_dir):
        npz_path = os.path.join(root, "reference_and_prediction.npz")
        if os.path.exists(npz_path) and any(
            f.startswith("burgers2d_moe_") and f.endswith(".pt") for f in files
        ):
            model_dirs.append((root, npz_path))
        dirs[:] = [d for d in dirs if d not in ("__pycache__",)]

    if not model_dirs:
        result["error"] = "no model dir with checkpoint + npz found"
        return result

    for model_dir, npz_path in model_dirs:
        xyt, u_exact = _load_grid(npz_path)
        batch_size = 65536
        pt_files = sorted(
            f
            for f in os.listdir(model_dir)
            if f.startswith("burgers2d_moe_") and f.endswith(".pt")
        )
        rel_dir = os.path.relpath(model_dir, run_dir)
        result.setdefault("models", {})[rel_dir] = {}
        for pt_name in pt_files:
            model_path = os.path.join(model_dir, pt_name)
            loaded = None
            gate_used = None
            dir_used = None
            wave_used = None
            for gate_variant in GATE_VARIANTS:
                for directional_variant in DIRECTIONAL_VARIANTS:
                    for wave_variant in WAVE_VARIANTS:
                        candidate = _build_and_load(
                            model_path,
                            gate_variant,
                            directional_variant,
                            wave_variant,
                        )
                        if candidate is not None:
                            loaded = candidate
                            gate_used = gate_variant
                            dir_used = directional_variant
                            wave_used = wave_variant
                            break
                        if loaded is not None:
                            break
                if loaded is not None:
                    break
            if loaded is None:
                result["models"][rel_dir][pt_name] = {"error": "could not load state dict"}
                continue

            branch, corr_out = _expert_outputs(loaded, xyt)
            names = loaded.expert_names

            with torch.no_grad():
                pred = []
                for start in range(0, xyt.shape[0], batch_size):
                    pred.append(loaded(xyt[start:start + batch_size]).cpu())
                pred = torch.cat(pred, dim=0).numpy()

            per_expert_l2 = []
            per_expert_max = []
            for k in range(branch.shape[1]):
                per_expert_l2.append(
                    float(l2_relative_error(torch.tensor(branch[:, k:k + 1]), u_exact))
                )
                per_expert_max.append(
                    float(max_absolute_error(torch.tensor(branch[:, k:k + 1]), u_exact))
                )

            load_stats = loaded.load_balance_stats(xyt)
            load_frac = load_stats["expert_load_frac"]
            entropy = load_stats["mean_entropy"]
            max_weight = load_stats["max_gate_weight"]
            load_arr = np.asarray(load_frac, dtype=np.float64)
            effective_experts = float(1.0 / np.sum(load_arr**2))
            mixture_l2 = float(l2_relative_error(torch.tensor(pred), u_exact))
            mixture_max = float(max_absolute_error(torch.tensor(pred), u_exact))

            branch_corr = _pairwise_correlation(branch)
            branch_cos = _pairwise_cosine(branch)
            corr_corr = _pairwise_correlation(corr_out)
            corr_cos = _pairwise_cosine(corr_out)
            param_cos = _parameter_cosine(loaded)
            branch_pr = _participation_ratio(branch)
            corr_pr = _participation_ratio(corr_out)

            result["models"][rel_dir][pt_name] = {
                "gate_variant_detected": gate_used,
                "directional_variant_detected": dir_used,
                "wave_variant_detected": wave_used,
                "mixture_l2_relative_error": mixture_l2,
                "mixture_max_absolute_error": mixture_max,
                "per_expert_l2_relative_error": dict(zip(names, per_expert_l2)),
                "per_expert_max_absolute_error": dict(zip(names, per_expert_max)),
                "expert_load_frac": dict(zip(names, load_frac)),
                "route_entropy": entropy,
                "route_max_weight": max_weight,
                "effective_experts": effective_experts,
                "branch_correlation": _summarize_matrix(branch_corr, names),
                "branch_cosine": _summarize_matrix(branch_cos, names),
                "correction_correlation": _summarize_matrix(corr_corr, names),
                "correction_cosine": _summarize_matrix(corr_cos, names),
                "parameter_cosine": _summarize_matrix(param_cos, names),
                "branch_participation_ratio": branch_pr,
                "correction_participation_ratio": corr_pr,
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    all_results = {}
    for run_dir in args.run_dirs:
        print(f"\n=== {run_dir} ===")
        res = analyze_dir(run_dir)
        all_results[os.path.basename(run_dir.rstrip("\\/"))] = res
        print(json.dumps(res, ensure_ascii=False, indent=2, default=float))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2, default=float)
        print(f"\n[OK] Saved: {args.out}")


if __name__ == "__main__":
    main()
