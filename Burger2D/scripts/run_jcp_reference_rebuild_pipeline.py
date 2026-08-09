"""Rebuild the 2-D Burgers reference with conservative WENO5 + SSP-RK3.

The pipeline is deliberately gated:
1. solve on nested grids;
2. require monotonically decreasing self-differences and a user-specified
   finest-pair tolerance;
3. only after the gate passes, re-evaluate the staged/co-adaptive and APINN
   checkpoints on a common offset subset of the finest reference.

It never retrains a neural model and never retries a failed numerical run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from Burger2D.equations.burgers2d import (  # noqa: E402
    ReferenceSolution2D,
    initial_profile_np,
)
from Burger2D.training.staged_burgers2d import flatten_reference_solution  # noqa: E402
from run_apinn_reproduction import (  # noqa: E402
    APINNConfig,
    _batched_outputs as apinn_batched_outputs,
    _build_model as build_apinn_model,
)
from run_equal_information_2x2 import (  # noqa: E402
    DTYPE,
    EqualInfoConfig,
    _batched_model_outputs,
    _build_model,
)


NU = 0.01 / np.pi
EXPERT_NAMES = ("smooth", "iso_shock", "directional_shock", "wave")


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(tmp, path)


def _weno5_left(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Left-biased WENO5 flux at all n+1 interfaces along the last axis."""
    n = values.shape[-1]
    pad_width = [(0, 0)] * values.ndim
    pad_width[-1] = (3, 3)
    v = np.pad(values, pad_width, mode="constant", constant_values=0.0)
    v0 = v[..., 0 : n + 1]
    v1 = v[..., 1 : n + 2]
    v2 = v[..., 2 : n + 3]
    v3 = v[..., 3 : n + 4]
    v4 = v[..., 4 : n + 5]

    q0 = (2.0 * v0 - 7.0 * v1 + 11.0 * v2) / 6.0
    q1 = (-v1 + 5.0 * v2 + 2.0 * v3) / 6.0
    q2 = (2.0 * v2 + 5.0 * v3 - v4) / 6.0
    b0 = (13.0 / 12.0) * (v0 - 2.0 * v1 + v2) ** 2 + 0.25 * (v0 - 4.0 * v1 + 3.0 * v2) ** 2
    b1 = (13.0 / 12.0) * (v1 - 2.0 * v2 + v3) ** 2 + 0.25 * (v1 - v3) ** 2
    b2 = (13.0 / 12.0) * (v2 - 2.0 * v3 + v4) ** 2 + 0.25 * (3.0 * v2 - 4.0 * v3 + v4) ** 2
    a0 = 0.1 / (eps + b0) ** 2
    a1 = 0.6 / (eps + b1) ** 2
    a2 = 0.3 / (eps + b2) ** 2
    total = a0 + a1 + a2
    return (a0 * q0 + a1 * q1 + a2 * q2) / total


def _conservative_flux_derivative(u: np.ndarray, spacing: float, axis: int) -> np.ndarray:
    v = np.moveaxis(u, axis, -1)
    alpha = max(float(np.max(np.abs(v))), 1e-12)
    flux = 0.5 * v * v
    f_plus = 0.5 * (flux + alpha * v)
    f_minus = 0.5 * (flux - alpha * v)
    h_plus = _weno5_left(f_plus)
    h_minus = np.flip(_weno5_left(np.flip(f_minus, axis=-1)), axis=-1)
    numerical_flux = h_plus + h_minus
    derivative = (numerical_flux[..., 1:] - numerical_flux[..., :-1]) / spacing
    return np.moveaxis(derivative, -1, axis)


def _fourth_order_second_derivative(u: np.ndarray, spacing: float, axis: int) -> np.ndarray:
    v = np.moveaxis(u, axis, -1)
    out = np.zeros_like(v)
    h2 = spacing * spacing
    out[..., 1] = (v[..., 0] - 2.0 * v[..., 1] + v[..., 2]) / h2
    out[..., -2] = (v[..., -3] - 2.0 * v[..., -2] + v[..., -1]) / h2
    out[..., 2:-2] = (
        -v[..., 4:] + 16.0 * v[..., 3:-1] - 30.0 * v[..., 2:-2]
        + 16.0 * v[..., 1:-3] - v[..., :-4]
    ) / (12.0 * h2)
    return np.moveaxis(out, -1, axis)


def _enforce_zero_boundary(u: np.ndarray) -> None:
    u[0, :] = 0.0
    u[-1, :] = 0.0
    u[:, 0] = 0.0
    u[:, -1] = 0.0


def _rhs(u: np.ndarray, dx: float, dy: float) -> np.ndarray:
    convection = _conservative_flux_derivative(u, dx, 1) + _conservative_flux_derivative(u, dy, 0)
    diffusion = _fourth_order_second_derivative(u, dx, 1) + _fourth_order_second_derivative(u, dy, 0)
    result = -convection + NU * diffusion
    result[0, :] = result[-1, :] = 0.0
    result[:, 0] = result[:, -1] = 0.0
    return result


def _ssprk3_step(u: np.ndarray, dt: float, dx: float, dy: float) -> np.ndarray:
    u1 = u + dt * _rhs(u, dx, dy)
    _enforce_zero_boundary(u1)
    u2 = 0.75 * u + 0.25 * (u1 + dt * _rhs(u1, dx, dy))
    _enforce_zero_boundary(u2)
    result = (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + dt * _rhs(u2, dx, dy))
    _enforce_zero_boundary(result)
    return result


def solve_reference(
    n: int,
    output_times: np.ndarray,
    *,
    cfl: float,
    status_path: Path,
    status: dict[str, Any],
) -> ReferenceSolution2D:
    x = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    y = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    u = initial_profile_np(xx, yy).astype(np.float64)
    _enforce_zero_boundary(u)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    outputs = np.empty((len(output_times), n, n), dtype=np.float32)
    outputs[0] = u.astype(np.float32)
    current = float(output_times[0])
    steps = 0
    started = time.perf_counter()

    for out_index in range(1, len(output_times)):
        target = float(output_times[out_index])
        while current < target - 1e-14:
            max_u = max(float(np.max(np.abs(u))), 1e-12)
            dt_adv = cfl * min(dx, dy) / max_u
            dt_diff = 0.20 / (NU * (1.0 / dx**2 + 1.0 / dy**2))
            dt = min(dt_adv, dt_diff, target - current)
            u = _ssprk3_step(u, dt, dx, dy)
            current += dt
            steps += 1
            if not np.isfinite(u).all():
                raise FloatingPointError(f"WENO reference became non-finite on n={n}, step={steps}")
            if float(np.max(np.abs(u))) > 10.0:
                raise FloatingPointError(f"WENO reference exceeded stability bound on n={n}, step={steps}")
        outputs[out_index] = u.astype(np.float32)
        status.update({
            "state": "running",
            "phase": "reference_convergence",
            "grid": n,
            "output_index": out_index,
            "output_count": len(output_times) - 1,
            "time": target,
            "steps": steps,
            "grid_elapsed_seconds": time.perf_counter() - started,
            "updated_at": time.time(),
        })
        _atomic_json(status_path, status)
    status.setdefault("grid_timings", {})[str(n)] = time.perf_counter() - started
    status.setdefault("grid_steps", {})[str(n)] = steps
    _atomic_json(status_path, status)
    return ReferenceSolution2D(x=x, y=y, t=output_times.copy(), u=outputs)


def _rel_l2(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if mask is not None:
        aa, bb = aa[mask], bb[mask]
    return float(np.linalg.norm(aa - bb) / max(np.linalg.norm(bb), 1e-14))


def _convergence_rows(references: dict[int, ReferenceSolution2D]) -> list[dict[str, Any]]:
    grids = sorted(references)
    rows: list[dict[str, Any]] = []
    for coarse_n, fine_n in zip(grids[:-1], grids[1:]):
        coarse = references[coarse_n].u
        ratio = (fine_n - 1) // (coarse_n - 1)
        if ratio * (coarse_n - 1) != fine_n - 1:
            raise ValueError("Nested grids must satisfy (fine-1)/(coarse-1) integer")
        fine = references[fine_n].u[:, ::ratio, ::ratio]
        gy, gx = np.gradient(fine.astype(np.float64), axis=(1, 2))
        grad = np.sqrt(gx * gx + gy * gy)
        steep = grad >= np.quantile(grad, 0.90)
        for region, mask in (("global", None), ("steep-gradient", steep)):
            diff = coarse.astype(np.float64) - fine.astype(np.float64)
            used = diff if mask is None else diff[mask]
            rows.append({
                "coarse_grid": coarse_n,
                "fine_grid": fine_n,
                "region": region,
                "relative_l2_difference": _rel_l2(coarse, fine, mask),
                "max_absolute_difference": float(np.max(np.abs(used))),
            })
    return rows


def _score(pred: torch.Tensor, truth: torch.Tensor) -> dict[str, float]:
    err = pred - truth
    return {
        "relative_l2": float(torch.linalg.vector_norm(err) / torch.linalg.vector_norm(truth).clamp_min(1e-12)),
        "rmse": float(err.square().mean().sqrt()),
        "max_abs": float(err.abs().max()),
    }


def _mix(branches: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
    return (branches * gates).sum(dim=1)


def _counterfactual_metrics(branches: torch.Tensor, gates: torch.Tensor, truth: torch.Tensor) -> dict[str, Any]:
    mixture = _mix(branches, gates)
    base = _score(mixture, truth)
    sqerr = (branches - truth[:, None]).square()
    denom = torch.linalg.vector_norm(truth).clamp_min(1e-12)
    per_branch = [float(torch.linalg.vector_norm(branches[:, k] - truth) / denom) for k in range(branches.shape[1])]
    weighted_mse = float((gates * sqerr).sum(dim=1).mean())
    mixture_mse = float((mixture - truth).square().mean())
    oracle = branches[torch.arange(branches.shape[0]), sqerr.argmin(dim=1)]
    mean_load = gates.mean(dim=0)
    top_idx = gates.argmax(dim=1)
    top_pred = branches[torch.arange(branches.shape[0]), top_idx]
    temp2 = gates.clamp_min(1e-12).pow(0.5)
    temp2 = temp2 / temp2.sum(dim=1, keepdim=True)
    most_loaded = int(mean_load.argmax())
    deleted = gates.clone()
    deleted[:, most_loaded] = 0.0
    deleted = deleted / deleted.sum(dim=1, keepdim=True).clamp_min(1e-12)
    top_score = _score(top_pred, truth)
    temp_score = _score(_mix(branches, temp2), truth)
    delete_score = _score(_mix(branches, deleted), truth)
    return {
        "soft": base,
        "per_branch_l2": per_branch,
        "worst_branch_l2": max(per_branch),
        "oracle_l2": _score(oracle, truth)["relative_l2"],
        "aggregation_gain": 1.0 - mixture_mse / max(weighted_mse, 1e-14),
        "soft_routing_regret": float(((gates * sqerr).sum(dim=1) - sqerr.min(dim=1).values).mean()),
        "effective_experts": float(torch.exp(-(mean_load.clamp_min(1e-12) * mean_load.clamp_min(1e-12).log()).sum())),
        "soft_load": [float(x) for x in mean_load],
        "top1": {**top_score, "ratio_vs_soft": top_score["relative_l2"] / base["relative_l2"]},
        "temperature_2": {**temp_score, "ratio_vs_soft": temp_score["relative_l2"] / base["relative_l2"]},
        "delete_highest_load": {
            **delete_score,
            "ratio_vs_soft": delete_score["relative_l2"] / base["relative_l2"],
            "deleted_expert": most_loaded,
        },
    }


def _evaluation_subset(reference: ReferenceSolution2D) -> ReferenceSolution2D:
    spatial_stride = max((len(reference.x) - 1) // 128, 1)
    x_idx = np.arange(1, len(reference.x) - 1, spatial_stride, dtype=int)[:128]
    y_idx = np.arange(1, len(reference.y) - 1, spatial_stride, dtype=int)[:128]
    t_idx = np.arange(0, len(reference.t), 2, dtype=int)
    return ReferenceSolution2D(
        x=reference.x[x_idx],
        y=reference.y[y_idx],
        t=reference.t[t_idx],
        u=reference.u[np.ix_(t_idx, y_idx, x_idx)],
    )


def _main_checkpoint_audit(reference: ReferenceSolution2D, device: torch.device, status_path: Path, status: dict[str, Any]) -> list[dict[str, Any]]:
    root = PROJECT_ROOT / "Burger2D" / "results" / "true_staged_vs_coadapt_20260806"
    coords, truth = flatten_reference_solution(reference, device=device, dtype=DTYPE)
    truth_cpu = truth.detach().cpu().reshape(-1)
    rows: list[dict[str, Any]] = []
    for seed in range(42, 52):
        for mode in ("staged", "coadapt"):
            run_dir = root / f"seed{seed}_{mode}"
            checkpoint = torch.load(run_dir / "pre_calibration_checkpoint.pt", map_location=device, weights_only=False)
            cfg = EqualInfoConfig(**checkpoint["config"])
            model = _build_model(cfg, device)
            model.load_state_dict(checkpoint["model_state"])
            _, branches, gates = _batched_model_outputs(model, coords)
            metrics = _counterfactual_metrics(branches, gates, truth_cpu)
            rows.append({"seed": seed, "mode": mode, **metrics})
            status.update({
                "state": "running",
                "phase": "main_checkpoint_audit",
                "completed_models": len(rows),
                "total_models": 20,
                "updated_at": time.time(),
            })
            _atomic_json(status_path, status)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return rows


def _apinn_checkpoint_dirs() -> list[Path]:
    root = PROJECT_ROOT / "Burger2D" / "results" / "apinn_reproduction"
    names = []
    for seed in (42, 43, 44):
        names.extend((f"matched2_seed{seed}", f"official2_seed{seed}", f"seed{seed}_spatial_matched"))
    return [root / name for name in names if (root / name / "checkpoint.pt").exists()]


def _apinn_audit(reference: ReferenceSolution2D, device: torch.device, status_path: Path, status: dict[str, Any]) -> list[dict[str, Any]]:
    coords, truth = flatten_reference_solution(reference, device=device, dtype=DTYPE)
    truth_cpu = truth.detach().cpu().reshape(-1)
    dirs = _apinn_checkpoint_dirs()
    rows: list[dict[str, Any]] = []
    for run_dir in dirs:
        checkpoint = torch.load(run_dir / "checkpoint.pt", map_location=device, weights_only=False)
        cfg = APINNConfig(**checkpoint["config"])
        model = build_apinn_model(cfg, device)
        model.load_state_dict(checkpoint["model_state"])
        _, branches, gates = apinn_batched_outputs(model, coords)
        metrics = _counterfactual_metrics(branches, gates, truth_cpu)
        if run_dir.name.startswith("matched2"):
            group = "two_subnet_parameter_matched"
        elif run_dir.name.startswith("official2"):
            group = "two_subnet_official_size"
        else:
            group = "four_subnet_parameter_matched"
        rows.append({"run": run_dir.name, "group": group, "seed": cfg.seed, **metrics})
        status.update({
            "state": "running",
            "phase": "apinn_checkpoint_audit",
            "completed_models": len(rows),
            "total_models": len(dirs),
            "updated_at": time.time(),
        })
        _atomic_json(status_path, status)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def _mean_sd(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "sample_std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0}


def _summarize(rows: list[dict[str, Any]], group_key: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for group in sorted({str(row[group_key]) for row in rows}):
        selected = [row for row in rows if str(row[group_key]) == group]
        output[group] = {
            "n": len(selected),
            "soft_relative_l2": _mean_sd([row["soft"]["relative_l2"] for row in selected]),
            "worst_branch_l2": _mean_sd([row["worst_branch_l2"] for row in selected]),
            "oracle_l2": _mean_sd([row["oracle_l2"] for row in selected]),
            "aggregation_gain": _mean_sd([row["aggregation_gain"] for row in selected]),
            "effective_experts": _mean_sd([row["effective_experts"] for row in selected]),
            "top1_ratio": _mean_sd([row["top1"]["ratio_vs_soft"] for row in selected]),
            "temperature_2_ratio": _mean_sd([row["temperature_2"]["ratio_vs_soft"] for row in selected]),
            "delete_highest_load_ratio": _mean_sd([row["delete_highest_load"]["ratio_vs_soft"] for row in selected]),
        }
    return output


def _write_flat_csv(path: Path, rows: list[dict[str, Any]], id_fields: list[str]) -> None:
    flattened: list[dict[str, Any]] = []
    for row in rows:
        item = {key: row[key] for key in id_fields}
        item.update({
            "soft_relative_l2": row["soft"]["relative_l2"],
            "worst_branch_l2": row["worst_branch_l2"],
            "oracle_l2": row["oracle_l2"],
            "aggregation_gain": row["aggregation_gain"],
            "soft_routing_regret": row["soft_routing_regret"],
            "effective_experts": row["effective_experts"],
            "top1_ratio": row["top1"]["ratio_vs_soft"],
            "temperature_2_ratio": row["temperature_2"]["ratio_vs_soft"],
            "delete_highest_load_ratio": row["delete_highest_load"]["ratio_vs_soft"],
        })
        flattened.append(item)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grids", nargs="+", type=int, default=[129, 257, 513])
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--cfl", type=float, default=0.35)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="Burger2D/results/jcp_reference_rebuild_20260808")
    args = parser.parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "status.json"
    status: dict[str, Any] = {
        "state": "running",
        "phase": "initialising",
        "started_at": time.time(),
        "updated_at": time.time(),
        "protocol": {
            "solver": "global-Lax-Friedrichs finite-difference WENO5 + SSP-RK3; fourth-order interior diffusion",
            "grids": args.grids,
            "output_times": 41,
            "cfl": args.cfl,
            "finest_pair_global_threshold": args.threshold,
            "gate": "global differences strictly decrease and finest-pair global difference <= threshold",
            "no_automatic_retry": True,
        },
    }
    _atomic_json(status_path, status)
    try:
        grids = sorted(set(args.grids))
        if len(grids) < 3:
            raise ValueError("At least three nested grids are required")
        for a, b in zip(grids[:-1], grids[1:]):
            if (b - 1) % (a - 1) != 0:
                raise ValueError(f"Grids are not nested: {a}, {b}")
        times = np.linspace(0.0, 1.0, 41, dtype=np.float64)
        references: dict[int, ReferenceSolution2D] = {}
        for n in grids:
            ref = solve_reference(n, times, cfl=args.cfl, status_path=status_path, status=status)
            references[n] = ref
            np.savez_compressed(out / f"burgers_weno5_reference_{n}.npz", x=ref.x, y=ref.y, t=ref.t, u=ref.u)

        convergence = _convergence_rows(references)
        _atomic_json(out / "convergence.json", {"rows": convergence})
        with (out / "convergence.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(convergence[0]))
            writer.writeheader()
            writer.writerows(convergence)
        global_diffs = [row["relative_l2_difference"] for row in convergence if row["region"] == "global"]
        monotone = all(right < left for left, right in zip(global_diffs[:-1], global_diffs[1:]))
        passed = monotone and global_diffs[-1] <= args.threshold
        gate = {
            "passed": passed,
            "monotonically_decreasing": monotone,
            "global_differences": global_diffs,
            "threshold": args.threshold,
        }
        _atomic_json(out / "convergence_gate.json", gate)
        status["convergence_gate"] = gate
        if not passed:
            status.update({
                "state": "blocked",
                "phase": "reference_convergence_gate",
                "reason": "Reference convergence gate failed; neural checkpoints were not re-evaluated.",
                "finished_at": time.time(),
                "updated_at": time.time(),
            })
            _atomic_json(status_path, status)
            return 0

        device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
        subset = _evaluation_subset(references[grids[-1]])
        np.savez_compressed(out / "evaluation_subset.npz", x=subset.x, y=subset.y, t=subset.t, u=subset.u)
        status["evaluation"] = {"device": str(device), "points": int(subset.u.size)}
        _atomic_json(status_path, status)

        main_rows = _main_checkpoint_audit(subset, device, status_path, status)
        main_payload = {"summary": _summarize(main_rows, "mode"), "runs": main_rows}
        _atomic_json(out / "main_checkpoint_reevaluation.json", main_payload)
        _write_flat_csv(out / "main_checkpoint_reevaluation.csv", main_rows, ["seed", "mode"])

        apinn_rows = _apinn_audit(subset, device, status_path, status)
        apinn_payload = {"summary": _summarize(apinn_rows, "group"), "runs": apinn_rows}
        _atomic_json(out / "apinn_checkpoint_reevaluation.json", apinn_payload)
        if apinn_rows:
            _write_flat_csv(out / "apinn_checkpoint_reevaluation.csv", apinn_rows, ["run", "group", "seed"])

        status.update({
            "state": "completed",
            "phase": "done",
            "finished_at": time.time(),
            "updated_at": time.time(),
            "outputs": [
                "convergence.json",
                "convergence_gate.json",
                "main_checkpoint_reevaluation.json",
                "apinn_checkpoint_reevaluation.json",
            ],
        })
        _atomic_json(status_path, status)
        return 0
    except Exception as exc:
        status.update({
            "state": "crashed",
            "phase": status.get("phase", "unknown"),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "finished_at": time.time(),
            "updated_at": time.time(),
        })
        _atomic_json(status_path, status)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
