"""Counterfactual robustness diagnostics for a trained APINN checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Burger2D.equations.burgers2d import Burgers2DProblem  # noqa: E402
from Burger2D.scripts.run_apinn_reproduction import (  # noqa: E402
    APINNConfig,
    DTYPE,
    NU,
    _batched_outputs,
    _build_model,
    _write_json,
)
from Burger2D.training.staged_burgers2d import flatten_reference_solution  # noqa: E402


def _score(pred: torch.Tensor, truth: torch.Tensor) -> dict[str, float]:
    error = pred - truth
    return {
        "relative_l2": float(torch.linalg.vector_norm(error) / torch.linalg.vector_norm(truth).clamp_min(1e-10)),
        "rmse": float(error.square().mean().sqrt()),
        "max_absolute_error": float(error.abs().max()),
    }


def _mix(branches: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (branches * weights).sum(dim=1)


def analyze(
    run_dir: Path,
    device: torch.device,
    *,
    nx: int = 82,
    ny: int = 83,
    nt: int = 32,
    noise_repeats: int = 8,
) -> dict[str, Any]:
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location=device, weights_only=False)
    cfg = APINNConfig(**checkpoint["config"])
    model = _build_model(cfg, device)
    model.load_state_dict(checkpoint["model_state"])
    problem = Burgers2DProblem(nu=NU, device=device, dtype=DTYPE, seed=cfg.seed + 90000)
    reference = problem.generate_reference_solution(nx, ny, nt)
    coords, truth_device = flatten_reference_solution(reference, device=device, dtype=DTYPE)
    soft_pred, branches, gates = _batched_outputs(model, coords)
    truth = truth_device.detach().cpu().reshape(-1)
    n_points, n_subnets = branches.shape
    row_idx = torch.arange(n_points)
    top1_idx = gates.argmax(dim=1)
    sq = (branches - truth[:, None]).square()
    oracle_idx = sq.argmin(dim=1)

    scenarios: dict[str, dict[str, float]] = {
        "soft_gate": _score(soft_pred, truth),
        "top1_hard_gate": _score(branches[row_idx, top1_idx], truth),
        "oracle_hard_gate": _score(branches[row_idx, oracle_idx], truth),
        "uniform_average": _score(branches.mean(dim=1), truth),
    }
    for temperature in (0.5, 2.0):
        adjusted = gates.clamp_min(1e-10).pow(1.0 / temperature)
        adjusted /= adjusted.sum(dim=1, keepdim=True)
        scenarios[f"gate_temperature_{temperature:g}"] = _score(_mix(branches, adjusted), truth)

    rolled = gates.roll(shifts=1, dims=1)
    scenarios["permuted_gate"] = _score(_mix(branches, rolled), truth)

    drop_results: dict[str, Any] = {}
    for idx in range(n_subnets):
        dropped = gates.clone()
        dropped[:, idx] = 0.0
        dropped /= dropped.sum(dim=1, keepdim=True).clamp_min(1e-10)
        drop_results[f"subnet_{idx + 1}"] = {
            **_score(_mix(branches, dropped), truth),
            "soft_load_removed": float(gates[:, idx].mean()),
        }

    generator = torch.Generator().manual_seed(cfg.seed + 77123)
    noise_results: dict[str, Any] = {}
    log_gates = gates.clamp_min(1e-10).log()
    for sigma in (0.1, 0.25, 0.5):
        scores = []
        for _ in range(noise_repeats):
            noisy = torch.softmax(log_gates + sigma * torch.randn(log_gates.shape, generator=generator), dim=1)
            scores.append(_score(_mix(branches, noisy), truth)["relative_l2"])
        noise_results[f"sigma_{sigma:g}"] = {
            "relative_l2_mean": float(np.mean(scores)),
            "relative_l2_std": float(np.std(scores)),
            "relative_l2_max": float(np.max(scores)),
        }

    routed_regions: dict[str, Any] = {}
    for idx in range(n_subnets):
        mask = top1_idx == idx
        if not bool(mask.any()):
            routed_regions[f"subnet_{idx + 1}"] = {"points": 0, "load": 0.0}
            continue
        branch_error = branches[mask, idx] - truth[mask]
        mixture_error = soft_pred[mask] - truth[mask]
        routed_regions[f"subnet_{idx + 1}"] = {
            "points": int(mask.sum()),
            "load": float(mask.float().mean()),
            "branch_rmse": float(branch_error.square().mean().sqrt()),
            "mixture_rmse": float(mixture_error.square().mean().sqrt()),
            "branch_pointwise_win_rate": float((branch_error.abs() < mixture_error.abs()).float().mean()),
        }

    baseline_l2 = scenarios["soft_gate"]["relative_l2"]
    for values in scenarios.values():
        values["l2_ratio_vs_soft"] = float(values["relative_l2"] / max(baseline_l2, 1e-12))
    for values in drop_results.values():
        values["l2_ratio_vs_soft"] = float(values["relative_l2"] / max(baseline_l2, 1e-12))

    result = {
        "seed": cfg.seed,
        "grid": [nx, ny, nt],
        "points": n_points,
        "scenarios": scenarios,
        "drop_one_subnet": drop_results,
        "gate_logit_noise": noise_results,
        "routed_region_diagnostics": routed_regions,
        "interpretation_guardrail": (
            "Soft-over-hard improvement establishes dependence on convex mixing, but it is not by itself proof of pathological pseudo-fitting. "
            "Use seed stability, perturbation sensitivity, branch health, and extrapolation jointly."
        ),
    }
    _write_json(run_dir / "counterfactual_metrics.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--nx", type=int, default=82)
    parser.add_argument("--ny", type=int, default=83)
    parser.add_argument("--nt", type=int, default=32)
    args = parser.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    result = analyze(Path(args.run_dir).resolve(), device, nx=args.nx, ny=args.ny, nt=args.nt)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

