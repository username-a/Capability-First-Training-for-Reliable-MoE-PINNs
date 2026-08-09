"""Train and evaluate an APINN strong baseline on the 2-D Burgers problem.

The implementation follows the public APINN/SXPINN recipe: pretrain a
partition-of-unity gate from a geometric prior, then jointly optimise the
shared trunk, subnetworks, and gate with the physics-informed objective.
Reference values are used only for periodic diagnostics and final evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Burger2D.core.apinn import APINN2D  # noqa: E402
from Burger2D.equations.burgers2d import Burgers2DProblem  # noqa: E402
from Burger2D.training.losses import LossConfig2D, PhysicsLoss2D  # noqa: E402
from Burger2D.training.staged_burgers2d import flatten_reference_solution  # noqa: E402


DTYPE = torch.float32
NU = 0.01 / np.pi


@dataclass
class APINNConfig:
    seed: int = 42
    n_subnets: int = 4
    preset: str = "matched"
    shared_width: int = 112
    shared_depth: int = 3
    subnet_width: int = 112
    subnet_depth: int = 3
    gate_width: int = 48
    gate_depth: int = 2
    gate_temperature: float = 1.0
    binary_scalar_gate: bool = False
    gate_prior: str = "spatial"
    prior_temperature: float = 0.32
    gate_pretrain_steps: int = 1000
    train_steps: int = 8000
    gate_lr: float = 1e-3
    train_lr: float = 8e-4
    weight_decay: float = 1e-3
    n_col: int = 6000
    n_ic: int = 1200
    n_bc_per_face: int = 400
    test_nx: int = 65
    test_ny: int = 65
    test_nt: int = 21
    eval_every: int = 100
    checkpoint_every: int = 1000
    w_res: float = 1.0
    w_ic: float = 10.0
    w_bc: float = 5.0


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temp, path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


def _build_model(cfg: APINNConfig, device: torch.device) -> APINN2D:
    return APINN2D(
        n_subnets=cfg.n_subnets,
        shared_width=cfg.shared_width,
        shared_depth=cfg.shared_depth,
        subnet_width=cfg.subnet_width,
        subnet_depth=cfg.subnet_depth,
        gate_width=cfg.gate_width,
        gate_depth=cfg.gate_depth,
        gate_temperature=cfg.gate_temperature,
        binary_scalar_gate=cfg.binary_scalar_gate,
    ).to(device=device, dtype=DTYPE)


def _spatial_centres(n_subnets: int, device: torch.device) -> torch.Tensor:
    side = math.ceil(math.sqrt(n_subnets))
    axis = torch.linspace(-0.7, 0.7, side, dtype=DTYPE, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)[:n_subnets]


def _gate_prior(coords: torch.Tensor, cfg: APINNConfig) -> torch.Tensor:
    if cfg.gate_prior == "uniform":
        return torch.full(
            (coords.shape[0], cfg.n_subnets),
            1.0 / cfg.n_subnets,
            dtype=coords.dtype,
            device=coords.device,
        )
    if cfg.gate_prior == "official_xpinn":
        if cfg.n_subnets != 2:
            raise ValueError("official_xpinn prior requires exactly two subnetworks")
        first = torch.exp(coords[:, 0:1] - 1.0).clamp(0.0, 1.0)
        return torch.cat([first, 1.0 - first], dim=1)
    if cfg.gate_prior != "spatial":
        raise ValueError(f"Unknown gate prior: {cfg.gate_prior}")
    centres = _spatial_centres(cfg.n_subnets, coords.device)
    squared_distance = (coords[:, None, :2] - centres[None, :, :]).square().sum(dim=-1)
    return torch.softmax(-squared_distance / max(cfg.prior_temperature, 1e-4), dim=1)


def _batched_outputs(
    model: APINN2D,
    coords: torch.Tensor,
    batch_size: int = 16384,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mixtures, branches, gates = [], [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, coords.shape[0], batch_size):
            chunk = coords[start : start + batch_size]
            branches.append(model.get_subnet_predictions(chunk).cpu())
            gates.append(model.compute_gate_weights(chunk).cpu())
            mixtures.append(model(chunk).squeeze(-1).cpu())
    return torch.cat(mixtures), torch.cat(branches), torch.cat(gates)


def _diagnostics(model: APINN2D, coords: torch.Tensor, truth: torch.Tensor) -> dict[str, Any]:
    pred, branches, gates = _batched_outputs(model, coords)
    truth_cpu = truth.detach().cpu().reshape(-1)
    sq = (branches - truth_cpu[:, None]).square()
    best_sq, best_idx = sq.min(dim=1)
    route_idx = gates.argmax(dim=1)
    hard_load = torch.bincount(route_idx, minlength=branches.shape[1]).float()
    hard_load /= hard_load.sum().clamp_min(1.0)
    soft_load = gates.mean(dim=0)
    per_subnet_l2 = [
        float(torch.linalg.vector_norm(branches[:, idx] - truth_cpu) / torch.linalg.vector_norm(truth_cpu).clamp_min(1e-10))
        for idx in range(branches.shape[1])
    ]
    weighted_expert_mse = float((gates * sq).sum(dim=1).mean().item())
    mixture_mse = float((pred - truth_cpu).square().mean().item())
    cancellation = max(weighted_expert_mse - mixture_mse, 0.0)
    entropy = -(gates * gates.clamp_min(1e-12).log()).sum(dim=1).mean()
    return {
        "mixture_l2": float(torch.linalg.vector_norm(pred - truth_cpu) / torch.linalg.vector_norm(truth_cpu).clamp_min(1e-10)),
        "mixture_mse": mixture_mse,
        "max_absolute_error": float((pred - truth_cpu).abs().max().item()),
        "per_subnet_l2": per_subnet_l2,
        "worst_subnet_l2": float(max(per_subnet_l2)),
        "soft_load": [float(value) for value in soft_load.tolist()],
        "top1_load": [float(value) for value in hard_load.tolist()],
        "effective_subnets": float(torch.exp(entropy).item()),
        "route_entropy": float(entropy.item()),
        "oracle_hit": float((route_idx == best_idx).float().mean().item()),
        "oracle_rmse": float(best_sq.mean().sqrt().item()),
        "routing_regret": float((gates * (sq - best_sq[:, None])).sum(dim=1).mean().item()),
        "weighted_expert_mse": weighted_expert_mse,
        "cancellation_mse": cancellation,
        "cancellation_ratio": float(cancellation / max(weighted_expert_mse, 1e-12)),
    }


def _save_checkpoint(path: Path, model: APINN2D, optimizer: torch.optim.Optimizer, cfg: APINNConfig, step: int) -> None:
    torch.save(
        {
            "config": asdict(cfg),
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )


def train(cfg: APINNConfig, output_dir: Path, device: torch.device) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.jsonl"
    history_path.unlink(missing_ok=True)
    _seed_everything(cfg.seed)
    started = time.time()
    problem = Burgers2DProblem(nu=NU, device=device, dtype=DTYPE, seed=cfg.seed)
    batch = problem.training_batch(cfg.n_col, cfg.n_ic, cfg.n_bc_per_face)
    reference = problem.generate_reference_solution(cfg.test_nx, cfg.test_ny, cfg.test_nt)
    test_coords, test_values = flatten_reference_solution(reference, device=device, dtype=DTYPE)
    model = _build_model(cfg, device)
    status: dict[str, Any] = {
        "state": "initialising",
        "phase": "gate_pretraining",
        "step": 0,
        "phase_step": 0,
        "total_steps": cfg.gate_pretrain_steps + cfg.train_steps,
        "started_at": started,
        "updated_at": time.time(),
        "device": str(device),
        "config": asdict(cfg),
        "parameter_counts": model.count_parameters(),
    }
    _write_json(output_dir / "config.json", status)
    _write_json(output_dir / "status.json", status)

    gate_optimizer = torch.optim.Adam(
        model.gating.parameters(), lr=cfg.gate_lr, weight_decay=cfg.weight_decay
    )
    gate_coords = torch.cat([batch["xt_col"], batch["xt_ic"], batch["xt_bc"]], dim=0)
    targets = _gate_prior(gate_coords, cfg)
    model.train()
    for step in range(1, cfg.gate_pretrain_steps + 1):
        gate_optimizer.zero_grad(set_to_none=True)
        probs = model.compute_gate_weights(gate_coords)
        if cfg.gate_prior == "official_xpinn":
            gate_loss = F.mse_loss(probs, targets)
        else:
            gate_loss = F.kl_div(probs.clamp_min(1e-8).log(), targets, reduction="batchmean")
        gate_loss.backward()
        gate_optimizer.step()
        if step == 1 or step % cfg.eval_every == 0 or step == cfg.gate_pretrain_steps:
            row = {
                "phase": "gate_pretraining",
                "step": step,
                "global_step": step,
                "elapsed_seconds": time.time() - started,
                "gate_prior_loss": float(gate_loss.item()),
                "soft_load": [float(x) for x in probs.detach().mean(dim=0).cpu().tolist()],
            }
            _append_jsonl(history_path, row)
            status.update({
                "state": "running",
                "phase": row["phase"],
                "step": row["global_step"],
                "phase_step": step,
                "updated_at": time.time(),
                "latest": row,
            })
            _write_json(output_dir / "status.json", status)

    loss_fn = PhysicsLoss2D(LossConfig2D(
        w_res=cfg.w_res,
        w_ic=cfg.w_ic,
        w_bc=cfg.w_bc,
        w_sparse=0.0,
        w_balance=0.0,
        nu=NU,
    ))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train_lr)
    last_metrics: dict[str, Any] = {}
    for step in range(1, cfg.train_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, parts = loss_fn.compute(model, batch)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        should_report = step == 1 or step % cfg.eval_every == 0 or step == cfg.train_steps
        if should_report:
            last_metrics = _diagnostics(model, test_coords, test_values)
            row = {
                "phase": "joint_physics_training",
                "step": step,
                "global_step": cfg.gate_pretrain_steps + step,
                "elapsed_seconds": time.time() - started,
                "total_loss": float(total_loss.item()),
                "residual_loss": float(parts["res"].item()),
                "initial_loss": float(parts["ic"].item()),
                "boundary_loss": float(parts["bc"].item()),
                **last_metrics,
            }
            _append_jsonl(history_path, row)
            status.update({
                "state": "running",
                "phase": row["phase"],
                "step": row["global_step"],
                "phase_step": step,
                "updated_at": time.time(),
                "latest": row,
            })
            _write_json(output_dir / "status.json", status)
        if step % cfg.checkpoint_every == 0 or step == cfg.train_steps:
            _save_checkpoint(output_dir / "checkpoint.pt", model, optimizer, cfg, step)

    results = {
        "state": "completed",
        "train_seconds": time.time() - started,
        "device": str(device),
        "parameter_counts": model.count_parameters(),
        "metrics": last_metrics,
        "config": asdict(cfg),
        "method_note": "K-way clean-room generalisation of the public two-subnetwork APINN/SXPINN recipe.",
    }
    _write_json(output_dir / "metrics.json", results)
    status.update({"state": "completed", "updated_at": time.time(), "latest": {**status.get("latest", {}), **last_metrics}})
    _write_json(output_dir / "status.json", status)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(PACKAGE_ROOT / "results" / "apinn_reproduction" / "latest"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-subnets", type=int, default=4)
    parser.add_argument("--preset", choices=("matched", "compact", "official2", "matched2"), default="matched")
    parser.add_argument("--gate-prior", choices=("spatial", "uniform", "official_xpinn"), default="spatial")
    parser.add_argument("--gate-pretrain-steps", type=int, default=1000)
    parser.add_argument("--train-steps", type=int, default=8000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = APINNConfig(
        seed=args.seed,
        n_subnets=args.n_subnets,
        preset=args.preset,
        gate_prior=args.gate_prior,
        gate_pretrain_steps=args.gate_pretrain_steps,
        train_steps=args.train_steps,
        eval_every=args.eval_every,
    )
    if args.preset == "compact":
        cfg.shared_width = 64
        cfg.subnet_width = 64
        cfg.gate_width = 32
    elif args.preset == "official2":
        cfg.n_subnets = 2
        cfg.shared_width = 20
        cfg.shared_depth = 2
        cfg.subnet_width = 20
        cfg.subnet_depth = 3
        cfg.gate_width = 20
        cfg.gate_depth = 1
        cfg.binary_scalar_gate = True
    elif args.preset == "matched2":
        # 186,429 parameters: within 0.2% of the 186,727-parameter main MoE.
        # Keep the official two-subnetwork scalar gate while matching capacity.
        cfg.n_subnets = 2
        cfg.shared_width = 143
        cfg.shared_depth = 3
        cfg.subnet_width = 143
        cfg.subnet_depth = 3
        cfg.gate_width = 48
        cfg.gate_depth = 1
        cfg.binary_scalar_gate = True
    if args.smoke:
        cfg.gate_pretrain_steps = 2
        cfg.train_steps = 3
        cfg.eval_every = 1
        cfg.checkpoint_every = 2
        cfg.n_col = 64
        cfg.n_ic = 32
        cfg.n_bc_per_face = 8
        cfg.test_nx = 17
        cfg.test_ny = 17
        cfg.test_nt = 5
        cfg.shared_width = 16
        cfg.subnet_width = 16
        cfg.gate_width = 12
        cfg.shared_depth = 2
        cfg.subnet_depth = 2

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).resolve()
    try:
        results = train(cfg, output_dir, device)
        print(json.dumps(results, ensure_ascii=False, indent=2))
    except Exception as exc:
        failure = {
            "state": "failed",
            "updated_at": time.time(),
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "config": asdict(cfg),
        }
        _write_json(output_dir / "status.json", failure)
        raise


if __name__ == "__main__":
    main()
