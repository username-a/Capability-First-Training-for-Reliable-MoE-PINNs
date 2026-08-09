"""Controlled sentinel-expert experiment using frozen 2x2 R-B experts."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Burger2D.equations.burgers2d import Burgers2DProblem, ReferenceSolution2D  # noqa: E402
from Burger2D.scripts.run_equal_information_2x2 import (  # noqa: E402
    DTYPE,
    NU,
    EqualInfoConfig,
    _build_model,
    _coords_hash,
    _set_trainable,
    evaluate_checkpoint,
)
from Burger2D.training.losses import LossConfig2D, PhysicsLoss2D  # noqa: E402
from Burger2D.training.staged_burgers2d import flatten_reference_solution  # noqa: E402


OBJECTIVES = {
    "physics_no_balance": 0.0,
    "physics_balance": 0.01,
    "capability_aware": None,
}
VARIANTS = ("healthy", "random_directional")


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _hash_state(state) -> str:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _load_frozen_source(
    source_dir: Path,
    variant: str,
    device: torch.device,
) -> tuple[torch.nn.Module, EqualInfoConfig, dict]:
    source = torch.load(source_dir / "train_checkpoint.pt", map_location=device, weights_only=False)
    cfg = EqualInfoConfig(**source["config"])
    # Fresh build supplies the same gate initialization to every probe cell.
    _seed(cfg.seed)
    model = _build_model(cfg, device)
    source_state = source["model_state"]
    partial = {}
    for key, value in source_state.items():
        if key.startswith("base_model."):
            partial[key] = value
        elif key.startswith("experts."):
            if variant == "random_directional" and key.startswith("experts.2."):
                continue
            partial[key] = value
    incompatible = model.load_state_dict(partial, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"Unexpected source keys: {unexpected}")
    _set_trainable(model, base=False, experts=False, gate=True)
    return model, cfg, source


def _coordinate_pool(cfg: EqualInfoConfig, device: torch.device) -> tuple[ReferenceSolution2D, torch.Tensor]:
    x = np.linspace(-1.0, 1.0, cfg.teacher_nx, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, cfg.teacher_ny, dtype=np.float32)
    t = np.linspace(0.0, 1.0, cfg.teacher_nt, dtype=np.float32)
    blank = ReferenceSolution2D(x=x, y=y, t=t, u=np.zeros((len(t), len(y), len(x)), dtype=np.float32))
    coords, _ = flatten_reference_solution(blank, device=device, dtype=DTYPE)
    return blank, coords


def run_probe(
    source_dir: Path,
    output_dir: Path,
    variant: str,
    objective: str,
    device: torch.device,
    steps: int,
) -> None:
    if variant not in VARIANTS or objective not in OBJECTIVES:
        raise ValueError((variant, objective))
    output_dir.mkdir(parents=True, exist_ok=True)
    model, cfg, source = _load_frozen_source(source_dir, variant, device)
    started_at = time.time()
    gate_initial_hash = _hash_state(model.gating.state_dict())
    problem = Burgers2DProblem(nu=NU, device=device, dtype=DTYPE, seed=cfg.seed + 3000)
    physics_batch = problem.training_batch(cfg.n_col, cfg.n_ic, cfg.n_bc_per_face)
    _, pool_coords = _coordinate_pool(cfg, device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(cfg.seed + 4000)
    count = min(cfg.gate_pool_size, pool_coords.shape[0])
    idx = torch.randperm(pool_coords.shape[0], generator=generator)[:count].to(device)
    gate_coords = pool_coords[idx]
    np.save(output_dir / "training_pool_coords.npy", gate_coords.detach().cpu().numpy())
    optimizer = torch.optim.Adam(model.gating.parameters(), lr=cfg.gate_lr)
    history = []
    teacher_queries = 0
    if objective == "capability_aware":
        teacher = problem.generate_reference_solution(cfg.teacher_nx, cfg.teacher_ny, cfg.teacher_nt)
        _, values = flatten_reference_solution(teacher, device=device, dtype=DTYPE)
        gate_values = values[idx]
        with torch.no_grad():
            branches = model.get_expert_predictions(gate_coords).squeeze(-1)
            error = (branches - gate_values).abs()
            relative = error / error.mean(dim=1, keepdim=True).clamp_min(1e-8)
            targets = torch.softmax(-cfg.gate_temperature * relative, dim=1)
        teacher_queries = int(gate_coords.shape[0])
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            probs = model.compute_gate_weights(gate_coords)
            loss = F.kl_div(probs.clamp_min(1e-8).log(), targets, reduction="batchmean")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.gating.parameters(), 1.0)
            optimizer.step()
            history.append(float(loss.item()))
    else:
        balance = float(OBJECTIVES[objective])
        loss_fn = PhysicsLoss2D(
            LossConfig2D(nu=NU, w_res=1.0, w_ic=5.0, w_bc=2.0, w_sparse=0.0, w_balance=balance)
        )
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = loss_fn.compute(model, physics_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.gating.parameters(), 1.0)
            optimizer.step()
            history.append(float(loss.item()))

    checkpoint = {
        "model_state": model.state_dict(),
        "config": source["config"],
        "expert_names": list(model.expert_names),
        "probe": {"variant": variant, "objective": objective, "steps": steps},
    }
    torch.save(checkpoint, output_dir / "train_checkpoint.pt")
    audit = {
        "seed": cfg.seed,
        "variant": variant,
        "objective": objective,
        "steps": steps,
        "source": str(source_dir),
        "source_base_hash": source["base_hash"],
        "gate_initial_hash": gate_initial_hash,
        "directional_state_hash": _hash_state(model.experts[2].state_dict()),
        "teacher_queries": teacher_queries,
        "test_loaded_during_training": False,
        "training_pool_coordinate_hash": _coords_hash(gate_coords.detach().cpu().numpy()),
        "train_seconds": time.time() - started_at,
    }
    (output_dir / "probe_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "probe_history.json").write_text(json.dumps(history), encoding="utf-8")
    metrics = evaluate_checkpoint(output_dir, device, test_nx=82, test_ny=83, test_nt=32)
    metrics["directional_expert_l2"] = metrics["per_expert_l2"][2]
    metrics["directional_top1_load"] = metrics["top1_load"][2]
    metrics["directional_soft_load"] = metrics["soft_load"][2]
    (output_dir / "test_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize(root: Path) -> None:
    rows = []
    for path in sorted(root.glob("seed*_*_*")):
        metric_path = path / "test_metrics.json"
        audit_path = path / "probe_audit.json"
        if not metric_path.exists() or not audit_path.exists():
            continue
        metrics = json.loads(metric_path.read_text(encoding="utf-8"))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        rows.append({**audit, **{key: metrics[key] for key in (
            "l2_relative_error", "directional_expert_l2", "directional_top1_load",
            "directional_soft_load", "routing_regret", "soft_routing_regret",
        )}})
    grouped = {}
    for variant in VARIANTS:
        for objective in OBJECTIVES:
            cell = [row for row in rows if row["variant"] == variant and row["objective"] == objective]
            grouped[f"{variant}/{objective}"] = {
                key: {"mean": float(np.mean([row[key] for row in cell])), "raw": [float(row[key]) for row in cell]}
                for key in ("l2_relative_error", "directional_expert_l2", "directional_top1_load", "directional_soft_load", "routing_regret")
            } if cell else {}
    result = {"completed": len(rows), "cells": grouped}
    (root / "probe_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--source-dir", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--variant", choices=VARIANTS, required=True)
    run.add_argument("--objective", choices=tuple(OBJECTIVES), required=True)
    run.add_argument("--device", default="cuda")
    run.add_argument("--steps", type=int, default=300)
    summary = sub.add_parser("summarize")
    summary.add_argument("--root", required=True)
    args = parser.parse_args()
    if args.command == "summarize":
        summarize(Path(args.root))
        return
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    run_probe(Path(args.source_dir), Path(args.output_dir), args.variant, args.objective, device, args.steps)


if __name__ == "__main__":
    main()
