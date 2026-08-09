"""Equal-information 2x2 experiment for the 2D Burgers MoE-PINN.

The confirmatory factors are:
  information_mode: physics_only | reference_guided
  schedule_mode:    blocked | interleaved

Each paired schedule receives the same initialized model, fixed training pools,
and the same counts of atomic E/G/M/C updates.  Only their execution order
changes.  Test labels are loaded exclusively by the separate ``evaluate``
command.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import sys
import time
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

from Burger2D.core.moe_pinn import build_burgers2d_moe  # noqa: E402
from Burger2D.equations.burgers2d import (  # noqa: E402
    Burgers2DProblem,
    ReferenceSolution2D,
    steep_region_mask,
)
from Burger2D.training.losses import LossConfig2D, PhysicsLoss2D  # noqa: E402
from Burger2D.training.staged_burgers2d import (  # noqa: E402
    StagedBurgers2DConfig,
    _expert_physics_loss,
    build_specialist_batches,
    compute_region_scores,
    flatten_reference_solution,
)


NU = 0.01 / np.pi
DTYPE = torch.float32
GROUPS = {
    "P-B": ("physics_only", "blocked"),
    "P-I": ("physics_only", "interleaved"),
    "R-B": ("reference_guided", "blocked"),
    "R-I": ("reference_guided", "interleaved"),
}


@dataclass
class EqualInfoConfig:
    seed: int = 42
    information_mode: str = "reference_guided"
    schedule_mode: str = "blocked"
    base_steps: int = 270
    expert_updates: int = 300
    gate_updates: int = 180
    # The confirmatory 2x2 isolates routing-target chronology.  Experts receive
    # exactly the same standalone E updates in both schedules and remain fixed
    # during the final routing phase.  Conventional simultaneous M updates are
    # intentionally reserved for the separate practical-baseline experiment.
    mixture_updates: int = 0
    refinement_updates: int = 750
    n_col: int = 6000
    n_ic: int = 1200
    n_bc_per_face: int = 400
    teacher_nx: int = 65
    teacher_ny: int = 65
    teacher_nt: int = 21
    gate_pool_size: int = 8192
    expert_sup_points: int = 1800
    expert_sup_weight: float = 8.0
    base_lr: float = 1e-3
    expert_lr: float = 1e-3
    gate_lr: float = 5e-4
    mixture_lr: float = 5e-4
    refinement_lr: float = 5e-4
    gate_temperature: float = 8.0
    gate_balance_weight: float = 0.01
    mixture_balance_weight: float = 0.01
    target_refresh: int = 10
    directional_expert_variant: str = "hybrid"
    wave_expert_variant: str = "mixed_lite"
    expert_layout_variant: str = "categorical"
    gate_variant: str = "local_conv"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _state_hash(state: dict[str, Any]) -> str:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _coordinate_rows(coords: np.ndarray) -> set[tuple[float, float, float]]:
    rounded = np.round(np.asarray(coords, dtype=np.float64), decimals=7)
    return {tuple(row) for row in rounded.tolist()}


def _coords_hash(coords: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.round(coords, decimals=7), dtype=np.float32)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _tensor_bundle_hash(bundle: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(bundle):
        value = bundle[key]
        if not isinstance(value, torch.Tensor):
            continue
        array = np.ascontiguousarray(value.detach().cpu().numpy())
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _set_trainable(model: torch.nn.Module, *, base: bool, experts: bool, gate: bool) -> None:
    for param in model.base_model.parameters():
        param.requires_grad_(base)
    for expert in model.experts:
        for param in expert.parameters():
            param.requires_grad_(experts)
    for param in model.gating.parameters():
        param.requires_grad_(gate)


def _build_model(cfg: EqualInfoConfig, device: torch.device) -> torch.nn.Module:
    return build_burgers2d_moe(
        balance_weight=cfg.gate_balance_weight,
        directional_expert_variant=cfg.directional_expert_variant,
        wave_expert_variant=cfg.wave_expert_variant,
        expert_layout_variant=cfg.expert_layout_variant,
        gate_variant=cfg.gate_variant,
        rotation_variant="none",
    ).to(device=device, dtype=DTYPE)


def _proxy_reference(
    model: torch.nn.Module,
    cfg: EqualInfoConfig,
    device: torch.device,
) -> ReferenceSolution2D:
    """Build physics-only region maps from the pretrained base prediction."""
    x = np.linspace(-1.0, 1.0, cfg.teacher_nx, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, cfg.teacher_ny, dtype=np.float32)
    t = np.linspace(0.0, 1.0, cfg.teacher_nt, dtype=np.float32)
    blank = ReferenceSolution2D(
        x=x,
        y=y,
        t=t,
        u=np.zeros((len(t), len(y), len(x)), dtype=np.float32),
    )
    coords, _ = flatten_reference_solution(blank, device=device, dtype=DTYPE)
    chunks = []
    model.base_model.eval()
    with torch.no_grad():
        for start in range(0, coords.shape[0], 32768):
            chunks.append(model.base_model(coords[start : start + 32768]).cpu())
    pred = torch.cat(chunks).numpy().reshape(blank.u.shape)
    return ReferenceSolution2D(x=x, y=y, t=t, u=pred.astype(np.float32))


def _branch_physics_scores(
    model: torch.nn.Module,
    coords: torch.Tensor,
) -> torch.Tensor:
    """Detached pointwise PDE residual magnitude for every branch."""
    scores = []
    for expert in model.experts:
        xt = coords.detach().clone().requires_grad_(True)
        base = model.base_model(xt)
        u = base + float(model.correction_scale) * expert(xt)
        grad = torch.autograd.grad(u.sum(), xt, create_graph=True, retain_graph=True)[0]
        u_x, u_y, u_t = grad[:, 0:1], grad[:, 1:2], grad[:, 2:3]
        u_xx = torch.autograd.grad(u_x.sum(), xt, create_graph=True, retain_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y.sum(), xt, create_graph=True, retain_graph=False)[0][:, 1:2]
        residual = u_t + u * (u_x + u_y) - NU * (u_xx + u_yy)
        scores.append(residual.detach().abs().squeeze(1))
    return torch.stack(scores, dim=1)


def _targets_from_scores(
    scores: torch.Tensor,
    region_prior: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    relative = scores / scores.mean(dim=1, keepdim=True).clamp_min(1e-8)
    targets = torch.softmax(-temperature * relative, dim=1)
    targets = targets * region_prior.clamp_min(1e-4).pow(1.15)
    targets = targets / targets.sum(dim=1, keepdim=True).clamp_min(1e-8)
    top2 = torch.topk(targets, k=min(2, targets.shape[1]), dim=1).values
    confidence = top2[:, 0] - top2[:, 1] if targets.shape[1] > 1 else torch.ones_like(top2[:, 0])
    return targets.detach(), confidence.detach()


def _make_gate_targets(
    model: torch.nn.Module,
    coords: torch.Tensor,
    teacher_values: torch.Tensor | None,
    region_prior: torch.Tensor,
    cfg: EqualInfoConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cfg.information_mode == "reference_guided":
        if teacher_values is None:
            raise RuntimeError("reference_guided gate targets require teacher values")
        with torch.no_grad():
            branch = model.get_expert_predictions(coords).squeeze(-1)
            scores = (branch - teacher_values.reshape(-1, 1)).abs()
    else:
        scores = _branch_physics_scores(model, coords)
    return _targets_from_scores(scores, region_prior, cfg.gate_temperature)


def _gate_atom(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    coords: torch.Tensor,
    targets: torch.Tensor,
    confidence: torch.Tensor,
    balance_weight: float,
) -> dict[str, float]:
    _set_trainable(model, base=False, experts=False, gate=True)
    optimizer.zero_grad(set_to_none=True)
    probs = model.compute_gate_weights(coords)
    sample_kl = F.kl_div(probs.clamp_min(1e-8).log(), targets, reduction="none").sum(dim=1)
    weights = (0.1 + confidence).detach()
    match = (weights * sample_kl).sum() / weights.sum().clamp_min(1e-8)
    balance = model.load_balance_loss(coords)
    loss = match + balance_weight * balance
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.gating.parameters(), 1.0)
    optimizer.step()
    return {"loss": float(loss.item()), "match": float(match.item()), "balance": float(balance.item())}


def _mixture_atom(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    loss_fn: PhysicsLoss2D,
    *,
    gate_only: bool,
) -> dict[str, float]:
    _set_trainable(model, base=False, experts=not gate_only, gate=True)
    optimizer.zero_grad(set_to_none=True)
    loss, parts = loss_fn.compute(model, batch)
    loss.backward()
    params = list(model.gating.parameters())
    if not gate_only:
        params += [p for expert in model.experts for p in expert.parameters()]
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    optimizer.step()
    return {"loss": float(loss.item()), **{key: float(value.detach().item()) for key, value in parts.items()}}


def _train_base(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    cfg: EqualInfoConfig,
) -> list[float]:
    _set_trainable(model, base=True, experts=False, gate=False)
    optimizer = torch.optim.Adam(model.base_model.parameters(), lr=cfg.base_lr)
    loss_fn = PhysicsLoss2D(LossConfig2D(nu=NU, w_res=1.0, w_ic=5.0, w_bc=2.0, w_sparse=0.0, w_balance=0.0))
    history = []
    for _ in range(cfg.base_steps):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = loss_fn.compute(model.base_model, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.base_model.parameters(), 1.0)
        optimizer.step()
        history.append(float(loss.item()))
    _set_trainable(model, base=False, experts=True, gate=True)
    return history


def train_group(cfg: EqualInfoConfig, output_dir: Path, device: torch.device) -> Path:
    if cfg.information_mode not in {"physics_only", "reference_guided"}:
        raise ValueError(cfg.information_mode)
    if cfg.schedule_mode not in {"blocked", "interleaved"}:
        raise ValueError(cfg.schedule_mode)
    output_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(cfg.seed)
    problem = Burgers2DProblem(nu=NU, device=device, dtype=DTYPE, seed=cfg.seed)
    base_batch = problem.training_batch(cfg.n_col, cfg.n_ic, cfg.n_bc_per_face)
    model = _build_model(cfg, device)
    initial_hash = _state_hash(model.state_dict())
    start = time.time()
    base_history = _train_base(model, base_batch, cfg)
    base_hash = _state_hash(model.base_model.state_dict())

    reference_access = {
        "teacher_values": cfg.information_mode == "reference_guided",
        "reference_region_maps": cfg.information_mode == "reference_guided",
        "validation_values": False,
        "test_values": False,
    }
    if cfg.information_mode == "reference_guided":
        map_reference = problem.generate_reference_solution(cfg.teacher_nx, cfg.teacher_ny, cfg.teacher_nt)
    else:
        map_reference = _proxy_reference(model, cfg, device)

    all_coords, all_values = flatten_reference_solution(map_reference, device=device, dtype=DTYPE)
    coords_np = all_coords.detach().cpu().numpy()
    np.save(output_dir / "training_pool_coords.npy", coords_np)
    region_scores = compute_region_scores(map_reference, layout_variant=cfg.expert_layout_variant)

    # Reset RNGs before information-dependent batch construction.  The paired
    # blocked/interleaved schedules therefore receive bitwise-identical pools.
    _seed_everything(cfg.seed + 1000)
    stage_cfg = StagedBurgers2DConfig(
        expert_steps=cfg.expert_updates,
        gate_steps=cfg.gate_updates,
        joint_steps=cfg.mixture_updates,
        base_steps=cfg.base_steps,
        rotation_steps=0,
        expert_lr=cfg.expert_lr,
        gate_lr=cfg.gate_lr,
        joint_lr=cfg.mixture_lr,
        expert_sup_points=cfg.expert_sup_points,
        expert_sup_weight=cfg.expert_sup_weight if cfg.information_mode == "reference_guided" else 0.0,
        use_gate_region_prior=True,
    )
    expert_batches, _, _, _ = build_specialist_batches(
        map_reference,
        base_batch,
        model.expert_names,
        cfg=stage_cfg,
        layout_variant=cfg.expert_layout_variant,
    )
    if cfg.information_mode == "physics_only":
        for expert_batch in expert_batches:
            expert_batch.pop("xt_sup", None)
            expert_batch.pop("u_sup", None)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(cfg.seed + 2000)
    pool_size = min(cfg.gate_pool_size, all_coords.shape[0])
    gate_indices_cpu = torch.randperm(all_coords.shape[0], generator=generator)[:pool_size]
    gate_indices = gate_indices_cpu.to(device=device)
    gate_coords = all_coords[gate_indices]
    gate_teacher = all_values[gate_indices] if cfg.information_mode == "reference_guided" else None
    prior_columns = []
    idx_np = gate_indices_cpu.numpy()
    for name in model.expert_names:
        prior_columns.append(torch.tensor(region_scores[name].reshape(-1)[idx_np], device=device, dtype=DTYPE))
    gate_prior = torch.stack(prior_columns, dim=1)

    expert_optimizers = [torch.optim.Adam(expert.parameters(), lr=cfg.expert_lr) for expert in model.experts]
    gate_optimizer = torch.optim.Adam(model.gating.parameters(), lr=cfg.gate_lr)
    mixture_params = [p for expert in model.experts for p in expert.parameters()] + list(model.gating.parameters())
    mixture_optimizer = torch.optim.Adam(mixture_params, lr=cfg.mixture_lr)
    refinement_optimizer = torch.optim.Adam(model.gating.parameters(), lr=cfg.refinement_lr)
    mixture_loss = PhysicsLoss2D(
        LossConfig2D(nu=NU, w_res=1.0, w_ic=5.0, w_bc=2.0, w_sparse=0.0, w_balance=cfg.mixture_balance_weight)
    )
    refine_loss = PhysicsLoss2D(
        LossConfig2D(nu=NU, w_res=1.0, w_ic=5.0, w_bc=2.0, w_sparse=0.0, w_balance=cfg.mixture_balance_weight)
    )

    counters = {
        "B": cfg.base_steps,
        **{f"E_{name}": 0 for name in model.expert_names},
        "G": 0,
        "M": 0,
        "C": 0,
        "teacher_gate_point_queries": 0,
        "expert_supervised_point_updates": 0,
        "pde_point_updates": cfg.base_steps * cfg.n_col,
    }
    history: dict[str, list[dict[str, float]]] = {"E": [], "G": [], "M": [], "C": []}
    target_cache: tuple[torch.Tensor, torch.Tensor] | None = None

    def expert_atom(k: int) -> None:
        nonlocal history
        _set_trainable(model, base=False, experts=False, gate=False)
        for param in model.experts[k].parameters():
            param.requires_grad_(True)
        optimizer = expert_optimizers[k]
        optimizer.zero_grad(set_to_none=True)
        sup_weight = cfg.expert_sup_weight if cfg.information_mode == "reference_guided" else 0.0
        loss, parts = _expert_physics_loss(
            model.experts[k],
            expert_batches[k],
            base_model=model.base_model,
            correction_scale=float(model.correction_scale),
            nu=NU,
            sup_weight=sup_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.experts[k].parameters(), 1.0)
        optimizer.step()
        name = model.expert_names[k]
        counters[f"E_{name}"] += 1
        counters["pde_point_updates"] += int(expert_batches[k]["xt_col"].shape[0])
        if sup_weight > 0:
            counters["expert_supervised_point_updates"] += int(expert_batches[k]["xt_sup"].shape[0])
        history["E"].append({"expert": float(k), "loss": float(loss.item()), "sup": float(parts["sup"].item())})

    def gate_atom() -> None:
        nonlocal target_cache
        if target_cache is None or counters["G"] % max(cfg.target_refresh, 1) == 0:
            target_cache = _make_gate_targets(model, gate_coords, gate_teacher, gate_prior, cfg)
            if cfg.information_mode == "reference_guided":
                counters["teacher_gate_point_queries"] += int(gate_coords.shape[0])
        row = _gate_atom(model, gate_optimizer, gate_coords, *target_cache, cfg.gate_balance_weight)
        counters["G"] += 1
        history["G"].append(row)

    def mixture_atom() -> None:
        row = _mixture_atom(model, mixture_optimizer, base_batch, mixture_loss, gate_only=False)
        counters["M"] += 1
        counters["pde_point_updates"] += cfg.n_col
        history["M"].append(row)

    def refinement_atom() -> None:
        row = _mixture_atom(model, refinement_optimizer, base_batch, refine_loss, gate_only=True)
        counters["C"] += 1
        counters["pde_point_updates"] += cfg.n_col
        history["C"].append(row)

    if cfg.schedule_mode == "blocked":
        for k in range(len(model.experts)):
            for _ in range(cfg.expert_updates):
                expert_atom(k)
        for _ in range(cfg.gate_updates):
            gate_atom()
        for _ in range(cfg.mixture_updates):
            mixture_atom()
        for _ in range(cfg.refinement_updates):
            refinement_atom()
    else:
        rounds = max(cfg.expert_updates, cfg.gate_updates, cfg.mixture_updates, cfg.refinement_updates)
        for round_idx in range(rounds):
            if round_idx < cfg.expert_updates:
                for k in range(len(model.experts)):
                    expert_atom(k)
            if round_idx < cfg.gate_updates:
                gate_atom()
            if round_idx < cfg.mixture_updates:
                mixture_atom()
            if round_idx < cfg.refinement_updates:
                refinement_atom()

    expected = {
        **{f"E_{name}": cfg.expert_updates for name in model.expert_names},
        "G": cfg.gate_updates,
        "M": cfg.mixture_updates,
        "C": cfg.refinement_updates,
    }
    for key, value in expected.items():
        if counters[key] != value:
            raise RuntimeError(f"Atomic update audit failed for {key}: {counters[key]} != {value}")

    train_seconds = time.time() - start
    checkpoint = output_dir / "train_checkpoint.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(cfg),
            "expert_names": list(model.expert_names),
            "initial_hash": initial_hash,
            "base_hash": base_hash,
        },
        checkpoint,
    )
    audit = {
        "valid": True,
        "config": asdict(cfg),
        "initial_hash": initial_hash,
        "base_hash": base_hash,
        "final_expert_hashes": {
            name: _state_hash(model.experts[idx].state_dict())
            for idx, name in enumerate(model.expert_names)
        },
        "training_pool_coordinate_hash": _coords_hash(coords_np),
        "training_pool_size": int(coords_np.shape[0]),
        "base_batch_hash": _tensor_bundle_hash(base_batch),
        "expert_batch_hashes": {
            model.expert_names[idx]: _tensor_bundle_hash(batch)
            for idx, batch in enumerate(expert_batches)
        },
        "gate_pool_coordinate_hash": _coords_hash(gate_coords.detach().cpu().numpy()),
        "reference_access": reference_access,
        "counters": counters,
        "train_seconds": train_seconds,
        "parameter_counts": model.count_parameters(),
        "test_loaded_during_training": False,
    }
    with open(output_dir / "audit.json", "w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
    with open(output_dir / "history.json", "w", encoding="utf-8") as handle:
        json.dump({"base": base_history, **history}, handle, ensure_ascii=False)
    print(json.dumps({"checkpoint": str(checkpoint), "audit": audit}, ensure_ascii=False, indent=2))
    return checkpoint


def _batched_model_outputs(
    model: torch.nn.Module,
    coords: torch.Tensor,
    batch_size: int = 16384,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_parts, branch_parts, gate_parts = [], [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, coords.shape[0], batch_size):
            chunk = coords[start : start + batch_size]
            branches = model.get_expert_predictions(chunk).squeeze(-1)
            gates = model.compute_gate_weights(chunk, expert_preds=branches.unsqueeze(-1))
            pred_parts.append(model(chunk).squeeze(-1).cpu())
            branch_parts.append(branches.cpu())
            gate_parts.append(gates.cpu())
    return torch.cat(pred_parts), torch.cat(branch_parts), torch.cat(gate_parts)


def evaluate_checkpoint(
    run_dir: Path,
    device: torch.device,
    *,
    test_nx: int,
    test_ny: int,
    test_nt: int,
    checkpoint_name: str = "train_checkpoint.pt",
    metrics_name: str = "test_metrics.json",
) -> dict[str, Any]:
    checkpoint = torch.load(run_dir / checkpoint_name, map_location=device, weights_only=False)
    cfg = EqualInfoConfig(**checkpoint["config"])
    model = _build_model(cfg, device)
    model.load_state_dict(checkpoint["model_state"])
    # Test labels first enter memory here, in this independent evaluation command.
    problem = Burgers2DProblem(nu=NU, device=device, dtype=DTYPE, seed=cfg.seed + 90000)
    reference = problem.generate_reference_solution(test_nx, test_ny, test_nt)
    test_coords, test_values = flatten_reference_solution(reference, device=device, dtype=DTYPE)
    training_coords = np.load(run_dir / "training_pool_coords.npy")
    train_rows = _coordinate_rows(training_coords)
    test_np = test_coords.detach().cpu().numpy()
    overlap = np.array([tuple(row) in train_rows for row in np.round(test_np, 7)], dtype=bool)
    keep = torch.tensor(~overlap, dtype=torch.bool)
    if not bool(keep.any()):
        raise RuntimeError("No disjoint test coordinates remain after overlap filtering.")
    pred, branches, gates = _batched_model_outputs(model, test_coords)
    truth = test_values.detach().cpu().reshape(-1)
    pred, branches, gates, truth = pred[keep], branches[keep], gates[keep], truth[keep]
    sq = (branches - truth[:, None]).square()
    best_sq, best_idx = sq.min(dim=1)
    route_idx = gates.argmax(dim=1)
    routed_sq = sq[torch.arange(sq.shape[0]), route_idx]
    load = torch.bincount(route_idx, minlength=branches.shape[1]).float()
    load = load / load.sum().clamp_min(1.0)
    soft_load = gates.mean(dim=0)
    effective = float(torch.exp(-(load * load.clamp_min(1e-12).log()).sum()).item())
    expert_l2 = [float(torch.linalg.vector_norm(branches[:, k] - truth) / torch.linalg.vector_norm(truth).clamp_min(1e-10)) for k in range(branches.shape[1])]
    top2 = torch.topk(sq, k=2, dim=1, largest=False).values
    gap = top2[:, 1] - top2[:, 0]
    metrics = {
        "l2_relative_error": float(torch.linalg.vector_norm(pred - truth) / torch.linalg.vector_norm(truth).clamp_min(1e-10)),
        "max_absolute_error": float((pred - truth).abs().max().item()),
        "per_expert_l2": expert_l2,
        "worst_expert_l2": float(max(expert_l2)),
        "expert_l2_std": float(np.std(expert_l2)),
        "oracle_rmse": float(best_sq.mean().sqrt().item()),
        "oracle_hit": float((route_idx == best_idx).float().mean().item()),
        "routing_regret": float((routed_sq - best_sq).mean().item()),
        "soft_routing_regret": float((gates * (sq - best_sq[:, None])).sum(dim=1).mean().item()),
        "top1_load": [float(x) for x in load.tolist()],
        "soft_load": [float(x) for x in soft_load.tolist()],
        "effective_experts": effective,
        "min_load": float(load.min().item()),
        "route_entropy": float((-(gates * gates.clamp_min(1e-12).log()).sum(dim=1)).mean().item()),
        "capability_gap_q25": float(torch.quantile(gap, 0.25).item()),
        "capability_gap_median": float(torch.quantile(gap, 0.50).item()),
        "test_grid": [test_nx, test_ny, test_nt],
        "test_total_points": int(test_coords.shape[0]),
        "test_overlap_excluded": int(overlap.sum()),
        "test_disjoint_points": int(keep.sum().item()),
        "test_coordinate_hash": _coords_hash(test_np[~overlap]),
    }
    with open(run_dir / metrics_name, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--group", choices=sorted(GROUPS), required=True)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--device", default="cuda")
    train.add_argument("--smoke", action="store_true")
    train.add_argument("--base-steps", type=int, default=None)
    train.add_argument("--expert-updates", type=int, default=None)
    train.add_argument("--gate-updates", type=int, default=None)
    train.add_argument("--mixture-updates", type=int, default=None)
    train.add_argument("--refinement-updates", type=int, default=None)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--run-dir", required=True)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--test-nx", type=int, default=82)
    evaluate.add_argument("--test-ny", type=int, default=83)
    evaluate.add_argument("--test-nt", type=int, default=32)
    evaluate.add_argument("--checkpoint-name", default="train_checkpoint.pt")
    evaluate.add_argument("--metrics-name", default="test_metrics.json")
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    if args.command == "evaluate":
        evaluate_checkpoint(
            Path(args.run_dir), device,
            test_nx=args.test_nx, test_ny=args.test_ny, test_nt=args.test_nt,
            checkpoint_name=args.checkpoint_name, metrics_name=args.metrics_name,
        )
        return

    information_mode, schedule_mode = GROUPS[args.group]
    cfg = EqualInfoConfig(seed=args.seed, information_mode=information_mode, schedule_mode=schedule_mode)
    if args.smoke:
        cfg.base_steps = 4
        cfg.expert_updates = 3
        cfg.gate_updates = 3
        cfg.mixture_updates = 3
        cfg.refinement_updates = 3
        cfg.n_col = 256
        cfg.n_ic = 96
        cfg.n_bc_per_face = 32
        cfg.teacher_nx = 17
        cfg.teacher_ny = 17
        cfg.teacher_nt = 7
        cfg.gate_pool_size = 128
        cfg.expert_sup_points = 64
        cfg.target_refresh = 1
    for name in ("base_steps", "expert_updates", "gate_updates", "mixture_updates", "refinement_updates"):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)
    train_group(cfg, Path(args.output_dir), device)


if __name__ == "__main__":
    main()
