"""True staged-vs-co-adaptation experiment with fixed reference information.

Both modes share initialization, base training, reference pool, specialist pools,
and parameter-step counts.  In ``staged``, each expert is first optimized as a
standalone residual branch and the frozen gate is trained afterwards.  In
``coadapt``, all experts and the gate are updated together through the mixture
physics + supervised loss, so expert gradients are explicitly gate-scaled.

Checkpoints are saved before and after an identical 750-step gate-only
calibration, preventing that calibration from hiding the main-phase effect.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_equal_information_2x2 import (
    DTYPE,
    NU,
    EqualInfoConfig,
    _build_model,
    _coords_hash,
    _expert_physics_loss,
    _gate_atom,
    _make_gate_targets,
    _mixture_atom,
    _seed_everything,
    _set_trainable,
    _state_hash,
    _tensor_bundle_hash,
    _train_base,
    evaluate_checkpoint,
)
from Burger2D.equations.burgers2d import Burgers2DProblem
from Burger2D.training.losses import LossConfig2D, PhysicsLoss2D
from Burger2D.training.staged_burgers2d import (
    StagedBurgers2DConfig,
    build_specialist_batches,
    compute_region_scores,
    flatten_reference_solution,
)


def _checkpoint(model: torch.nn.Module, cfg: EqualInfoConfig, mode: str, base_hash: str) -> dict:
    return {
        "model_state": model.state_dict(),
        "config": asdict(cfg),
        "expert_names": list(model.expert_names),
        "mode": mode,
        "base_hash": base_hash,
    }


def train(mode: str, seed: int, output_dir: Path, device: torch.device, smoke: bool) -> None:
    if mode not in {"staged", "coadapt"}:
        raise ValueError(mode)
    cfg = EqualInfoConfig(
        seed=seed,
        information_mode="reference_guided",
        schedule_mode="blocked",
        gate_updates=300,
        mixture_updates=0,
        refinement_updates=750,
    )
    if smoke:
        cfg.base_steps = 4
        cfg.expert_updates = 3
        cfg.gate_updates = 3
        cfg.refinement_updates = 3
        cfg.n_col = 256
        cfg.n_ic = 96
        cfg.n_bc_per_face = 32
        cfg.teacher_nx = 17
        cfg.teacher_ny = 17
        cfg.teacher_nt = 7
        cfg.gate_pool_size = 128
        cfg.expert_sup_points = 64

    output_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(seed)
    problem = Burgers2DProblem(nu=NU, device=device, dtype=DTYPE, seed=seed)
    base_batch = problem.training_batch(cfg.n_col, cfg.n_ic, cfg.n_bc_per_face)
    model = _build_model(cfg, device)
    initial_hash = _state_hash(model.state_dict())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    base_history = _train_base(model, base_batch, cfg)
    base_hash = _state_hash(model.base_model.state_dict())

    reference = problem.generate_reference_solution(cfg.teacher_nx, cfg.teacher_ny, cfg.teacher_nt)
    all_coords, all_values = flatten_reference_solution(reference, device=device, dtype=DTYPE)
    coords_np = all_coords.detach().cpu().numpy()
    np.save(output_dir / "training_pool_coords.npy", coords_np)
    region_scores = compute_region_scores(reference, layout_variant=cfg.expert_layout_variant)

    _seed_everything(seed + 1000)
    stage_cfg = StagedBurgers2DConfig(
        base_steps=cfg.base_steps,
        expert_steps=cfg.expert_updates,
        gate_steps=cfg.gate_updates,
        joint_steps=0,
        rotation_steps=0,
        expert_lr=cfg.expert_lr,
        gate_lr=cfg.gate_lr,
        expert_sup_points=cfg.expert_sup_points,
        expert_sup_weight=cfg.expert_sup_weight,
        use_gate_region_prior=True,
    )
    expert_batches, _, _, _ = build_specialist_batches(
        reference, base_batch, model.expert_names, cfg=stage_cfg,
        layout_variant=cfg.expert_layout_variant,
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 2000)
    pool_size = min(cfg.gate_pool_size, all_coords.shape[0])
    gate_idx_cpu = torch.randperm(all_coords.shape[0], generator=generator)[:pool_size]
    gate_idx = gate_idx_cpu.to(device=device)
    gate_coords = all_coords[gate_idx]
    gate_teacher = all_values[gate_idx]
    prior_cols = []
    idx_np = gate_idx_cpu.numpy()
    for name in model.expert_names:
        prior_cols.append(torch.tensor(region_scores[name].reshape(-1)[idx_np], device=device, dtype=DTYPE))
    gate_prior = torch.stack(prior_cols, dim=1)

    # The co-adaptive supervised stream uses exactly the union of the staged
    # expert supervision sets.  A mixture evaluation touches all four experts,
    # so 1800 mixture points match 4 x 1800 single-expert evaluations per macro.
    sup_coords = torch.cat([batch["xt_sup"] for batch in expert_batches], dim=0)
    sup_values = torch.cat([batch["u_sup"] for batch in expert_batches], dim=0)
    sup_gen = torch.Generator(device="cpu")
    sup_gen.manual_seed(seed + 3000)
    sup_perm = torch.randperm(sup_coords.shape[0], generator=sup_gen).to(device=device)

    physics_loss = PhysicsLoss2D(
        LossConfig2D(
            nu=NU, w_res=1.0, w_ic=5.0, w_bc=2.0,
            w_sparse=0.0, w_balance=cfg.mixture_balance_weight,
        )
    )
    history: dict[str, list] = {"base": base_history, "main": [], "calibration": []}

    if mode == "staged":
        expert_opts = [torch.optim.Adam(expert.parameters(), lr=cfg.expert_lr) for expert in model.experts]
        for k, expert in enumerate(model.experts):
            for step in range(cfg.expert_updates):
                _set_trainable(model, base=False, experts=False, gate=False)
                for p in expert.parameters():
                    p.requires_grad_(True)
                expert_opts[k].zero_grad(set_to_none=True)
                loss, parts = _expert_physics_loss(
                    expert, expert_batches[k], base_model=model.base_model,
                    correction_scale=float(model.correction_scale), nu=NU,
                    sup_weight=cfg.expert_sup_weight,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(expert.parameters(), 1.0)
                expert_opts[k].step()
                if step % max(cfg.expert_updates // 10, 1) == 0:
                    history["main"].append({
                        "phase": "expert", "expert": k, "step": step,
                        "loss": float(loss.item()), "sup": float(parts["sup"].item()),
                    })
        gate_opt = torch.optim.Adam(model.gating.parameters(), lr=cfg.gate_lr)
        targets, confidence = _make_gate_targets(model, gate_coords, gate_teacher, gate_prior, cfg)
        for step in range(cfg.gate_updates):
            row = _gate_atom(model, gate_opt, gate_coords, targets, confidence, cfg.gate_balance_weight)
            if step % max(cfg.gate_updates // 20, 1) == 0:
                history["main"].append({"phase": "gate", "step": step, **row})
    else:
        joint_opt = torch.optim.Adam(
            [
                {"params": [p for expert in model.experts for p in expert.parameters()], "lr": cfg.expert_lr},
                {"params": list(model.gating.parameters()), "lr": cfg.gate_lr},
            ]
        )
        batch_size = cfg.expert_sup_points
        target_cache = None
        for step in range(cfg.expert_updates):
            _set_trainable(model, base=False, experts=True, gate=True)
            joint_opt.zero_grad(set_to_none=True)
            pde_loss, parts = physics_loss.compute(model, base_batch)
            start = (step * batch_size) % sup_perm.shape[0]
            pos = (torch.arange(batch_size, device=device) + start) % sup_perm.shape[0]
            ids = sup_perm[pos]
            pred_sup = model(sup_coords[ids])
            sup_loss = F.mse_loss(pred_sup, sup_values[ids])
            # Give the co-adaptive gate the same direct reference teacher used
            # by staged training.  Targets are refreshed as experts move.
            if target_cache is None or step % max(cfg.target_refresh, 1) == 0:
                target_cache = _make_gate_targets(model, gate_coords, gate_teacher, gate_prior, cfg)
            targets, confidence = target_cache
            probs = model.compute_gate_weights(gate_coords)
            sample_kl = F.kl_div(probs.clamp_min(1e-8).log(), targets, reduction="none").sum(dim=1)
            weights = (0.1 + confidence).detach()
            gate_match = (weights * sample_kl).sum() / weights.sum().clamp_min(1e-8)
            loss = pde_loss + cfg.expert_sup_weight * sup_loss + gate_match
            loss.backward()
            params = [p for expert in model.experts for p in expert.parameters()] + list(model.gating.parameters())
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            joint_opt.step()
            if step % max(cfg.expert_updates // 20, 1) == 0:
                history["main"].append({
                    "phase": "coadapt", "step": step, "loss": float(loss.item()),
                    "physics": float(pde_loss.item()), "sup": float(sup_loss.item()),
                    "gate_match": float(gate_match.item()),
                    "balance": float(parts["balance"].item()),
                })

    pre_expert_hashes = {name: _state_hash(model.experts[k].state_dict()) for k, name in enumerate(model.expert_names)}
    torch.save(_checkpoint(model, cfg, mode, base_hash), output_dir / "pre_calibration_checkpoint.pt")

    calibration_opt = torch.optim.Adam(model.gating.parameters(), lr=cfg.refinement_lr)
    for step in range(cfg.refinement_updates):
        row = _mixture_atom(model, calibration_opt, base_batch, physics_loss, gate_only=True)
        if step % max(cfg.refinement_updates // 20, 1) == 0:
            history["calibration"].append({"step": step, **row})

    final_expert_hashes = {name: _state_hash(model.experts[k].state_dict()) for k, name in enumerate(model.expert_names)}
    torch.save(_checkpoint(model, cfg, mode, base_hash), output_dir / "train_checkpoint.pt")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2))
    else:
        peak_memory_mb = None
    audit = {
        "valid": pre_expert_hashes == final_expert_hashes,
        "mode": mode,
        "seed": seed,
        "initial_hash": initial_hash,
        "base_hash": base_hash,
        "pre_expert_hashes": pre_expert_hashes,
        "final_expert_hashes": final_expert_hashes,
        "training_pool_coordinate_hash": _coords_hash(coords_np),
        "base_batch_hash": _tensor_bundle_hash(base_batch),
        "expert_batch_hashes": {
            model.expert_names[k]: _tensor_bundle_hash(batch) for k, batch in enumerate(expert_batches)
        },
        "gate_pool_coordinate_hash": _coords_hash(gate_coords.detach().cpu().numpy()),
        "supervision_union_coordinate_hash": _coords_hash(sup_coords.detach().cpu().numpy()),
        "parameter_steps": {
            "base": cfg.base_steps,
            "each_expert_main": cfg.expert_updates,
            "gate_main": cfg.gate_updates,
            "gate_calibration": cfg.refinement_updates,
        },
        "reference_access": {
            "staged": "specialist supervision + gate teacher",
            "coadapt": "same specialist-supervision union through mixture loss",
            "test_values": False,
        },
        "train_seconds": time.time() - started,
        "peak_cuda_memory_mb": peak_memory_mb,
    }
    with open(output_dir / "audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--mode", choices=["staged", "coadapt"], required=True)
    tr.add_argument("--seed", type=int, required=True)
    tr.add_argument("--output-dir", required=True)
    tr.add_argument("--device", default="cuda")
    tr.add_argument("--smoke", action="store_true")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--run-dir", required=True)
    ev.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    if args.command == "train":
        run_dir = Path(args.output_dir)
        train(args.mode, args.seed, run_dir, device, args.smoke)
    else:
        run_dir = Path(args.run_dir)
        evaluate_checkpoint(
            run_dir, device, test_nx=82, test_ny=83, test_nt=32,
            checkpoint_name="pre_calibration_checkpoint.pt", metrics_name="pre_metrics.json",
        )
        evaluate_checkpoint(
            run_dir, device, test_nx=82, test_ny=83, test_nt=32,
            checkpoint_name="train_checkpoint.pt", metrics_name="test_metrics.json",
        )


if __name__ == "__main__":
    main()
