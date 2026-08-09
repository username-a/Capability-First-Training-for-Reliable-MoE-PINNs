"""
Logged trajectory runs under the final protocol (seed 42, WENO5-consistent
evaluation set) to validate two theory predictions:

  P3: during co-adaptation, the capability gap shrinks and oracle labels flip;
  P1: persistently low soft weights starve the expert gradient.

The run re-trains seed 42 in the exact final staged/co-adaptation protocol and
logs, every LOG_EVERY steps on a fixed 20,000-point subset of the WENO5
evaluation set:
  - capability-gap quantiles, oracle coverage, oracle-label flip rate
  - per-expert mean soft weights and raw gradient norms (co-adaptation only)

Outputs:
    Burger2D/results/jcp_reference_rebuild_20260808/trajectory_logged.json
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import torch.nn.functional as F  # noqa: E402

from run_equal_information_2x2 import (  # noqa: E402
    DTYPE,
    NU,
    EqualInfoConfig,
    _build_model,
    _expert_physics_loss,
    _make_gate_targets,
    _seed_everything,
    _set_trainable,
    _train_base,
)
from Burger2D.equations.burgers2d import Burgers2DProblem  # noqa: E402
from Burger2D.training.losses import LossConfig2D, PhysicsLoss2D  # noqa: E402
from Burger2D.training.staged_burgers2d import (  # noqa: E402
    StagedBurgers2DConfig,
    build_specialist_batches,
    compute_region_scores,
    flatten_reference_solution,
)


RESULTS = os.path.join(PACKAGE_ROOT, "results")
OUT_DIR = os.path.join(RESULTS, "jcp_reference_rebuild_20260808")
LOG_PATH = os.path.join(OUT_DIR, "trajectory_logged_progress.log")
LOG_EVERY = 25
EVAL_N = 20000
SEED = 42


def log(msg: str) -> None:
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    print(msg, flush=True)


def eval_stats(model, coords: torch.Tensor, truth: np.ndarray) -> dict:
    with torch.no_grad():
        branches = model.get_expert_predictions(coords).cpu().numpy()[:, :, 0]
        gates = model.get_gate_weights(coords).cpu().numpy()
    sq = (branches.astype(np.float64) - truth[:, None]) ** 2
    kstar = sq.argmin(axis=1)
    s = np.sort(sq, axis=1)
    gap = s[:, 1] - s[:, 0]
    return {
        "gap_q10": float(np.percentile(gap, 10)),
        "gap_q50": float(np.median(gap)),
        "gap_q90": float(np.percentile(gap, 90)),
        "oracle_coverage": float(gates[np.arange(len(gates)), kstar].mean()),
        "kstar": kstar,
        "soft_mean": gates.mean(axis=0).tolist(),
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    # fixed WENO5 evaluation subset
    sub = np.load(os.path.join(OUT_DIR, "evaluation_subset.npz"), mmap_mode="r")
    x, y, t, u = sub["x"], sub["y"], sub["t"], sub["u"]
    tt, yy, xx = np.meshgrid(t, y, x, indexing="ij")
    coords_all = np.stack([xx.ravel(), yy.ravel(), tt.ravel()], axis=1)
    rng = np.random.default_rng(0)
    idx = rng.choice(coords_all.shape[0], size=EVAL_N, replace=False)
    eval_coords = torch.tensor(coords_all[idx], dtype=torch.float32, device=device)
    eval_truth = np.asarray(u, dtype=np.float64).ravel()[idx]

    out = {"seed": SEED, "eval_points": EVAL_N, "reference": "conservative WENO5 subset",
           "staged": {}, "coadapt": {}}

    cfg = EqualInfoConfig(
        seed=SEED,
        information_mode="reference_guided",
        schedule_mode="blocked",
        gate_updates=300,
        mixture_updates=0,
        refinement_updates=750,
    )

    # ---------- co-adaptation ----------
    _seed_everything(SEED)
    problem = Burgers2DProblem(nu=NU, device=device, dtype=DTYPE, seed=SEED)
    base_batch = problem.training_batch(cfg.n_col, cfg.n_ic, cfg.n_bc_per_face)
    model = _build_model(cfg, device)
    _train_base(model, base_batch, cfg)
    reference = problem.generate_reference_solution(cfg.teacher_nx, cfg.teacher_ny, cfg.teacher_nt)
    all_coords, all_values = flatten_reference_solution(reference, device=device, dtype=DTYPE)

    _seed_everything(SEED + 1000)
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
    generator.manual_seed(SEED + 2000)
    pool_size = min(cfg.gate_pool_size, all_coords.shape[0])
    gate_idx = torch.randperm(all_coords.shape[0], generator=generator)[:pool_size]
    gate_coords = all_coords[gate_idx]
    gate_teacher = all_values[gate_idx]
    gate_prior = torch.stack([torch.tensor(
        compute_region_scores(reference, layout_variant=cfg.expert_layout_variant)[name].reshape(-1)[gate_idx.cpu().numpy()],
        device=device, dtype=DTYPE) for name in model.expert_names], dim=1)

    sup_coords = torch.cat([b["xt_sup"] for b in expert_batches], dim=0)
    sup_values = torch.cat([b["u_sup"] for b in expert_batches], dim=0)
    sup_gen = torch.Generator(device="cpu")
    sup_gen.manual_seed(SEED + 3000)
    sup_perm = torch.randperm(sup_coords.shape[0], generator=sup_gen).to(device=device)

    physics_loss = PhysicsLoss2D(LossConfig2D(
        nu=NU, w_res=1.0, w_ic=5.0, w_bc=2.0, w_sparse=0.0, w_balance=cfg.mixture_balance_weight))

    joint_opt = torch.optim.Adam([
        {"params": [p for expert in model.experts for p in expert.parameters()], "lr": cfg.expert_lr},
        {"params": list(model.gating.parameters()), "lr": cfg.gate_lr},
    ])
    batch_size = cfg.expert_sup_points
    target_cache = None
    prev_kstar = None
    co_rows = []
    for step in range(cfg.expert_updates):
        _set_trainable(model, base=False, experts=True, gate=True)
        joint_opt.zero_grad(set_to_none=True)
        pde_loss, parts = physics_loss.compute(model, base_batch)
        start = (step * batch_size) % sup_perm.shape[0]
        pos = (torch.arange(batch_size, device=device) + start) % sup_perm.shape[0]
        ids = sup_perm[pos]
        pred_sup = model(sup_coords[ids])
        sup_loss = F.mse_loss(pred_sup, sup_values[ids])
        if target_cache is None or step % max(cfg.target_refresh, 1) == 0:
            target_cache = _make_gate_targets(model, gate_coords, gate_teacher, gate_prior, cfg)
        targets, confidence = target_cache
        probs = model.compute_gate_weights(gate_coords)
        sample_kl = F.kl_div(probs.clamp_min(1e-8).log(), targets, reduction="none").sum(dim=1)
        weights = (0.1 + confidence).detach()
        gate_match = (weights * sample_kl).sum() / weights.sum().clamp_min(1e-8)
        loss = pde_loss + cfg.expert_sup_weight * sup_loss + gate_match
        loss.backward()
        grad_norms = [
            float(sum(float(p.grad.norm().item() ** 2) for p in expert.parameters() if p.grad is not None) ** 0.5)
            for expert in model.experts
        ]
        joint_opt.step()
        if step % LOG_EVERY == 0 or step == cfg.expert_updates - 1:
            st = eval_stats(model, eval_coords, eval_truth)
            flip = float(np.mean(prev_kstar != st["kstar"])) if prev_kstar is not None else 0.0
            prev_kstar = st["kstar"]
            row = {"step": step, "gap_q10": st["gap_q10"], "gap_q50": st["gap_q50"],
                   "gap_q90": st["gap_q90"], "oracle_coverage": st["oracle_coverage"],
                   "flip_rate": flip, "soft_mean": st["soft_mean"], "grad_norms": grad_norms,
                   "loss": float(loss.item())}
            co_rows.append(row)
            log(f"coadapt step{step}: gap50={row['gap_q50']:.2e} cov={row['oracle_coverage']:.3f} "
                f"flip={row['flip_rate']:.3f} soft={[round(v,3) for v in st['soft_mean']]}")
    out["coadapt"]["rows"] = co_rows
    log("coadapt done")

    # ---------- staged gate phase (contrast) ----------
    _seed_everything(SEED)
    problem = Burgers2DProblem(nu=NU, device=device, dtype=DTYPE, seed=SEED)
    base_batch = problem.training_batch(cfg.n_col, cfg.n_ic, cfg.n_bc_per_face)
    model = _build_model(cfg, device)
    _train_base(model, base_batch, cfg)
    reference = problem.generate_reference_solution(cfg.teacher_nx, cfg.teacher_ny, cfg.teacher_nt)
    all_coords, all_values = flatten_reference_solution(reference, device=device, dtype=DTYPE)
    _seed_everything(SEED + 1000)
    expert_batches, _, _, _ = build_specialist_batches(
        reference, base_batch, model.expert_names, cfg=stage_cfg,
        layout_variant=cfg.expert_layout_variant)
    expert_opts = [torch.optim.Adam(expert.parameters(), lr=cfg.expert_lr) for expert in model.experts]
    for k, expert in enumerate(model.experts):
        for step in range(cfg.expert_updates):
            _set_trainable(model, base=False, experts=False, gate=False)
            for p in expert.parameters():
                p.requires_grad_(True)
            expert_opts[k].zero_grad(set_to_none=True)
            loss, _ = _expert_physics_loss(
                expert, expert_batches[k], base_model=model.base_model,
                correction_scale=float(model.correction_scale), nu=NU,
                sup_weight=cfg.expert_sup_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(expert.parameters(), 1.0)
            expert_opts[k].step()
    log("staged expert pretraining done")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED + 2000)
    gate_idx = torch.randperm(all_coords.shape[0], generator=generator)[:pool_size]
    gate_coords = all_coords[gate_idx]
    gate_teacher = all_values[gate_idx]
    gate_prior = torch.stack([torch.tensor(
        compute_region_scores(reference, layout_variant=cfg.expert_layout_variant)[name].reshape(-1)[gate_idx.cpu().numpy()],
        device=device, dtype=DTYPE) for name in model.expert_names], dim=1)
    targets, confidence = _make_gate_targets(model, gate_coords, gate_teacher, gate_prior, cfg)
    gate_opt = torch.optim.Adam(model.gating.parameters(), lr=cfg.gate_lr)
    prev_kstar = None
    st_rows = []
    for step in range(cfg.gate_updates):
        _set_trainable(model, base=False, experts=False, gate=True)
        gate_opt.zero_grad(set_to_none=True)
        probs = model.compute_gate_weights(gate_coords)
        sample_kl = F.kl_div(probs.clamp_min(1e-8).log(), targets, reduction="none").sum(dim=1)
        weights = (0.1 + confidence).detach()
        match = (weights * sample_kl).sum() / weights.sum().clamp_min(1e-8)
        balance = model.load_balance_loss(gate_coords)
        loss = match + cfg.gate_balance_weight * balance
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.gating.parameters(), 1.0)
        gate_opt.step()
        if step % LOG_EVERY == 0 or step == cfg.gate_updates - 1:
            st = eval_stats(model, eval_coords, eval_truth)
            flip = float(np.mean(prev_kstar != st["kstar"])) if prev_kstar is not None else 0.0
            prev_kstar = st["kstar"]
            st_rows.append({"step": step, "gap_q10": st["gap_q10"], "gap_q50": st["gap_q50"],
                            "gap_q90": st["gap_q90"], "oracle_coverage": st["oracle_coverage"],
                            "flip_rate": flip, "soft_mean": st["soft_mean"]})
            log(f"staged gate step{step}: gap50={st['gap_q50']:.2e} flip={flip:.3f}")
    out["staged"]["rows"] = st_rows
    log("staged done")

    with open(os.path.join(OUT_DIR, "trajectory_logged.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    log("ALL DONE")


if __name__ == "__main__":
    main()
