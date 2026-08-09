"""
Long mechanism training runs with snapshots and per-expert gradient logging.

Run A (e2e_long): 8000-step end-to-end MoE training from scratch.
Run B (joint_long): fully pretrained staged model (base + experts + Stage B
    gate), then 8000 joint fine-tuning steps with base frozen and experts+gate
    updated, starting from the pre-joint checkpoint.

Both save a model snapshot + quick metrics every `snap_every` steps and record
per-expert / gate gradient norms, so the results support:
    - the joint-fine-tuning length curve (accuracy / effective experts /
      division degree vs joint steps),
    - gradient-imbalance analysis,
    - long-budget end-to-end controls.

Usage:
    python Burger2D/scripts/run_mechanism_training.py [--smoke]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Burger2D.core.moe_pinn import build_burgers2d_moe  # noqa: E402
from Burger2D.equations.burgers2d import Burgers2DProblem  # noqa: E402
from Burger2D.experiments.run_burgers2d import (  # noqa: E402
    NU,
    DTYPE,
    DEVICE,
    _set_global_seed,
    run_experiment,
)
from Burger2D.training.losses import (  # noqa: E402
    LossConfig2D,
    PhysicsLoss2D,
    l2_relative_error,
)
from Burger2D.training.staged_burgers2d import (  # noqa: E402
    StagedBurgers2DConfig,
    flatten_reference_solution,
)


RESULTS = os.path.join(PACKAGE_ROOT, "results")
STAGED_VARIANT = "stronger_expert_route_sharp"
GATE_VARIANT = "local_conv"
WAVE_VARIANT = "mixed_lite"
DIRECTIONAL_VARIANT = "hybrid"


def _build_model() -> torch.nn.Module:
    model = build_burgers2d_moe(
        directional_expert_variant=DIRECTIONAL_VARIANT,
        wave_expert_variant=WAVE_VARIANT,
        expert_layout_variant="categorical",
        attribute_expert_variant="base",
        gate_variant=GATE_VARIANT,
        rotation_variant="none",
    ).to(DEVICE).to(DTYPE)
    return model


def _batched(fn, xyt: torch.Tensor, batch_size: int = 65536) -> torch.Tensor:
    outs = []
    for start in range(0, xyt.shape[0], batch_size):
        outs.append(fn(xyt[start:start + batch_size]))
    return torch.cat(outs, dim=0)


def _grad_norms(model: torch.nn.Module) -> dict:
    norms: dict[str, float] = {}
    for i, expert in enumerate(model.experts):
        total = 0.0
        for p in expert.parameters():
            if p.grad is not None:
                total += float(p.grad.detach().norm().item() ** 2)
        norms[f"expert_{i}"] = float(np.sqrt(total))
    gate_total = 0.0
    for p in model.gating.parameters():
        if p.grad is not None:
            gate_total += float(p.grad.detach().norm().item() ** 2)
    norms["gate"] = float(np.sqrt(gate_total))
    return norms


def _snapshot_metrics(
    model: torch.nn.Module,
    xyt: torch.Tensor,
    u_exact: torch.Tensor,
) -> dict:
    with torch.no_grad():
        pred = _batched(model, xyt)
        branch = _batched(model.get_expert_predictions, xyt)[:, :, 0].cpu().numpy()
        load_stats = model.load_balance_stats(xyt)
    u = u_exact.cpu().numpy()[:, 0]
    per_expert_l2 = [
        float(np.linalg.norm(branch[:, k] - u) / (np.linalg.norm(u) + 1e-10))
        for k in range(branch.shape[1])
    ]
    corr = np.corrcoef(branch, rowvar=False)
    n = corr.shape[0]
    off = [corr[i, j] for i in range(n) for j in range(i + 1, n)]
    load_frac = np.asarray(load_stats["expert_load_frac"])
    return {
        "l2": float(l2_relative_error(pred, u_exact)),
        "per_expert_l2": per_expert_l2,
        "effective_experts": float(1.0 / np.sum(load_frac**2)),
        "min_load": float(load_frac.min()),
        "load_frac": load_frac.tolist(),
        "route_entropy": float(load_stats["mean_entropy"]),
        "expert_output_corr_mean": float(np.mean(off)) if off else 0.0,
    }


def _training_batch(problem: Burgers2DProblem, n_col: int, n_ic: int, n_bc: int) -> dict:
    return problem.training_batch(n_col=n_col, n_ic=n_ic, n_bc_per_face=n_bc)


def run_custom_training(
    *,
    model: torch.nn.Module,
    problem: Burgers2DProblem,
    xyt_full: torch.Tensor,
    u_full: torch.Tensor,
    xyt_snap: torch.Tensor,
    u_snap: torch.Tensor,
    loss_fn,
    lr: float,
    n_steps: int,
    snap_every: int,
    grad_log_every: int,
    freeze_base: bool,
    save_dir: str,
    tag: str,
) -> dict:
    os.makedirs(save_dir, exist_ok=True)
    if freeze_base:
        for p in model.base_model.parameters():
            p.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)
    batch = _training_batch(problem, n_col=12000, n_ic=3000, n_bc=800)
    model.train()

    grad_fh = open(os.path.join(save_dir, "grad_norms.jsonl"), "a", encoding="utf-8")
    snap_fh = open(os.path.join(save_dir, "snapshots.jsonl"), "a", encoding="utf-8")
    t0 = time.time()
    for step in range(1, n_steps + 1):
        if step % 20 == 1:
            batch = _training_batch(problem, n_col=12000, n_ic=3000, n_bc=800)
        batch = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        optimizer.zero_grad()
        total_loss, _ = loss_fn.compute(model, batch)
        total_loss.backward()
        if step % grad_log_every == 0:
            row = {"step": step, **{k: round(v, 6) for k, v in _grad_norms(model).items()}}
            grad_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            grad_fh.flush()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        if step % snap_every == 0 or step == n_steps:
            model.eval()
            with torch.no_grad():
                m = _snapshot_metrics(model, xyt_snap, u_snap)
            model.train()
            m["step"] = step
            snap_fh.write(json.dumps(m, ensure_ascii=False) + "\n")
            snap_fh.flush()
            torch.save(
                {"model_state": model.state_dict(), "step": step, "metrics": m},
                os.path.join(save_dir, f"snap_{step}.pt"),
            )
            print(
                f"[{tag}] step={step}/{n_steps} L2={m['l2']:.4f} "
                f"eff={m['effective_experts']:.2f} minLoad={m['min_load']:.3f} "
                f"elapsed={(time.time() - t0) / 60:.1f}min",
                flush=True,
            )
    grad_fh.close()
    snap_fh.close()

    with torch.no_grad():
        model.eval()
        final = _snapshot_metrics(model, xyt_full, u_full)
    with open(os.path.join(save_dir, "final_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"tag": tag, "n_steps": n_steps, **final}, f, ensure_ascii=False, indent=2)
    torch.save(
        {"model_state": model.state_dict(), "step": n_steps},
        os.path.join(save_dir, "final.pt"),
    )
    print(f"[{tag}] FINAL L2={final['l2']:.4f} eff={final['effective_experts']:.2f}", flush=True)
    return final


def run_e2e_long(*, seed: int, root: str, smoke: bool) -> dict:
    tag = f"e2e8000_seed{seed}"
    save_dir = os.path.join(root, tag)
    _set_global_seed(seed)
    problem = Burgers2DProblem(nu=NU, device=DEVICE, dtype=DTYPE, seed=seed)
    reference = problem.generate_reference_solution(nx=33 if smoke else 81, ny=33 if smoke else 81, nt=11 if smoke else 31)
    xyt_full, u_full = flatten_reference_solution(reference, device=DEVICE, dtype=DTYPE)
    ref_snap = problem.generate_reference_solution(nx=25 if smoke else 65, ny=25 if smoke else 65, nt=9 if smoke else 21)
    xyt_snap, u_snap = flatten_reference_solution(ref_snap, device=DEVICE, dtype=DTYPE)
    model = _build_model()
    loss_fn = PhysicsLoss2D(
        LossConfig2D(
            nu=NU,
            w_res=1.0,
            w_ic=5.0,
            w_bc=2.0,
            w_sparse=model.sparsity_weight,
            w_balance=model.balance_weight,
        )
    )
    return run_custom_training(
        model=model,
        problem=problem,
        xyt_full=xyt_full,
        u_full=u_full,
        xyt_snap=xyt_snap,
        u_snap=u_snap,
        loss_fn=loss_fn,
        lr=1e-3,
        n_steps=400 if smoke else 8000,
        snap_every=50 if smoke else 200,
        grad_log_every=20 if smoke else 20,
        freeze_base=False,
        save_dir=save_dir,
        tag=tag,
    )


def run_joint_long(*, seed: int, root: str, smoke: bool) -> dict:
    tag = f"joint8000_seed{seed}"
    save_dir = os.path.join(root, tag)
    _set_global_seed(seed)
    pre_dir = os.path.join(save_dir, "pretrain")
    run_experiment(
        train_mode="staged",
        n_steps=200 if smoke else 1500,
        n_col=3000 if smoke else 12000,
        n_ic=800 if smoke else 3000,
        n_bc_per_face=200 if smoke else 800,
        nx=33 if smoke else 81,
        ny=33 if smoke else 81,
        nt=11 if smoke else 31,
        results_root=pre_dir,
        staged_variant=STAGED_VARIANT,
        gate_variant=GATE_VARIANT,
        wave_expert_variant=WAVE_VARIANT,
        directional_expert_variant=DIRECTIONAL_VARIANT,
        expert_layout_variant="categorical",
        exclude_experts="",
        expert_pretrain_fraction=1.0,
        joint_steps_override=0,
        seed=seed,
    )
    model = _build_model()
    ckpt = torch.load(
        os.path.join(pre_dir, "burgers2d_moe_staged", "burgers2d_pre_joint.pt"),
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(ckpt["model_state"])
    problem = Burgers2DProblem(nu=NU, device=DEVICE, dtype=DTYPE, seed=seed)
    reference = problem.generate_reference_solution(nx=33 if smoke else 81, ny=33 if smoke else 81, nt=11 if smoke else 31)
    xyt_full, u_full = flatten_reference_solution(reference, device=DEVICE, dtype=DTYPE)
    ref_snap = problem.generate_reference_solution(nx=25 if smoke else 65, ny=25 if smoke else 65, nt=9 if smoke else 21)
    xyt_snap, u_snap = flatten_reference_solution(ref_snap, device=DEVICE, dtype=DTYPE)
    loss_fn = PhysicsLoss2D(
        LossConfig2D(
            nu=NU,
            w_res=1.0,
            w_ic=5.0,
            w_bc=2.0,
        )
    )
    return run_custom_training(
        model=model,
        problem=problem,
        xyt_full=xyt_full,
        u_full=u_full,
        xyt_snap=xyt_snap,
        u_snap=u_snap,
        loss_fn=loss_fn,
        lr=5e-4,
        n_steps=400 if smoke else 8000,
        snap_every=50 if smoke else 200,
        grad_log_every=20 if smoke else 20,
        freeze_base=True,
        save_dir=save_dir,
        tag=tag,
    )


def popup(title: str, message: str) -> None:
    ps = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        f"[System.Windows.Forms.MessageBox]::Show('{message}', '{title}')"
    )
    try:
        subprocess.Popen(
            ["powershell", "-NoProfile", "-Command", ps],
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[popup failed] {exc}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = os.path.join(RESULTS, f"mechanism_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(root, exist_ok=True)

    results: dict[str, dict] = {}
    for seed in (42, 43):
        for fn, name in [
            (run_e2e_long, "e2e"),
            (run_joint_long, "joint"),
        ]:
            key = f"{name}_{seed}"
            try:
                results[key] = fn(seed=seed, root=root, smoke=args.smoke)
            except Exception as exc:  # noqa: BLE001
                print(f"[FAILED] {key}: {type(exc).__name__}: {exc}", flush=True)
    with open(os.path.join(root, "mechanism_summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(os.path.join(root, "DONE"), "w", encoding="utf-8") as f:
        f.write(datetime.now().isoformat() + "\n")
    print(f"[OK] Summary: {os.path.join(root, 'mechanism_summary.json')}", flush=True)
    popup(
        "Mechanism Training 完成",
        f"e2e 8000 步 × 2 seeds + joint 8000 步 × 2 seeds 已结束。\n结果目录：{root}",
    )


if __name__ == "__main__":
    main()
