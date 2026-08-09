"""
KdV two-soliton gate-introduction continuum (f = 0/25/50/75/100%, 3 seeds).

Writes per-run metrics and a status file for the dashboard.
Usage:
    python burger1D/scripts/run_gate_intro_kdv.py [--smoke]
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
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from core.moe_pinn import build_kdv_moe  # noqa: E402
from equations.kdv import KdVEquation  # noqa: E402
from training.loss_functions import l2_relative_error  # noqa: E402
from training.staged_kdv import (  # noqa: E402
    StagedKdvConfig,
    build_kdv_gate_targets,
    build_kdv_specialist_batches,
    joint_finetune_kdv,
    pretrain_kdv_experts,
    train_kdv_gate,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
RESULTS = os.path.join(PACKAGE_ROOT, "results")
FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEEDS = [42, 43, 44]
STATUS_PATH = os.path.join(RESULTS, "kdv_gate_intro_status.json")


def run_one(*, fraction: float, seed: int, smoke: bool, root: str) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    eq = KdVEquation(device=DEVICE, dtype=DTYPE)
    n_col, n_ic, n_bc = (1200, 150, 60) if smoke else (5000, 400, 100)
    xt_ic, u_ic = eq.initial_condition(n_ic)
    xt_bc, u_bc = eq.boundary_condition(n_bc)
    xt_col = eq.collocation_points(n_col, method="lhs")
    batch = {
        "xt_col": xt_col,
        "xt_ic": xt_ic,
        "u_ic": u_ic,
        "xt_bc": xt_bc,
        "u_bc": u_bc,
    }
    nx, nt = (64, 20) if smoke else (128, 60)
    X, T, U_exact = eq.test_grid(nx=nx, nt=nt)
    xt_test = torch.stack(
        [
            torch.tensor(X.ravel(), dtype=DTYPE, device=DEVICE),
            torch.tensor(T.ravel(), dtype=DTYPE, device=DEVICE),
        ],
        dim=-1,
    )
    u_exact_flat = torch.tensor(U_exact.ravel(), dtype=DTYPE, device=DEVICE).unsqueeze(-1)

    model = build_kdv_moe(
        num_experts=3,
        sparsity_weight=1e-3,
        balance_weight=5e-3,
        gate_temperature=0.9,
    ).to(DEVICE).to(DTYPE)
    total = 1500 if smoke else 4000
    cfg = StagedKdvConfig.from_total_steps(total)
    if fraction < 1.0:
        cfg.expert_pretrain_fraction = fraction
        cfg.gate_steps = 0

    save_dir = os.path.join(root, f"f{int(fraction * 100)}_seed{seed}")
    os.makedirs(save_dir, exist_ok=True)

    batches = build_kdv_specialist_batches(X, T, U_exact, batch, cfg, seed=seed)
    if cfg.expert_pretrain_fraction > 0:
        pretrain_kdv_experts(
            model, batches, cfg, eq, xt_test, u_exact_flat, save_dir,
        )
    if cfg.gate_steps > 0:
        targets = build_kdv_gate_targets(model, xt_test, u_exact_flat, cfg.gate_target_temperature)
        train_kdv_gate(model, xt_test, targets, cfg)

    joint_finetune_kdv(model, batch, cfg, eq, xt_test, u_exact_flat, save_dir)

    with torch.no_grad():
        pred = model(xt_test)
        expert_preds = torch.stack([e(xt_test) for e in model.experts], dim=1).squeeze(-1)
        gates = model.get_gate_weights(xt_test)
    l2 = float(l2_relative_error(pred, u_exact_flat))
    maxerr = float((pred - u_exact_flat).abs().max().item())
    per_expert_l2 = [
        float(l2_relative_error(expert_preds[:, k:k + 1], u_exact_flat))
        for k in range(3)
    ]
    arg = gates.argmax(dim=1)
    load_frac = [float((arg == k).float().mean()) for k in range(3)]
    entropy = float(-(gates * gates.clamp_min(1e-8).log()).sum(dim=1).mean().item())
    entry = {
        "fraction": fraction,
        "seed": seed,
        "l2": l2,
        "maxerr": maxerr,
        "per_expert_l2": per_expert_l2,
        "load_frac": load_frac,
        "route_entropy": entropy,
        "elapsed_sec": 0.0,
    }
    with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False, indent=2)
    return entry


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
    root = os.path.join(RESULTS, f"gate_intro_kdv_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(root, exist_ok=True)
    status = {"root": root, "done": {}, "current": None, "total": len(FRACTIONS) * len(SEEDS)}

    def write_status() -> None:
        status["updated"] = datetime.now().isoformat()
        with open(STATUS_PATH, "w", encoding="utf-8") as f:
            json.dump(status, f, ensure_ascii=False, indent=2)

    for fraction in FRACTIONS:
        for seed in SEEDS:
            tag = f"f{int(fraction * 100)}_seed{seed}"
            print(f"[KdV gate-intro] {tag} starting", flush=True)
            status["current"] = tag
            write_status()
            t0 = time.time()
            try:
                entry = run_one(fraction=fraction, seed=seed, smoke=args.smoke, root=root)
                entry["elapsed_sec"] = time.time() - t0
                entry["tag"] = tag
                status["done"][tag] = entry
                print(f"[KdV gate-intro] {tag} done L2={entry['l2']:.4f} in {entry['elapsed_sec']:.0f}s", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[FAILED] {tag}: {type(exc).__name__}: {exc}", flush=True)
                status["done"][tag] = {"tag": tag, "error": str(exc)}
            status["current"] = None
            status["completed"] = len(status["done"])
            write_status()
    with open(os.path.join(root, "DONE"), "w", encoding="utf-8") as f:
        f.write(datetime.now().isoformat() + "\n")
    popup("KdV Gate-Intro 完成", f"KdV gate 引入点连续谱已完成。\n结果目录：{root}")


if __name__ == "__main__":
    main()
