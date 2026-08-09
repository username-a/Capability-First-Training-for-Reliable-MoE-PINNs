"""
1D Burgers gate-introduction continuum (f = 0/25/50/75/100%, 3 seeds).

Reuses the 1D staged pipeline: base pretrain -> experts pretrained to f ->
gate introduced (cold at f<1, Stage B at f=1) -> joint fine-tuning
(base frozen for f<1, experts+gate updated; mainline behavior at f=1).

Writes per-run metrics and a status file for the dashboard.
Usage:
    python burger1D/scripts/run_gate_intro_1d.py [--smoke]
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

from core.moe_pinn import build_burgers_moe  # noqa: E402
from equations.burgers import BurgersEquation  # noqa: E402
from training.loss_functions import l2_relative_error  # noqa: E402
from training.staged_burgers import (  # noqa: E402
    StagedBurgersConfig,
    build_gate_targets,
    build_specialist_batches,
    joint_finetune,
    pretrain_base_model,
    pretrain_experts,
    train_gate,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
NU = 0.01 / np.pi
RESULTS = os.path.join(PACKAGE_ROOT, "results")
FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEEDS = [42, 43, 44]
STATUS_PATH = os.path.join(RESULTS, "gate_intro_1d_status.json")


def _freeze(module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)


def _unfreeze(module) -> None:
    for p in module.parameters():
        p.requires_grad_(True)


def run_one(*, fraction: float, seed: int, smoke: bool, root: str) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    eq = BurgersEquation(nu=NU, device=DEVICE, dtype=DTYPE)
    n_col, n_ic, n_bc = (600, 100, 50) if smoke else (3000, 200, 100)
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
    nx, nt = (40, 20) if smoke else (100, 50)
    X, T, U_exact = eq.test_grid(nx=nx, nt=nt)
    xt_test = torch.stack(
        [
            torch.tensor(X.ravel(), dtype=DTYPE, device=DEVICE),
            torch.tensor(T.ravel(), dtype=DTYPE, device=DEVICE),
        ],
        dim=-1,
    )
    u_exact_flat = torch.tensor(U_exact.ravel(), dtype=DTYPE, device=DEVICE).unsqueeze(-1)

    model = build_burgers_moe(
        num_experts=3,
        use_fourier=True,
        sparsity_weight=1e-3,
        balance_weight=2e-2,
        gate_temperature=0.9,
    ).to(DEVICE).to(DTYPE)
    total = 2000 if smoke else 8000
    cfg = StagedBurgersConfig.from_total_steps(total)
    if fraction < 1.0:
        cfg.expert_steps = int(cfg.expert_steps * fraction)
        cfg.gate_steps = 0

    save_dir = os.path.join(root, f"f{int(fraction * 100)}_seed{seed}")
    os.makedirs(save_dir, exist_ok=True)

    pretrain_base_model(
        model,
        batch,
        cfg=cfg,
        nu=NU,
        save_dir=save_dir,
        xt_test=xt_test,
        u_exact_flat=u_exact_flat,
    )
    expert_batches = build_specialist_batches(X, T, U_exact, batch, cfg=cfg)
    x_vals = X[0, :]
    grad_x = np.abs(np.gradient(U_exact, x_vals, axis=1))
    dx = float(np.mean(np.diff(x_vals)))
    margin = max(6.0 * dx, 0.12 * (x_vals.max() - x_vals.min()))
    interior = (x_vals >= x_vals.min() + margin) & (x_vals <= x_vals.max() - margin)
    center_idx = np.flatnonzero(interior)[np.abs(grad_x[:, interior]).argmax(axis=1)]
    shock_band = np.abs(X - x_vals[center_idx][:, None]) <= max(
        4.0 * dx,
        0.08 * (x_vals.max() - x_vals.min()),
    )
    shock_mask_flat = torch.tensor(shock_band.ravel(), dtype=torch.bool, device=xt_test.device)
    expert_histories = {}
    if cfg.expert_steps > 0:
        expert_histories = pretrain_experts(
            model,
            expert_batches,
            cfg=cfg,
            nu=NU,
            save_dir=save_dir,
            xt_test=xt_test,
            u_exact_flat=u_exact_flat,
            shock_mask_flat=shock_mask_flat,
        )
    if cfg.gate_steps > 0:
        gate_targets = build_gate_targets(
            model,
            xt_test,
            u_exact_flat,
            temperature=cfg.gate_target_temperature,
            cfg=cfg,
        )
        train_gate(model, xt_test, gate_targets, cfg=cfg)
    else:
        _freeze(model.base_model)
    joint_history = joint_finetune(
        model,
        batch,
        cfg=cfg,
        nu=NU,
        use_ntk=True,
        device=DEVICE,
        save_dir=save_dir,
        xt_test=xt_test,
        u_exact_flat=u_exact_flat,
    )
    _unfreeze(model.base_model)

    with torch.no_grad():
        pred = model(xt_test)
    l2 = float(l2_relative_error(pred, u_exact_flat))
    maxerr = float((pred - u_exact_flat).abs().max().item())
    load_stats = model.load_balance_stats(xt_test) if hasattr(model, "load_balance_stats") else {}
    entry = {
        "fraction": fraction,
        "seed": seed,
        "l2": l2,
        "maxerr": maxerr,
        "elapsed_sec": 0.0,
        "load_frac": load_stats.get("expert_load_frac", []),
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
    root = os.path.join(RESULTS, f"gate_intro_1d_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(root, exist_ok=True)
    status = {"root": root, "done": {}, "current": None, "total": len(FRACTIONS) * len(SEEDS)}

    def write_status() -> None:
        status["updated"] = datetime.now().isoformat()
        with open(STATUS_PATH, "w", encoding="utf-8") as f:
            json.dump(status, f, ensure_ascii=False, indent=2)

    for fraction in FRACTIONS:
        for seed in SEEDS:
            tag = f"f{int(fraction * 100)}_seed{seed}"
            t0 = time.time()
            print(f"[1D gate-intro] {tag} starting", flush=True)
            status["current"] = tag
            write_status()
            try:
                entry = run_one(fraction=fraction, seed=seed, smoke=args.smoke, root=root)
                entry["elapsed_sec"] = time.time() - t0
                entry["tag"] = tag
                status["done"][tag] = entry
                print(f"[1D gate-intro] {tag} done L2={entry['l2']:.4f} in {entry['elapsed_sec']:.0f}s", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[FAILED] {tag}: {type(exc).__name__}: {exc}", flush=True)
                status["done"][tag] = {"tag": tag, "error": str(exc)}
            status["current"] = None
            status["completed"] = len(status["done"])
            write_status()
    with open(os.path.join(root, "DONE"), "w", encoding="utf-8") as f:
        f.write(datetime.now().isoformat() + "\n")
    popup("1D Gate-Intro 完成", f"1D gate 引入点连续谱已完成。\n结果目录：{root}")


if __name__ == "__main__":
    main()
