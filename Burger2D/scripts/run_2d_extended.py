"""
2D Burgers extended experiments:

A. balance sweep   : e2e 8000 steps x balance {0.1, 0.5} x 3 seeds
B. K sensitivity   : K in {2, 3, 4, 6} x {staged, e2e} x 2 seeds (1500 config)

Writes `extended_status.json` (updated after every run) for the dashboard.
Usage:
    python Burger2D/scripts/run_2d_extended.py [--smoke]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Burger2D.experiments.run_burgers2d import run_experiment  # noqa: E402


RESULTS = os.path.join(PACKAGE_ROOT, "results")
BALANCE_SWEEP = [0.1, 0.5]
BALANCE_SEEDS = [42, 43, 44]
KS = [2, 3, 4, 6]
K_SEEDS = [42, 43]
PROTOCOLS = ["staged", "end_to_end"]
STAGED_VARIANT = "stronger_expert_route_sharp"


def k_config(k: int) -> dict:
    if k == 2:
        return {"exclude_experts": "directional_shock,wave", "extra_experts": ""}
    if k == 3:
        return {"exclude_experts": "directional_shock", "extra_experts": ""}
    if k == 4:
        return {"exclude_experts": "", "extra_experts": ""}
    if k == 6:
        return {"exclude_experts": "", "extra_experts": "vortex,wave2"}
    raise ValueError(k)


def write_status(path: str, status: dict) -> None:
    status["updated"] = datetime.now().isoformat()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)


def run_job(job: dict, root: str, status_path: str, status: dict, smoke: bool) -> dict:
    tag = job["tag"]
    run_dir = os.path.join(root, tag)
    os.makedirs(run_dir, exist_ok=True)
    t0 = time.time()
    print(f"\n{'='*72}\n[{job['stream']}] {tag} starting\n{'='*72}", flush=True)
    status["current"] = {"tag": tag, "stream": job["stream"], "started": datetime.now().isoformat()}
    write_status(status_path, status)

    steps = 200 if smoke else 8000
    grid = 33 if smoke else 81
    if job["stream"] == "balance_sweep":
        metrics = run_experiment(
            train_mode="end_to_end",
            n_steps=steps,
            n_col=3000 if smoke else 12000,
            n_ic=800 if smoke else 3000,
            n_bc_per_face=200 if smoke else 800,
            nx=grid,
            ny=grid,
            nt=11 if smoke else 31,
            results_root=run_dir,
            directional_expert_variant="hybrid",
            wave_expert_variant="mixed_lite",
            expert_layout_variant="categorical",
            gate_variant="local_conv",
            rotation_variant="none",
            exclude_experts="",
            extra_experts="",
            balance_weight_override=job["balance"],
            seed=job["seed"],
        )["end_to_end"]
    else:  # k_sensitivity
        kc = k_config(job["k"])
        metrics = run_experiment(
            train_mode=job["protocol"],
            n_steps=200 if smoke else 1500,
            n_col=3000 if smoke else 12000,
            n_ic=800 if smoke else 3000,
            n_bc_per_face=200 if smoke else 800,
            nx=grid,
            ny=grid,
            nt=11 if smoke else 31,
            results_root=run_dir,
            staged_variant=STAGED_VARIANT if job["protocol"] == "staged" else "gate_only_joint",
            directional_expert_variant="hybrid",
            wave_expert_variant="mixed_lite",
            expert_layout_variant="categorical",
            gate_variant="local_conv",
            rotation_variant="none",
            exclude_experts=kc["exclude_experts"],
            extra_experts=kc["extra_experts"],
            seed=job["seed"],
        )[job["protocol"] if job["protocol"] != "staged" else "staged"]

    elapsed = time.time() - t0
    entry = {
        "tag": tag,
        "stream": job["stream"],
        "elapsed_sec": elapsed,
        "metrics": {k: v for k, v in metrics.items() if not isinstance(v, (list, dict))},
        "load_frac": metrics.get("expert_load_frac", []),
        "per_expert_l2": metrics.get("per_expert_l2", []),
    }
    status["done"][tag] = entry
    status["completed"] = len(status["done"])
    status["current"] = None
    write_status(status_path, status)
    print(
        f"[{tag}] done in {elapsed:.0f}s L2={metrics.get('l2_relative_error', float('nan')):.4f}",
        flush=True,
    )
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


def build_queue() -> list[dict]:
    queue = []
    for bal in BALANCE_SWEEP:
        for seed in BALANCE_SEEDS:
            queue.append(
                {
                    "stream": "balance_sweep",
                    "tag": f"bal{bal}_seed{seed}",
                    "balance": bal,
                    "seed": seed,
                }
            )
    for k in KS:
        for proto in PROTOCOLS:
            for seed in K_SEEDS:
                queue.append(
                    {
                        "stream": "k_sensitivity",
                        "tag": f"k{k}_{proto}_seed{seed}",
                        "k": k,
                        "protocol": proto,
                        "seed": seed,
                    }
                )
    return queue


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    root = os.path.join(RESULTS, f"extended2d_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(root, exist_ok=True)
    status_path = os.path.join(RESULTS, "extended_status.json")
    queue = build_queue()
    status = {
        "root": root,
        "queue": queue,
        "current": None,
        "done": {},
        "completed": 0,
        "total": len(queue),
    }
    write_status(status_path, status)

    for job in queue:
        try:
            run_job(job, root, status_path, status, args.smoke)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAILED] {job['tag']}: {type(exc).__name__}: {exc}", flush=True)
            status["done"][job["tag"]] = {"tag": job["tag"], "error": str(exc)}
            status["completed"] = len(status["done"])
            write_status(status_path, status)

    with open(os.path.join(root, "DONE"), "w", encoding="utf-8") as f:
        f.write(datetime.now().isoformat() + "\n")
    popup("2D Extended Experiments 完成", f"balance 扫描 + K 敏感性已完成。\n结果目录：{root}")


if __name__ == "__main__":
    main()
