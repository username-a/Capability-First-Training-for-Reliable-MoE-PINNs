"""
Re-evaluate the Burgers gate-introduction checkpoints on the conservative WENO5
reference subset (the same 344,064-point evaluation set used for the main
results), and write an aggregated summary.

Outputs:
    Burger2D/results/jcp_reference_rebuild_20260808/gate_intro_reevaluation.json
    Burger2D/results/jcp_reference_rebuild_20260808/gate_intro_summary_weno5.json
"""

from __future__ import annotations

import argparse
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

from Burger2D.scripts.plot_gate_intro_maps import load_model  # noqa: E402


RESULTS = os.path.join(PACKAGE_ROOT, "results")
RUN_ROOT = os.path.join(RESULTS, "gate_intro_ablation_20260803_122825")
OUT_DIR = os.path.join(RESULTS, "jcp_reference_rebuild_20260808")
LOG_PATH = os.path.join(OUT_DIR, "gate_intro_reeval_progress.log")
FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEEDS = [42, 43, 44]


def log(msg: str) -> None:
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    print(msg, flush=True)


def _batched(fn, x: torch.Tensor, batch: int = 65536) -> torch.Tensor:
    out = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch):
            out.append(fn(x[i : i + batch]).cpu())
    return torch.cat(out, dim=0)


def evaluate_one(model, coords: torch.Tensor, u_ref: torch.Tensor) -> dict:
    u_mix = _batched(model, coords).squeeze(-1).numpy()
    expert = _batched(model.get_expert_predictions, coords).numpy()[:, :, 0]
    gates = _batched(model.get_gate_weights, coords).numpy()
    ur = np.asarray(u_ref, dtype=np.float64).ravel()
    um = u_mix.astype(np.float64)
    denom = np.sqrt(np.sum(ur**2))
    l2 = float(np.sqrt(np.sum((um - ur) ** 2)) / denom)
    maxerr = float(np.max(np.abs(um - ur)))
    per_expert = [
        float(np.sqrt(np.sum((expert[:, k].astype(np.float64) - ur) ** 2)) / denom)
        for k in range(expert.shape[1])
    ]
    ent = float(-(gates * np.log(gates + 1e-12)).sum(axis=1).mean())
    mxw = float(gates.max(axis=1).mean())
    mean_g = gates.mean(axis=0)
    eff = float(np.exp(-(mean_g * np.log(mean_g + 1e-12)).sum()))
    return {
        "l2_relative_error": l2,
        "max_absolute_error": maxerr,
        "per_expert_l2": per_expert,
        "max_expert_l2": float(max(per_expert)),
        "std_expert_l2": float(np.std(per_expert)),
        "route_entropy": ent,
        "route_max_weight": mxw,
        "effective_experts": eff,
        "min_load_frac": float(mean_g.min()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    sub = np.load(os.path.join(OUT_DIR, "evaluation_subset.npz"), mmap_mode="r")
    x, y, t, u = sub["x"], sub["y"], sub["t"], sub["u"]
    # evaluation_subset.npz stores u as (t, y, x); build coordinates in the
    # same axis order so ravel() indexing matches u_ref.
    tt, yy, xx = np.meshgrid(t, y, x, indexing="ij")
    coords = torch.tensor(
        np.stack([xx.ravel(), yy.ravel(), tt.ravel()], axis=1),
        dtype=torch.float32,
        device=device,
    )
    u_ref = np.asarray(u).ravel()

    entries = []
    for f in FRACTIONS:
        for s in SEEDS:
            tag = f"f{int(f * 100)}_seed{s}"
            ckpt = os.path.join(RUN_ROOT, tag, "burgers2d_moe_staged", "burgers2d_moe_staged.pt")
            t0 = time.time()
            model = load_model(ckpt)
            if model is None:
                raise RuntimeError(f"could not load model for {tag}")
            model = model.to(device).eval()
            metrics = evaluate_one(model, coords, u_ref)
            metrics.update({"fraction": f, "seed": s, "elapsed_sec": time.time() - t0})
            entries.append(metrics)
            log(f"[{tag}] L2={metrics['l2_relative_error']:.4f} "
                f"worst={metrics['max_expert_l2']:.4f} eff={metrics['effective_experts']:.3f} "
                f"entropy={metrics['route_entropy']:.3f} min_load={metrics['min_load_frac']:.3f}")

    with open(os.path.join(OUT_DIR, "gate_intro_reevaluation.json"), "w", encoding="utf-8") as fh:
        json.dump({"protocol": {"reference": "conservative WENO5 513-grid subset (21 times, 128x128 spatial indices 1,5,...,509)",
                                "points": int(coords.shape[0]), "checkpoints": "gate_intro_ablation_20260803_122825"},
                   "runs": entries}, fh, ensure_ascii=False, indent=2)

    keys = ["l2_relative_error", "max_absolute_error", "route_entropy", "route_max_weight",
            "effective_experts", "min_load_frac", "max_expert_l2", "std_expert_l2"]
    rows = {}
    for f in FRACTIONS:
        rows[str(f)] = {"n": len(SEEDS)}
        for k in keys:
            vals = [e[k] for e in entries if abs(e["fraction"] - f) < 1e-9]
            rows[str(f)][k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1))}
    with open(os.path.join(OUT_DIR, "gate_intro_summary_weno5.json"), "w", encoding="utf-8") as fh:
        json.dump({"fractions": FRACTIONS, "seeds": SEEDS, "rows": rows,
                   "reference": "conservative WENO5"}, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT_DIR, "gate_intro_reeval_done"), "w", encoding="utf-8") as fh:
        fh.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    log("ALL DONE -> " + os.path.join(OUT_DIR, "gate_intro_summary_weno5.json"))


if __name__ == "__main__":
    main()
