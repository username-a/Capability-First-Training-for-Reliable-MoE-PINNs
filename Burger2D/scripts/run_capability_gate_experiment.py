"""Screen and confirm capability-aware gate objectives on frozen staged experts."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
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
    _set_trainable,
    evaluate_checkpoint,
)
from Burger2D.equations.burgers2d import Burgers2DProblem
from Burger2D.training.losses import LossConfig2D, PhysicsLoss2D
from Burger2D.training.staged_burgers2d import flatten_reference_solution


SCREEN_CANDIDATES = {
    "cap": {"kind": "capability", "lambda_cap": 1.0, "lambda_bal": 0.0},
    "hyb_0p1": {"kind": "hybrid", "lambda_cap": 0.1, "lambda_bal": 0.0},
    "hyb_0p3": {"kind": "hybrid", "lambda_cap": 0.3, "lambda_bal": 0.0},
    "hyb_1": {"kind": "hybrid", "lambda_cap": 1.0, "lambda_bal": 0.0},
    "hyb_3": {"kind": "hybrid", "lambda_cap": 3.0, "lambda_bal": 0.0},
    "hyb_bal_0p1": {"kind": "hybrid", "lambda_cap": 0.1, "lambda_bal": 0.01},
    "hyb_bal_0p3": {"kind": "hybrid", "lambda_cap": 0.3, "lambda_bal": 0.01},
    "hyb_bal_1": {"kind": "hybrid", "lambda_cap": 1.0, "lambda_bal": 0.01},
    "hyb_bal_3": {"kind": "hybrid", "lambda_cap": 3.0, "lambda_bal": 0.01},
}


def _hash_state(state: dict) -> str:
    buf = io.BytesIO()
    torch.save(state, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def _load_source(source_dir: Path, device: torch.device):
    ckpt = torch.load(source_dir / "pre_calibration_checkpoint.pt", map_location=device, weights_only=False)
    cfg = EqualInfoConfig(**ckpt["config"])
    torch.manual_seed(cfg.seed)
    model = _build_model(cfg, device)
    model.load_state_dict(ckpt["model_state"])
    _set_trainable(model, base=False, experts=False, gate=True)
    return model, cfg, ckpt


def _gate_data(model, cfg, device):
    problem = Burgers2DProblem(nu=NU, device=device, dtype=DTYPE, seed=cfg.seed)
    physics_batch = problem.training_batch(cfg.n_col, cfg.n_ic, cfg.n_bc_per_face)
    reference = problem.generate_reference_solution(cfg.teacher_nx, cfg.teacher_ny, cfg.teacher_nt)
    coords, values = flatten_reference_solution(reference, device=device, dtype=DTYPE)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(cfg.seed + 2000)
    count = min(cfg.gate_pool_size, coords.shape[0])
    ids = torch.randperm(coords.shape[0], generator=gen)[:count].to(device)
    gate_coords, gate_values = coords[ids], values[ids]
    with torch.no_grad():
        branches = model.get_expert_predictions(gate_coords).squeeze(-1)
        error = (branches - gate_values.reshape(-1, 1)).abs()
        relative = error / error.mean(dim=1, keepdim=True).clamp_min(1e-8)
        targets = torch.softmax(-cfg.gate_temperature * relative, dim=1)
        top2 = torch.topk(targets, k=2, dim=1).values
        confidence = (top2[:, 0] - top2[:, 1]).detach()
    return physics_batch, coords, gate_coords, targets.detach(), confidence


def _cap_loss(model, coords, targets, confidence):
    probs = model.compute_gate_weights(coords)
    sample = F.kl_div(probs.clamp_min(1e-8).log(), targets, reduction="none").sum(dim=1)
    weights = 0.1 + confidence
    return (weights * sample).sum() / weights.sum().clamp_min(1e-8)


def train_candidate(source_dir: Path, output_dir: Path, candidate: str, steps: int,
                    eval_steps: list[int], device: torch.device,
                    val_grid: tuple[int, int, int]) -> None:
    spec = SCREEN_CANDIDATES[candidate]
    output_dir.mkdir(parents=True, exist_ok=True)
    model, cfg, source = _load_source(source_dir, device)
    initial_gate_hash = _hash_state(model.gating.state_dict())
    initial_expert_hashes = [_hash_state(e.state_dict()) for e in model.experts]
    physics_batch, full_pool, gate_coords, targets, confidence = _gate_data(model, cfg, device)
    # Exclude the full source teacher pool, not merely the sampled gate points,
    # when validation/test grids are evaluated.
    source_pool = np.load(source_dir / "training_pool_coords.npy")
    np.save(output_dir / "training_pool_coords.npy", source_pool)

    physics_fn = PhysicsLoss2D(
        LossConfig2D(nu=NU, w_res=1.0, w_ic=5.0, w_bc=2.0, w_sparse=0.0, w_balance=0.0)
    )
    optimizer = torch.optim.Adam(model.gating.parameters(), lr=cfg.gate_lr)
    history = []
    # Normalize hybrid components at the common initial gate so lambda_cap has
    # a stable interpretation across seeds.
    with torch.enable_grad():
        p0, _ = physics_fn.compute(model, physics_batch)
        c0 = _cap_loss(model, gate_coords, targets, confidence)
        b0 = model.load_balance_loss(gate_coords)
    scales = {
        "physics": max(float(p0.detach().item()), 1e-8),
        "capability": max(float(c0.detach().item()), 1e-8),
        "balance": max(abs(float(b0.detach().item())), 1e-8),
    }
    eval_set = set(eval_steps)
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        cap = _cap_loss(model, gate_coords, targets, confidence)
        if spec["kind"] == "capability":
            physics = torch.zeros((), device=device, dtype=DTYPE)
            balance = torch.zeros((), device=device, dtype=DTYPE)
            loss = cap
        else:
            physics, _ = physics_fn.compute(model, physics_batch)
            balance = model.load_balance_loss(gate_coords)
            loss = (
                physics / scales["physics"]
                + spec["lambda_cap"] * cap / scales["capability"]
                + spec["lambda_bal"] * balance / scales["balance"]
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.gating.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 25 == 0 or step in eval_set:
            history.append({
                "step": step, "loss": float(loss.item()),
                "physics": float(physics.detach().item()),
                "capability": float(cap.detach().item()),
                "balance": float(balance.detach().item()),
            })
        if step in eval_set:
            ckpt_name = f"checkpoint_step{step}.pt"
            torch.save({
                "model_state": model.state_dict(), "config": source["config"],
                "expert_names": list(model.expert_names), "candidate": candidate, "step": step,
            }, output_dir / ckpt_name)
            evaluate_checkpoint(
                output_dir, device, test_nx=val_grid[0], test_ny=val_grid[1], test_nt=val_grid[2],
                checkpoint_name=ckpt_name, metrics_name=f"val_step{step}.json",
            )

    final_expert_hashes = [_hash_state(e.state_dict()) for e in model.experts]
    audit = {
        "valid": initial_expert_hashes == final_expert_hashes,
        "candidate": candidate, "spec": spec, "seed": cfg.seed, "steps": steps,
        "source": str(source_dir), "initial_gate_hash": initial_gate_hash,
        "initial_expert_hashes": initial_expert_hashes,
        "final_expert_hashes": final_expert_hashes,
        "source_pool_hash": _coords_hash(source_pool),
        "gate_pool_hash": _coords_hash(gate_coords.detach().cpu().numpy()),
        "scales": scales, "test_loaded_during_training": False,
    }
    (output_dir / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False), encoding="utf-8")


def select_candidate(root: Path, seeds: list[int], eval_steps: list[int]) -> dict:
    rows = []
    for candidate in SCREEN_CANDIDATES:
        for step in eval_steps:
            metrics = []
            valid = True
            for seed in seeds:
                run = root / "screen" / candidate / f"seed{seed}"
                audit_path, metric_path = run / "audit.json", run / f"val_step{step}.json"
                if not audit_path.exists() or not metric_path.exists():
                    valid = False
                    continue
                valid &= bool(json.loads(audit_path.read_text(encoding="utf-8"))["valid"])
                metrics.append(json.loads(metric_path.read_text(encoding="utf-8")))
            if not metrics:
                continue
            mean = lambda key: float(np.mean([m[key] for m in metrics]))
            row = {
                "candidate": candidate, "step": step, "valid": valid and len(metrics) == len(seeds),
                "l2": mean("l2_relative_error"), "soft_regret": mean("soft_routing_regret"),
                "effective_experts": mean("effective_experts"), "min_load": mean("min_load"),
            }
            row["feasible"] = (
                row["valid"] and row["soft_regret"] <= 0.005
                and row["effective_experts"] >= 3.0 and row["min_load"] >= 0.05
            )
            rows.append(row)
    feasible = [r for r in rows if r["feasible"]]
    if feasible:
        best = min(feasible, key=lambda r: r["l2"])
    else:
        # Deterministic fallback: heavily penalize violations while retaining
        # validation L2 as the main objective.
        def score(r):
            return (r["l2"] + 5 * max(r["soft_regret"] - 0.005, 0)
                    + 0.05 * max(3.0 - r["effective_experts"], 0)
                    + 0.5 * max(0.05 - r["min_load"], 0))
        best = min([r for r in rows if r["valid"]], key=score)
    cap_rows = [r for r in rows if r["candidate"] == "cap" and r["valid"]]
    cap_feasible = [r for r in cap_rows if r["feasible"]]
    best_cap = min(cap_feasible or cap_rows, key=lambda r: r["l2"])
    result = {"screen_seeds": seeds, "rows": rows, "selected": best, "selected_capability": best_cap}
    (root / "selection.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def evaluate_test(run_dir: Path, device: torch.device, step: int) -> None:
    evaluate_checkpoint(
        run_dir, device, test_nx=82, test_ny=83, test_nt=32,
        checkpoint_name=f"checkpoint_step{step}.pt", metrics_name="test_metrics.json",
    )


def _stats(values):
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)), "raw": a.tolist()}


def summarize(root: Path, source_root: Path, seeds: list[int]) -> None:
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    methods = {
        "no_calibration": [(source_root / f"seed{s}_staged" / "pre_metrics.json") for s in seeds],
        "physics_750": [(source_root / f"seed{s}_staged" / "test_metrics.json") for s in seeds],
        "capability": [(root / "confirm" / "capability" / f"seed{s}" / "test_metrics.json") for s in seeds],
        "selected_hybrid": [(root / "confirm" / "selected" / f"seed{s}" / "test_metrics.json") for s in seeds],
    }
    keys = ["l2_relative_error", "soft_routing_regret", "worst_expert_l2", "effective_experts", "min_load", "oracle_hit"]
    out = {"selection": selection["selected"], "selection_capability": selection["selected_capability"], "groups": {}, "paired_vs_no_calibration": {}}
    cache = {}
    for method, paths in methods.items():
        rows = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
        cache[method] = rows
        out["groups"][method] = {key: _stats([r[key] for r in rows]) for key in keys}
    for method in ("physics_750", "capability", "selected_hybrid"):
        out["paired_vs_no_calibration"][method] = {}
        for key in keys:
            diffs = [cache[method][i][key] - cache["no_calibration"][i][key] for i in range(len(seeds))]
            st = _stats(diffs)
            half = 1.96 * st["std"] / math.sqrt(len(diffs))
            st["ci95_normal"] = [st["mean"] - half, st["mean"] + half]
            out["paired_vs_no_calibration"][method][key] = st
    (root / "confirmatory_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--source-dir", required=True); tr.add_argument("--output-dir", required=True)
    tr.add_argument("--candidate", choices=SCREEN_CANDIDATES, required=True)
    tr.add_argument("--steps", type=int, default=750); tr.add_argument("--eval-steps", nargs="+", type=int, default=[50,100,200,400,750])
    tr.add_argument("--device", default="cuda"); tr.add_argument("--val-grid", nargs=3, type=int, default=[66,67,22])
    sel = sub.add_parser("select")
    sel.add_argument("--root", required=True); sel.add_argument("--seeds", nargs="+", type=int, required=True)
    sel.add_argument("--eval-steps", nargs="+", type=int, default=[50,100,200,400,750])
    ev = sub.add_parser("evaluate-test")
    ev.add_argument("--run-dir", required=True); ev.add_argument("--step", type=int, required=True); ev.add_argument("--device", default="cuda")
    sm = sub.add_parser("summarize")
    sm.add_argument("--root", required=True); sm.add_argument("--source-root", required=True); sm.add_argument("--seeds", nargs="+", type=int, required=True)
    args = ap.parse_args()
    if args.command == "select": select_candidate(Path(args.root), args.seeds, args.eval_steps); return
    if args.command == "summarize": summarize(Path(args.root), Path(args.source_root), args.seeds); return
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    if args.command == "evaluate-test": evaluate_test(Path(args.run_dir), device, args.step); return
    train_candidate(Path(args.source_dir), Path(args.output_dir), args.candidate, args.steps, args.eval_steps, device, tuple(args.val_grid))


if __name__ == "__main__":
    main()
