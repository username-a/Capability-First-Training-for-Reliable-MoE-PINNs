"""Aggregate theory metrics across seeds and render the theory-validation figures."""

from __future__ import annotations

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = r"Burger2D\results\mechanism_training_20260803_230356"
FIG = r"docs\paper1_figures"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def agg(kind):
    series = {}
    for seed in (42, 43, 44):
        rows = read_jsonl(os.path.join(ROOT, f"{kind}8000_seed{seed}", "theory_metrics.jsonl"))
        for r in rows:
            series.setdefault(r["step"], []).append(r)
    steps = sorted(series)
    out = {}
    for key in [
        "orc_mse", "eta", "zeta", "mix_mse", "cancel_ratio",
        "gap_q10", "gap_q50", "gap_q90", "gap_steep_q50",
        "flip_adjacent", "flip_from_endpoint",
    ]:
        out[key] = {
            "mean": np.array([np.mean([r[key] for r in series[s]]) for s in steps]),
            "std": np.array([np.std([r[key] for r in series[s]]) for s in steps]),
        }
    return steps, out


def main():
    steps_j, j = agg("joint")
    steps_e, e = agg("e2e")

    # Fig A: error decomposition for joint degradation
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    ax = axes[0]
    ax.errorbar(steps_j, j["mix_mse"]["mean"], yerr=j["mix_mse"]["std"], marker="o", ms=3,
                capsize=3, lw=1.6, color="#C44E52", label="mixture MSE")
    ax.errorbar(steps_j, j["orc_mse"]["mean"], yerr=j["orc_mse"]["std"], marker="s", ms=3,
                capsize=3, lw=1.6, color="#4C72B0", label="oracle MSE (eps_orc^2)")
    ax.errorbar(steps_j, j["zeta"]["mean"], yerr=j["zeta"]["std"], marker="^", ms=3,
                capsize=3, lw=1.6, color="#55A868", label="non-oracle routing cost zeta")
    ax.set_xlabel("Joint fine-tuning steps"); ax.set_ylabel("MSE")
    ax.set_title("(a) Error decomposition: oracle vs routing cost")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1]
    ax.errorbar(steps_j, j["cancel_ratio"]["mean"], yerr=j["cancel_ratio"]["std"],
                marker="o", ms=3, capsize=3, lw=1.6, color="#9467BD", label="joint 8000")
    ax.errorbar(steps_e, e["cancel_ratio"]["mean"], yerr=e["cancel_ratio"]["std"],
                marker="s", ms=3, capsize=3, lw=1.6, color="#FF7F0E", label="e2e 8000")
    ax.axhline(1.0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("Training steps"); ax.set_ylabel("Error cancellation ratio")
    ax.set_title("(b) Mixture improvement from error cancellation")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[2]
    ax.errorbar(steps_j, j["flip_from_endpoint"]["mean"], yerr=j["flip_from_endpoint"]["std"],
                marker="o", ms=3, capsize=3, lw=1.6, color="#C44E52", label="joint: flip vs Stage-B endpoint")
    ax.errorbar(steps_j, j["flip_adjacent"]["mean"], yerr=j["flip_adjacent"]["std"],
                marker="s", ms=3, capsize=3, lw=1.6, color="#4C72B0", label="joint: adjacent snapshots")
    ax.errorbar(steps_e, e["flip_adjacent"]["mean"], yerr=e["flip_adjacent"]["std"],
                marker="^", ms=3, capsize=3, lw=1.6, color="#FF7F0E", label="e2e: adjacent snapshots")
    ax.set_xlabel("Training steps"); ax.set_ylabel("Oracle label flip rate")
    ax.set_title("(c) Capability-terrain drift (oracle label instability)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Theory validation (Eq 31 / Prop 2 / Thm 2): joint fine-tuning degrades routing", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(FIG, "theory_joint_decomposition.png"), dpi=150)
    plt.close(fig)
    print("[OK] theory_joint_decomposition.png")

    # Fig B: capability gap quantiles
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.errorbar(steps_j, j["gap_q50"]["mean"], yerr=j["gap_q50"]["std"],
                marker="o", ms=3, capsize=3, lw=1.6, color="#4C72B0", label="joint: median gap")
    ax.errorbar(steps_j, j["gap_q10"]["mean"], yerr=j["gap_q10"]["std"],
                marker="s", ms=3, capsize=3, lw=1.6, color="#55A868", label="joint: q10 gap")
    ax.set_yscale("log")
    ax.set_xlabel("Joint fine-tuning steps"); ax.set_ylabel("Capability gap Delta(z)")
    ax.set_title("Capability gap distribution shrinks as experts drift")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "theory_capability_gap.png"), dpi=150)
    plt.close(fig)
    print("[OK] theory_capability_gap.png")

    # summary table (final-step values)
    print(f"\n{'quantity':<24}{'joint final':>14}{'e2e final':>14}")
    for key, label in [
        ("orc_mse", "oracle MSE"),
        ("eta", "soft misroute eta"),
        ("zeta", "routing cost zeta"),
        ("mix_mse", "mixture MSE"),
        ("cancel_ratio", "cancellation ratio"),
        ("gap_q50", "median capability gap"),
        ("flip_from_endpoint", "flip vs Stage-B endpoint (joint)"),
    ]:
        jv = j[key]["mean"][-1]
        ev = e[key]["mean"][-1] if len(e[key]["mean"]) else float("nan")
        print(f"{label:<24}{jv:>14.4e}{ev:>14.4e}")


if __name__ == "__main__":
    main()
