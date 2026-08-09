"""Soft-weight mean/RMS and raw gradient norms vs steps (gradient starvation)."""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = r"Burger2D\results\mechanism_training_20260803_230356"
FIG = r"docs\paper1_figures"
EXPERT_NAMES = ["smooth", "iso_shock", "directional_shock", "wave"]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


kind = "e2e8000"
seed = 42
theory = read_jsonl(os.path.join(ROOT, f"{kind}_seed{seed}", "theory_metrics.jsonl"))
grad = read_jsonl(os.path.join(ROOT, f"{kind}_seed{seed}", "grad_norms.jsonl"))
step_th = [r["step"] for r in theory]
step_g = [r["step"] for r in grad]

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
ax = axes[0]
for k, name in enumerate(EXPERT_NAMES):
    vals = [r["soft_mean"][k] for r in theory]
    ax.plot(step_th, vals, marker="o", ms=3, lw=1.6, color=COLORS[k], label=name)
ax.set_xlabel("End-to-end training steps")
ax.set_ylabel("Mean soft weight g_k")
ax.set_title("(a) Soft weights: iso_shock is starved from step 1")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

ax = axes[1]
for k, name in enumerate(EXPERT_NAMES):
    vals = [r.get(f"expert_{k}", np.nan) for r in grad]
    ax.plot(step_g, vals, lw=1.4, color=COLORS[k], label=name)
ax.set_xlabel("End-to-end training steps")
ax.set_ylabel("Raw gradient norm before optimizer step")
ax.set_title("(b) Raw gradient norms: starved expert receives ~zero gradient")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

fig.suptitle("Gradient starvation (Eq 13): low soft weight -> low expert gradient (e2e seed42)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(os.path.join(FIG, "theory_grad_starvation.png"), dpi=150)
plt.close(fig)
print("[OK] theory_grad_starvation.png")
