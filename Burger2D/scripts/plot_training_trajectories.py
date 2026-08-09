"""
Plot gate routing trajectories (expert load fractions + route entropy) from
saved Burger2D checkpoints: end-to-end vs staged, and the Stage C ablation.

Outputs:
    Burger2D/results/figures/trajectory_e2e_vs_staged.png
    Burger2D/results/figures/trajectory_stagec_ablation.png
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

RESULTS = os.path.join(PACKAGE_ROOT, "results")
FIG_DIR = os.path.join(RESULTS, "figures")
EXPERT_NAMES = ["smooth", "iso_shock", "directional_shock", "wave"]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def _history(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt["history"]


def _staged_info(path: str) -> tuple[dict, dict]:
    info = torch.load(path, map_location="cpu", weights_only=False)
    return info["gate_history"], info["stage_config"]


def _load_fracs(history: dict) -> np.ndarray:
    return np.stack([np.asarray(history[f"gate_load_{i}"]) for i in range(4)], axis=1)


def _style_ax(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("训练步数", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.3, linewidth=0.6)
    ax.tick_params(labelsize=8)


def plot_e2e_vs_staged() -> None:
    e2e = _history(
        os.path.join(
            RESULTS,
            "full_compare_local_conv_route_sharp_20260414_233000",
            "burgers2d_moe_end_to_end",
            "burgers2d_moe_end_to_end.pt",
        )
    )
    staged = _history(
        os.path.join(
            RESULTS,
            "full_compare_local_conv_route_sharp_20260414_233000",
            "burgers2d_moe_staged",
            "burgers2d_moe_staged.pt",
        )
    )
    gate_hist, cfg = _staged_info(
        os.path.join(
            RESULTS,
            "full_compare_local_conv_route_sharp_20260414_233000",
            "burgers2d_moe_staged",
            "burgers2d_staged_training.pt",
        )
    )

    n_e2e = len(e2e["total"])
    e2e_steps = (np.arange(len(e2e["gate_entropy"])) + 1) * (n_e2e / len(e2e["gate_entropy"]))
    e2e_loads = _load_fracs(e2e)
    e2e_entropy = np.asarray(e2e["gate_entropy"])
    e2e_max = np.asarray(e2e["gate_max"])

    gate_steps_b = int(cfg["gate_steps"])
    joint_steps_c = int(cfg["joint_steps"])
    staged_steps = gate_steps_b + (np.arange(len(staged["gate_entropy"])) + 1) * (
        joint_steps_c / len(staged["gate_entropy"])
    )
    staged_loads = _load_fracs(staged)
    b_entropy = np.asarray(gate_hist["entropy"])
    b_max = np.asarray(gate_hist["max_gate"])

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6))

    # (a) e2e load fractions
    ax = axes[0, 0]
    for i, name in enumerate(EXPERT_NAMES):
        ax.plot(e2e_steps, e2e_loads[:, i], color=COLORS[i], lw=1.8,
                label=f"{name} (终值 {e2e_loads[-1, i]:.1%})")
    ax.set_ylim(-0.02, 1.02)
    _style_ax(ax, "(a) 端到端：专家负载轨迹（1500 步）", "top-1 负载占比")
    ax.legend(fontsize=8, loc="center right")
    ax.annotate("iso_shock 自始至终 < 5%", xy=(e2e_steps[-1], e2e_loads[-1, 1]),
                xytext=(0.30, 0.55), textcoords="axes fraction", fontsize=9,
                color=COLORS[1], arrowprops=dict(arrowstyle="->", color=COLORS[1], lw=1.2))

    # (b) e2e entropy / max weight
    ax = axes[0, 1]
    ax.plot(e2e_steps, e2e_entropy, color="#7F7F7F", lw=1.8, label="路由熵")
    ax.plot(e2e_steps, e2e_max, color="#9467BD", lw=1.8, ls="--", label="平均 top-1 权重")
    ax.set_ylim(0.0, 1.5)
    _style_ax(ax, "(b) 端到端：路由熵与门控决断度", "数值")
    ax.legend(fontsize=9)
    ax.annotate("熵长期停在 0.9 附近，门控学不会决断路由",
                xy=(e2e_steps[-1], e2e_entropy[-1]), xytext=(0.25, 0.68),
                textcoords="axes fraction", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#7F7F7F", lw=1.2))

    # (c) staged load fractions (Stage B -> C)
    ax = axes[1, 0]
    for i, name in enumerate(EXPERT_NAMES):
        ax.plot(staged_steps, staged_loads[:, i], color=COLORS[i], lw=1.8,
                label=f"{name} (终值 {staged_loads[-1, i]:.1%})")
    ax.axvline(gate_steps_b, color="k", ls=":", lw=1.2)
    ax.text(gate_steps_b + 8, 0.94, "Stage B→C", fontsize=8, color="k")
    ax.set_ylim(-0.02, 1.02)
    _style_ax(ax, "(c) 分阶段：专家负载轨迹（Stage B 门控训练 → Stage C 仅门控微调）",
              "top-1 负载占比")
    ax.legend(fontsize=8, loc="lower right")

    # (d) staged entropy (Stage B + C)
    ax = axes[1, 1]
    b_steps = np.arange(1, len(b_entropy) + 1)
    ax.plot(b_steps, b_entropy, color="#7F7F7F", lw=1.6, label="路由熵（Stage B）")
    ax.plot(staged_steps, staged["gate_entropy"], color="#7F7F7F", lw=1.8,
            ls="-", marker="o", ms=3, label="路由熵（Stage C）")
    ax.plot(b_steps, b_max, color="#9467BD", lw=1.4, ls="--", label="平均 top-1 权重（Stage B）")
    ax.plot(staged_steps, staged["gate_max"], color="#9467BD", lw=1.6, ls="--",
            marker="s", ms=3, label="平均 top-1 权重（Stage C）")
    ax.axvline(gate_steps_b, color="k", ls=":", lw=1.2)
    ax.text(gate_steps_b + 8, 1.42, "Stage B→C", fontsize=8, color="k")
    ax.set_ylim(0.0, 1.5)
    _style_ax(ax, "(d) 分阶段：路由熵与门控决断度", "数值")
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("端到端 vs 分阶段：门控路由轨迹（2D Burgers 正式对比）",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "trajectory_e2e_vs_staged.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("[OK]", out)


def plot_stagec_ablation() -> None:
    root = os.path.join(RESULTS, "stagec_ablation_20260412_1835")
    variants = {
        "default（联合更新专家）": os.path.join(root, "default"),
        "gate_only（仅更新门控）": os.path.join(root, "gate_only_joint_retry"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    saved = {}
    for ax, (label, run_dir) in zip(axes, variants.items()):
        hist = _history(os.path.join(run_dir, "burgers2d_moe_staged", "burgers2d_moe_staged.pt"))
        n = len(hist["total"])
        steps = (np.arange(len(hist["gate_entropy"])) + 1) * (n / len(hist["gate_entropy"]))
        loads = _load_fracs(hist)
        saved[label] = (steps, loads)
        for i, name in enumerate(EXPERT_NAMES):
            ax.plot(steps, loads[:, i], color=COLORS[i], lw=1.8,
                    label=f"{name} (终值 {loads[-1, i]:.1%})")
        ax.set_ylim(-0.02, 1.02)
        _style_ax(ax, label, "top-1 负载占比")
        ax.legend(fontsize=8)
    steps_d, loads_d = saved["default（联合更新专家）"]
    axes[0].annotate("directional_shock 飙升到 61.2%\nsmooth 被饿到 8.8%",
                     xy=(steps_d[-1], loads_d[-1, 2]), xytext=(0.32, 0.72),
                     textcoords="axes fraction", fontsize=9,
                     color=COLORS[2], arrowprops=dict(arrowstyle="->", color=COLORS[2], lw=1.2))
    steps_g, loads_g = saved["gate_only（仅更新门控）"]
    axes[1].annotate("四专家保持均衡（≥15%）", xy=(steps_g[-1], loads_g[-1, 2]),
                     xytext=(0.30, 0.68), textcoords="axes fraction", fontsize=9,
                     color=COLORS[2], arrowprops=dict(arrowstyle="->", color=COLORS[2], lw=1.2))
    fig.suptitle("Stage C 联合微调消融：是否继续更新专家（1250 步，同一套专家预训练产物）",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "trajectory_stagec_ablation.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("[OK]", out)


def main() -> None:
    plot_e2e_vs_staged()
    plot_stagec_ablation()


if __name__ == "__main__":
    main()
