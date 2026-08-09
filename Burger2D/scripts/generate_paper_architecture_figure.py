from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float
    title: str
    body: str
    face: str


def add_box(ax, box: Box) -> None:
    rect = FancyBboxPatch(
        (box.x, box.y),
        box.w,
        box.h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.8,
        edgecolor="#243140",
        facecolor=box.face,
    )
    ax.add_patch(rect)
    ax.text(
        box.x + box.w / 2,
        box.y + box.h * 0.70,
        box.title,
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color="#13202d",
    )
    ax.text(
        box.x + box.w / 2,
        box.y + box.h * 0.34,
        box.body,
        ha="center",
        va="center",
        fontsize=9.8,
        color="#223344",
        linespacing=1.3,
    )


def add_arrow(ax, start: tuple[float, float], end: tuple[float, float], text: str | None = None) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=1.8,
        color="#2e4a62",
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)
    if text:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y + 0.02, text, ha="center", va="bottom", fontsize=9.4, color="#2e4a62")


def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(__file__))
    out_dir = os.path.join(repo_root, "paper", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "burger2_formal_architecture.png")
    out_pdf = os.path.join(out_dir, "burger2_formal_architecture.pdf")

    fig, ax = plt.subplots(figsize=(15.5, 8.8), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#f7f3ea")
    ax.set_facecolor("#f7f3ea")

    title_color = "#102233"
    ax.text(
        0.5,
        0.965,
        "Staged Local-Context MoE-PINN for 2D Burgers",
        ha="center",
        va="top",
        fontsize=19,
        fontweight="bold",
        color=title_color,
    )
    ax.text(
        0.5,
        0.93,
        "Architecture emphasis: expert formation, local routing, misroute suppression, and guarded base pullback",
        ha="center",
        va="top",
        fontsize=11,
        color="#355066",
    )

    boxes = [
        Box(0.05, 0.69, 0.17, 0.16, "Input Layer", "coords (x, y, t)\nPDE / IC / BC samples\nreference-derived regions", "#f8d9b6"),
        Box(0.28, 0.72, 0.16, 0.12, "Base PINN", "global smooth prior\nhard IC / BC structure", "#f4c7a6"),
        Box(0.48, 0.72, 0.20, 0.12, "Expert Bank", "smooth\niso shock\ndirectional shock\nwave", "#f1deb1"),
        Box(0.73, 0.69, 0.20, 0.16, "Local-Conv Gate", "kNN neighborhood query\nlocal token builder\nlightweight conv aggregator\nsoftmax expert weights", "#cfe4d4"),
        Box(0.28, 0.46, 0.22, 0.12, "Residual Mixture", "base + weighted expert corrections\nfinal mixture prediction", "#d7ead2"),
        Box(0.58, 0.46, 0.26, 0.12, "Routing Calibration", "branch-aware gate supervision\nmisroute penalty\nroute sharpening", "#d1e3ef"),
        Box(0.18, 0.22, 0.24, 0.14, "Guarded Base Anchor", "pull back only if base remains competitive\nblock pullback in wave / directional advantage zones\nuncertainty-aware activation", "#d8d9f1"),
        Box(0.50, 0.22, 0.32, 0.14, "Diagnostics and Evaluation", "global metrics\nsteep / background / stress metrics\nwave routing regret\nstress routing regret\n3D dataset scene", "#cfdcf2"),
    ]

    for box in boxes:
        add_box(ax, box)

    add_arrow(ax, (0.22, 0.77), (0.28, 0.78))
    add_arrow(ax, (0.22, 0.77), (0.48, 0.78), "shared coordinates")
    add_arrow(ax, (0.22, 0.77), (0.73, 0.77), "region + local context cues")
    add_arrow(ax, (0.44, 0.78), (0.50, 0.78))
    add_arrow(ax, (0.68, 0.78), (0.73, 0.78))
    add_arrow(ax, (0.36, 0.72), (0.38, 0.58))
    add_arrow(ax, (0.58, 0.72), (0.46, 0.58), "expert residuals")
    add_arrow(ax, (0.83, 0.69), (0.46, 0.52), "gating weights")
    add_arrow(ax, (0.71, 0.46), (0.60, 0.36), "joint-stage calibration")
    add_arrow(ax, (0.34, 0.36), (0.39, 0.46), "smooth-region pullback")
    add_arrow(ax, (0.39, 0.46), (0.66, 0.36), "prediction traces")
    add_arrow(ax, (0.71, 0.22), (0.71, 0.46), "routing reports")

    ax.text(0.08, 0.61, "Stage 0\nbase pretrain", ha="center", va="center", fontsize=10, color="#6b4f2a")
    ax.text(0.53, 0.61, "Stage A\nexpert specialization", ha="center", va="center", fontsize=10, color="#6b5a25")
    ax.text(0.86, 0.61, "Stage B\nlocal gate training", ha="center", va="center", fontsize=10, color="#2e5c41")
    ax.text(0.58, 0.395, "Stage C\nconstrained fine-tuning", ha="center", va="center", fontsize=10, color="#2d4d65")

    ax.text(
        0.5,
        0.06,
        "Current best configuration: local-conv gate + stronger-expert-aware route sharpening + guarded / uncertainty-aware base anchor",
        ha="center",
        va="center",
        fontsize=11,
        color="#213546",
    )

    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(out_png)
    print(out_pdf)


if __name__ == "__main__":
    main()
