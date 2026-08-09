"""Paired statistics and the audited routing-counterfactual figure for the JCP paper.

All values are read from the WENO5 513-grid checkpoint re-evaluation.  The
figure shows every paired seed and uses the same source rows as the manuscript
table; no neural model is re-evaluated here.
"""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT_ROOT / "Burger2D/results/jcp_reference_rebuild_20260808/main_checkpoint_reevaluation.json"
OUT = PROJECT_ROOT / "Burger2D/results/jcp_reviewed_20260809"

METRICS = {
    "soft_relative_l2": lambda r: r["soft"]["relative_l2"],
    "worst_branch_l2": lambda r: r["worst_branch_l2"],
    "oracle_l2": lambda r: r["oracle_l2"],
    "aggregation_gain": lambda r: r["aggregation_gain"],
    "soft_routing_regret": lambda r: r["soft_routing_regret"],
    "effective_experts": lambda r: r["effective_experts"],
    "top1_ratio": lambda r: r["top1"]["ratio_vs_soft"],
    "tau2_ratio": lambda r: r["temperature_2"]["ratio_vs_soft"],
    "delete_highest_load_ratio": lambda r: r["delete_highest_load"]["ratio_vs_soft"],
}


def exact_sign_flip_p(diff: np.ndarray) -> float:
    observed = abs(float(diff.mean()))
    values = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(diff)):
        values.append(abs(float((diff * np.asarray(signs)).mean())))
    return float(np.mean(np.asarray(values) >= observed - 1e-15))


def bootstrap_ci(diff: np.ndarray, rng: np.random.Generator, draws: int = 200_000) -> tuple[float, float]:
    indices = rng.integers(0, len(diff), size=(draws, len(diff)))
    means = diff[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def holm_adjust(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (len(pvalues) - rank) * pvalues[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted.tolist()


def load_pairs() -> tuple[list[int], dict[int, dict[str, dict]]]:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    paired: dict[int, dict[str, dict]] = {}
    for row in payload["runs"]:
        paired.setdefault(int(row["seed"]), {})[str(row["mode"])] = row
    seeds = sorted(paired)
    for seed in seeds:
        if set(paired[seed]) != {"staged", "coadapt"}:
            raise RuntimeError(f"Incomplete pair for seed {seed}")
    return seeds, paired


def make_statistics(seeds: list[int], paired: dict[int, dict[str, dict]]) -> list[dict]:
    rng = np.random.default_rng(20260809)
    rows = []
    for name, getter in METRICS.items():
        staged = np.asarray([getter(paired[s]["staged"]) for s in seeds], dtype=float)
        coadapt = np.asarray([getter(paired[s]["coadapt"]) for s in seeds], dtype=float)
        diff = coadapt - staged
        low, high = bootstrap_ci(diff, rng)
        rows.append({
            "metric": name,
            "contrast": "coadapt_minus_staged",
            "n_pairs": len(seeds),
            "staged_mean": float(staged.mean()),
            "staged_sample_sd": float(staged.std(ddof=1)),
            "coadapt_mean": float(coadapt.mean()),
            "coadapt_sample_sd": float(coadapt.std(ddof=1)),
            "paired_mean_difference": float(diff.mean()),
            "bootstrap_95ci_low": low,
            "bootstrap_95ci_high": high,
            "exact_two_sided_sign_flip_p": exact_sign_flip_p(diff),
            "raw_differences": diff.tolist(),
        })
    adjusted = holm_adjust([row["exact_two_sided_sign_flip_p"] for row in rows])
    for row, value in zip(rows, adjusted):
        row["holm_adjusted_p_across_nine_endpoints"] = value
    return rows


def make_counterfactual_figure(seeds: list[int], paired: dict[int, dict[str, dict]], output: Path) -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.75,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    specs = [
        ("Top-1 routing", lambda r: r["top1"]["ratio_vs_soft"]),
        (r"Post-hoc $\tau=2$", lambda r: r["temperature_2"]["ratio_vs_soft"]),
        ("Delete highest-load", lambda r: r["delete_highest_load"]["ratio_vs_soft"]),
    ]
    staged_color = "#2F6B8A"
    coadapt_color = "#D07A5F"
    line_color = "#AAB2B8"
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.45), constrained_layout=True)
    for panel, (ax, (title, getter)) in enumerate(zip(axes, specs)):
        staged = np.asarray([getter(paired[s]["staged"]) for s in seeds], dtype=float)
        coadapt = np.asarray([getter(paired[s]["coadapt"]) for s in seeds], dtype=float)
        for index in range(len(seeds)):
            ax.plot([0, 1], [staged[index], coadapt[index]], color=line_color, lw=0.65, zorder=1)
        ax.scatter(np.zeros(len(seeds)), staged, s=17, color=staged_color, edgecolor="white", linewidth=0.35,
                   zorder=3)
        ax.scatter(np.ones(len(seeds)), coadapt, s=17, marker="s", color=coadapt_color, edgecolor="white", linewidth=0.35,
                   zorder=3)
        for xpos, values, color in ((0, staged, staged_color), (1, coadapt, coadapt_color)):
            mean = float(values.mean())
            sd = float(values.std(ddof=1))
            ax.errorbar(xpos, mean, yerr=sd, fmt="_", ms=15, mew=1.6, color=color,
                        elinewidth=1.25, capsize=3.0, capthick=1.0, zorder=4)
        ax.axhline(1.0, color="#4D4D4D", ls="--", lw=0.8)
        ax.set_xticks([0, 1], ["Staged", "Co-adaptive"])
        ax.set_title(title)
        ax.grid(axis="y", color="#D9DEE3", lw=0.55, alpha=0.9)
        ax.text(-0.14, 1.04, chr(ord("a") + panel), transform=ax.transAxes, fontweight="bold", fontsize=8.5)
    axes[0].set_ylabel(r"Counterfactual / soft-mixture relative $L_2$")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    seeds, paired = load_pairs()
    rows = make_statistics(seeds, paired)
    source_label = SOURCE.relative_to(PROJECT_ROOT).as_posix()
    (OUT / "paired_statistics.json").write_text(json.dumps({"source": source_label, "rows": rows}, indent=2), encoding="utf-8")
    flat_rows = [{key: value for key, value in row.items() if key != "raw_differences"} for row in rows]
    with (OUT / "paired_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    make_counterfactual_figure(seeds, paired, OUT / "main_counterfactuals_reviewed")


if __name__ == "__main__":
    main()
