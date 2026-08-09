"""
Gate-introduction timing ablation for 2D Burgers staged MoE-PINN.

Experts are pretrained to a fraction f of their full Stage A budget, then the
gate is introduced and gate+experts are jointly trained (base frozen). f=0 is
therefore end-to-end-like, f=1.0 is the current mainline (gate-only joint).

    f = 0.00, 0.25, 0.50, 0.75, 1.00
    seeds = 42, 43, 44

Outputs (in the run root):
    - gate_intro_ablation_summary.json
    - gate_intro_trajectory.png (2x3 line charts with mean+/-std error bars)
    - a Windows popup when everything finishes

Usage:
    python Burger2D/scripts/run_gate_intro_ablation.py [--smoke]
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from Burger2D.experiments.run_burgers2d import run_experiment  # noqa: E402


FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEEDS = [42, 43, 44]
STAGED_VARIANT = "stronger_expert_route_sharp"
GATE_VARIANT = "local_conv"
WAVE_VARIANT = "mixed_lite"
DIRECTIONAL_VARIANT = "hybrid"
LAYOUT_VARIANT = "categorical"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_one(
    *,
    results_root: str,
    fraction: float,
    seed: int,
    smoke: bool,
) -> dict:
    tag = f"f{int(fraction * 100)}"
    run_dir = os.path.join(results_root, f"{tag}_seed{seed}")
    t0 = time.time()
    print(f"\n{'='*72}\n[{tag} seed={seed}] fraction={fraction}\n{'='*72}", flush=True)
    run_experiment(
        train_mode="staged",
        n_steps=200 if smoke else 1500,
        n_col=3000 if smoke else 12000,
        n_ic=800 if smoke else 3000,
        n_bc_per_face=200 if smoke else 800,
        nx=33 if smoke else 81,
        ny=33 if smoke else 81,
        nt=11 if smoke else 31,
        results_root=run_dir,
        staged_variant=STAGED_VARIANT,
        gate_variant=GATE_VARIANT,
        wave_expert_variant=WAVE_VARIANT,
        directional_expert_variant=DIRECTIONAL_VARIANT,
        expert_layout_variant=LAYOUT_VARIANT,
        exclude_experts="",
        expert_pretrain_fraction=fraction,
        seed=seed,
    )
    elapsed = time.time() - t0
    metrics = _load_json(os.path.join(run_dir, "burgers2d_moe_staged", "metrics.json"))
    expert_metrics = _load_json(
        os.path.join(run_dir, "burgers2d_moe_staged", "expert_metrics.json")
    )
    entry = {
        "fraction": fraction,
        "seed": seed,
        "elapsed_sec": elapsed,
        "metrics": metrics,
        "expert_metrics": expert_metrics,
    }
    with open(os.path.join(run_dir, "gate_intro_entry.json"), "w", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False, indent=2)
    print(
        f"[{tag} seed={seed}] done in {elapsed:.0f}s, "
        f"L2={metrics.get('l2_relative_error', float('nan')):.4f}",
        flush=True,
    )
    return entry


def _summary_table(entries: list[dict]) -> dict:
    by_frac: dict[float, list[dict]] = {f: [] for f in FRACTIONS}
    for e in entries:
        by_frac.setdefault(e["fraction"], []).append(e)

    rows = {}
    for f, es in by_frac.items():
        metrics_list = [e["metrics"] for e in es]
        row: dict = {"n": len(es)}
        for key in [
            "l2_relative_error",
            "max_absolute_error",
            "steep_mae",
            "background_mae",
            "route_entropy",
            "route_max_weight",
        ]:
            vals = [m[key] for m in metrics_list if key in m]
            if vals:
                row[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        load_fracs = np.asarray([m["expert_load_frac"] for m in metrics_list if "expert_load_frac" in m])
        if len(load_fracs):
            eff = 1.0 / np.sum(load_fracs**2, axis=1)
            row["effective_experts"] = {
                "mean": float(np.mean(eff)),
                "std": float(np.std(eff)),
            }
            row["min_load_frac"] = {
                "mean": float(np.mean(load_fracs.min(axis=1))),
                "std": float(np.std(load_fracs.min(axis=1))),
            }
        expert_l2 = np.asarray(
            [[v["l2_relative_error"] for v in e["expert_metrics"].values()] for e in es]
        )
        if len(expert_l2):
            row["max_expert_l2"] = {
                "mean": float(np.mean(expert_l2.max(axis=1))),
                "std": float(np.std(expert_l2.max(axis=1))),
            }
            row["std_expert_l2"] = {
                "mean": float(np.mean(expert_l2.std(axis=1))),
                "std": float(np.std(expert_l2.std(axis=1))),
            }
        rows[str(f)] = row
    return rows


def _plot(rows: dict, out_path: str) -> None:
    xs = np.arange(len(FRACTIONS))
    labels = [f"{int(f * 100)}%" for f in FRACTIONS]

    def series(key: str):
        mean = np.array([rows[str(f)][key]["mean"] for f in FRACTIONS])
        std = np.array([rows[str(f)][key]["std"] for f in FRACTIONS])
        return mean, std

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2))
    panels = [
        ("l2_relative_error", "相对 L2 误差（越低越好）"),
        ("max_absolute_error", "最大绝对误差（越低越好）"),
        ("effective_experts", "有效专家数 1/Σp²（越高越好）"),
        ("min_load_frac", "最小专家负载（越高越均衡）"),
        ("route_entropy", "路由熵（越低越果断）"),
        ("max_expert_l2", "最差单专家 L2（越低越健康）"),
    ]
    for ax, (key, title) in zip(axes.flat, panels):
        if key not in rows["0.0"]:
            ax.set_title(title)
            ax.text(0.5, 0.5, "数据缺失", ha="center", va="center", transform=ax.transAxes)
            continue
        mean, std = series(key)
        ax.errorbar(xs, mean, yerr=std, marker="o", capsize=4, lw=1.8, color="#4C72B0")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("gate 引入点（专家训练完成度）", fontsize=9)
        ax.grid(alpha=0.3, linewidth=0.6)
        ax.tick_params(labelsize=8)
        for x, m, s in zip(xs, mean, std):
            ax.annotate(f"{m:.3f}±{s:.3f}", (x, m), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7)
    fig.suptitle("gate 引入时机 vs 预测精度与专家利用（2D Burgers，3 seeds 均值±std）",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


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

    results_root = os.path.join(
        PACKAGE_ROOT,
        "results",
        f"gate_intro_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(results_root, exist_ok=True)

    entries: list[dict] = []
    for fraction in FRACTIONS:
        for seed in SEEDS:
            try:
                entries.append(
                    run_one(
                        results_root=results_root,
                        fraction=fraction,
                        seed=seed,
                        smoke=args.smoke,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[FAILED] fraction={fraction} seed={seed}: {type(exc).__name__}: {exc}",
                    flush=True,
                )

    rows = _summary_table(entries)
    summary = {
        "results_root": results_root,
        "fractions": FRACTIONS,
        "seeds": SEEDS,
        "rows": rows,
        "entries": entries,
    }
    summary_path = os.path.join(results_root, "gate_intro_ablation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    chart_path = os.path.join(results_root, "gate_intro_trajectory.png")
    if len({e["fraction"] for e in entries}) == len(FRACTIONS) and len(rows.get("0.0", {})) > 3:
        _plot(rows, chart_path)

    print("\n" + "=" * 100)
    print("GATE-INTRO ABLATION SUMMARY")
    print("=" * 100)
    for f in FRACTIONS:
        row = rows.get(str(f), {})
        if not row:
            continue
        print(
            f"f={f:>4} n={row['n']} "
            f"L2={row.get('l2_relative_error', {}).get('mean', float('nan')):.4f}±"
            f"{row.get('l2_relative_error', {}).get('std', 0):.4f} "
            f"MaxErr={row.get('max_absolute_error', {}).get('mean', float('nan')):.4f} "
            f"eff={row.get('effective_experts', {}).get('mean', float('nan')):.2f} "
            f"minLoad={row.get('min_load_frac', {}).get('mean', float('nan')):.3f}"
        )
    done_path = os.path.join(results_root, "DONE")
    with open(done_path, "w", encoding="utf-8") as f:
        f.write(f"finished at {datetime.now().isoformat()}\n")
    print(f"[OK] Summary: {summary_path}", flush=True)
    print(f"[OK] Chart: {chart_path}", flush=True)
    popup(
        "Gate-Intro Ablation 完成",
        f"全部 {len(entries)} 个 run 已结束（失败 {15 - len(entries)} 个）。\n"
        f"结果目录：{results_root}",
    )


if __name__ == "__main__":
    main()
