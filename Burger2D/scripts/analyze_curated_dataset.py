"""Analyze curated Burger2D classification results."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Burger2D.data import REGION_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze curated Burger2D dataset classification quality")
    parser.add_argument("--dataset-dir", required=True, help="Curated dataset directory")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for analysis artifacts; defaults to <dataset-dir>/analysis",
    )
    return parser.parse_args()


def _load_manifest(dataset_dir: str) -> dict[str, Any]:
    with open(os.path.join(dataset_dir, "manifest.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def _summarize_cases(dataset_dir: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    cases_dir = os.path.join(dataset_dir, "cases")
    case_files = sorted(
        os.path.join(cases_dir, name)
        for name in os.listdir(cases_dir)
        if name.endswith(".npz")
    )
    per_label_conf: dict[str, list[np.ndarray]] = {name: [] for name in REGION_NAMES}
    per_label_margin: dict[str, list[np.ndarray]] = {name: [] for name in REGION_NAMES}
    per_label_time: dict[str, list[np.ndarray]] = {name: [] for name in REGION_NAMES}
    representatives: dict[str, dict[str, Any]] = {}

    for case_path in case_files:
        case = np.load(case_path)
        u = case["u"]
        labels = case["labels"]
        confidence = case["label_confidence"]
        score_margin = case["score_margin"]
        probs = case["probs"]
        t = case["t"]
        time_frac = np.linspace(0.0, 1.0, len(t), dtype=np.float32)[:, None, None]
        case_id = int(os.path.splitext(os.path.basename(case_path))[0].split("_")[-1])

        for label_idx, label_name in enumerate(REGION_NAMES):
            mask = labels == label_idx
            if not np.any(mask):
                continue
            per_label_conf[label_name].append(confidence[mask])
            per_label_margin[label_name].append(score_margin[mask])
            per_label_time[label_name].append(np.broadcast_to(time_frac, labels.shape)[mask])

            flat_mask = mask.reshape(mask.shape[0], -1)
            flat_conf = (confidence * mask.astype(np.float32)).reshape(mask.shape[0], -1)
            counts_by_t = flat_mask.sum(axis=1)
            conf_sum_by_t = flat_conf.sum(axis=1)
            best_t = int(np.argmax(counts_by_t + 0.5 * conf_sum_by_t))
            count = int(counts_by_t[best_t])
            if count == 0:
                continue
            mean_conf = float(confidence[best_t][mask[best_t]].mean())
            score = float(count * max(mean_conf, 1e-6))
            previous = representatives.get(label_name)
            if previous is None or score > previous["score"]:
                representatives[label_name] = {
                    "case_id": case_id,
                    "time_index": best_t,
                    "time_value": float(t[best_t]),
                    "count": count,
                    "mean_confidence": mean_conf,
                    "score": score,
                    "u_slice": u[best_t].astype(np.float32),
                    "mask_slice": mask[best_t].astype(np.float32),
                    "prob_slice": probs[best_t, :, :, label_idx].astype(np.float32),
                    "confidence_slice": confidence[best_t].astype(np.float32),
                }

    label_summary: dict[str, dict[str, Any]] = {}
    for label_name in REGION_NAMES:
        conf = np.concatenate(per_label_conf[label_name], axis=0) if per_label_conf[label_name] else np.array([], dtype=np.float32)
        margin = (
            np.concatenate(per_label_margin[label_name], axis=0)
            if per_label_margin[label_name]
            else np.array([], dtype=np.float32)
        )
        time_values = (
            np.concatenate(per_label_time[label_name], axis=0)
            if per_label_time[label_name]
            else np.array([], dtype=np.float32)
        )
        if conf.size == 0:
            label_summary[label_name] = {
                "num_points": 0,
                "mean_confidence": 0.0,
                "median_confidence": 0.0,
                "p90_confidence": 0.0,
                "mean_score_margin": 0.0,
                "mean_time_frac": 0.0,
            }
            continue
        label_summary[label_name] = {
            "num_points": int(conf.size),
            "mean_confidence": float(conf.mean()),
            "median_confidence": float(np.median(conf)),
            "p90_confidence": float(np.quantile(conf, 0.90)),
            "mean_score_margin": float(margin.mean()),
            "mean_time_frac": float(time_values.mean()),
        }
    return label_summary, representatives


def _plot_label_stats(
    manifest: dict[str, Any],
    label_summary: dict[str, dict[str, Any]],
    save_path: str,
) -> None:
    labels = manifest["region_names"]
    counts = np.array([manifest["label_histogram"][name] for name in labels], dtype=np.float64)
    ratios = counts / np.clip(counts.sum(), 1.0, None)
    mean_conf = np.array([label_summary[name]["mean_confidence"] for name in labels], dtype=np.float64)
    mean_margin = np.array([label_summary[name]["mean_score_margin"] for name in labels], dtype=np.float64)
    mean_time = np.array([label_summary[name]["mean_time_frac"] for name in labels], dtype=np.float64)
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        "Curated Dataset Classification Summary\n"
        "Top-left shows label proportions; the other panels summarize confidence, score margin, and time location.",
        fontsize=12,
        y=0.98,
    )

    axes[0, 0].bar(x, ratios, color=["#4c78a8", "#f58518", "#54a24b", "#e45756", "#b279a2"])
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels, rotation=20)
    axes[0, 0].set_ylabel("fraction of points")
    axes[0, 0].set_title("Label Proportion")
    axes[0, 0].grid(axis="y", alpha=0.3)

    axes[0, 1].bar(x, mean_conf, color="#72b7b2")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels, rotation=20)
    axes[0, 1].set_ylabel("mean confidence")
    axes[0, 1].set_title("Mean Confidence by Label")
    axes[0, 1].grid(axis="y", alpha=0.3)

    axes[1, 0].bar(x, mean_margin, color="#eeca3b")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels, rotation=20)
    axes[1, 0].set_ylabel("mean score margin")
    axes[1, 0].set_title("Mean Margin by Label")
    axes[1, 0].grid(axis="y", alpha=0.3)

    axes[1, 1].bar(x, mean_time, color="#ff9da6")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels, rotation=20)
    axes[1, 1].set_ylabel("mean normalized time")
    axes[1, 1].set_title("Mean Time Location by Label")
    axes[1, 1].grid(axis="y", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_representatives(
    representatives: dict[str, dict[str, Any]],
    save_path: str,
) -> None:
    shown_labels = [name for name in REGION_NAMES if name in representatives]
    ncols = len(shown_labels)
    fig, axes = plt.subplots(3, ncols, figsize=(3.6 * ncols, 10.0))
    if ncols == 1:
        axes = np.expand_dims(axes, axis=1)
    fig.suptitle(
        "Representative slices for each label\n"
        "Rows: reference field, binary region mask, and per-label probability map.",
        fontsize=12,
        y=0.98,
    )

    for col, label_name in enumerate(shown_labels):
        rep = representatives[label_name]
        u_slice = rep["u_slice"]
        mask_slice = rep["mask_slice"]
        prob_slice = rep["prob_slice"]
        case_title = f"{label_name}\ncase={rep['case_id']:03d}, t={rep['time_value']:.2f}"

        im0 = axes[0, col].imshow(u_slice, origin="lower", cmap="coolwarm", aspect="auto")
        axes[0, col].set_title(case_title)
        axes[0, col].set_xlabel("x index")
        axes[0, col].set_ylabel("y index")
        cbar0 = fig.colorbar(im0, ax=axes[0, col], fraction=0.046)
        cbar0.set_label("u")

        im1 = axes[1, col].imshow(mask_slice, origin="lower", cmap="gray_r", vmin=0.0, vmax=1.0, aspect="auto")
        axes[1, col].set_xlabel("x index")
        axes[1, col].set_ylabel("y index")
        axes[1, col].set_title(
            f"mask | count={rep['count']}\nmean conf={rep['mean_confidence']:.3f}"
        )
        cbar1 = fig.colorbar(im1, ax=axes[1, col], fraction=0.046)
        cbar1.set_label("label mask")

        im2 = axes[2, col].imshow(prob_slice, origin="lower", cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
        axes[2, col].set_xlabel("x index")
        axes[2, col].set_ylabel("y index")
        axes[2, col].set_title("probability map")
        cbar2 = fig.colorbar(im2, ax=axes[2, col], fraction=0.046)
        cbar2.set_label("class probability")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_dir = os.path.abspath(args.dataset_dir)
    output_dir = os.path.abspath(args.output_dir or os.path.join(dataset_dir, "analysis"))
    os.makedirs(output_dir, exist_ok=True)

    manifest = _load_manifest(dataset_dir)
    label_summary, representatives = _summarize_cases(dataset_dir)
    summary = {
        "dataset_dir": dataset_dir,
        "region_rules": manifest.get("region_rules", {}),
        "label_histogram": manifest["label_histogram"],
        "label_summary": label_summary,
        "representatives": {
            name: {
                "case_id": info["case_id"],
                "time_index": info["time_index"],
                "time_value": info["time_value"],
                "count": info["count"],
                "mean_confidence": info["mean_confidence"],
            }
            for name, info in representatives.items()
        },
    }

    with open(os.path.join(output_dir, "classification_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    _plot_label_stats(manifest, label_summary, os.path.join(output_dir, "classification_stats.png"))
    _plot_representatives(representatives, os.path.join(output_dir, "classification_representatives.png"))

    print("=" * 72)
    print("Curated dataset analysis complete")
    print("=" * 72)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
