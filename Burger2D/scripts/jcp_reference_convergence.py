"""Reference-solution self-convergence checks for the JCP manuscript.

The script compares nested spatial grids and, separately, time-step refinements
for the Burgers and Allen--Cahn reference solvers used by the manuscript.
It writes machine-readable JSON/CSV source data and a compact audit figure.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Burger2D.equations.allen_cahn import generate_reference as generate_allen
from Burger2D.equations.burgers2d import Burgers2DProblem


def _rel_l2(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if mask is not None:
        aa, bb = aa[mask], bb[mask]
    return float(np.linalg.norm(aa - bb) / max(np.linalg.norm(bb), 1e-14))


def _record(rows: list[dict], *, equation: str, axis: str, coarse: str, fine: str,
            coarse_u: np.ndarray, fine_on_coarse: np.ndarray, region: str,
            mask: np.ndarray | None = None) -> None:
    diff = np.asarray(coarse_u, dtype=np.float64) - np.asarray(fine_on_coarse, dtype=np.float64)
    used = diff if mask is None else diff[mask]
    rows.append({
        "equation": equation,
        "refinement_axis": axis,
        "coarse": coarse,
        "fine": fine,
        "region": region,
        "relative_l2_difference": _rel_l2(coarse_u, fine_on_coarse, mask),
        "max_absolute_difference": float(np.max(np.abs(used))),
    })


def _burgers(rows: list[dict], timings: list[dict]) -> None:
    spatial = {}
    for n in (65, 129, 257, 513):
        t0 = time.perf_counter()
        ref = Burgers2DProblem(seed=42).generate_reference_solution(n, n, 41, cfl=0.22)
        timings.append({"equation": "Burgers2D", "configuration": f"n={n}, cfl=0.22", "seconds": time.perf_counter() - t0})
        spatial[n] = ref
    for coarse, fine in ((65, 129), (129, 257), (257, 513)):
        cu = spatial[coarse].u
        fu = spatial[fine].u[:, ::2, ::2]
        _, gy, gx = np.gradient(fu.astype(np.float64), axis=(0, 1, 2))
        steep = np.sqrt(gx * gx + gy * gy) >= np.quantile(np.sqrt(gx * gx + gy * gy), 0.90)
        _record(rows, equation="Burgers2D", axis="space", coarse=f"{coarse}x{coarse}", fine=f"{fine}x{fine}", coarse_u=cu, fine_on_coarse=fu, region="global")
        _record(rows, equation="Burgers2D", axis="space", coarse=f"{coarse}x{coarse}", fine=f"{fine}x{fine}", coarse_u=cu, fine_on_coarse=fu, region="steep-gradient", mask=steep)

    temporal = {}
    for cfl in (0.22, 0.11, 0.055):
        t0 = time.perf_counter()
        ref = Burgers2DProblem(seed=42).generate_reference_solution(129, 129, 41, cfl=cfl)
        timings.append({"equation": "Burgers2D", "configuration": f"n=129, cfl={cfl}", "seconds": time.perf_counter() - t0})
        temporal[cfl] = ref.u
    for coarse, fine in ((0.22, 0.11), (0.11, 0.055)):
        _record(rows, equation="Burgers2D", axis="time", coarse=f"cfl={coarse}", fine=f"cfl={fine}", coarse_u=temporal[coarse], fine_on_coarse=temporal[fine], region="global")


def _allen(rows: list[dict], timings: list[dict]) -> None:
    coupled = {}
    settings = ((101, 4e-4), (201, 2e-4), (401, 1e-4))
    for n, dt in settings:
        t0 = time.perf_counter()
        ref = generate_allen(nx=n, ny=n, nt=41, t_max=0.25, eps=0.08, r0=0.8, dt=dt)
        timings.append({"equation": "AllenCahn2D", "configuration": f"n={n}, dt={dt:g}", "seconds": time.perf_counter() - t0})
        coupled[(n, dt)] = ref.u
    for (cn, cdt), (fn, fdt) in zip(settings[:-1], settings[1:]):
        cu = coupled[(cn, cdt)]
        fu = coupled[(fn, fdt)][:, ::2, ::2]
        interface = np.abs(fu) <= 0.5
        _record(rows, equation="AllenCahn2D", axis="space+time", coarse=f"{cn}x{cn},dt={cdt:g}", fine=f"{fn}x{fn},dt={fdt:g}", coarse_u=cu, fine_on_coarse=fu, region="global")
        _record(rows, equation="AllenCahn2D", axis="space+time", coarse=f"{cn}x{cn},dt={cdt:g}", fine=f"{fn}x{fn},dt={fdt:g}", coarse_u=cu, fine_on_coarse=fu, region="interface", mask=interface)

    temporal = {}
    for dt in (4e-4, 2e-4, 1e-4):
        t0 = time.perf_counter()
        ref = generate_allen(nx=201, ny=201, nt=41, t_max=0.25, eps=0.08, r0=0.8, dt=dt)
        timings.append({"equation": "AllenCahn2D", "configuration": f"n=201, dt={dt:g}", "seconds": time.perf_counter() - t0})
        temporal[dt] = ref.u
    for coarse, fine in ((4e-4, 2e-4), (2e-4, 1e-4)):
        interface = np.abs(temporal[fine]) <= 0.5
        _record(rows, equation="AllenCahn2D", axis="time", coarse=f"dt={coarse:g}", fine=f"dt={fine:g}", coarse_u=temporal[coarse], fine_on_coarse=temporal[fine], region="global")
        _record(rows, equation="AllenCahn2D", axis="time", coarse=f"dt={coarse:g}", fine=f"dt={fine:g}", coarse_u=temporal[coarse], fine_on_coarse=temporal[fine], region="interface", mask=interface)


def _plot(rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    for ax, equation in zip(axes, ("Burgers2D", "AllenCahn2D")):
        subset = [r for r in rows if r["equation"] == equation and r["region"] in {"global", "interface", "steep-gradient"}]
        labels = [f"{r['refinement_axis']}\n{r['coarse']}->{r['fine']}\n{r['region']}" for r in subset]
        values = [r["relative_l2_difference"] for r in subset]
        colors = ["#3977b8" if r["region"] == "global" else "#d65f5f" for r in subset]
        ax.bar(np.arange(len(values)), values, color=colors)
        ax.set_yscale("log")
        ax.set_xticks(np.arange(len(values)), labels, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel("Relative self-difference")
        ax.set_title("2D Burgers" if equation == "Burgers2D" else "2D Allen-Cahn")
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="Burger2D/results/jcp_validation_20260808")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    timings: list[dict] = []
    _burgers(rows, timings)
    _allen(rows, timings)
    with (out / "reference_convergence.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with (out / "reference_convergence.json").open("w", encoding="utf-8") as f:
        json.dump({"comparisons": rows, "timings": timings}, f, indent=2)
    _plot(rows, out / "reference_convergence.png")
    print(json.dumps({"comparisons": rows, "timings": timings}, indent=2))


if __name__ == "__main__":
    main()
