"""Batch runner: Allen-Cahn staged-vs-coadapt for seeds 45..51 (both modes).

Launched in the background; pops up a message box when all runs finish.
"""

from __future__ import annotations

import subprocess
import sys
import traceback
import time
from pathlib import Path

from Burger2D.scripts.run_allen_cahn_staged_vs_coadapt import (
    RESULTS_DIR,
    AllenConfig,
    train,
)


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
    log_path = RESULTS_DIR / "batch_seeds45_51.log"
    err_path = RESULTS_DIR / "batch_seeds45_51.err.log"
    log_file = open(log_path, "w", encoding="utf-8")
    err_file = open(err_path, "w", encoding="utf-8")
    sys.stdout = log_file
    sys.stderr = err_file
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    import torch

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    started = time.time()
    results = []
    try:
        for seed in range(45, 52):
            for mode in ("staged", "coadapt"):
                cfg = AllenConfig(seed=seed, mode=mode)
                out = RESULTS_DIR / f"{mode}_seed{seed}"
                train(mode, cfg, dev, out)
                import json

                m = json.loads((out / "metrics.json").read_text())
                results.append(
                    f"{mode} seed{seed}: l2={m['l2_mixed']:.4f} "
                    f"eff={m['effective_experts']:.3f} intf_load={m['load_interface']:.4f}"
                )
                print(results[-1], flush=True)
        elapsed = time.time() - started
        print(f"all done in {elapsed/60:.1f} min", flush=True)
        popup(
            "Allen-Cahn batch done",
            f"14 runs (seeds 45-51) finished in {elapsed/60:.1f} min.\n"
            + "\n".join(results),
        )
    except Exception:
        traceback.print_exc()
        print("BATCH FAILED", flush=True)
        popup("Allen-Cahn batch failed", "see batch_seeds45_51.err.log")
    finally:
        log_file.flush()
        err_file.flush()


if __name__ == "__main__":
    main()
