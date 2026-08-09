"""
Append an additional seed (44) to the running mechanism training queue.

Waits until the current mechanism-training driver writes its DONE marker,
then runs e2e8000_seed44 and joint8000_seed44 into the same results root
(so the total becomes 3 seeds per condition), writes DONE_EXT and pops up.

Usage:
    python Burger2D/scripts/append_more_seeds.py
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Burger2D.scripts.run_mechanism_training import (  # noqa: E402
    RESULTS,
    run_e2e_long,
    run_joint_long,
)


def latest_root() -> str | None:
    dirs = sorted(
        glob.glob(os.path.join(RESULTS, "mechanism_training_*")),
        key=os.path.getmtime,
    )
    return dirs[-1] if dirs else None


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
    root = latest_root()
    if root is None:
        print("[ERROR] no mechanism_training_* root found", flush=True)
        return
    done_marker = os.path.join(root, "DONE")
    print(f"[wait] root={root}", flush=True)
    while not os.path.exists(done_marker):
        time.sleep(60)
        print(f"[wait] {datetime.now().strftime('%H:%M:%S')} still waiting...", flush=True)
    print("[wait] DONE found, starting appended seeds", flush=True)

    results: dict[str, dict] = {}
    for name, fn in [("e2e", run_e2e_long), ("joint", run_joint_long)]:
        key = f"{name}_44"
        try:
            results[key] = fn(seed=44, root=root, smoke=False)
            print(f"[OK] {key} finished", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAILED] {key}: {type(exc).__name__}: {exc}", flush=True)

    with open(os.path.join(root, "mechanism_summary_ext.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(os.path.join(root, "DONE_EXT"), "w", encoding="utf-8") as f:
        f.write(datetime.now().isoformat() + "\n")
    popup(
        "追加种子完成",
        f"e2e/joint seed 44 已追加完成（共 3 seeds）。\n结果目录：{root}",
    )
    print(f"[OK] appended seeds finished in {root}", flush=True)


if __name__ == "__main__":
    main()
