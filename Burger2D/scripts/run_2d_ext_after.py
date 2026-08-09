"""Wait for dam-break + 1D gate-intro to finish, then run the 2D extended suite."""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Burger2D.scripts.run_2d_extended import main as run_2d_ext  # noqa: E402


def wait_for(path: str, label: str) -> None:
    print(f"[wait] {label}: {path}", flush=True)
    while not os.path.exists(path):
        time.sleep(60)
        print(f"[wait] {datetime.now().strftime('%H:%M:%S')} still waiting on {label}...", flush=True)
    print(f"[wait] {label} DONE found", flush=True)


def main() -> None:
    # Dam-break root from its status file
    try:
        with open(
            os.path.join(PROJECT_ROOT, "ShallowWater2D", "results", "dam_break_gate_intro_status.json"),
            encoding="utf-8",
        ) as f:
            dam_status = json.load(f)
        dam_done = os.path.join(dam_status["root"], "DONE")
    except (OSError, KeyError, json.JSONDecodeError):
        dam_done = None
        print("[wait] dam status not readable yet", flush=True)
    try:
        with open(
            os.path.join(PROJECT_ROOT, "burger1D", "results", "gate_intro_1d_status.json"),
            encoding="utf-8",
        ) as f:
            one_d_status = json.load(f)
        one_d_done = os.path.join(one_d_status["root"], "DONE")
    except (OSError, KeyError, json.JSONDecodeError):
        one_d_done = None
        print("[wait] 1d status not readable yet", flush=True)

    if dam_done:
        wait_for(dam_done, "dam-break")
    if one_d_done:
        wait_for(one_d_done, "1D")
    print("[wait] prerequisites done, starting 2D extended suite", flush=True)
    run_2d_ext()


if __name__ == "__main__":
    main()
