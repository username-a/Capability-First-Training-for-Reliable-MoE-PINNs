"""
One-click full Burger2D experiment runner.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Burger2D.experiments.run_burgers2d import run_experiment


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = os.path.join(PACKAGE_ROOT, "results", f"full_run_{timestamp}")

    config = {
        "train_mode": "all",
        "n_steps": 2500,
        "n_col": 12000,
        "n_ic": 3000,
        "n_bc_per_face": 800,
        "nx": 81,
        "ny": 81,
        "nt": 31,
        "device_override": None,
        "results_root": results_root,
    }

    print("=" * 72)
    print("Burger2D full experiment")
    print("=" * 72)
    for key, value in config.items():
        print(f"{key}: {value}")

    run_experiment(**config)
    print(f"[OK] Full experiment finished. Results: {results_root}")


if __name__ == "__main__":
    main()
