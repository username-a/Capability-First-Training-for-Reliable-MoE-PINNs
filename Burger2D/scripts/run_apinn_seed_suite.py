"""Run additional APINN seeds sequentially while keeping the dashboard live."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent


def _write(path: Path, payload: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[43, 44])
    parser.add_argument("--root", default=str(PACKAGE_ROOT / "results" / "apinn_reproduction"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preset", default="matched")
    parser.add_argument("--gate-prior", default="spatial")
    parser.add_argument("--gate-pretrain-steps", type=int, default=1000)
    parser.add_argument("--train-steps", type=int, default=8000)
    parser.add_argument("--active-dir-name", default="latest")
    parser.add_argument("--archive-template", default="seed{seed}_spatial_matched")
    parser.add_argument("--status-name", default="suite_status.json")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    latest = root / args.active_dir_name
    suite_status = root / args.status_name
    train_script = SCRIPT_DIR / "run_apinn_reproduction.py"
    counterfactual_script = SCRIPT_DIR / "analyze_apinn_counterfactuals.py"
    completed: list[int] = []

    for seed in args.seeds:
        _write(suite_status, {"state": "running", "active_seed": seed, "completed_seeds": completed, "updated_at": time.time()})
        command = [
            sys.executable, str(train_script), "--preset", args.preset, "--gate-prior", args.gate_prior,
            "--seed", str(seed), "--gate-pretrain-steps", str(args.gate_pretrain_steps), "--train-steps", str(args.train_steps),
            "--eval-every", "100", "--output-dir", str(latest), "--device", args.device,
        ]
        subprocess.run(command, check=True)
        archive = root / args.archive_template.format(seed=seed)
        if archive.exists():
            raise FileExistsError(f"Refusing to overwrite existing archive: {archive}")
        shutil.copytree(latest, archive)
        subprocess.run([
            sys.executable, str(counterfactual_script), "--run-dir", str(archive), "--device", args.device,
        ], check=True)
        completed.append(seed)

    _write(suite_status, {"state": "completed", "active_seed": None, "completed_seeds": completed, "updated_at": time.time()})


if __name__ == "__main__":
    main()
