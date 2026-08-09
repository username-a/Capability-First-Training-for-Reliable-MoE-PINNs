"""
Regenerate Burgers evaluation figures from existing checkpoints.

This is useful after updating plotting code so we do not need to retrain.
"""

from __future__ import annotations

import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.moe_pinn import VanillaPINN, build_burgers_moe
from equations.burgers import BurgersEquation
from experiments.run_burgers import (
    DEVICE,
    DTYPE,
    NU,
    _evaluate_model,
    _make_eval_tensors,
    _save_metrics_and_plots,
)


RESULT_DIR = os.path.join("results", "burgers_staged")
FIGURE_DIR = "figures"


def _load_model_states():
    moe_path = os.path.join(RESULT_DIR, "burgers_moe_staged.pt")
    van_path = os.path.join(RESULT_DIR, "burgers_vanilla.pt")
    if not os.path.exists(moe_path):
        raise FileNotFoundError(f"Missing checkpoint: {moe_path}")
    if not os.path.exists(van_path):
        raise FileNotFoundError(f"Missing checkpoint: {van_path}")

    moe_model = build_burgers_moe().to(DEVICE).to(DTYPE)
    moe_ckpt = torch.load(moe_path, map_location=DEVICE)
    moe_model.load_state_dict(moe_ckpt["model_state"])

    vanilla_model = VanillaPINN(
        in_dim=2,
        out_dim=1,
        hidden=64,
        depth=5,
        output_transform="burgers_hard_bc",
    ).to(DEVICE).to(DTYPE)
    van_ckpt = torch.load(van_path, map_location=DEVICE)
    vanilla_model.load_state_dict(van_ckpt["model_state"])

    return moe_model, vanilla_model, moe_ckpt.get("history", {}), van_ckpt.get("history", {})


def _copy_paper_figures():
    os.makedirs(FIGURE_DIR, exist_ok=True)
    mapping = {
        "metrics_bar.png": "burgers_metrics_bar.png",
        "solution_comparison.png": "solution_comparison.png",
        "shock_zoom_comparison.png": "shock_zoom_comparison.png",
        "shock_diagnostics.png": "shock_diagnostics.png",
        "expert_specialization.png": "expert_specialization.png",
        "gating_weights.png": "gating_weights.png",
    }
    for src_name, dst_name in mapping.items():
        src = os.path.join(RESULT_DIR, src_name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(FIGURE_DIR, dst_name))
            print(f"[OK] Copied: {src} -> {os.path.join(FIGURE_DIR, dst_name)}")


def main():
    print(f"Device: {DEVICE}")
    moe_model, vanilla_model, history_moe, history_van = _load_model_states()

    eq = BurgersEquation(nu=NU, device=DEVICE, dtype=DTYPE)
    nx, nt = 100, 50
    X, T, U_exact = eq.test_grid(nx=nx, nt=nt)
    xt_test, _ = _make_eval_tensors(X, T, U_exact, device=DEVICE, dtype=DTYPE)

    U_moe = _evaluate_model(moe_model, xt_test, nt=nt, nx=nx)
    U_vanilla = _evaluate_model(vanilla_model, xt_test, nt=nt, nx=nx)

    _save_metrics_and_plots(
        save_dir=RESULT_DIR,
        X=X,
        T=T,
        U_exact=U_exact,
        U_moe=U_moe,
        U_vanilla=U_vanilla,
        history_moe=history_moe,
        history_van=history_van,
        moe_model=moe_model,
        xt_test=xt_test,
    )
    _copy_paper_figures()


if __name__ == "__main__":
    main()
