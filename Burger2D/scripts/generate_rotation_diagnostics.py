"""
Regenerate evaluation artifacts for a saved staged Burger2D checkpoint.

This is mainly used for the rotation side branch so that new diagnostics can be
added without rerunning full training.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Burger2D.core.moe_pinn import build_burgers2d_moe
from Burger2D.equations.burgers2d import Burgers2DProblem
from Burger2D.experiments.run_burgers2d import (
    DTYPE,
    NU,
    _compute_directional_stress_metrics,
    _evaluate_metrics,
    _save_artifacts,
    _set_global_seed,
)
from Burger2D.training.staged_burgers2d import flatten_reference_solution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate Burger2D rotation diagnostics from a saved checkpoint.")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Directory containing burgers2d_moe_staged.pt.")
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Optional output directory. Defaults to <checkpoint-dir>\\rotation_diagnostics_replay.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device for evaluation, for example cpu or cuda.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for reference regeneration.")
    parser.add_argument("--nx", type=int, default=65)
    parser.add_argument("--ny", type=int, default=65)
    parser.add_argument("--nt", type=int, default=21)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    save_dir = os.path.abspath(args.save_dir) if args.save_dir else os.path.join(checkpoint_dir, "rotation_diagnostics_replay")
    model_ckpt = os.path.join(checkpoint_dir, "burgers2d_moe_staged.pt")
    stage_ckpt = os.path.join(checkpoint_dir, "burgers2d_staged_training.pt")
    if not os.path.exists(model_ckpt):
        raise FileNotFoundError(f"Missing checkpoint: {model_ckpt}")
    if not os.path.exists(stage_ckpt):
        raise FileNotFoundError(f"Missing staged training record: {stage_ckpt}")

    device = torch.device(args.device)
    _set_global_seed(args.seed)

    stage_info = torch.load(stage_ckpt, map_location="cpu")
    model_state = torch.load(model_ckpt, map_location="cpu")

    model = build_burgers2d_moe(
        directional_expert_variant=stage_info.get("directional_expert_variant", "hybrid"),
        wave_expert_variant=stage_info.get("wave_expert_variant", "base"),
        expert_layout_variant=stage_info.get("expert_layout_variant", "categorical"),
        attribute_expert_variant=stage_info.get("attribute_expert_variant", "base"),
        gate_variant=stage_info.get("gate_variant", "pointwise"),
        rotation_variant=stage_info.get("rotation_variant", "none"),
    ).to(device).to(DTYPE)
    load_result = model.load_state_dict(model_state["model_state"], strict=False)
    missing = set(load_result.missing_keys)
    unexpected = set(load_result.unexpected_keys)
    allowed_missing = {
        "rotation_route_adapter.0.weight",
        "rotation_route_adapter.0.bias",
        "rotation_route_adapter.2.weight",
        "rotation_route_adapter.2.bias",
    }
    if unexpected or (missing - allowed_missing):
        raise RuntimeError(
            f"Checkpoint compatibility error. Missing={sorted(missing)} | Unexpected={sorted(unexpected)}"
        )
    model.eval()

    problem = Burgers2DProblem(nu=NU, device=device, dtype=DTYPE, seed=args.seed)
    reference = problem.generate_reference_solution(nx=args.nx, ny=args.ny, nt=args.nt)
    xyt_test, u_exact_flat = flatten_reference_solution(reference, device=device, dtype=DTYPE)

    metrics, u_pred = _evaluate_metrics(model, reference, xyt_test=xyt_test, u_exact_flat=u_exact_flat)
    stress_metrics = _compute_directional_stress_metrics(model, reference, u_pred=u_pred, xyt_test=xyt_test)
    history = model_state.get("history", {})
    _save_artifacts(
        save_dir=save_dir,
        reference=reference,
        model=model,
        history=history,
        metrics=metrics,
        stress_metrics=stress_metrics,
        u_pred=u_pred,
        xyt_test=xyt_test,
        u_exact_flat=u_exact_flat,
    )

    print(f"[OK] Regenerated diagnostics at: {save_dir}")
    print(f"[metrics] L2={metrics['l2_relative_error']:.6f} | Steep={metrics['steep_mae']:.6f} | Background={metrics['background_mae']:.6f}")
    if "rotation_focus_axis_error_deg" in metrics:
        print(
            "[rotation] "
            f"axis_error={metrics['rotation_focus_axis_error_deg']:.2f} deg | "
            f"activation={metrics.get('rotation_activation_mean', 0.0):.4f} | "
            f"concentration={metrics.get('rotation_concentration_mean', 0.0):.6f}"
        )


if __name__ == "__main__":
    main()
