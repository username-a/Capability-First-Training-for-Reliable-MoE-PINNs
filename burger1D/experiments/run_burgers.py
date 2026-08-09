"""
Full Burgers experiment.

Supports two training modes:
- end_to_end: standard MoE training from scratch
- staged: expert pretraining -> gate training -> joint fine-tuning
"""

from __future__ import annotations

import os
import sys
from typing import Dict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.moe_pinn import VanillaPINN, build_burgers_moe
from equations.burgers import BurgersEquation
from training.loss_functions import LossConfig, PhysicsLoss, l2_relative_error
from training.staged_burgers import (
    StagedBurgersConfig,
    build_gate_targets,
    build_specialist_batches,
    joint_finetune,
    pretrain_base_model,
    pretrain_experts,
    train_gate,
)
from training.trainer import Trainer
from visualization.plots import (
    plot_error_distribution,
    plot_expert_specialization,
    plot_gating_weights,
    plot_loss_curves,
    plot_metrics_bar,
    plot_ntk_weight_evolution,
    plot_shock_diagnostics,
    plot_shock_zoom_comparison,
    plot_solution_comparison,
)


torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
NU = 0.01 / np.pi

N_COL = 3000
N_IC = 200
N_BC = 100
N_STEPS = 8000
LR = 1e-3

SAVE_DIR = "results/burgers"
SAVE_DIR_STAGED = "results/burgers_staged"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR_STAGED, exist_ok=True)


def _make_eval_tensors(
    X: np.ndarray,
    T: np.ndarray,
    U_exact: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    xt_test = torch.stack(
        [
            torch.tensor(X.ravel(), dtype=dtype, device=device),
            torch.tensor(T.ravel(), dtype=dtype, device=device),
        ],
        dim=-1,
    )
    u_exact_flat = torch.tensor(U_exact.ravel(), dtype=dtype, device=device).unsqueeze(-1)
    return xt_test, u_exact_flat


def _evaluate_model(
    model: torch.nn.Module,
    xt_test: torch.Tensor,
    *,
    nt: int,
    nx: int,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred = model(xt_test).cpu().numpy().reshape(nt, nx)
    return pred


def _train_moe_end_to_end(
    batch: Dict[str, torch.Tensor],
    xt_test: torch.Tensor,
    u_exact_flat: torch.Tensor,
    *,
    save_dir: str,
    use_ntk: bool,
) -> tuple[torch.nn.Module, Dict[str, list]]:
    print("\n" + "=" * 60)
    print("Training MoE-PINN end-to-end")
    print("=" * 60)

    moe_model = build_burgers_moe(
        num_experts=3,
        use_fourier=True,
        sparsity_weight=1e-3,
        balance_weight=2e-2,
        gate_temperature=0.8,
    ).to(DEVICE).to(DTYPE)
    print(f"MoE-PINN parameters: {sum(p.numel() for p in moe_model.parameters()):,}")

    loss_fn = PhysicsLoss(
        LossConfig(
            equation="burgers",
            nu=NU,
            use_ntk=use_ntk,
            w_res=1.0,
            w_ic=10.0,
            w_bc=5.0,
        )
    )

    def eval_l2() -> float:
        with torch.no_grad():
            return l2_relative_error(moe_model(xt_test), u_exact_flat)

    trainer = Trainer(
        moe_model,
        loss_fn,
        lr=LR,
        n_steps=N_STEPS,
        ntk_update_freq=200,
        use_ntk=use_ntk,
        device=DEVICE,
        save_dir=save_dir,
    )
    history = trainer.train(batch, eval_fn=eval_l2, eval_freq=500)
    trainer.save_checkpoint("burgers_moe")
    return moe_model, history


def _train_moe_staged(
    batch: Dict[str, torch.Tensor],
    X: np.ndarray,
    T: np.ndarray,
    U_exact: np.ndarray,
    xt_test: torch.Tensor,
    u_exact_flat: torch.Tensor,
    *,
    save_dir: str,
    use_ntk: bool,
) -> tuple[torch.nn.Module, Dict[str, list]]:
    print("\n" + "=" * 60)
    print("Training MoE-PINN with staged specialists")
    print("=" * 60)

    moe_model = build_burgers_moe(
        num_experts=3,
        use_fourier=True,
        sparsity_weight=1e-3,
        balance_weight=2e-2,
        gate_temperature=0.9,
    ).to(DEVICE).to(DTYPE)
    print(f"MoE-PINN parameters: {sum(p.numel() for p in moe_model.parameters()):,}")

    cfg = StagedBurgersConfig.from_total_steps(N_STEPS)
    print(
        f"Stage config | base={cfg.base_steps} | expert={cfg.expert_steps} each | "
        f"gate={cfg.gate_steps} | joint={cfg.joint_steps}"
    )

    base_history = pretrain_base_model(
        moe_model,
        batch,
        cfg=cfg,
        nu=NU,
        save_dir=save_dir,
        xt_test=xt_test,
        u_exact_flat=u_exact_flat,
    )
    expert_batches = build_specialist_batches(X, T, U_exact, batch, cfg=cfg)
    x_vals = X[0, :]
    grad_x = np.abs(np.gradient(U_exact, x_vals, axis=1))
    dx = float(np.mean(np.diff(x_vals)))
    margin = max(6.0 * dx, 0.12 * (x_vals.max() - x_vals.min()))
    interior = (x_vals >= x_vals.min() + margin) & (x_vals <= x_vals.max() - margin)
    grad_inner = grad_x[:, interior]
    center_idx_inner = grad_inner.argmax(axis=1)
    center_idx = np.flatnonzero(interior)[center_idx_inner]
    center_x = x_vals[center_idx]
    shock_band = np.abs(X - center_x[:, None]) <= max(
        4.0 * dx,
        0.08 * (x_vals.max() - x_vals.min()),
    )
    shock_mask_flat = torch.tensor(
        shock_band.ravel(),
        dtype=torch.bool,
        device=xt_test.device,
    )
    expert_histories = pretrain_experts(
        moe_model,
        expert_batches,
        cfg=cfg,
        nu=NU,
        save_dir=save_dir,
        xt_test=xt_test,
        u_exact_flat=u_exact_flat,
        shock_mask_flat=shock_mask_flat,
    )

    gate_targets = build_gate_targets(
        moe_model,
        xt_test,
        u_exact_flat,
        temperature=cfg.gate_target_temperature,
        cfg=cfg,
    )
    gate_history = train_gate(
        moe_model,
        xt_test,
        gate_targets,
        cfg=cfg,
    )

    joint_history = joint_finetune(
        moe_model,
        batch,
        cfg=cfg,
        nu=NU,
        use_ntk=use_ntk,
        device=DEVICE,
        save_dir=save_dir,
        xt_test=xt_test,
        u_exact_flat=u_exact_flat,
    )

    merged_history = dict(joint_history)
    if "l2_error" in base_history:
        merged_history["base_l2_error"] = base_history["l2_error"]
    for name, history in expert_histories.items():
        if "l2_error" in history:
            merged_history[f"{name}_l2_error"] = history["l2_error"]
    merged_history["gate_stage_loss"] = gate_history["loss"]
    merged_history["gate_stage_entropy"] = gate_history["entropy"]
    merged_history["gate_stage_max"] = gate_history["max_gate"]
    return moe_model, merged_history


def _train_vanilla(
    batch: Dict[str, torch.Tensor],
    *,
    save_dir: str,
) -> tuple[torch.nn.Module, Dict[str, list]]:
    print("\n" + "=" * 60)
    print("Baseline: Training Vanilla PINN")
    print("=" * 60)

    van_model = VanillaPINN(
        in_dim=2,
        out_dim=1,
        hidden=64,
        depth=5,
        output_transform="burgers_hard_bc",
    ).to(DEVICE).to(DTYPE)
    loss_fn = PhysicsLoss(
        LossConfig(
            equation="burgers",
            nu=NU,
            use_ntk=False,
            w_res=1.0,
            w_ic=10.0,
            w_bc=5.0,
        )
    )
    trainer = Trainer(
        van_model,
        loss_fn,
        lr=LR,
        n_steps=N_STEPS,
        use_ntk=False,
        device=DEVICE,
        save_dir=save_dir,
    )
    history = trainer.train(batch, eval_freq=500)
    trainer.save_checkpoint("burgers_vanilla")
    return van_model, history


def _save_metrics_and_plots(
    *,
    save_dir: str,
    X: np.ndarray,
    T: np.ndarray,
    U_exact: np.ndarray,
    U_moe: np.ndarray,
    U_vanilla: np.ndarray,
    history_moe: Dict[str, list],
    history_van: Dict[str, list],
    moe_model: torch.nn.Module,
    xt_test: torch.Tensor,
) -> None:
    l2_moe = np.linalg.norm(U_moe - U_exact) / (np.linalg.norm(U_exact) + 1e-10)
    l2_van = np.linalg.norm(U_vanilla - U_exact) / (np.linalg.norm(U_exact) + 1e-10)
    mae_moe = np.abs(U_moe - U_exact).max()
    mae_van = np.abs(U_vanilla - U_exact).max()

    print("\n" + "=" * 60)
    print("Evaluation")
    print("=" * 60)
    print(f"MoE-PINN  L2 rel. error: {l2_moe:.4e} | Max abs error: {mae_moe:.4e}")
    print(f"Vanilla   L2 rel. error: {l2_van:.4e} | Max abs error: {mae_van:.4e}")
    print(f"Error reduction (L2): {l2_van / max(l2_moe, 1e-12):.2f}x")

    metrics_path = os.path.join(save_dir, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as file:
        file.write(f"Burgers Equation (nu={NU:.6f})\n")
        file.write(f"Training steps: {N_STEPS}\n\n")
        file.write(f"MoE-PINN  L2: {l2_moe:.6e}  MaxErr: {mae_moe:.6e}\n")
        file.write(f"Vanilla   L2: {l2_van:.6e}  MaxErr: {mae_van:.6e}\n")
        file.write(
            f"Improvement: {l2_van / max(l2_moe, 1e-12):.2f}x (L2)  "
            f"{mae_van / max(mae_moe, 1e-12):.2f}x (MaxErr)\n"
        )
    print(f"[OK] Saved metrics: {metrics_path}")

    print("\nGenerating plots...")
    t_slices = [s for s in [0.25, 0.5, 0.75, 1.0] if s <= T.max()]
    plot_solution_comparison(
        X,
        T,
        U_exact,
        U_moe,
        U_vanilla,
        equation="Burgers",
        t_slices=t_slices,
        save_path=os.path.join(save_dir, "solution_comparison.png"),
    )
    plot_error_distribution(
        X,
        T,
        U_exact,
        U_moe,
        U_vanilla,
        save_path=os.path.join(save_dir, "error_distribution.png"),
    )
    plot_loss_curves(
        history_moe,
        history_van,
        save_path=os.path.join(save_dir, "loss_curves.png"),
    )
    plot_ntk_weight_evolution(
        history_moe,
        save_path=os.path.join(save_dir, "ntk_evolution.png"),
    )
    plot_shock_diagnostics(
        X,
        T,
        U_exact,
        U_moe,
        U_vanilla,
        save_path=os.path.join(save_dir, "shock_diagnostics.png"),
    )
    plot_shock_zoom_comparison(
        X,
        T,
        U_exact,
        U_moe,
        U_vanilla,
        save_path=os.path.join(save_dir, "shock_zoom_comparison.png"),
    )
    plot_metrics_bar(
        l2_moe,
        l2_van,
        mae_moe,
        mae_van,
        save_path=os.path.join(save_dir, "metrics_bar.png"),
    )

    moe_model.eval()
    with torch.no_grad():
        gate_flat = moe_model.get_gate_weights(xt_test).cpu().numpy()
    gates = gate_flat.reshape(T.shape[0], T.shape[1], gate_flat.shape[-1])
    plot_gating_weights(
        X,
        T,
        gates,
        num_experts=gates.shape[-1],
        expert_names=["ShockExpert (Swish)", "SmoothExpert (tanh)", "DispersionExpert (sin)"],
        save_path=os.path.join(save_dir, "gating_weights.png"),
    )

    if hasattr(moe_model, "experts"):
        with torch.no_grad():
            if hasattr(moe_model, "get_expert_predictions"):
                expert_preds = [
                    pred.cpu().numpy().reshape(T.shape[0], T.shape[1])
                    for pred in moe_model.get_expert_predictions(xt_test).unbind(dim=1)
                ]
            else:
                expert_preds = [
                    expert(xt_test).cpu().numpy().reshape(T.shape[0], T.shape[1])
                    for expert in moe_model.experts
                ]
        plot_expert_specialization(
            X,
            T,
            U_exact,
            expert_preds,
            expert_names=["ShockExpert (Swish)", "SmoothExpert (tanh)", "DispersionExpert (sin)"],
            save_path=os.path.join(save_dir, "expert_specialization.png"),
        )

        expert_metrics_path = os.path.join(save_dir, "expert_metrics.txt")
        with open(expert_metrics_path, "w", encoding="utf-8") as file:
            file.write("Per-expert diagnostics\n")
            x_vals = X[0, :]
            grad_x = np.abs(np.gradient(U_exact, x_vals, axis=1))
            dx = float(np.mean(np.diff(x_vals)))
            margin = max(6.0 * dx, 0.12 * (x_vals.max() - x_vals.min()))
            interior = (x_vals >= x_vals.min() + margin) & (x_vals <= x_vals.max() - margin)
            grad_inner = grad_x[:, interior]
            center_idx_inner = grad_inner.argmax(axis=1)
            center_idx = np.flatnonzero(interior)[center_idx_inner]
            center_x = x_vals[center_idx]
            shock_band = np.abs(X - center_x[:, None]) <= max(
                4.0 * dx,
                0.08 * (x_vals.max() - x_vals.min()),
            )
            for name, pred in zip(
                ["ShockExpert", "SmoothExpert", "DispersionExpert"],
                expert_preds,
            ):
                abs_err = np.abs(pred - U_exact)
                l2 = np.linalg.norm(pred - U_exact) / (np.linalg.norm(U_exact) + 1e-10)
                max_err = abs_err.max()
                shock_mae = np.nanmean(np.where(shock_band, abs_err, np.nan))
                bg_mae = np.nanmean(np.where(~shock_band, abs_err, np.nan))
                file.write(
                    f"{name}  L2: {l2:.6e}  MaxErr: {max_err:.6e}  "
                    f"ShockMAE: {shock_mae:.6e}  BgMAE: {bg_mae:.6e}\n"
                )
        print(f"[OK] Saved metrics: {expert_metrics_path}")

    print(f"\n[OK] All results saved to: {save_dir}/")


def main(use_ntk: bool = True, train_mode: str = "end_to_end") -> None:
    save_dir = SAVE_DIR_STAGED if train_mode == "staged" else SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Training mode: {train_mode}")

    eq = BurgersEquation(nu=NU, device=DEVICE, dtype=DTYPE)
    xt_ic, u_ic = eq.initial_condition(N_IC)
    xt_bc, u_bc = eq.boundary_condition(N_BC)
    xt_col = eq.collocation_points(N_COL, method="lhs")
    batch = {
        "xt_col": xt_col,
        "xt_ic": xt_ic,
        "u_ic": u_ic,
        "xt_bc": xt_bc,
        "u_bc": u_bc,
    }

    print("Computing reference exact solution on test grid...")
    nx, nt = 100, 50
    X, T, U_exact = eq.test_grid(nx=nx, nt=nt)
    xt_test, u_exact_flat = _make_eval_tensors(
        X,
        T,
        U_exact,
        device=DEVICE,
        dtype=DTYPE,
    )

    if train_mode == "staged":
        moe_model, history_moe = _train_moe_staged(
            batch,
            X,
            T,
            U_exact,
            xt_test,
            u_exact_flat,
            save_dir=save_dir,
            use_ntk=use_ntk,
        )
    else:
        moe_model, history_moe = _train_moe_end_to_end(
            batch,
            xt_test,
            u_exact_flat,
            save_dir=save_dir,
            use_ntk=use_ntk,
        )

    van_model, history_van = _train_vanilla(batch, save_dir=save_dir)

    U_moe = _evaluate_model(moe_model, xt_test, nt=nt, nx=nx)
    U_vanilla = _evaluate_model(van_model, xt_test, nt=nt, nx=nx)
    _save_metrics_and_plots(
        save_dir=save_dir,
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


if __name__ == "__main__":
    main()
