"""
run_kdv.py — KdV 双孤子碰撞完整实验（战略第三步：冲击篇）

运行：
    python experiments/run_kdv.py

输出：
    results/kdv/solution_comparison.png
    results/kdv/soliton_collision.png
    results/kdv/error_distribution.png
    results/kdv/loss_curves.png
    results/kdv/gating_weights.png
    results/kdv/metrics.txt
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float32
print(f"Device: {DEVICE}")

from equations.kdv import KdVEquation, hirota_two_soliton
from core.moe_pinn import build_kdv_moe, VanillaPINN
from training.loss_functions import PhysicsLoss, LossConfig, l2_relative_error
from training.trainer import Trainer
from visualization.plots import (
    plot_solution_comparison, plot_error_distribution,
    plot_loss_curves, plot_gating_weights, plot_soliton_collision,
)

# ── 超参数 ────────────────────────────────────────────────────────────────────
K1, K2      = 1.0, 0.5         # 孤子波数
DELTA1      = 0.0
DELTA2      = 4.0              # 初始相位：让两孤子在计算域中段碰撞
X_RANGE     = (-20.0, 20.0)
T_RANGE     = (0.0, 6.0)
N_COL       = 5000
N_IC        = 400
N_BC        = 100
N_STEPS     = 8000
LR          = 1e-3
SAVE_DIR    = "results/kdv"
os.makedirs(SAVE_DIR, exist_ok=True)


def cast(t):
    """Cast tensor to experiment dtype and device."""
    return t.to(device=DEVICE, dtype=DTYPE)


def main(use_ntk: bool = True):
    # ═══════════════════════════════════════════════════════════════════════
    # 1. 生成数据
    # ═══════════════════════════════════════════════════════════════════════
    eq = KdVEquation(
        x_range=X_RANGE, t_range=T_RANGE,
        k1=K1, k2=K2, delta1=DELTA1, delta2=DELTA2,
        device=DEVICE, dtype=DTYPE,
    )

    xt_ic, u_ic = eq.initial_condition(N_IC)
    xt_bc, u_bc = eq.boundary_condition(N_BC)
    xt_col      = eq.collocation_points(N_COL, method="lhs")

    batch = {
        "xt_col": xt_col,   # 已是正确 dtype/device
        "xt_ic":  xt_ic,
        "u_ic":   u_ic,
        "xt_bc":  xt_bc,
        "u_bc":   u_bc,
    }

    print("Building test grid (Hirota exact solution)...")
    NX, NT = 128, 60
    X, T, U_exact = eq.test_grid(nx=NX, nt=NT)

    x_flat = torch.tensor(X.ravel(), dtype=DTYPE, device=DEVICE)
    t_flat = torch.tensor(T.ravel(), dtype=DTYPE, device=DEVICE)
    xt_test = torch.stack([x_flat, t_flat], dim=-1)
    u_exact_flat = torch.tensor(U_exact.ravel(), dtype=DTYPE, device=DEVICE).unsqueeze(-1)

    def make_pred(model):
        model.eval()
        with torch.no_grad():
            return model(xt_test).cpu().numpy().reshape(NT, NX)

    # ═══════════════════════════════════════════════════════════════════════
    # 2. 训练 MoE-PINN（碰撞专家 + NTK）
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("Step 3 (Breakthrough): Training KdV MoE-PINN")
    print(f"  DispersionExpert (LearnableSin) handles collision window")
    print(f"  Hirota exact ground truth: k1={K1}, k2={K2}")
    print("="*60)

    moe_model = build_kdv_moe(
        num_experts=3,
        sparsity_weight=1e-3,
        balance_weight=5e-3,
        gate_temperature=0.9,
    ).to(DEVICE).to(DTYPE)
    print(f"MoE-PINN parameters: {sum(p.numel() for p in moe_model.parameters()):,}")

    loss_cfg = LossConfig(equation="kdv", use_ntk=use_ntk, w_res=1.0, w_ic=10.0, w_bc=5.0)
    loss_fn  = PhysicsLoss(loss_cfg)

    def eval_moe():
        with torch.no_grad():
            return l2_relative_error(moe_model(xt_test), u_exact_flat)

    trainer_moe = Trainer(
        moe_model, loss_fn, lr=LR, n_steps=N_STEPS,
        ntk_update_freq=200, use_ntk=use_ntk,
        device=DEVICE, save_dir=SAVE_DIR,
    )
    history_moe = trainer_moe.train(batch, eval_fn=eval_moe, eval_freq=500)
    trainer_moe.save_checkpoint("kdv_moe")

    # ═══════════════════════════════════════════════════════════════════════
    # 3. 训练 Vanilla-PINN（对照）
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("Baseline: Vanilla PINN (tanh, no MoE)")
    print("="*60)

    van_model = VanillaPINN(in_dim=2, out_dim=1, hidden=64, depth=5).to(DEVICE).to(DTYPE)
    loss_fn_v = PhysicsLoss(LossConfig(equation="kdv", use_ntk=False,
                                       w_res=1.0, w_ic=10.0, w_bc=5.0))
    trainer_van = Trainer(van_model, loss_fn_v, lr=LR, n_steps=N_STEPS,
                          use_ntk=False, device=DEVICE, save_dir=SAVE_DIR)
    history_van = trainer_van.train(batch, eval_freq=500)
    trainer_van.save_checkpoint("kdv_vanilla")

    # ═══════════════════════════════════════════════════════════════════════
    # 4. 误差评估
    # ═══════════════════════════════════════════════════════════════════════
    U_moe     = make_pred(moe_model)
    U_vanilla = make_pred(van_model)

    l2_moe = np.linalg.norm(U_moe - U_exact) / (np.linalg.norm(U_exact) + 1e-10)
    l2_van = np.linalg.norm(U_vanilla - U_exact) / (np.linalg.norm(U_exact) + 1e-10)
    mae_moe = np.abs(U_moe - U_exact).max()
    mae_van = np.abs(U_vanilla - U_exact).max()

    print(f"\nMoE-PINN  — L2 rel: {l2_moe:.4e} | MaxErr: {mae_moe:.4e}")
    print(f"Vanilla   — L2 rel: {l2_van:.4e} | MaxErr: {mae_van:.4e}")
    print(f"Improvement: {l2_van/l2_moe:.1f}× (L2)")

    with open(os.path.join(SAVE_DIR, "metrics.txt"), "w") as f:
        f.write(f"KdV Two-Soliton Collision\n")
        f.write(f"k1={K1}, k2={K2}, delta1={DELTA1}, delta2={DELTA2}\n")
        f.write(f"Training steps: {N_STEPS}\n\n")
        f.write(f"MoE-PINN L2: {l2_moe:.6e}  MaxErr: {mae_moe:.6e}\n")
        f.write(f"Vanilla  L2: {l2_van:.6e}  MaxErr: {mae_van:.6e}\n")
        f.write(f"Improvement: {l2_van/l2_moe:.2f}× (L2)\n")

    # ═══════════════════════════════════════════════════════════════════════
    # 5. 可视化
    # ═══════════════════════════════════════════════════════════════════════
    print("\nGenerating plots...")

    plot_solution_comparison(
        X, T, U_exact, U_moe, U_vanilla,
        equation="KdV Two-Soliton",
        t_slices=[T_RANGE[1]*0.1, T_RANGE[1]*0.3, T_RANGE[1]*0.6, T_RANGE[1]*0.9],
        save_path=os.path.join(SAVE_DIR, "solution_comparison.png"),
    )
    plot_error_distribution(
        X, T, U_exact, U_moe, U_vanilla,
        save_path=os.path.join(SAVE_DIR, "error_distribution.png"),
    )
    plot_loss_curves(
        history_moe, history_van,
        save_path=os.path.join(SAVE_DIR, "loss_curves.png"),
    )
    plot_soliton_collision(
        X, T, U_exact, U_moe,
        t_snapshots=[0, NT//5, 2*NT//5, 3*NT//5, NT-1],
        save_path=os.path.join(SAVE_DIR, "soliton_collision.png"),
    )

    # 门控权重
    moe_model.eval()
    with torch.no_grad():
        g_flat = moe_model.get_gate_weights(xt_test).cpu().numpy()
    G = g_flat.reshape(NT, NX, 3)
    plot_gating_weights(
        X, T, G, num_experts=3,
        expert_names=["DispersionExpert (sin)", "SmoothExpert (tanh)", "ShockExpert (Swish)"],
        save_path=os.path.join(SAVE_DIR, "gating_weights.png"),
    )

    print(f"\n✓ All results saved to: {SAVE_DIR}/")


if __name__ == "__main__":
    main()
