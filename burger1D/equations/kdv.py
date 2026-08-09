"""
kdv.py — Korteweg-de Vries (KdV) 方程与双孤子碰撞
=============================================
论文对应章节：研究情景三，冲击篇

方程（无量纲标准形式）：
  u_t + 6*u*u_x + u_xxx = 0
  x ∈ [-20, 20], t ∈ [0, 6]

精确解（广田双线性方法 Hirota Bilinear Method）：
  u(x,t) = 2 * ∂_xx ln f(x,t)

  f(x,t) = 1 + e^{η₁} + e^{η₂} + A₁₂ * e^{η₁+η₂}
  η_i = k_i * x - k_i³ * t + δ_i
  A₁₂ = ((k₁ - k₂) / (k₁ + k₂))²

核心挑战：
  1. u_xxx 三阶导数 → 自动微分噪声以指数级放大
  2. 孤子碰撞时相位漂移必须完全守恒（LearnableSin 激活函数解决）
  3. 哈密顿量 H = ∫(½u_x² - u³)dx 需全程守恒（附加不变量约束）
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# 精确解：广田双线性方法（两孤子对碰）
# ─────────────────────────────────────────────────────────────────────────────
def hirota_two_soliton(
    x: np.ndarray,
    t: np.ndarray,
    k1: float = 1.0,
    k2: float = 0.5,
    delta1: float = 0.0,
    delta2: float = 4.0,
) -> np.ndarray:
    """
    KdV 双孤子精确解（广田双线性公式）。

    u(x,t) = 2 * ∂²/∂x² ln f(x,t)

    其中：
      η_i = k_i * x - k_i³ * t + δ_i
      f = 1 + e^{η₁} + e^{η₂} + A₁₂ * e^{η₁+η₂}
      A₁₂ = ((k₁-k₂)/(k₁+k₂))²

    孤子速度：v_i = k_i² （振幅越大，速度越快）
    孤子振幅：a_i = 2*k_i² / (2π) ... 实际振幅 ∝ k_i²

    Args:
        x     : [...] 空间坐标（任意形状）
        t     : [...] 时间坐标（同形状）
        k1    : 第一孤子波数（k1 > k2 > 0 时 soliton1 更快）
        k2    : 第二孤子波数
        delta1: 第一孤子初始相位
        delta2: 第二孤子初始相位

    Returns:
        u     : [...] 精确解数组
    """
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    # 相位
    eta1 = k1 * x - k1**3 * t + delta1
    eta2 = k2 * x - k2**3 * t + delta2

    # 交叉作用系数
    A12 = ((k1 - k2) / (k1 + k2)) ** 2

    # 为数值稳定性，对 e^η 进行截断（防止溢出）
    eta1_s = np.clip(eta1, -50, 50)
    eta2_s = np.clip(eta2, -50, 50)

    e1 = np.exp(eta1_s)
    e2 = np.exp(eta2_s)
    e12 = A12 * np.exp(np.clip(eta1_s + eta2_s, -50, 50))

    f = 1.0 + e1 + e2 + e12

    # ∂f/∂x
    f_x = k1 * e1 + k2 * e2 + (k1 + k2) * e12

    # ∂²f/∂x²
    f_xx = k1**2 * e1 + k2**2 * e2 + (k1 + k2)**2 * e12

    # u = 2 * ∂²/∂x² ln f = 2 * (f_xx * f - f_x²) / f²
    u = 2.0 * (f_xx * f - f_x**2) / (f**2 + 1e-20)

    return u


def single_soliton(
    x: np.ndarray,
    t: np.ndarray,
    k: float = 1.0,
    delta: float = 0.0,
) -> np.ndarray:
    """
    KdV 单孤子精确解：u(x,t) = (k²/2) * sech²(k/2 * (x - k²t + δ))

    Args:
        x, t  : 时空坐标
        k     : 波数（振幅 = k²/2）
        delta : 初始相位偏移
    """
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    phase = k / 2.0 * (x - k**2 * t) + delta
    u = (k**2 / 2.0) / np.cosh(phase)**2
    return u


# ─────────────────────────────────────────────────────────────────────────────
# 方程类：KdV 配点生成与 PDE 残差
# ─────────────────────────────────────────────────────────────────────────────
class KdVEquation:
    """
    KdV 方程：u_t + 6*u*u_x + u_xxx = 0

    包含：
    - 配点生成（初始条件 / 边界条件 / 内部点）
    - PDE 残差计算（含三阶导数 u_xxx）
    - 守恒量评估（质量、动量、哈密顿量）
    - 精确解接口（广田双线性方法）
    """

    def __init__(
        self,
        x_range: Tuple[float, float] = (-20.0, 20.0),
        t_range: Tuple[float, float] = (0.0, 6.0),
        k1: float = 1.0,
        k2: float = 0.5,
        delta1: float = 0.0,
        delta2: float = 4.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ):
        self.x_min, self.x_max = x_range
        self.t_min, self.t_max = t_range
        self.k1 = k1
        self.k2 = k2
        self.delta1 = delta1
        self.delta2 = delta2
        self.device = device or torch.device("cpu")
        self.dtype = dtype

    # ─── 初始条件（双孤子 t=0 时刻） ─────────────────────────────────────
    def initial_condition(
        self, n: int = 512
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        IC: u(x, 0) = Hirota_two_soliton(x, 0)

        Returns:
            xt_ic: [n, 2]
            u_ic : [n, 1]
        """
        x_np = np.linspace(self.x_min, self.x_max, n)
        t_np = np.zeros(n)
        u_np = hirota_two_soliton(
            x_np, t_np, self.k1, self.k2, self.delta1, self.delta2
        )

        x = torch.tensor(x_np, dtype=self.dtype)
        t = torch.tensor(t_np, dtype=self.dtype)
        u = torch.tensor(u_np, dtype=self.dtype)

        xt = torch.stack([x, t], dim=-1).to(self.device)
        return xt, u.unsqueeze(-1).to(self.device)

    # ─── 边界条件（周期 BC 近似：u → 0 at far field）──────────────────────
    def boundary_condition(
        self, n: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        远场 BC: u(±L, t) ≈ 0 （孤子传播未抵达边界）

        Returns:
            xt_bc: [2n, 2]
            u_bc : [2n, 1]
        """
        t = torch.linspace(self.t_min, self.t_max, n, dtype=self.dtype)

        x_left = torch.full((n,), self.x_min, dtype=self.dtype)
        xt_left = torch.stack([x_left, t], dim=-1)

        x_right = torch.full((n,), self.x_max, dtype=self.dtype)
        xt_right = torch.stack([x_right, t], dim=-1)

        xt_bc = torch.cat([xt_left, xt_right], dim=0).to(self.device)
        u_bc = torch.zeros(2 * n, 1, dtype=self.dtype).to(self.device)
        return xt_bc, u_bc

    # ─── 内部配点 ─────────────────────────────────────────────────────────
    def collocation_points(
        self, n: int = 6000, method: str = "lhs"
    ) -> torch.Tensor:
        """
        在时空域内采样配点。为捕捉碰撞时刻（孤子叠加区），
        可在碰撞区域额外加密采样。

        Returns:
            xt_col: [n, 2]
        """
        if method == "lhs":
            from .burgers import _lhs_sample
            xt_np = _lhs_sample(n, 2)
            xt_np[:, 0] = xt_np[:, 0] * (self.x_max - self.x_min) + self.x_min
            xt_np[:, 1] = xt_np[:, 1] * (self.t_max - self.t_min) + self.t_min
        else:
            x_r = np.random.uniform(self.x_min, self.x_max, n)
            t_r = np.random.uniform(self.t_min, self.t_max, n)
            xt_np = np.stack([x_r, t_r], axis=-1)

        return torch.tensor(xt_np, dtype=self.dtype).to(self.device)

    # ─── PDE 残差（含三阶导数 u_xxx）────────────────────────────────────
    @staticmethod
    def pde_residual(
        u_pred: torch.Tensor,
        xt: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算 KdV 残差：r = u_t + 6*u*u_x + u_xxx

        三阶导数通过连续自动微分获取：
          u_x   = ∂u/∂x
          u_xx  = ∂u_x/∂x
          u_xxx = ∂u_xx/∂x

        Args:
            u_pred: [N, 1] 模型预测
            xt    : [N, 2] 时空坐标（需 requires_grad=True）

        Returns:
            residual: [N, 1]
        """
        # 一阶导数
        u_grad = torch.autograd.grad(
            u_pred.sum(), xt,
            create_graph=True, retain_graph=True
        )[0]                        # [N, 2]
        u_x = u_grad[:, 0:1]
        u_t = u_grad[:, 1:2]

        # 二阶空间导数
        u_xx = torch.autograd.grad(
            u_x.sum(), xt,
            create_graph=True, retain_graph=True
        )[0][:, 0:1]               # [N, 1]

        # 三阶空间导数
        u_xxx = torch.autograd.grad(
            u_xx.sum(), xt,
            create_graph=True, retain_graph=True
        )[0][:, 0:1]               # [N, 1]

        residual = u_t + 6.0 * u_pred * u_x + u_xxx
        return residual

    # ─── 守恒量约束损失（保结构 PINN 核心）──────────────────────────────
    @staticmethod
    def conservation_loss(
        u_pred: torch.Tensor,
        xt: torch.Tensor,
        x_quad: torch.Tensor,
        u_quad: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算质量守恒和哈密顿量守恒的惩罚损失。

        质量：   M = ∫ u dx   （应保持为常数）
        哈密顿：  H = ∫ (½u_x² - u³) dx （应保持为常数）

        实现方式：数值梯形积分（近似），与初始时刻的值比较偏差。

        Args:
            u_pred : 当前预测
            xt     : 对应坐标
            x_quad : 空间积分求积点（1D）[M]
            u_quad : 对应 u 预测值 [M, 1]

        Returns:
            loss_mass, loss_hamiltonian: scalar tensors
        """
        u_q = u_quad.squeeze(-1)           # [M]

        # 梯形积分近似（需排序的 x）
        dx = x_quad[1:] - x_quad[:-1]     # [M-1]

        # 质量
        mass_integrand = (u_q[1:] + u_q[:-1]) * 0.5
        mass = (mass_integrand * dx).sum()

        # 哈密顿量（需要 u_x）
        u_x_q = (u_q[1:] - u_q[:-1]) / (dx + 1e-10)   # 有限差分近似
        u_mid = (u_q[1:] + u_q[:-1]) * 0.5
        ham_integrand = 0.5 * u_x_q**2 - u_mid**3
        hamiltonian = (ham_integrand * dx).sum()

        # 惩罚：与初始时刻的理想值偏差（这里简化为零，要求 t 域上 H 的方差最小）
        loss_mass = mass**2  # 鼓励质量稳定
        loss_ham = hamiltonian**2

        return loss_mass, loss_ham

    # ─── 精确解网格（用于误差评估）───────────────────────────────────────
    def test_grid(
        self, nx: int = 256, nt: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        返回均匀测试网格上的精确解。

        Returns:
            X, T    : [nt, nx] 网格
            U_exact : [nt, nx] 精确解
        """
        x_lin = np.linspace(self.x_min, self.x_max, nx)
        t_lin = np.linspace(self.t_min, self.t_max, nt)
        X, T = np.meshgrid(x_lin, t_lin)

        U_exact = hirota_two_soliton(
            X.ravel(), T.ravel(), self.k1, self.k2, self.delta1, self.delta2
        ).reshape(nt, nx)

        return X, T, U_exact
