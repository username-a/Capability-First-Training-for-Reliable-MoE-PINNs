"""
burgers.py — 1D 粘性伯格斯方程（Viscous Burgers' Equation）
=============================================
论文对应章节：研究情景一，基础篇 + 强化篇

方程：
  u_t + u * u_x = ν * u_xx
  x ∈ [-1, 1], t ∈ [0, 1]
  IC: u(x, 0) = -sin(πx)
  BC: u(-1, t) = u(1, t) = 0

精确解（Cole-Hopf 变换 + Gauss-Hermite 数值积分）：
  u(x,t) = -2ν * ∂_x [ln φ(x,t)]

核心挑战：
  ν = 0.01/π ≈ 0.00318 时，t > 0.5 处形成近垂直激波。
  标准 PINN 产生严重 Gibbs 振荡；MoE-PINN 通过路由机制精准捕获。
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional
from scipy.special import roots_hermite


# ─────────────────────────────────────────────────────────────────────────────
# 精确解：Cole-Hopf 变换 + Gauss-Hermite 求积
# ─────────────────────────────────────────────────────────────────────────────
def cole_hopf_exact(
    x: np.ndarray,
    t: np.ndarray,
    nu: float = 0.01 / np.pi,
    n_quad: int = 200,
) -> np.ndarray:
    """
    通过 Cole-Hopf 变换精确求解 Burgers 方程（数值积分版）。

    φ(x,t) = ∫ φ₀(y) * G(x-y, t) dy
    u(x,t) = -∫ (x-y)/t * φ₀(y) * G(x-y, t) dy  /  ∫ φ₀(y) * G(x-y, t) dy

    其中：
      G(x-y, t) = exp(-(x-y)²/(4νt))（热核）
      φ₀(y) = exp((cos(πy) - 1) / (2πν))（初始条件变换）

    使用 Gauss-Hermite 求积高效计算无限域积分：
      ∫_{-∞}^{∞} f(y) e^{-y²} dy ≈ Σ_k w_k f(y_k)

    Args:
        x      : [N] 空间坐标数组
        t      : [N] 时间坐标数组（同形状）
        nu     : 粘性系数（默认 0.01/π）
        n_quad : Gauss-Hermite 求积点数（≥100 确保高精度）

    Returns:
        u_exact: [N] 精确解数组
    """
    # Gauss-Hermite 求积节点和权重
    gh_nodes, gh_weights = roots_hermite(n_quad)

    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    u_exact = np.zeros_like(x)

    for idx in range(len(x)):
        xi, ti = x[idx], t[idx]

        if ti <= 1e-8:
            # t=0 直接返回初始条件
            u_exact[idx] = -np.sin(np.pi * xi)
            continue

        # 变量替换：y = xi + 2*sqrt(nu*ti) * s，将高斯积分标准化
        sigma = np.sqrt(4 * nu * ti)  # 热核标准差×√2

        # 求积变量 s（Gauss-Hermite 节点）对应的 y 值
        y_nodes = xi + sigma * gh_nodes  # [n_quad]

        # φ₀(y) = exp((cos(πy) - 1) / (2πν))
        log_phi0 = (1.0 - np.cos(np.pi * y_nodes)) / (2.0 * np.pi * nu)
        log_phi0 -= np.max(log_phi0)
        phi0 = np.exp(log_phi0)

        # 分子：∫ (x-y)/t * φ₀(y) * G(x-y,t) dy（变换后 Gauss-Hermite 权重已包含 e^{-s²}）
        numerator = np.sum(gh_weights * ((xi - y_nodes) / ti) * phi0)

        # 分母：∫ φ₀(y) * G(x-y,t) dy
        denominator = np.sum(gh_weights * phi0)

        if abs(denominator) < 1e-100:
            u_exact[idx] = 0.0
        else:
            u_exact[idx] = numerator / denominator

    return u_exact


# ─────────────────────────────────────────────────────────────────────────────
# 方程类：生成配点和计算 PDE 残差
# ─────────────────────────────────────────────────────────────────────────────
class BurgersEquation:
    """
    1D Burgers 方程的配点生成与物理残差计算。

    用法示例：
        eq = BurgersEquation(nu=0.01/np.pi)
        x_ic, u_ic = eq.initial_condition(n=200)
        x_bc, u_bc = eq.boundary_condition(n=100)
        x_col = eq.collocation_points(n=4000)
    """

    def __init__(
        self,
        nu: float = 0.01 / np.pi,
        x_range: Tuple[float, float] = (-1.0, 1.0),
        t_range: Tuple[float, float] = (0.0, 1.0),
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ):
        self.nu = nu
        self.x_min, self.x_max = x_range
        self.t_min, self.t_max = t_range
        self.device = device or torch.device("cpu")
        self.dtype = dtype

    # ─── 初始条件配点 ─────────────────────────────────────────────────────
    def initial_condition(
        self, n: int = 256, noise: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回初始条件采样点。

        IC: u(x, 0) = -sin(πx),  x ∈ [x_min, x_max]

        Returns:
            xt_ic : [n, 2]  时空坐标 (x, t=0)
            u_ic  : [n, 1]  精确 u 值
        """
        x = torch.linspace(self.x_min, self.x_max, n, dtype=self.dtype)
        t = torch.zeros_like(x)
        u = -torch.sin(torch.tensor(np.pi, dtype=self.dtype) * x)

        if noise > 0:
            u = u + noise * torch.randn_like(u)

        xt = torch.stack([x, t], dim=-1).to(self.device)
        return xt, u.unsqueeze(-1).to(self.device)

    # ─── 边界条件配点 ─────────────────────────────────────────────────────
    def boundary_condition(
        self, n: int = 128
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dirichlet BC: u(±1, t) = 0

        Returns:
            xt_bc : [2n, 2]  左右边界时空点
            u_bc  : [2n, 1]  零值
        """
        t = torch.linspace(self.t_min, self.t_max, n, dtype=self.dtype)

        # 左边界 x=-1
        x_left = torch.full((n,), self.x_min, dtype=self.dtype)
        xt_left = torch.stack([x_left, t], dim=-1)

        # 右边界 x=+1
        x_right = torch.full((n,), self.x_max, dtype=self.dtype)
        xt_right = torch.stack([x_right, t], dim=-1)

        xt_bc = torch.cat([xt_left, xt_right], dim=0).to(self.device)
        u_bc = torch.zeros(2 * n, 1, dtype=self.dtype).to(self.device)
        return xt_bc, u_bc

    # ─── 内部配点（用于 PDE 残差）─────────────────────────────────────────
    def collocation_points(
        self,
        n: int = 4000,
        method: str = "lhs",
    ) -> torch.Tensor:
        """
        在计算域内均匀/LHS 采样配点。

        Args:
            n      : 配点数量
            method : "random" | "lhs"（拉丁超立方采样）| "uniform"

        Returns:
            xt_col: [n, 2] 配点坐标（需设置 requires_grad=True）
        """
        if method == "lhs":
            # 拉丁超立方采样（更均匀的覆盖）
            xt = _lhs_sample(n, 2)  # [n, 2] ∈ [0,1]²
            xt[:, 0] = xt[:, 0] * (self.x_max - self.x_min) + self.x_min
            xt[:, 1] = xt[:, 1] * (self.t_max - self.t_min) + self.t_min
        elif method == "uniform":
            nx = int(np.sqrt(n))
            nt = n // nx
            x_lin = np.linspace(self.x_min, self.x_max, nx)
            t_lin = np.linspace(self.t_min, self.t_max, nt)
            xx, tt = np.meshgrid(x_lin, t_lin)
            xt = np.stack([xx.ravel(), tt.ravel()], axis=-1)
        else:
            x_r = np.random.uniform(self.x_min, self.x_max, n)
            t_r = np.random.uniform(self.t_min, self.t_max, n)
            xt = np.stack([x_r, t_r], axis=-1)

        tensor = torch.tensor(xt, dtype=self.dtype).to(self.device)
        return tensor

    # ─── PDE 残差计算（核心，用于 loss_res）────────────────────────────────
    @staticmethod
    def pde_residual(
        u_pred: torch.Tensor,
        xt: torch.Tensor,
        nu: float,
    ) -> torch.Tensor:
        """
        计算 Burgers PDE 残差：r = u_t + u*u_x - ν*u_xx

        Args:
            u_pred: [N,1] 模型预测（xt 需有 requires_grad=True）
            xt    : [N,2] 时空坐标
            nu    : 粘性系数

        Returns:
            residual: [N,1]
        """
        u = u_pred

        # 一阶导数
        grads = torch.autograd.grad(
            u.sum(), xt,
            create_graph=True, retain_graph=True
        )[0]                        # [N, 2]
        u_x = grads[:, 0:1]
        u_t = grads[:, 1:2]

        # 二阶空间导数
        u_xx = torch.autograd.grad(
            u_x.sum(), xt,
            create_graph=True, retain_graph=True
        )[0][:, 0:1]                # [N, 1]

        residual = u_t + u * u_x - nu * u_xx
        return residual

    # ─── 生成测试网格（用于误差评估和可视化）───────────────────────────────
    def test_grid(
        self, nx: int = 200, nt: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        返回均匀网格的坐标和精确解（用于 L2 误差计算）。

        Returns:
            X, T : [nt, nx] 网格坐标数组
            U_exact: [nt, nx] 精确解（通过 Cole-Hopf 积分计算）
        """
        x_lin = np.linspace(self.x_min, self.x_max, nx)
        t_lin = np.linspace(self.t_min, self.t_max, nt)
        X, T = np.meshgrid(x_lin, t_lin)

        x_flat = X.ravel()
        t_flat = T.ravel()

        print(f"Computing Cole-Hopf exact solution on {nx}×{nt} grid...")
        u_flat = cole_hopf_exact(x_flat, t_flat, nu=self.nu, n_quad=100)
        U_exact = u_flat.reshape(nt, nx)

        return X, T, U_exact


# ─────────────────────────────────────────────────────────────────────────────
# 辅助：拉丁超立方采样
# ─────────────────────────────────────────────────────────────────────────────
def _lhs_sample(n: int, d: int) -> np.ndarray:
    """
    简单拉丁超立方采样（Latin Hypercube Sampling）。

    将 [0,1]^d 均匀分层，每层随机一点，确保全域均匀覆盖。
    """
    result = np.zeros((n, d))
    for dim in range(d):
        perm = np.random.permutation(n)
        result[:, dim] = (perm + np.random.uniform(size=n)) / n
    return result
