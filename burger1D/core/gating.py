"""
gating.py — 门控网络（Gating Network）
=============================================
论文对应章节：MoE域分解与探针机制

功能：
  1. 接收时空坐标 x 作为输入
  2. 输出 m 维 Softmax 概率向量 g(x)，作为各专家的混合权重
  3. 支持稀疏正则化损失 L_sp = (1/|B|) * sum(g_i^p)，p=0.5
  4. 提供"探针模式"：从雅可比矩阵计算散度/旋度，进行物理特征识别

数学定义：
  g(x) = Softmax(W_L * σ(... σ(W_1 * x + b_1) ...) + b_L)
  u_hat(x) = sum_i { g_i(x) * u_i(x) }         ← MoE加权聚合
  L_sp = (1/|B|) * sum_x sum_i { g_i(x)^p }     ← 稀疏正则，p ∈ (0,1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class GatingNetwork(nn.Module):
    """
    MoE 门控网络。

    Args:
        in_dim       : 输入坐标维度（1D Burgers=2, KdV=2, 2D Euler=3）
        num_experts  : 专家数量 m
        hidden       : 门控网络隐藏层宽度（通常比专家网络小）
        depth        : 隐藏层层数
        sparsity_p   : L_sp 稀疏指数，p∈(0,1)，论文取 0.5
        temperature  : Softmax 温度（<1 → 更稀疏，>1 → 更均匀）
    """

    def __init__(
        self,
        in_dim: int = 2,
        num_experts: int = 3,
        hidden: int = 32,
        depth: int = 3,
        sparsity_p: float = 0.5,
        temperature: float = 0.5,
        init_gain: float = 0.5,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.sparsity_p = sparsity_p
        self.temperature = temperature
        self.init_gain = init_gain
        self.in_dim = in_dim
        self.feature_dim = in_dim * 4

        # 构建轻量门控 MLP
        layers = [nn.Linear(self.feature_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, num_experts))
        self.mlp = nn.Sequential(*layers)

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=init_gain)
                nn.init.zeros_(m.bias)

    def _expand_features(self, x: torch.Tensor) -> torch.Tensor:
        """Lift coordinates into a richer routing feature space."""
        return torch.cat(
            [
                x,
                x * x,
                torch.sin(torch.pi * x),
                torch.cos(torch.pi * x),
            ],
            dim=-1,
        )

    # ─── 前向传播 ────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, in_dim] 时空坐标批次
        Returns:
            g: [N, num_experts] Softmax 门控权重
        """
        feats = self._expand_features(x)
        logits = self.mlp(feats)                      # [N, num_experts]
        g = F.softmax(logits / self.temperature, dim=-1)
        return g

    # ─── 稀疏正则化损失 ──────────────────────────────────────────────────────
    def sparsity_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        L_sp = (1/|B|) * Σ_x Σ_i  g_i(x)^p

        鼓励门控权重向 one-hot 集中（迫使工作分配专业化）。
        参数：p=0.5 时对接近 0 的权重惩罚极强，接近 1 的权重惩罚轻。

        Args:
            x: [N, in_dim]
        Returns:
            scalar 稀疏损失
        """
        g = self.forward(x)                            # [N, m]
        # 数值稳定：避免 0^0.5 梯度奇异，加小 eps
        eps = 1e-8
        sparse_penalty = torch.pow(g.clamp(min=eps), self.sparsity_p)
        return sparse_penalty.sum(dim=-1).mean()       # scalar

    def load_balance_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encourage balanced expert usage at the batch level while keeping
        per-sample routing free to become selective.
        """
        g = self.forward(x)
        mean_load = g.mean(dim=0)
        target = torch.full_like(mean_load, 1.0 / self.num_experts)
        return ((mean_load - target) ** 2).mean()

    # ─── 物理探针：计算散度（用于激波/接触间断识别）─────────────────────────
    def compute_divergence_proxy(
        self,
        velocity_pred: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        近似流场散度：∇·v ≈ ∂u/∂x + ∂v/∂y（2D）或 ∂u/∂x（1D）

        激波区域：∇·v << 0（强压缩）→ 将其配点路由给激波专家
        接触间断：密度突变但速度连续

        Args:
            velocity_pred: [N, d] 速度场预测（需带 requires_grad）
            coords       : [N, in_dim] 对应坐标
        Returns:
            divergence: [N, 1] 散度标量
        """
        if not coords.requires_grad:
            coords = coords.requires_grad_(True)

        divs = []
        d = velocity_pred.shape[-1]
        for i in range(d):
            grad_i = torch.autograd.grad(
                velocity_pred[:, i].sum(), coords,
                create_graph=True, retain_graph=True
            )[0]
            # grad_i: [N, in_dim]；取第 i 个空间维度的偏导
            divs.append(grad_i[:, i:i+1])

        return sum(divs)                               # [N, 1]

    # ─── 获取最活跃专家的索引（推理时用于分析）──────────────────────────────
    def get_routing_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        返回每个配点被路由到的主专家索引。

        Returns:
            indices: [N] 整数张量，值域 [0, num_experts-1]
        """
        with torch.no_grad():
            g = self.forward(x)
        return g.argmax(dim=-1)

    # ─── 获取专家负载均衡统计 ────────────────────────────────────────────────
    def load_balance_stats(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: [N, in_dim]
        Returns:
            dict: {"entropy": 平均熵, "max_weight": 各专家最大权重, ...}
        """
        with torch.no_grad():
            g = self.forward(x)                        # [N, m]
            routing = g.argmax(dim=-1)                 # [N]
            counts = torch.bincount(routing, minlength=self.num_experts)

        entropy = -(g * (g + 1e-10).log()).sum(dim=-1).mean().item()
        return {
            "mean_entropy":     entropy,
            "expert_load_frac": (counts.float() / counts.sum()).tolist(),
            "max_gate_weight":  g.max(dim=-1).values.mean().item(),
        }
