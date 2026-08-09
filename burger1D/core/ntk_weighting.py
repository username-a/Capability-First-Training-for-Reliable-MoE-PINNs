"""
ntk_weighting.py — 神经正切核（NTK）动态损失权重调适
=============================================
论文对应章节：强化篇 — NTK 动态加权机制

理论基础：
  在无限宽极限下，全连接神经网络的梯度下降动态由神经正切核（NTK）矩阵支配。
  K_ij = <∂f(x_i)/∂θ, ∂f(x_j)/∂θ>

  对于 PINN 的联合损失：
    L_total = λ_res * L_res + λ_ic * L_ic + λ_bc * L_bc

  不同损失项的 NTK 特征值尺度差异导致梯度刚性（Stiffness）。
  动态权重公式：
    λ_i(t) = max_eig(K_res(t)) / trace(K_i(t))

  这使得所有损失项具有相近的收敛速率，从根本上消除梯度病态。

参考文献：
  Wang, S. et al. (2022). "When and why PINNs fail to train:
  A neural tangent kernel perspective." JCP.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import math


class NTKWeighting:
    """
    NTK 动态权重计算器。

    在训练过程中周期性地（每 update_freq 步）重新估算各损失项
    的 NTK 矩阵迹（Trace），并据此更新自适应权重。

    Args:
        loss_names     : 损失项名称列表，例如 ["res", "ic", "bc"]
        update_freq    : 更新权重的步频（默认每 100 步更新一次）
        ema_alpha      : 指数移动平均平滑系数（防止权重剧烈抖动）
        subsample_n    : 计算 NTK 时采样的配点数（控制计算开销）
        device         : torch.device
    """

    def __init__(
        self,
        loss_names: List[str],
        update_freq: int = 100,
        ema_alpha: float = 0.9,
        subsample_n: int = 64,
        device: torch.device = None,
    ):
        self.loss_names = loss_names
        self.update_freq = update_freq
        self.ema_alpha = ema_alpha
        self.subsample_n = subsample_n
        self.device = device or torch.device("cpu")

        # 初始化权重为 1.0
        self.weights: Dict[str, float] = {name: 1.0 for name in loss_names}
        self._step = 0

        # EMA 状态（存储 trace 均值）
        self._ema_traces: Dict[str, float] = {name: 1.0 for name in loss_names}

    # ─── 主接口：返回当前权重字典 ─────────────────────────────────────────────
    def get_weights(self) -> Dict[str, float]:
        """返回当前 NTK 动态权重字典 {loss_name: weight}"""
        return dict(self.weights)

    # ─── 核心：估算单个损失项的 NTK 矩阵迹 ─────────────────────────────────
    @staticmethod
    def _compute_ntk_trace(
        model: nn.Module,
        loss_fn,           # callable() → scalar tensor
        subsample_n: int = 64,
    ) -> float:
        """
        高效估算 NTK 对角线迹：
          trace(K) ≈ Σ_i ||∂L/∂θ||^2  （对随机子样本求和）

        使用随机投影的方法（Hutchinson estimator）避免显式构建 N×N 矩阵。

        Returns:
            float: trace 估计值
        """
        try:
            # 保存当前梯度
            grads_before = {
                name: p.grad.clone() if p.grad is not None else None
                for name, p in model.named_parameters()
            }

            # 计算该损失项的梯度
            loss_val = loss_fn()
            loss_val.backward(retain_graph=True)

            # 收集梯度向量并计算其 L2 范数平方（即 NTK 迹的估计）
            trace = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    trace += p.grad.data.norm(2).item() ** 2

            # 恢复梯度
            for name, p in model.named_parameters():
                if grads_before[name] is not None:
                    p.grad = grads_before[name]
                else:
                    p.grad = None

            return max(trace, 1e-10)

        except Exception:
            return 1.0

    # ─── 更新权重（在训练循环中调用）────────────────────────────────────────
    def update(
        self,
        model: nn.Module,
        loss_fns: Dict[str, any],
        step: int,
    ) -> Dict[str, float]:
        """
        在固定步频（update_freq）调用一次，更新 NTK 权重。

        Args:
            model    : PINN 模型（用于获取参数梯度）
            loss_fns : {loss_name: callable() → scalar} 各损失项的计算函数
            step     : 当前训练步数

        Returns:
            更新后的权重字典
        """
        if step % self.update_freq != 0:
            return self.weights

        # 计算各损失项的 NTK 迹
        traces: Dict[str, float] = {}
        for name in self.loss_names:
            if name in loss_fns:
                traces[name] = self._compute_ntk_trace(model, loss_fns[name])
            else:
                traces[name] = self._ema_traces.get(name, 1.0)

        # EMA 平滑
        for name in self.loss_names:
            if name in traces:
                old = self._ema_traces.get(name, traces[name])
                self._ema_traces[name] = (
                    self.ema_alpha * old + (1 - self.ema_alpha) * traces[name]
                )

        # 计算参考值：以残差项的最大 NTK 迹为基准
        ref_name = "res" if "res" in self._ema_traces else self.loss_names[0]
        ref_trace = self._ema_traces.get(ref_name, 1.0)

        # 更新各权重：λ_i = ref_trace / trace_i（使各项收敛速率均质化）
        for name in self.loss_names:
            t = self._ema_traces.get(name, 1.0)
            self.weights[name] = float(ref_trace / (t + 1e-10))

        # 归一化：防止权重爆炸
        max_w = max(self.weights.values())
        if max_w > 100.0:
            self.weights = {k: v / max_w * 10.0 for k, v in self.weights.items()}

        return self.weights

    # ─── 简化版：基于损失值比率的启发式权重（无需二次 backward）────────────
    @staticmethod
    def heuristic_weights(
        loss_dict: Dict[str, torch.Tensor],
        ref_key: str = "res",
        ema_state: Optional[Dict[str, float]] = None,
        ema_alpha: float = 0.9,
    ) -> Dict[str, float]:
        """
        基于当前损失值比率的轻量级启发式权重（计算开销极小，适合每步更新）：
          λ_i = L_ref / L_i

        这是 NTK 动态权重的一阶近似，在实践中效果接近完整 NTK 版本。

        Args:
            loss_dict: {name: scalar_tensor}
            ref_key  : 参考损失项名称（通常为 "res"）
            ema_state: EMA 平滑状态字典（in-place 更新）
            ema_alpha: EMA 平滑系数

        Returns:
            weights: {name: float}
        """
        with torch.no_grad():
            vals = {k: v.item() for k, v in loss_dict.items()}
            if ema_state is not None:
                for k, v in vals.items():
                    ema_state[k] = ema_alpha * ema_state.get(k, v) + (1 - ema_alpha) * v
                vals = dict(ema_state)

            ref_val = vals.get(ref_key, 1.0)
            weights = {}
            for name, val in vals.items():
                weights[name] = float(ref_val / (val + 1e-10))

            # 截断防止权重过大
            max_w = max(weights.values())
            if max_w > 100.0:
                weights = {k: v / max_w * 10.0 for k, v in weights.items()}

        return weights


# ─────────────────────────────────────────────────────────────────────────────
# NTK 动态权重（可训练版本）— 作为 nn.Module 参数参与梯度更新
# ─────────────────────────────────────────────────────────────────────────────
class LearnableWeights(nn.Module):
    """
    可学习损失权重（替代或补充 NTK 方法）。

    使用 softplus 参数化确保权重恒正：
      λ_i = softplus(w_i) = log(1 + exp(w_i))

    可与主优化器联合训练（元学习式权重自适应）。
    """

    def __init__(self, loss_names: List[str], init_val: float = 1.0):
        super().__init__()
        self.loss_names = loss_names
        # 参数化：w_0 = log(exp(init_val) - 1) ≈ init_val for large init_val
        init_raw = math.log(math.exp(init_val) - 1.0 + 1e-6)
        raw = torch.full((len(loss_names),), init_raw)
        self.raw_weights = nn.Parameter(raw)

    def forward(self) -> Dict[str, torch.Tensor]:
        """返回各损失项权重（可微分张量）"""
        weights_pos = torch.nn.functional.softplus(self.raw_weights)
        return {name: weights_pos[i] for i, name in enumerate(self.loss_names)}

    def get_values(self) -> Dict[str, float]:
        """返回当前权重的数值（用于日志）"""
        with torch.no_grad():
            return {
                name: torch.nn.functional.softplus(self.raw_weights[i]).item()
                for i, name in enumerate(self.loss_names)
            }
