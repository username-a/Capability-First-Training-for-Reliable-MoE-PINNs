"""
Core MoE-PINN and baseline PINN definitions.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .experts import (
    DispersionExpert,
    ExpertNetwork,
    ShockExpert,
    SmoothExpert,
    apply_output_transform,
)
from .gating import GatingNetwork


class MoEPINN(nn.Module):
    def __init__(
        self,
        experts: List[ExpertNetwork],
        gating: GatingNetwork,
        sparsity_weight: float = 1e-3,
        balance_weight: float = 1e-2,
        routing_prior_strength: float = 1.0,
        use_shock_aware_routing: bool = True,
    ):
        super().__init__()
        assert len(experts) == gating.num_experts
        self.experts = nn.ModuleList(experts)
        self.gating = gating
        self.sparsity_weight = sparsity_weight
        self.balance_weight = balance_weight
        self.num_experts = len(experts)
        self.routing_prior_strength = routing_prior_strength
        self.use_shock_aware_routing = use_shock_aware_routing

    def _routing_prior(self, x: torch.Tensor, expert_preds: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        if not self.use_shock_aware_routing or self.num_experts < 3 or x.shape[-1] < 2:
            return None

        x_coord = x[:, 0]
        t_coord = x[:, 1]

        center_focus = torch.exp(-torch.square(x_coord / 0.18))
        ring_focus = torch.exp(-torch.square((torch.abs(x_coord) - 0.24) / 0.16))
        edge_focus = torch.sigmoid((torch.abs(x_coord) - 0.45) * 7.0)
        time_focus = torch.sigmoid((t_coord - 0.10) * 8.0)

        disagreement_focus = torch.zeros_like(center_focus)
        if expert_preds is not None:
            pred_scalar = expert_preds.detach().squeeze(-1)
            disagreement = pred_scalar.std(dim=1, unbiased=False)
            disagreement_mean = disagreement.mean()
            disagreement_std = disagreement.std(unbiased=False) + 1e-6
            disagreement_focus = torch.sigmoid(
                (disagreement - disagreement_mean) / (0.5 * disagreement_std + 1e-6)
            )

        shock = (0.10 + 0.90 * center_focus) * (0.35 + 0.65 * time_focus) * (
            0.55 + 0.45 * disagreement_focus
        )
        smooth = 0.15 + 0.75 * (1.0 - center_focus) + 0.45 * edge_focus + 0.10 * (1.0 - time_focus)
        transition = 0.15 + 0.85 * ring_focus * (0.35 + 0.65 * time_focus) * (
            0.60 + 0.40 * disagreement_focus
        )

        prior = torch.stack([shock, smooth, transition], dim=1).clamp_min(1e-4)
        return prior.pow(self.routing_prior_strength)

    def compute_gate_weights(
        self,
        x: torch.Tensor,
        expert_preds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        base_weights = self.gating(x)
        if expert_preds is None and self.use_shock_aware_routing and self.num_experts >= 3:
            expert_preds = torch.stack([expert(x) for expert in self.experts], dim=1)
        prior = self._routing_prior(x, expert_preds=expert_preds)
        if prior is None:
            return base_weights
        weights = base_weights * prior
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def load_balance_loss(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.compute_gate_weights(x)
        mean_load = weights.mean(dim=0)
        target = torch.full_like(mean_load, 1.0 / self.num_experts)
        return ((mean_load - target) ** 2).mean()

    def load_balance_stats(self, x: torch.Tensor) -> Dict[str, object]:
        with torch.no_grad():
            weights = self.compute_gate_weights(x)
            routing = weights.argmax(dim=-1)
            counts = torch.bincount(routing, minlength=self.num_experts)
            entropy = -(weights * (weights + 1e-10).log()).sum(dim=-1).mean().item()
            return {
                "mean_entropy": entropy,
                "expert_load_frac": (counts.float() / counts.sum().clamp_min(1)).tolist(),
                "max_gate_weight": weights.max(dim=-1).values.mean().item(),
            }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expert_preds = torch.stack([expert(x) for expert in self.experts], dim=1)
        gate_weights = self.compute_gate_weights(x, expert_preds=expert_preds)
        return torch.einsum("ne,neo->no", gate_weights, expert_preds)

    def forward_with_sparsity(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pred = self.forward(x)
        sparse_loss = self.gating.sparsity_loss(x) * self.sparsity_weight
        return pred, sparse_loss

    def get_gate_weights(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            expert_preds = torch.stack([expert(x) for expert in self.experts], dim=1)
            return self.compute_gate_weights(x, expert_preds=expert_preds)

    def get_expert_predictions(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.stack([expert(x) for expert in self.experts], dim=1)

    def count_parameters(self) -> Dict[str, int]:
        gating_params = sum(p.numel() for p in self.gating.parameters())
        expert_params = [sum(p.numel() for p in expert.parameters()) for expert in self.experts]
        return {
            "gating": gating_params,
            "experts": expert_params,
            "experts_total": sum(expert_params),
            "total": gating_params + sum(expert_params),
        }


class ResidualMoEPINN(MoEPINN):
    def __init__(
        self,
        base_model: nn.Module,
        experts: List[ExpertNetwork],
        gating: GatingNetwork,
        correction_scale: float = 0.5,
        **kwargs,
    ):
        super().__init__(experts=experts, gating=gating, **kwargs)
        self.base_model = base_model
        self.correction_scale = correction_scale

    def get_expert_corrections(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([expert(x) for expert in self.experts], dim=1)

    def compute_gate_weights(
        self,
        x: torch.Tensor,
        expert_preds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if expert_preds is None:
            expert_preds = self.get_expert_predictions(x)
        return super().compute_gate_weights(x, expert_preds=expert_preds)

    def get_expert_predictions(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            base = self.base_model(x).unsqueeze(1)
            corrections = self.get_expert_corrections(x)
            return base + self.correction_scale * corrections

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_model(x)
        corrections = self.get_expert_corrections(x)
        branch_preds = base.unsqueeze(1) + self.correction_scale * corrections
        gate_weights = self.compute_gate_weights(x, expert_preds=branch_preds)
        mixed_correction = torch.einsum("ne,neo->no", gate_weights, corrections)
        return base + self.correction_scale * mixed_correction

    def get_gate_weights(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            branch_preds = self.get_expert_predictions(x)
            return self.compute_gate_weights(x, expert_preds=branch_preds)

    def count_parameters(self) -> Dict[str, int]:
        counts = super().count_parameters()
        base_params = sum(p.numel() for p in self.base_model.parameters())
        counts["base"] = base_params
        counts["total"] += base_params
        return counts


class VanillaPINN(nn.Module):
    def __init__(
        self,
        in_dim: int = 2,
        out_dim: int = 1,
        hidden: int = 64,
        depth: int = 5,
        activation: str = "tanh",
        output_transform: Optional[str] = None,
    ):
        super().__init__()
        self.output_transform = output_transform

        from .experts import LearnableSin

        act_map = {
            "tanh": nn.Tanh,
            "sin": LearnableSin,
            "swish": nn.SiLU,
            "relu": nn.ReLU,
            "silu": nn.SiLU,
        }
        act_factory = act_map.get(activation, nn.Tanh)

        layers = [nn.Linear(in_dim, hidden), act_factory()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), act_factory()]
        layers.append(nn.Linear(hidden, out_dim))
        self.network = nn.Sequential(*layers)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.network(x)
        return apply_output_transform(self.output_transform, x, raw)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_burgers_moe(
    num_experts: int = 3,
    sparsity_weight: float = 1e-3,
    balance_weight: float = 2e-2,
    use_fourier: bool = True,
    gate_temperature: float = 0.8,
) -> MoEPINN:
    base_model = VanillaPINN(
        in_dim=2,
        out_dim=1,
        hidden=64,
        depth=5,
        activation="tanh",
        output_transform="burgers_hard_bc",
    )
    experts: List[ExpertNetwork] = [
        ShockExpert(
            in_dim=2,
            out_dim=1,
            hidden=96,
            depth=5,
            use_fourier_embed=use_fourier,
            output_transform="burgers_hard_bc",
        ),
        SmoothExpert(
            in_dim=2,
            out_dim=1,
            hidden=48,
            depth=3,
            output_transform="burgers_hard_bc",
        ),
        DispersionExpert(
            in_dim=2,
            out_dim=1,
            hidden=48,
            depth=4,
            output_transform="burgers_hard_bc",
        ),
    ]
    if num_experts > 3:
        for _ in range(num_experts - 3):
            experts.append(
                ExpertNetwork(
                    in_dim=2,
                    out_dim=1,
                    hidden=64,
                    depth=4,
                    output_transform="burgers_hard_bc",
                )
            )

    gating = GatingNetwork(
        in_dim=2,
        num_experts=len(experts),
        hidden=32,
        depth=3,
        sparsity_p=0.5,
        temperature=gate_temperature,
        init_gain=0.5,
    )
    return ResidualMoEPINN(
        base_model=base_model,
        experts=experts,
        gating=gating,
        correction_scale=0.35,
        sparsity_weight=sparsity_weight,
        balance_weight=balance_weight,
        routing_prior_strength=1.4,
        use_shock_aware_routing=True,
    )


def build_kdv_moe(
    num_experts: int = 3,
    sparsity_weight: float = 2e-3,
    balance_weight: float = 5e-3,
    gate_temperature: float = 0.7,
) -> MoEPINN:
    experts: List[ExpertNetwork] = [
        DispersionExpert(in_dim=2, out_dim=1, hidden=96, depth=5),
        SmoothExpert(in_dim=2, out_dim=1, hidden=64, depth=3),
        ShockExpert(in_dim=2, out_dim=1, hidden=96, depth=5, use_fourier_embed=False),
    ]
    gating = GatingNetwork(
        in_dim=2,
        num_experts=len(experts),
        hidden=32,
        depth=3,
        sparsity_p=0.5,
        temperature=gate_temperature,
        init_gain=0.5,
    )
    return MoEPINN(
        experts,
        gating,
        sparsity_weight=sparsity_weight,
        balance_weight=balance_weight,
        routing_prior_strength=0.0,
        use_shock_aware_routing=False,
    )
