"""Augmented PINN (APINN) baseline for the 2-D Burgers benchmark.

This is a clean K-way generalisation of the two-subnetwork SXPINN/APINN
architecture released by Hu et al.: a shared feature extractor, several
subnetwork heads, and a trainable partition-of-unity gate.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from Burger2D.core.models import apply_output_transform


def _make_mlp(in_dim: int, out_dim: int, hidden: int, depth: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = in_dim
    for _ in range(depth):
        layers.extend([nn.Linear(current, hidden), nn.Tanh()])
        current = hidden
    layers.append(nn.Linear(current, out_dim))
    network = nn.Sequential(*layers)
    for module in network.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)
    return network


class APINN2D(nn.Module):
    """Shared-trunk APINN with a soft partition-of-unity gate.

    The output transform is applied to every branch. Because the gate weights
    sum to one, mixing transformed branches is algebraically identical to
    transforming the mixed raw output.
    """

    def __init__(
        self,
        *,
        in_dim: int = 3,
        n_subnets: int = 4,
        shared_width: int = 64,
        shared_depth: int = 3,
        subnet_width: int = 64,
        subnet_depth: int = 3,
        gate_width: int = 32,
        gate_depth: int = 2,
        gate_temperature: float = 1.0,
        binary_scalar_gate: bool = False,
        output_transform: Optional[str] = "burgers2d_hard_icbc",
    ) -> None:
        super().__init__()
        if n_subnets < 2:
            raise ValueError("APINN requires at least two subnetworks.")
        if gate_temperature <= 0:
            raise ValueError("gate_temperature must be positive.")

        self.n_subnets = n_subnets
        self.num_experts = n_subnets
        self.expert_names = [f"subnet_{idx + 1}" for idx in range(n_subnets)]
        self.output_transform = output_transform
        self.gate_temperature = gate_temperature
        self.binary_scalar_gate = bool(binary_scalar_gate and n_subnets == 2)

        self.shared = _make_mlp(in_dim, shared_width, shared_width, shared_depth)
        self.subnets = nn.ModuleList(
            _make_mlp(shared_width, 1, subnet_width, subnet_depth)
            for _ in range(n_subnets)
        )
        gate_outputs = 1 if self.binary_scalar_gate else n_subnets
        self.gating = _make_mlp(in_dim, gate_outputs, gate_width, gate_depth)

    def compute_gate_weights(
        self,
        coords: torch.Tensor,
        expert_preds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del expert_preds
        logits = self.gating(coords) / self.gate_temperature
        if self.binary_scalar_gate:
            first = torch.sigmoid(logits)
            return torch.cat([first, 1.0 - first], dim=1)
        return torch.softmax(logits, dim=-1)

    def _raw_subnet_predictions(self, coords: torch.Tensor) -> torch.Tensor:
        features = torch.tanh(self.shared(coords))
        return torch.cat([subnet(features) for subnet in self.subnets], dim=1)

    def get_subnet_predictions(self, coords: torch.Tensor) -> torch.Tensor:
        raw = self._raw_subnet_predictions(coords)
        return apply_output_transform(self.output_transform, coords, raw)

    def get_expert_predictions(self, coords: torch.Tensor) -> torch.Tensor:
        return self.get_subnet_predictions(coords).unsqueeze(-1)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        branches = self.get_subnet_predictions(coords)
        weights = self.compute_gate_weights(coords)
        return (weights * branches).sum(dim=1, keepdim=True)

    def load_balance_loss(self, coords: torch.Tensor) -> torch.Tensor:
        mean_load = self.compute_gate_weights(coords).mean(dim=0)
        target = torch.full_like(mean_load, 1.0 / self.n_subnets)
        return (mean_load - target).square().mean()

    def count_parameters(self) -> dict[str, object]:
        shared = sum(param.numel() for param in self.shared.parameters())
        gate = sum(param.numel() for param in self.gating.parameters())
        subnet_counts = [sum(param.numel() for param in subnet.parameters()) for subnet in self.subnets]
        return {
            "shared": shared,
            "gating": gate,
            "subnets": subnet_counts,
            "subnets_total": sum(subnet_counts),
            "total": shared + gate + sum(subnet_counts),
        }
