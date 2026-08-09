"""
Model definitions for Burger2D.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from Burger2D.equations.burgers2d import initial_profile_torch


def apply_output_transform(name: Optional[str], xyt: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
    if name is None:
        return raw
    if name != "burgers2d_hard_icbc":
        raise ValueError(f"Unknown output transform: {name}")

    x = xyt[:, 0:1]
    y = xyt[:, 1:2]
    t = xyt[:, 2:3]
    boundary_factor = (1.0 - x.square()) * (1.0 - y.square())
    initial = initial_profile_torch(x, y)
    return (1.0 - t) * initial + t * boundary_factor * raw


class VanillaPINN(nn.Module):
    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 1,
        hidden: int = 96,
        depth: int = 5,
        activation: str = "tanh",
        output_transform: Optional[str] = None,
    ):
        super().__init__()
        self.output_transform = output_transform

        act_map = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "swish": nn.SiLU,
            "silu": nn.SiLU,
        }
        act_factory = act_map.get(activation, nn.Tanh)

        layers = [nn.Linear(in_dim, hidden), act_factory()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), act_factory()])
        layers.append(nn.Linear(hidden, out_dim))
        self.network = nn.Sequential(*layers)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        raw = self.network(xyt)
        return apply_output_transform(self.output_transform, xyt, raw)

    def count_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters())
