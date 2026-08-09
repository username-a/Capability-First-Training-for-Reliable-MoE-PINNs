"""
Expert definitions for MoE-PINN.
"""

from typing import Optional

import torch
import torch.nn as nn


def apply_output_transform(
    output_transform: Optional[str],
    coords: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """Apply lightweight equation-specific constraints to the raw network output."""
    if output_transform == "burgers_hard_bc":
        x = coords[:, 0:1]
        return (1.0 - x * x) * values
    return values


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class LearnableSin(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w * torch.sin(x)


class ExpertNetwork(nn.Module):
    def __init__(
        self,
        in_dim: int = 2,
        out_dim: int = 1,
        hidden: int = 64,
        depth: int = 4,
        activation_factory=None,
        use_residual: bool = False,
        output_transform: Optional[str] = None,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.output_transform = output_transform

        if activation_factory is None:
            activation_factory = nn.Tanh
        if isinstance(activation_factory, nn.Module):
            instance = activation_factory
            activation_factory = lambda: type(instance)()

        layers = [nn.Linear(in_dim, hidden), activation_factory()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), activation_factory()]
        layers.append(nn.Linear(hidden, out_dim))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.network(x)
        return apply_output_transform(self.output_transform, x, raw)


class ShockExpert(ExpertNetwork):
    def __init__(
        self,
        in_dim: int = 2,
        out_dim: int = 1,
        hidden: int = 128,
        depth: int = 6,
        use_fourier_embed: bool = False,
        embed_scale: float = 10.0,
        output_transform: Optional[str] = None,
    ):
        self.use_fourier_embed = use_fourier_embed
        self.embed_scale = embed_scale

        if use_fourier_embed:
            embed_dim = 32
            actual_in = embed_dim * 2
        else:
            embed_dim = 0
            actual_in = in_dim

        super().__init__(
            in_dim=actual_in,
            out_dim=out_dim,
            hidden=hidden,
            depth=depth,
            activation_factory=Swish,
            output_transform=output_transform,
        )

        if use_fourier_embed:
            fourier_matrix = torch.randn(in_dim, embed_dim) * embed_scale
            self.register_buffer("fourier_B", fourier_matrix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coords = x
        if self.use_fourier_embed:
            proj = x @ self.fourier_B
            x = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        raw = self.network(x)
        return apply_output_transform(self.output_transform, coords, raw)


class SmoothExpert(ExpertNetwork):
    def __init__(
        self,
        in_dim: int = 2,
        out_dim: int = 1,
        hidden: int = 64,
        depth: int = 3,
        output_transform: Optional[str] = None,
    ):
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden=hidden,
            depth=depth,
            activation_factory=nn.Tanh,
            output_transform=output_transform,
        )


class DispersionExpert(ExpertNetwork):
    def __init__(
        self,
        in_dim: int = 2,
        out_dim: int = 1,
        hidden: int = 96,
        depth: int = 5,
        output_transform: Optional[str] = None,
    ):
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden=hidden,
            depth=depth,
            activation_factory=LearnableSin,
            output_transform=output_transform,
        )


def build_expert(
    expert_type: str,
    in_dim: int,
    out_dim: int,
    **kwargs,
) -> ExpertNetwork:
    expert_map = {
        "shock": ShockExpert,
        "smooth": SmoothExpert,
        "dispersion": DispersionExpert,
        "generic": ExpertNetwork,
    }
    if expert_type not in expert_map:
        raise ValueError(
            f"Unknown expert_type: {expert_type}. Available: {list(expert_map.keys())}"
        )
    return expert_map[expert_type](in_dim=in_dim, out_dim=out_dim, **kwargs)
