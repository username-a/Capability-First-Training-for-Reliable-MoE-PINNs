"""
Expert definitions for Burger2D MoE-PINN.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from Burger2D.equations.burgers2d import initial_profile_torch


def apply_output_transform(
    output_transform: Optional[str],
    coords: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    if output_transform is None:
        return values
    if output_transform not in {"burgers2d_hard_icbc", "burgers2d_residual_icbc"}:
        raise ValueError(f"Unknown output transform: {output_transform}")

    x = coords[:, 0:1]
    y = coords[:, 1:2]
    t = coords[:, 2:3]
    boundary_factor = (1.0 - x.square()) * (1.0 - y.square())
    if output_transform == "burgers2d_residual_icbc":
        return t * boundary_factor * values
    initial = initial_profile_torch(x, y)
    return (1.0 - t) * initial + t * boundary_factor * values


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class LearnableSin(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w * torch.sin(x)


def _init_linear_layers(module: nn.Module) -> None:
    for submodule in module.modules():
        if isinstance(submodule, nn.Linear):
            nn.init.xavier_normal_(submodule.weight)
            nn.init.zeros_(submodule.bias)


class ExpertNetwork(nn.Module):
    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 1,
        hidden: int = 64,
        depth: int = 4,
        activation_factory=None,
        output_transform: Optional[str] = None,
    ):
        super().__init__()
        self.input_dim = in_dim
        self.out_dim = out_dim
        self.output_transform = output_transform

        if activation_factory is None:
            activation_factory = nn.Tanh
        if isinstance(activation_factory, nn.Module):
            instance = activation_factory
            activation_factory = lambda: type(instance)()

        layers = [nn.Linear(in_dim, hidden), activation_factory()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), activation_factory()])
        layers.append(nn.Linear(hidden, out_dim))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        _init_linear_layers(self)

    def encode(self, coords: torch.Tensor) -> torch.Tensor:
        return coords

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        feats = self.encode(coords)
        raw = self.network(feats)
        return apply_output_transform(self.output_transform, coords, raw)


class SmoothExpert2D(ExpertNetwork):
    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 1,
        hidden: int = 48,
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


class IsoShockExpert2D(ExpertNetwork):
    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 1,
        hidden: int = 96,
        depth: int = 5,
        embed_dim: int = 24,
        embed_scale: float = 8.0,
        output_transform: Optional[str] = None,
    ):
        self.use_fourier = True
        self.embed_dim = embed_dim
        self.embed_scale = embed_scale
        super().__init__(
            in_dim=embed_dim * 2,
            out_dim=out_dim,
            hidden=hidden,
            depth=depth,
            activation_factory=Swish,
            output_transform=output_transform,
        )
        fourier_matrix = torch.randn(in_dim, embed_dim) * embed_scale
        self.register_buffer("fourier_B", fourier_matrix)

    def encode(self, coords: torch.Tensor) -> torch.Tensor:
        proj = coords @ self.fourier_B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class DirectionalShockExpert2D(ExpertNetwork):
    class DirectionalEncoder(nn.Module):
        def __init__(
            self,
            num_directions: int = 8,
            tangent_frequencies: tuple[float, ...] = (1.0, 2.0, 4.0),
            normal_frequencies: tuple[float, ...] = (1.0, 2.0),
            selector_hidden: int = 48,
            direction_temperature: float = 0.80,
            normal_decay: float = 10.0,
            shock_steepness: float = 4.0,
        ):
            super().__init__()
            self.num_directions = num_directions
            self.legacy_dim = 10
            self.direction_temperature = direction_temperature
            self.normal_decay = normal_decay
            self.shock_steepness = shock_steepness

            angles = torch.arange(num_directions, dtype=torch.float32) * (np.pi / num_directions)
            direction_bank = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
            normal_bank = torch.stack([-torch.sin(angles), torch.cos(angles)], dim=1)
            self.register_buffer("direction_bank", direction_bank)
            self.register_buffer("normal_bank", normal_bank)
            self.register_buffer("tangent_frequencies", torch.tensor(tangent_frequencies, dtype=torch.float32))
            self.register_buffer("normal_frequencies", torch.tensor(normal_frequencies, dtype=torch.float32))

            self.selector = nn.Sequential(
                nn.Linear(12, selector_hidden),
                Swish(),
                nn.Linear(selector_hidden, num_directions),
            )
            _init_linear_layers(self.selector)

            directional_dim = 9 + 2 * len(tangent_frequencies) + 2 * len(normal_frequencies)
            local_dim = 8
            self.output_dim = self.legacy_dim + local_dim + directional_dim + num_directions

        def legacy_features(self, coords: torch.Tensor) -> torch.Tensor:
            x = coords[:, 0:1]
            y = coords[:, 1:2]
            t = coords[:, 2:3]
            diag_p = x + y
            diag_m = x - y
            return torch.cat(
                [
                    x,
                    y,
                    t,
                    diag_p,
                    diag_m,
                    x * y,
                    torch.sin(np.pi * diag_p),
                    torch.cos(np.pi * diag_p),
                    torch.sin(np.pi * diag_m),
                    torch.cos(np.pi * diag_m),
                ],
                dim=-1,
            )

        def forward(self, coords: torch.Tensor) -> torch.Tensor:
            x = coords[:, 0:1]
            y = coords[:, 1:2]
            t = coords[:, 2:3]
            diag_p = x + y
            diag_m = x - y
            r2 = x.square() + y.square()
            xy = coords[:, :2]
            legacy_features = self.legacy_features(coords)

            tangent = xy @ self.direction_bank.t()
            normal = xy @ self.normal_bank.t()
            tangent_exp = tangent.unsqueeze(-1)
            normal_exp = normal.unsqueeze(-1)
            t_exp = t.unsqueeze(1)

            tangent_phase = np.pi * tangent_exp * self.tangent_frequencies.view(1, 1, -1)
            normal_phase = np.pi * normal_exp * self.normal_frequencies.view(1, 1, -1)
            tangent_periodic = torch.cat([torch.sin(tangent_phase), torch.cos(tangent_phase)], dim=-1)
            normal_periodic = torch.cat([torch.sin(normal_phase), torch.cos(normal_phase)], dim=-1)

            shock_core = torch.exp(-self.normal_decay * normal_exp.square())
            shock_sign = torch.tanh(self.shock_steepness * normal_exp)
            directional_features = torch.cat(
                [
                    tangent_exp,
                    normal_exp,
                    tangent_exp * normal_exp,
                    normal_exp.square(),
                    normal_exp.abs(),
                    shock_core,
                    shock_sign,
                    t_exp * shock_core,
                    t_exp * shock_sign,
                    tangent_periodic,
                    normal_periodic,
                ],
                dim=-1,
            )

            selector_inputs = torch.cat(
                [
                    x,
                    y,
                    t,
                    diag_p,
                    diag_m,
                    r2,
                    x * y,
                    x.square() - y.square(),
                    torch.sin(np.pi * diag_p),
                    torch.cos(np.pi * diag_p),
                    torch.sin(np.pi * diag_m),
                    torch.cos(np.pi * diag_m),
                ],
                dim=-1,
            )
            direction_logits = self.selector(selector_inputs)
            direction_weights = torch.softmax(direction_logits / self.direction_temperature, dim=-1)
            fused_directional = torch.sum(
                directional_features * direction_weights.unsqueeze(-1),
                dim=1,
            )

            local_features = torch.cat(
                [
                    x,
                    y,
                    t,
                    diag_p,
                    diag_m,
                    x * y,
                    r2,
                    x.square() - y.square(),
                ],
                dim=-1,
            )
            return torch.cat([legacy_features, local_features, fused_directional, direction_weights], dim=-1)

    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 1,
        hidden: int = 96,
        depth: int = 5,
        num_directions: int = 8,
        output_transform: Optional[str] = None,
    ):
        del in_dim
        nn.Module.__init__(self)
        self.out_dim = out_dim
        self.output_transform = output_transform
        self.encoder = self.DirectionalEncoder(
            num_directions=num_directions,
            selector_hidden=max(32, hidden // 2),
        )
        self.input_dim = self.encoder.output_dim
        self.legacy_dim = self.encoder.legacy_dim

        layers = [nn.Linear(self.input_dim, hidden), Swish()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), Swish()])
        layers.append(nn.Linear(hidden, out_dim))
        self.network = nn.Sequential(*layers)
        shortcut_hidden = max(24, hidden // 2)
        self.legacy_head = nn.Sequential(
            nn.Linear(self.legacy_dim, shortcut_hidden),
            Swish(),
            nn.Linear(shortcut_hidden, out_dim),
        )
        self.legacy_scale = nn.Parameter(torch.tensor(0.20))
        _init_linear_layers(self.network)
        _init_linear_layers(self.legacy_head)

    def encode(self, coords: torch.Tensor) -> torch.Tensor:
        return self.encoder(coords)

    def encode_legacy(self, coords: torch.Tensor) -> torch.Tensor:
        return self.encoder.legacy_features(coords)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        feats = self.encode(coords)
        raw_main = self.network(feats)
        raw_shortcut = self.legacy_head(self.encode_legacy(coords))
        raw = raw_main + self.legacy_scale * raw_shortcut
        return apply_output_transform(self.output_transform, coords, raw)


class LegacyDirectionalShockExpert2D(ExpertNetwork):
    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 1,
        hidden: int = 96,
        depth: int = 5,
        output_transform: Optional[str] = None,
    ):
        del in_dim
        super().__init__(
            in_dim=10,
            out_dim=out_dim,
            hidden=hidden,
            depth=depth,
            activation_factory=Swish,
            output_transform=output_transform,
        )

    def encode(self, coords: torch.Tensor) -> torch.Tensor:
        x = coords[:, 0:1]
        y = coords[:, 1:2]
        t = coords[:, 2:3]
        diag_p = x + y
        diag_m = x - y
        return torch.cat(
            [
                x,
                y,
                t,
                diag_p,
                diag_m,
                x * y,
                torch.sin(np.pi * diag_p),
                torch.cos(np.pi * diag_p),
                torch.sin(np.pi * diag_m),
                torch.cos(np.pi * diag_m),
            ],
            dim=-1,
        )


class WavePacketExpert2D(ExpertNetwork):
    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 1,
        hidden: int = 72,
        depth: int = 4,
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


class VortexExpert2D(ExpertNetwork):
    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 1,
        hidden: int = 72,
        depth: int = 4,
        output_transform: Optional[str] = None,
    ):
        super().__init__(
            in_dim=9,
            out_dim=out_dim,
            hidden=hidden,
            depth=depth,
            activation_factory=nn.Tanh,
            output_transform=output_transform,
        )

    def encode(self, coords: torch.Tensor) -> torch.Tensor:
        x = coords[:, 0:1]
        y = coords[:, 1:2]
        t = coords[:, 2:3]
        r2 = x.square() + y.square()
        r = torch.sqrt(r2 + 1e-8)
        return torch.cat(
            [
                x,
                y,
                t,
                r,
                r2,
                x * y,
                torch.sin(np.pi * r),
                torch.cos(np.pi * r),
                torch.sin(np.pi * (x + y)),
            ],
            dim=-1,
        )
