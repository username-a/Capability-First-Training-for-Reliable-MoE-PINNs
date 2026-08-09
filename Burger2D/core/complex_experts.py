"""
Experimental complex-inspired experts for Burger2D.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from Burger2D.core.experts import Swish, _init_linear_layers, apply_output_transform


class ComplexFrameDirectionalShockExpert2D(nn.Module):
    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 1,
        hidden: int = 96,
        depth: int = 5,
        harmonic_order: int = 3,
        spectral_frequencies: tuple[float, ...] = (1.0, 2.0, 4.0),
        output_transform: str | None = None,
    ):
        super().__init__()
        del in_dim
        self.out_dim = out_dim
        self.output_transform = output_transform
        self.harmonic_order = harmonic_order
        self.spectral_frequencies = spectral_frequencies

        self.frame_predictor = nn.Sequential(
            nn.Linear(12, hidden // 2),
            Swish(),
            nn.Linear(hidden // 2, hidden // 2),
            Swish(),
            nn.Linear(hidden // 2, 3),
        )
        _init_linear_layers(self.frame_predictor)

        input_dim = 20 + 4 * harmonic_order + 4 * len(spectral_frequencies)
        self.input_dim = input_dim
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden), Swish()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), Swish()])
        layers.append(nn.Linear(hidden, out_dim))
        self.network = nn.Sequential(*layers)
        _init_linear_layers(self.network)

    def _frame_inputs(self, coords: torch.Tensor) -> torch.Tensor:
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
                x.square() + y.square(),
                x.square() - y.square(),
                torch.sin(np.pi * diag_p),
                torch.cos(np.pi * diag_p),
                torch.sin(np.pi * diag_m),
                torch.cos(np.pi * diag_m),
            ],
            dim=-1,
        )

    def encode(self, coords: torch.Tensor) -> torch.Tensor:
        x = coords[:, 0]
        y = coords[:, 1]
        t = coords[:, 2:3]
        frame_inputs = self._frame_inputs(coords)
        frame_raw = self.frame_predictor(frame_inputs)
        theta = np.pi * torch.tanh(frame_raw[:, 0])
        shock_scale = 4.0 + 6.0 * torch.sigmoid(frame_raw[:, 1])
        tangent_bias = torch.tanh(frame_raw[:, 2])

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        tangent = (x * cos_theta + y * sin_theta).unsqueeze(-1)
        normal = (-x * sin_theta + y * cos_theta).unsqueeze(-1)
        radius = torch.sqrt(tangent.square() + normal.square() + 1e-8)
        unit_real = tangent / radius
        unit_imag = normal / radius

        envelope = torch.exp(-shock_scale.unsqueeze(-1) * normal.square())
        signed_normal = torch.tanh((shock_scale.unsqueeze(-1) + 1.0) * normal)
        aligned_tangent = tangent + tangent_bias.unsqueeze(-1) * normal

        features = [
            coords,
            tangent,
            normal,
            radius,
            unit_real,
            unit_imag,
            envelope,
            signed_normal,
            t * envelope,
            t * signed_normal,
            aligned_tangent,
            aligned_tangent * normal,
            normal.square(),
            radius.square(),
            cos_theta.unsqueeze(-1),
            sin_theta.unsqueeze(-1),
            shock_scale.unsqueeze(-1) / 10.0,
            tangent_bias.unsqueeze(-1),
        ]

        harm_real = unit_real
        harm_imag = unit_imag
        radius_safe = radius.squeeze(-1)
        for order in range(1, self.harmonic_order + 1):
            if order > 1:
                next_real = harm_real * unit_real - harm_imag * unit_imag
                next_imag = harm_real * unit_imag + harm_imag * unit_real
                harm_real, harm_imag = next_real, next_imag
            features.append(harm_real)
            features.append(harm_imag)
            features.append((radius_safe.pow(order)).unsqueeze(-1))
            features.append((radius_safe.pow(order) * harm_imag.squeeze(-1).abs()).unsqueeze(-1))

        for freq in self.spectral_frequencies:
            tangent_phase = np.pi * freq * tangent
            normal_phase = np.pi * freq * normal
            features.append(torch.cos(tangent_phase))
            features.append(torch.sin(tangent_phase))
            features.append(torch.cos(normal_phase) * envelope)
            features.append(torch.sin(normal_phase) * envelope)

        return torch.cat(features, dim=-1)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        raw = self.network(self.encode(coords))
        return apply_output_transform(self.output_transform, coords, raw)
