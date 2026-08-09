"""
Attribute-oriented expert definitions for Burger2D MoE-PINN.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from Burger2D.core.experts import (
    DirectionalShockExpert2D,
    ExpertNetwork,
    LearnableSin,
    SmoothExpert2D,
    Swish,
    _init_linear_layers,
    apply_output_transform,
)


class SmoothBackgroundExpert2D(SmoothExpert2D):
    """Semantic alias for the smooth/background capability axis."""


class NormalGradientExpert2D(ExpertNetwork):
    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 1,
        hidden: int = 88,
        depth: int = 5,
        output_transform: Optional[str] = None,
    ):
        del in_dim
        super().__init__(
            in_dim=20,
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
        diag_p = (x + y) / math.sqrt(2.0)
        diag_m = (x - y) / math.sqrt(2.0)
        r2 = x.square() + y.square()
        r = torch.sqrt(r2 + 1e-8)
        projections = torch.cat([x, y, diag_p, diag_m], dim=-1)
        envelope = torch.exp(-6.0 * projections.square())
        signed = torch.tanh(4.0 * projections)
        return torch.cat(
            [
                x,
                y,
                t,
                diag_p,
                diag_m,
                r,
                r2,
                (x * y).abs(),
                projections.abs(),
                envelope,
                signed,
            ],
            dim=-1,
        )


class _SelectorMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            Swish(),
            nn.Linear(hidden, hidden),
            Swish(),
            nn.Linear(hidden, out_dim),
        )
        _init_linear_layers(self.net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MixedNormalGradientExpert2D(nn.Module):
    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 1,
        hidden: int = 88,
        depth: int = 5,
        output_transform: Optional[str] = None,
    ):
        super().__init__()
        del in_dim
        self.out_dim = out_dim
        self.output_transform = output_transform
        self.selector_in_dim = 12
        self.axial_dim = 17
        self.diagonal_dim = 18
        self.radial_dim = 15
        self.base_dim = 12

        self.selector = _SelectorMLP(self.selector_in_dim, max(24, hidden // 2), 3)
        self.axial_net = self._make_branch(self.axial_dim, hidden, depth)
        self.diagonal_net = self._make_branch(self.diagonal_dim, hidden, depth)
        self.radial_net = self._make_branch(self.radial_dim, hidden, depth)
        self.shortcut = nn.Sequential(
            nn.Linear(self.base_dim, max(24, hidden // 2)),
            Swish(),
            nn.Linear(max(24, hidden // 2), out_dim),
        )
        self.shortcut_scale = nn.Parameter(torch.tensor(0.20))
        _init_linear_layers(self.shortcut)

    def _make_branch(self, in_dim: int, hidden: int, depth: int) -> nn.Sequential:
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), Swish()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), Swish()])
        layers.append(nn.Linear(hidden, self.out_dim))
        branch = nn.Sequential(*layers)
        _init_linear_layers(branch)
        return branch

    def _core_features(self, coords: torch.Tensor) -> dict[str, torch.Tensor]:
        x = coords[:, 0:1]
        y = coords[:, 1:2]
        t = coords[:, 2:3]
        diag_p = (x + y) / math.sqrt(2.0)
        diag_m = (x - y) / math.sqrt(2.0)
        r2 = x.square() + y.square()
        r = torch.sqrt(r2 + 1e-8)
        xy = x * y
        return {
            "x": x,
            "y": y,
            "t": t,
            "diag_p": diag_p,
            "diag_m": diag_m,
            "r": r,
            "r2": r2,
            "xy": xy,
        }

    def _selector_features(self, core: dict[str, torch.Tensor]) -> torch.Tensor:
        x = core["x"]
        y = core["y"]
        t = core["t"]
        diag_p = core["diag_p"]
        diag_m = core["diag_m"]
        r2 = core["r2"]
        return torch.cat(
            [
                x,
                y,
                t,
                diag_p,
                diag_m,
                r2,
                x.square() - y.square(),
                core["xy"],
                torch.sin(np.pi * diag_p),
                torch.cos(np.pi * diag_p),
                torch.sin(np.pi * diag_m),
                torch.cos(np.pi * diag_m),
            ],
            dim=-1,
        )

    def _axial_features(self, core: dict[str, torch.Tensor]) -> torch.Tensor:
        x = core["x"]
        y = core["y"]
        t = core["t"]
        axial = torch.cat([x, y], dim=-1)
        envelope = torch.exp(-6.0 * axial.square())
        signed = torch.tanh(4.0 * axial)
        return torch.cat(
            [
                x,
                y,
                t,
                core["r"],
                core["r2"],
                axial.abs(),
                envelope,
                signed,
                t * envelope,
                t * signed,
                core["xy"].abs(),
                x.square() - y.square(),
            ],
            dim=-1,
        )

    def _diagonal_features(self, core: dict[str, torch.Tensor]) -> torch.Tensor:
        diag_p = core["diag_p"]
        diag_m = core["diag_m"]
        t = core["t"]
        diagonal = torch.cat([diag_p, diag_m], dim=-1)
        envelope = torch.exp(-5.5 * diagonal.square())
        signed = torch.tanh(3.5 * diagonal)
        return torch.cat(
            [
                core["x"],
                core["y"],
                t,
                core["r"],
                diagonal,
                diagonal.abs(),
                envelope,
                signed,
                t * envelope,
                t * signed,
                core["xy"],
                core["x"].square() - core["y"].square(),
            ],
            dim=-1,
        )

    def _radial_features(self, core: dict[str, torch.Tensor]) -> torch.Tensor:
        r = core["r"]
        t = core["t"]
        radial_env = torch.exp(-4.0 * r.square())
        radial_phase = np.pi * r
        return torch.cat(
            [
                core["x"],
                core["y"],
                t,
                r,
                core["r2"],
                radial_env,
                torch.tanh(3.0 * r),
                torch.sin(radial_phase),
                torch.cos(radial_phase),
                t * torch.sin(radial_phase),
                t * torch.cos(radial_phase),
                torch.sin(2.0 * radial_phase),
                torch.cos(2.0 * radial_phase),
                core["diag_p"].abs(),
                core["diag_m"].abs(),
            ],
            dim=-1,
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        core = self._core_features(coords)
        selector_logits = self.selector(self._selector_features(core))
        selector_weights = torch.softmax(selector_logits / 0.75, dim=-1)
        branch_outputs = torch.stack(
            [
                self.axial_net(self._axial_features(core)),
                self.diagonal_net(self._diagonal_features(core)),
                self.radial_net(self._radial_features(core)),
            ],
            dim=1,
        )
        base_feats = torch.cat(
            [
                core["x"],
                core["y"],
                core["t"],
                core["diag_p"],
                core["diag_m"],
                core["r"],
                core["r2"],
                core["xy"],
                torch.sin(np.pi * core["diag_p"]),
                torch.cos(np.pi * core["diag_p"]),
                torch.sin(np.pi * core["diag_m"]),
                torch.cos(np.pi * core["diag_m"]),
            ],
            dim=-1,
        )
        raw = torch.einsum("nb,nbo->no", selector_weights, branch_outputs)
        raw = raw + self.shortcut_scale * self.shortcut(base_feats)
        return apply_output_transform(self.output_transform, coords, raw)


class AnisotropyDirectionalExpert2D(DirectionalShockExpert2D):
    """Directional expert reused as an anisotropy capability branch."""


class CurvatureWaveExpert2D(ExpertNetwork):
    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 1,
        hidden: int = 80,
        depth: int = 4,
        output_transform: Optional[str] = None,
    ):
        del in_dim
        super().__init__(
            in_dim=19,
            out_dim=out_dim,
            hidden=hidden,
            depth=depth,
            activation_factory=LearnableSin,
            output_transform=output_transform,
        )

    def encode(self, coords: torch.Tensor) -> torch.Tensor:
        x = coords[:, 0:1]
        y = coords[:, 1:2]
        t = coords[:, 2:3]
        diag_p = x + y
        diag_m = x - y
        r2 = x.square() + y.square()
        r = torch.sqrt(r2 + 1e-8)
        radial_phase = np.pi * r
        return torch.cat(
            [
                x,
                y,
                t,
                diag_p,
                diag_m,
                r,
                r2,
                x * y,
                x.square() - y.square(),
                torch.sin(np.pi * diag_p),
                torch.cos(np.pi * diag_p),
                torch.sin(np.pi * diag_m),
                torch.cos(np.pi * diag_m),
                torch.sin(radial_phase),
                torch.cos(radial_phase),
                t * torch.sin(radial_phase),
                t * torch.cos(radial_phase),
                torch.sin(2.0 * radial_phase),
                torch.cos(2.0 * radial_phase),
            ],
            dim=-1,
        )


class MixedCurvatureWaveExpert2D(nn.Module):
    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 1,
        hidden: int = 80,
        depth: int = 4,
        output_transform: Optional[str] = None,
        selector_temperature: float = 0.75,
        shortcut_scale_init: float = 0.18,
        branch_output_scale_init: float = 1.00,
    ):
        super().__init__()
        del in_dim
        self.out_dim = out_dim
        self.output_transform = output_transform
        self.selector_temperature = selector_temperature
        self.selector = _SelectorMLP(12, max(24, hidden // 2), 3)
        self.diagonal_branch = self._make_branch(20, hidden, depth)
        self.radial_branch = self._make_branch(20, hidden, depth)
        self.dispersive_branch = self._make_branch(20, hidden, depth)
        self.shortcut = nn.Sequential(
            nn.Linear(12, max(24, hidden // 2)),
            Swish(),
            nn.Linear(max(24, hidden // 2), out_dim),
        )
        self.shortcut_scale = nn.Parameter(torch.tensor(shortcut_scale_init))
        self.branch_output_scale = nn.Parameter(torch.tensor(branch_output_scale_init))
        _init_linear_layers(self.shortcut)

    def _make_branch(self, in_dim: int, hidden: int, depth: int) -> nn.Sequential:
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), LearnableSin()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), LearnableSin()])
        layers.append(nn.Linear(hidden, self.out_dim))
        branch = nn.Sequential(*layers)
        _init_linear_layers(branch)
        return branch

    def _core(self, coords: torch.Tensor) -> dict[str, torch.Tensor]:
        x = coords[:, 0:1]
        y = coords[:, 1:2]
        t = coords[:, 2:3]
        diag_p = x + y
        diag_m = x - y
        r2 = x.square() + y.square()
        r = torch.sqrt(r2 + 1e-8)
        return {
            "x": x,
            "y": y,
            "t": t,
            "diag_p": diag_p,
            "diag_m": diag_m,
            "r": r,
            "r2": r2,
            "xy": x * y,
        }

    def _selector_features(self, core: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            [
                core["x"],
                core["y"],
                core["t"],
                core["diag_p"],
                core["diag_m"],
                core["r"],
                core["r2"],
                core["xy"],
                torch.sin(np.pi * core["diag_p"]),
                torch.cos(np.pi * core["diag_p"]),
                torch.sin(np.pi * core["diag_m"]),
                torch.cos(np.pi * core["diag_m"]),
            ],
            dim=-1,
        )

    def _diagonal_features(self, core: dict[str, torch.Tensor]) -> torch.Tensor:
        diag_p = core["diag_p"]
        diag_m = core["diag_m"]
        t = core["t"]
        return torch.cat(
            [
                core["x"],
                core["y"],
                t,
                diag_p,
                diag_m,
                core["r"],
                core["r2"],
                torch.sin(np.pi * diag_p),
                torch.cos(np.pi * diag_p),
                torch.sin(np.pi * diag_m),
                torch.cos(np.pi * diag_m),
                torch.sin(2.0 * np.pi * diag_p),
                torch.cos(2.0 * np.pi * diag_p),
                t * torch.sin(np.pi * diag_p),
                t * torch.cos(np.pi * diag_p),
                t * torch.sin(np.pi * diag_m),
                t * torch.cos(np.pi * diag_m),
                core["xy"],
                core["x"].square() - core["y"].square(),
                diag_p * diag_m,
            ],
            dim=-1,
        )

    def _radial_features(self, core: dict[str, torch.Tensor]) -> torch.Tensor:
        r = core["r"]
        t = core["t"]
        radial_phase = np.pi * r
        return torch.cat(
            [
                core["x"],
                core["y"],
                t,
                core["diag_p"],
                core["diag_m"],
                r,
                core["r2"],
                torch.sin(radial_phase),
                torch.cos(radial_phase),
                torch.sin(2.0 * radial_phase),
                torch.cos(2.0 * radial_phase),
                torch.sin(4.0 * radial_phase),
                torch.cos(4.0 * radial_phase),
                t * torch.sin(radial_phase),
                t * torch.cos(radial_phase),
                t * torch.sin(2.0 * radial_phase),
                t * torch.cos(2.0 * radial_phase),
                core["xy"],
                core["x"].square() - core["y"].square(),
                torch.exp(-2.5 * r.square()),
            ],
            dim=-1,
        )

    def _dispersive_features(self, core: dict[str, torch.Tensor]) -> torch.Tensor:
        x = core["x"]
        y = core["y"]
        t = core["t"]
        diag_p = core["diag_p"]
        diag_m = core["diag_m"]
        r = core["r"]
        return torch.cat(
            [
                x,
                y,
                t,
                diag_p,
                diag_m,
                r,
                core["r2"],
                torch.sin(np.pi * (diag_p + t)),
                torch.cos(np.pi * (diag_p + t)),
                torch.sin(np.pi * (diag_m - t)),
                torch.cos(np.pi * (diag_m - t)),
                torch.sin(np.pi * (x + y + 0.5 * t)),
                torch.cos(np.pi * (x - y - 0.5 * t)),
                torch.sin(np.pi * (r + t)),
                torch.cos(np.pi * (r + t)),
                torch.sin(2.0 * np.pi * (r - t)),
                torch.cos(2.0 * np.pi * (r - t)),
                core["xy"],
                x.square() - y.square(),
                torch.exp(-2.0 * r.square()),
            ],
            dim=-1,
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        core = self._core(coords)
        weights = torch.softmax(self.selector(self._selector_features(core)) / self.selector_temperature, dim=-1)
        branch_outputs = torch.stack(
            [
                self.diagonal_branch(self._diagonal_features(core)),
                self.radial_branch(self._radial_features(core)),
                self.dispersive_branch(self._dispersive_features(core)),
            ],
            dim=1,
        )
        raw = self.branch_output_scale * torch.einsum("nb,nbo->no", weights, branch_outputs)
        raw = raw + self.shortcut_scale * self.shortcut(self._selector_features(core))
        return apply_output_transform(self.output_transform, coords, raw)
