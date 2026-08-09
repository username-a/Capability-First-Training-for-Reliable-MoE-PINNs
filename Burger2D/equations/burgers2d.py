"""
2D scalar viscous Burgers problem definition and reference solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import torch


def initial_profile_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    envelope = (1.0 - x**2) * (1.0 - y**2)
    carrier = (
        0.65 * np.sin(np.pi * (x + y))
        - 0.35 * np.sin(np.pi * (x - y))
        + 0.20 * np.sin(2.0 * np.pi * x) * np.sin(np.pi * y)
    )
    return -envelope * carrier


def initial_profile_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    pi = torch.tensor(np.pi, dtype=x.dtype, device=x.device)
    envelope = (1.0 - x.square()) * (1.0 - y.square())
    carrier = (
        0.65 * torch.sin(pi * (x + y))
        - 0.35 * torch.sin(pi * (x - y))
        + 0.20 * torch.sin(2.0 * pi * x) * torch.sin(pi * y)
    )
    return -envelope * carrier


def _lhs_sample(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    result = np.zeros((n, d), dtype=np.float64)
    for dim in range(d):
        perm = rng.permutation(n)
        result[:, dim] = (perm + rng.random(n)) / n
    return result


@dataclass
class ReferenceSolution2D:
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    u: np.ndarray


class Burgers2DProblem:
    def __init__(
        self,
        nu: float = 0.01 / np.pi,
        x_range: Tuple[float, float] = (-1.0, 1.0),
        y_range: Tuple[float, float] = (-1.0, 1.0),
        t_range: Tuple[float, float] = (0.0, 1.0),
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        seed: int = 42,
        initial_profile_np_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        initial_profile_torch_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        self.nu = nu
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.t_min, self.t_max = t_range
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.initial_profile_np_fn = initial_profile_np_fn or initial_profile_np
        self.initial_profile_torch_fn = initial_profile_torch_fn or initial_profile_torch

    def initial_condition(
        self,
        n: int = 1024,
        method: str = "lhs",
        noise: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xy = self._sample_box(
            n=n,
            mins=(self.x_min, self.y_min),
            maxs=(self.x_max, self.y_max),
            method=method,
            dims=2,
        )
        x = torch.tensor(xy[:, 0:1], dtype=self.dtype, device=self.device)
        y = torch.tensor(xy[:, 1:2], dtype=self.dtype, device=self.device)
        t = torch.zeros_like(x)
        u = self.initial_profile_torch_fn(x, y)
        if noise > 0:
            u = u + noise * torch.randn_like(u)
        xyt = torch.cat([x, y, t], dim=1)
        return xyt, u

    def boundary_condition(
        self,
        n_per_face: int = 256,
        method: str = "lhs",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        st_left = self._sample_box(
            n=n_per_face,
            mins=(self.y_min, self.t_min),
            maxs=(self.y_max, self.t_max),
            method=method,
            dims=2,
        )
        st_right = self._sample_box(
            n=n_per_face,
            mins=(self.y_min, self.t_min),
            maxs=(self.y_max, self.t_max),
            method=method,
            dims=2,
        )
        st_bottom = self._sample_box(
            n=n_per_face,
            mins=(self.x_min, self.t_min),
            maxs=(self.x_max, self.t_max),
            method=method,
            dims=2,
        )
        st_top = self._sample_box(
            n=n_per_face,
            mins=(self.x_min, self.t_min),
            maxs=(self.x_max, self.t_max),
            method=method,
            dims=2,
        )

        left = np.column_stack(
            [np.full(n_per_face, self.x_min), st_left[:, 0], st_left[:, 1]]
        )
        right = np.column_stack(
            [np.full(n_per_face, self.x_max), st_right[:, 0], st_right[:, 1]]
        )
        bottom = np.column_stack(
            [st_bottom[:, 0], np.full(n_per_face, self.y_min), st_bottom[:, 1]]
        )
        top = np.column_stack(
            [st_top[:, 0], np.full(n_per_face, self.y_max), st_top[:, 1]]
        )
        xyt_bc = np.concatenate([left, right, bottom, top], axis=0)
        u_bc = np.zeros((xyt_bc.shape[0], 1), dtype=np.float32)
        return (
            torch.tensor(xyt_bc, dtype=self.dtype, device=self.device),
            torch.tensor(u_bc, dtype=self.dtype, device=self.device),
        )

    def collocation_points(
        self,
        n: int = 6000,
        method: str = "lhs",
    ) -> torch.Tensor:
        xyt = self._sample_box(
            n=n,
            mins=(self.x_min, self.y_min, self.t_min),
            maxs=(self.x_max, self.y_max, self.t_max),
            method=method,
            dims=3,
        )
        return torch.tensor(xyt, dtype=self.dtype, device=self.device)

    def training_batch(
        self,
        n_col: int = 6000,
        n_ic: int = 1200,
        n_bc_per_face: int = 400,
    ) -> dict[str, torch.Tensor]:
        xt_ic, u_ic = self.initial_condition(n=n_ic)
        xt_bc, u_bc = self.boundary_condition(n_per_face=n_bc_per_face)
        xt_col = self.collocation_points(n=n_col)
        return {
            "xt_col": xt_col,
            "xt_ic": xt_ic,
            "u_ic": u_ic,
            "xt_bc": xt_bc,
            "u_bc": u_bc,
        }

    def generate_reference_solution(
        self,
        nx: int = 65,
        ny: int = 65,
        nt: int = 21,
        cfl: float = 0.22,
        max_steps: int = 200000,
    ) -> ReferenceSolution2D:
        x = np.linspace(self.x_min, self.x_max, nx, dtype=np.float64)
        y = np.linspace(self.y_min, self.y_max, ny, dtype=np.float64)
        t = np.linspace(self.t_min, self.t_max, nt, dtype=np.float64)
        xx, yy = np.meshgrid(x, y, indexing="xy")
        u = self.initial_profile_np_fn(xx, yy)
        u[0, :] = 0.0
        u[-1, :] = 0.0
        u[:, 0] = 0.0
        u[:, -1] = 0.0

        dx = float(x[1] - x[0])
        dy = float(y[1] - y[0])
        outputs = np.zeros((nt, ny, nx), dtype=np.float32)
        outputs[0] = u.astype(np.float32)

        cur_t = t[0]
        out_idx = 1
        n_steps = 0

        while cur_t < t[-1] - 1e-12:
            max_u = max(float(np.abs(u).max()), 1e-6)
            adv_dt = min(dx, dy) / max_u
            diff_dt = 0.25 * min(dx * dx, dy * dy) / max(self.nu, 1e-8)
            dt = min(cfl * adv_dt, 0.95 * diff_dt, t[-1] - cur_t)
            prev_u = u.copy()
            prev_t = cur_t
            next_u = self._advance_one_step(u, dx=dx, dy=dy, dt=dt)
            cur_t += dt
            n_steps += 1

            if not np.isfinite(next_u).all():
                raise RuntimeError("Reference solver became unstable.")
            if n_steps > max_steps:
                raise RuntimeError("Reference solver exceeded max_steps.")

            while out_idx < nt and t[out_idx] <= cur_t + 1e-12:
                alpha = (t[out_idx] - prev_t) / max(cur_t - prev_t, 1e-12)
                outputs[out_idx] = ((1.0 - alpha) * prev_u + alpha * next_u).astype(np.float32)
                out_idx += 1

            u = next_u

        if out_idx != nt:
            outputs[-1] = u.astype(np.float32)
        return ReferenceSolution2D(
            x=x.astype(np.float32),
            y=y.astype(np.float32),
            t=t.astype(np.float32),
            u=outputs,
        )

    def _advance_one_step(
        self,
        u: np.ndarray,
        *,
        dx: float,
        dy: float,
        dt: float,
    ) -> np.ndarray:
        u_new = u.copy()
        center = u[1:-1, 1:-1]

        ux_f = (u[1:-1, 2:] - center) / dx
        ux_b = (center - u[1:-1, :-2]) / dx
        uy_f = (u[2:, 1:-1] - center) / dy
        uy_b = (center - u[:-2, 1:-1]) / dy

        ux_adv = np.where(center >= 0.0, ux_b, ux_f)
        uy_adv = np.where(center >= 0.0, uy_b, uy_f)

        u_xx = (u[1:-1, 2:] - 2.0 * center + u[1:-1, :-2]) / (dx * dx)
        u_yy = (u[2:, 1:-1] - 2.0 * center + u[:-2, 1:-1]) / (dy * dy)

        convection = center * (ux_adv + uy_adv)
        diffusion = self.nu * (u_xx + u_yy)
        u_new[1:-1, 1:-1] = center - dt * convection + dt * diffusion

        u_new[0, :] = 0.0
        u_new[-1, :] = 0.0
        u_new[:, 0] = 0.0
        u_new[:, -1] = 0.0
        return u_new

    def _sample_box(
        self,
        *,
        n: int,
        mins: tuple[float, ...],
        maxs: tuple[float, ...],
        method: str,
        dims: int,
    ) -> np.ndarray:
        if method == "lhs":
            raw = _lhs_sample(n, dims, self.rng)
        elif method == "uniform":
            side = int(np.ceil(n ** (1.0 / dims)))
            axes = [np.linspace(lo, hi, side) for lo, hi in zip(mins, maxs)]
            mesh = np.meshgrid(*axes, indexing="xy")
            raw = np.stack([axis.ravel() for axis in mesh], axis=-1)
            return raw[:n].astype(np.float64)
        else:
            raw = self.rng.random((n, dims))

        mins_arr = np.asarray(mins, dtype=np.float64)
        maxs_arr = np.asarray(maxs, dtype=np.float64)
        return raw * (maxs_arr - mins_arr) + mins_arr


def steep_region_mask(
    u_ref: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    quantile: float = 0.9,
) -> tuple[np.ndarray, float]:
    grad_mag = np.empty_like(u_ref, dtype=np.float32)
    for i in range(u_ref.shape[0]):
        du_dy, du_dx = np.gradient(u_ref[i], y, x, edge_order=1)
        grad_mag[i] = np.sqrt(du_dx**2 + du_dy**2)
    threshold = float(np.quantile(grad_mag, quantile))
    return grad_mag >= threshold, threshold
