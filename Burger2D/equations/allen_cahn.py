"""
2D Allen-Cahn (phase-field / sharp-interface) equation:

    u_t = Delta u + (1/eps^2) (u - u^3)

on (x,y) in [-1,1]^2, t in [0,T], with a circular interface and
Dirichlet u = -1 at the boundary.  The 1/eps^2 scaling is the standard
phase-field scaling: the interface moves by mean curvature at O(1)
speed (a circle of radius r0 collapses in time ~ r0^2 / 2), while its
width is O(eps).  This is a qualitatively different mechanism from the
viscous-Burgers shock: a metastable, reaction-driven moving front.

Reference solution by operator splitting:
    reaction step  : exact ODE solution of du/dt = (1/eps^2) (u - u^3)
    diffusion step : Crank-Nicolson for u_t = Delta u with homogeneous
                     Dirichlet BC, solved exactly via DST (sine transform).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.fft import dst as _dst, idst as _idst


@dataclass
class AllenCahnReference:
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    u: np.ndarray  # (nt, ny, nx)


def initial_interface_np(
    x: np.ndarray,
    y: np.ndarray,
    *,
    r0: float = 0.8,
    eps: float = 0.05,
) -> np.ndarray:
    xx, yy = np.meshgrid(x, y, indexing="xy")
    r = np.sqrt(xx**2 + yy**2)
    return np.tanh((r0 - r) / (np.sqrt(2.0) * eps))


def _reaction_exact(u: np.ndarray, dt: float, eps: float) -> np.ndarray:
    """Exact solution of du/dt = (1/eps^2) (u - u^3) over time dt."""
    u2 = u**2
    denom = np.sqrt(np.exp(-2.0 * dt / (eps**2)) * (1.0 - u2) + u2 + 1e-30)
    return u / denom


def _diffusion_cn(
    u: np.ndarray,
    dt: float,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Crank-Nicolson step for u_t = Delta u with Dirichlet u=-1.

    Shifts v = u + 1 so v = 0 on the boundary, then solves the constant-
    coefficient implicit problem exactly with a 2D discrete sine transform.
    """
    ny, nx = u.shape
    v = u + 1.0
    v_int = v[1:-1, 1:-1]
    m, n = v_int.shape

    # discrete Laplacian of the interior (second differences)
    lap = (
        (v[2:, 1:-1] - 2.0 * v_int + v[:-2, 1:-1]) / dy**2
        + (v[1:-1, 2:] - 2.0 * v_int + v[1:-1, :-2]) / dx**2
    )
    rhs = v_int + 0.5 * dt * lap

    # sine-transform the RHS
    rhs_t = _dst(_dst(rhs, type=1, axis=0, orthogonalize=True),
                 type=1, axis=1, orthogonalize=True)

    # eigenvalues of the 1D Dirichlet Laplacian
    py = np.pi * np.arange(1, m + 1) / (m + 1.0)
    px = np.pi * np.arange(1, n + 1) / (n + 1.0)
    ly = 2.0 * (np.cos(py) - 1.0) / dy**2
    lx = 2.0 * (np.cos(px) - 1.0) / dx**2
    denom = 1.0 - 0.5 * dt * (ly[:, None] + lx[None, :])
    sol_t = rhs_t / denom

    sol_int = _idst(_idst(sol_t, type=1, axis=0, orthogonalize=True),
                    type=1, axis=1, orthogonalize=True)
    out = u.copy()
    out[1:-1, 1:-1] = sol_int - 1.0
    out[0, :] = out[-1, :] = -1.0
    out[:, 0] = out[:, -1] = -1.0
    return out


def generate_reference(
    *,
    nx: int = 201,
    ny: int = 201,
    nt: int = 41,
    t_max: float = 0.25,
    eps: float = 0.05,
    r0: float = 0.8,
    dt: float = 1e-4,
) -> AllenCahnReference:
    x = np.linspace(-1.0, 1.0, nx)
    y = np.linspace(-1.0, 1.0, ny)
    t = np.linspace(0.0, t_max, nt)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    u = initial_interface_np(x, y, r0=r0, eps=eps)
    outputs = np.zeros((nt, ny, nx), dtype=np.float32)
    outputs[0] = u.astype(np.float32)
    cur_t = 0.0
    out_idx = 1
    while cur_t < t_max - 1e-12:
        step = min(dt, t_max - cur_t)
        u = _reaction_exact(u, step, eps)
        u = _diffusion_cn(u, step, dx, dy)
        cur_t += step
        while out_idx < nt and t[out_idx] <= cur_t + 1e-12:
            outputs[out_idx] = u.astype(np.float32)
            out_idx += 1
    if out_idx < nt:
        outputs[out_idx] = u.astype(np.float32)
    return AllenCahnReference(x=x.astype(np.float32), y=y.astype(np.float32),
                             t=t.astype(np.float32), u=outputs)


class AllenCahnEquation:
    """Sampling + residual for the 2D Allen-Cahn equation."""

    def __init__(
        self,
        *,
        x_range: Tuple[float, float] = (-1.0, 1.0),
        y_range: Tuple[float, float] = (-1.0, 1.0),
        t_range: Tuple[float, float] = (0.0, 0.25),
        eps: float = 0.05,
        r0: float = 0.8,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.t_min, self.t_max = t_range
        self.eps = eps
        self.r0 = r0
        self.device = device or torch.device("cpu")
        self.dtype = dtype

    def _lhs(self, n: int):
        return np.random.uniform(-1.0, 1.0, n)

    def collocation_points(self, n: int = 8000) -> torch.Tensor:
        x = np.random.uniform(self.x_min, self.x_max, n)
        y = np.random.uniform(self.y_min, self.y_max, n)
        t = np.random.uniform(self.t_min, self.t_max, n)
        return torch.tensor(np.stack([x, y, t], axis=-1), dtype=self.dtype, device=self.device)

    def initial_condition(self, n: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        x = np.random.uniform(self.x_min, self.x_max, n)
        y = np.random.uniform(self.y_min, self.y_max, n)
        xt = torch.tensor(np.stack([x, y, np.zeros(n)], axis=-1), dtype=self.dtype, device=self.device)
        u = initial_interface_np(x, y, r0=self.r0, eps=self.eps)
        return xt, torch.tensor(u[:, None], dtype=self.dtype, device=self.device)

    def boundary_condition(self, n: int = 400) -> Tuple[torch.Tensor, torch.Tensor]:
        # Dirichlet u = -1 on all four boundaries
        t = np.random.uniform(self.t_min, self.t_max, n)
        xb = np.random.uniform(-1.0, 1.0, n)
        yb = np.random.uniform(-1.0, 1.0, n)
        pts = []
        for x0 in (-1.0, 1.0):
            pts.append(np.stack([np.full(n, x0), yb, t], axis=-1))
        for y0 in (-1.0, 1.0):
            pts.append(np.stack([xb, np.full(n, y0), t], axis=-1))
        xt = torch.tensor(np.concatenate(pts, axis=0), dtype=self.dtype, device=self.device)
        u = torch.full((xt.shape[0], 1), -1.0, dtype=self.dtype, device=self.device)
        return xt, u

    def pde_residual(self, u_pred: torch.Tensor, xyt: torch.Tensor) -> torch.Tensor:
        grad = torch.autograd.grad(u_pred.sum(), xyt, create_graph=True, retain_graph=True)[0]
        u_t = grad[:, 2:3]
        u_x = grad[:, 0:1]
        u_y = grad[:, 1:2]
        u_xx = torch.autograd.grad(u_x.sum(), xyt, create_graph=True, retain_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y.sum(), xyt, create_graph=True, retain_graph=True)[0][:, 1:2]
        return u_t - self.eps**2 * (u_xx + u_yy) - u_pred + u_pred**3

    def test_grid(self, nx: int = 129, ny: int = 129, nt: int = 33) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ref = generate_reference(nx=nx, ny=ny, nt=nt, t_max=self.t_max, eps=self.eps, r0=self.r0)
        xx, yy = np.meshgrid(ref.x, ref.y, indexing="xy")
        coords = np.stack([np.tile(xx, (nt, 1, 1)), np.tile(yy, (nt, 1, 1)),
                           np.repeat(ref.t[:, None, None], ny, axis=1)], axis=-1)
        return ref.x, ref.y, ref.t, ref.u


def region_scores(u_ref: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Region scores from the reference: interface / bulk / transition."""
    grad_mag = np.zeros_like(u_ref, dtype=np.float32)
    for i in range(u_ref.shape[0]):
        uy, ux = np.gradient(u_ref[i], y, x, edge_order=1)
        grad_mag[i] = np.sqrt(ux**2 + uy**2)
    g = grad_mag / (grad_mag.max() + 1e-8)
    interface = np.clip(g, 0.0, 1.5)
    bulk = np.clip(1.1 - g, 0.0, 1.5)
    transition = np.clip(2.0 * np.minimum(g, 1.0 - g), 0.0, 1.5)
    return {"interface": interface, "bulk": bulk, "transition": transition}
