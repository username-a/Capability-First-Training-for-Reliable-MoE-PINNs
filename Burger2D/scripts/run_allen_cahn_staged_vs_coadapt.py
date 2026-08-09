"""Staged-vs-coadaptation experiment on the 2D Allen-Cahn equation.

Third-equation robustness check for the paper's central mechanism claim:
with an identical reference-guided information stream, ``staged`` training
(region-pretrained experts, then gate-only adaptation) preserves expert
health, while ``coadapt`` (gate + experts updated jointly) produces
gate-scaled expert gradients and degrades the least-used experts even when
the mixture error stays low (pseudo-fit via error cancellation).

Equation:  u_t = Delta u + (1/eps^2) (u - u^3)   on [-1,1]^2 x [0,0.25],
circular interface r0 = 0.8 shrinking by mean curvature; Dirichlet u = -1.
Three experts: interior (+1 phase), exterior (-1 phase), interface.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Burger2D.core.gating import GatingNetwork2D
from Burger2D.equations.allen_cahn import (
    AllenCahnReference,
    generate_reference,
    initial_interface_np,
)

EPS = 0.08
R0 = 0.8
T_MAX = 0.25
DT_REF = 2e-4
EXPERT_NAMES = ["interior", "exterior", "interface"]
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "allen_cahn"


@dataclass
class AllenConfig:
    seed: int = 42
    mode: str = "staged"          # staged | coadapt
    expert_steps: int = 1500      # stage-A steps per expert
    main_steps: int = 1500        # gate-only (staged) or joint (coadapt) steps
    expert_lr: float = 1e-3
    gate_lr: float = 5e-4
    pde_weight: float = 0.1
    sup_weight: float = 1.0
    expert_sup_weight: float = 8.0
    gate_match_weight: float = 1.0
    balance_weight: float = 0.01
    sup_batch: int = 2048
    pde_batch: int = 1024
    gate_pool_size: int = 8192
    gate_target_temp: float = 0.10
    gate_oracle_frac: float = 0.6
    target_refresh: int = 20
    temperature: float = 0.8
    gate_hidden: int = 48
    gate_depth: int = 3
    expert_hidden: int = 96
    expert_depth: int = 5
    nx: int = 201
    ny: int = 201
    nt: int = 41
    log_freq: int = 100
    smoke: bool = False


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_reference(cfg: AllenConfig) -> AllenCahnReference:
    """Build (and cache) the operator-splitting reference solution."""
    cache = RESULTS_DIR / f"allen_cahn_ref_e{EPS}_nx{cfg.nx}_nt{cfg.nt}.npz"
    if cache.exists():
        data = np.load(cache)
        return AllenCahnReference(
            x=data["x"], y=data["y"], t=data["t"], u=data["u"],
        )
    cache.parent.mkdir(parents=True, exist_ok=True)
    ref = generate_reference(
        nx=cfg.nx, ny=cfg.ny, nt=cfg.nt, t_max=T_MAX,
        eps=EPS, r0=R0, dt=DT_REF,
    )
    np.savez_compressed(
        cache, x=ref.x, y=ref.y, t=ref.t, u=ref.u,
    )
    return ref


def flatten_reference(ref: AllenCahnReference) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nt_, ny_, nx_ = ref.u.shape
    xx, yy = np.meshgrid(ref.x, ref.y, indexing="xy")
    tt = np.tile(ref.t[:, None, None], (1, ny_, nx_))
    coords = np.stack(
        [np.tile(xx, (nt_, 1, 1)), np.tile(yy, (nt_, 1, 1)), tt], axis=-1,
    ).reshape(-1, 3).astype(np.float32)
    values = ref.u.reshape(-1, 1).astype(np.float32)
    region = np.zeros(values.shape[0], dtype=np.int64)
    uf = ref.u.reshape(-1)
    region[uf > 0.5] = 0          # interior
    region[uf < -0.5] = 1         # exterior
    region[np.abs(uf) <= 0.5] = 2 # interface
    return coords, values, region


class ExpertMLP(nn.Module):
    def __init__(self, hidden: int = 96, depth: int = 5):
        super().__init__()
        layers = [nn.Linear(3, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        return self.net(xyt)


class AllenCahnMoE(nn.Module):
    def __init__(
        self,
        experts: list[ExpertMLP],
        gating: GatingNetwork2D,
        expert_names: Optional[list[str]] = None,
    ):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.gating = gating
        self.expert_names = expert_names or [f"expert_{i}" for i in range(len(experts))]

    def expert_predictions(self, xyt: torch.Tensor) -> torch.Tensor:
        return torch.stack([expert(xyt) for expert in self.experts], dim=1)  # (N,K,1)

    def gate_weights(self, xyt: torch.Tensor) -> torch.Tensor:
        return self.gating(xyt)  # (N,K)

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        preds = self.expert_predictions(xyt)
        weights = self.gate_weights(xyt)
        return torch.einsum("nk,nko->no", weights, preds)


def pde_residual(model: nn.Module, xyt: torch.Tensor, eps: float) -> torch.Tensor:
    xyt = xyt.clone().requires_grad_(True)
    u = model(xyt)
    g = torch.autograd.grad(u.sum(), xyt, create_graph=True, retain_graph=True)[0]
    u_t = g[:, 2:3]
    u_x = g[:, 0:1]
    u_y = g[:, 1:2]
    u_xx = torch.autograd.grad(u_x.sum(), xyt, create_graph=True, retain_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y.sum(), xyt, create_graph=True, retain_graph=True)[0][:, 1:2]
    return u_t - (u_xx + u_yy) - (1.0 / eps**2) * (u - u**3)


def _relative_l2(pred: torch.Tensor, ref: torch.Tensor) -> float:
    return float(torch.sqrt(((pred - ref) ** 2).mean() / (ref**2).mean()))


def build_model(cfg: AllenConfig, device: torch.device) -> AllenCahnMoE:
    experts = [ExpertMLP(cfg.expert_hidden, cfg.expert_depth) for _ in EXPERT_NAMES]
    gating = GatingNetwork2D(
        in_dim=3,
        num_experts=len(experts),
        hidden=cfg.gate_hidden,
        depth=cfg.gate_depth,
        sparsity_p=0.5,
        temperature=cfg.temperature,
    )
    return AllenCahnMoE(experts, gating, EXPERT_NAMES).to(device)


def make_gate_targets(
    model: AllenCahnMoE,
    gate_coords: torch.Tensor,
    gate_values: torch.Tensor,
    gate_region: torch.Tensor,
    cfg: AllenConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Oracle (lowest single-expert error) + region prior targets."""
    with torch.no_grad():
        preds = model.expert_predictions(gate_coords).squeeze(-1)          # (N,K)
        errs = (preds - gate_values).abs()                                 # (N,K)
        oracle = F.softmax(-errs / cfg.gate_target_temp, dim=-1)
    prior = F.one_hot(gate_region, num_classes=len(EXPERT_NAMES)).float()
    targets = cfg.gate_oracle_frac * oracle + (1.0 - cfg.gate_oracle_frac) * prior
    targets = targets / targets.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    confidence = (1.0 + 0.1 * oracle.max(dim=-1).values)  # oracle-quality weight
    return targets, confidence


def expert_region_loss(
    model: AllenCahnMoE,
    k: int,
    sup_coords: torch.Tensor,
    sup_values: torch.Tensor,
    cfg: AllenConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Stage-A loss: region-restricted supervised regression only.

    The full Allen-Cahn PDE residual has a large gradient barrier around the
    trivial state u = 0 (u - u^3 vanishes there), which pins pure-PDE expert
    pretraining to the degenerate solution.  We therefore give each expert its
    region's reference values during pretraining; physics enters through the
    mixture residual in the main phase (identical information stream for both
    modes).
    """
    expert = model.experts[k]
    sup_pred = expert(sup_coords)
    sup_loss = F.mse_loss(sup_pred, sup_values)
    loss = cfg.expert_sup_weight * sup_loss
    return loss, {"sup": float(sup_loss.item())}


def evaluate(
    model: AllenCahnMoE,
    coords: torch.Tensor,
    values: torch.Tensor,
    batch: int = 65536,
) -> dict[str, float]:
    """Full-grid metrics (relative L2s, expert health, routing, cancellation)."""
    device = next(model.parameters()).device
    n = coords.shape[0]
    pred_list, preds_list, weights_list = [], [], []
    with torch.no_grad():
        for start in range(0, n, batch):
            chunk = coords[start:start + batch].to(device)
            preds = model.expert_predictions(chunk).cpu()
            weights = model.gate_weights(chunk).cpu()
            pred_list.append(torch.einsum("nk,nko->no", weights, preds))
            preds_list.append(preds)
            weights_list.append(weights)
    pred = torch.cat(pred_list, dim=0)
    preds = torch.cat(preds_list, dim=0).squeeze(-1)     # (N,K)
    weights = torch.cat(weights_list, dim=0)             # (N,K)
    ref = values.cpu()

    l2_mixed = _relative_l2(pred, ref)
    l2_experts = [_relative_l2(preds[:, k:k + 1], ref) for k in range(len(EXPERT_NAMES))]
    worst_expert = max(l2_experts)

    routing = weights.argmax(dim=-1)
    counts = torch.bincount(routing, minlength=len(EXPERT_NAMES)).float()
    load = counts / counts.sum().clamp_min(1.0)
    effective = float(torch.exp(-(load * (load + 1e-12).log()).sum()))
    min_load = float(load.min())

    errs = (preds - ref).abs()
    oracle_idx = errs.argmin(dim=-1)
    oracle_pred = preds.gather(1, oracle_idx[:, None])
    l2_oracle = _relative_l2(oracle_pred, ref)
    regret = l2_mixed - l2_oracle

    e_k = preds - ref                                      # (N,K)
    weighted = weights * e_k                               # (N,K)
    num = torch.sqrt((weighted.sum(dim=1) ** 2).mean())
    den = sum(torch.sqrt((weighted[:, k] ** 2).mean()) for k in range(len(EXPERT_NAMES)))
    cancellation = 1.0 - float(num / den.clamp_min(1e-12))

    entropy = -(weights * (weights + 1e-12).log()).sum(dim=-1).mean()
    return {
        "l2_mixed": l2_mixed,
        "l2_oracle": l2_oracle,
        "routing_regret": regret,
        "worst_expert_l2": worst_expert,
        **{f"expert_l2_{name}": l2_experts[k] for k, name in enumerate(EXPERT_NAMES)},
        "effective_experts": effective,
        "min_load": min_load,
        "mean_gate_entropy": float(entropy),
        "error_cancellation": cancellation,
        **{f"load_{name}": float(load[k]) for k, name in enumerate(EXPERT_NAMES)},
    }


def train(mode: str, cfg: AllenConfig, device: torch.device, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(cfg.seed)
    ref = get_reference(cfg)
    coords_np, values_np, region_np = flatten_reference(ref)

    coords = torch.tensor(coords_np, dtype=torch.float32)
    values = torch.tensor(values_np, dtype=torch.float32)
    region = torch.tensor(region_np, dtype=torch.long)

    model = build_model(cfg, device)

    # ---------------- Stage A: region-restricted expert pretraining ----------------
    rng = np.random.default_rng(cfg.seed + 1000)
    history: dict[str, list] = {"stage_a": [], "main": []}
    for k, name in enumerate(EXPERT_NAMES):
        mask = (region_np == k)
        idx_k = np.flatnonzero(mask)
        opt = torch.optim.Adam(model.experts[k].parameters(), lr=cfg.expert_lr)
        for step in range(cfg.expert_steps):
            opt.zero_grad(set_to_none=True)
            sup_ids = idx_k[rng.integers(0, len(idx_k), cfg.sup_batch)]
            sup_coords = coords[sup_ids].to(device)
            sup_values_t = values[sup_ids].to(device)
            loss, parts = expert_region_loss(model, k, sup_coords, sup_values_t, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.experts[k].parameters(), 1.0)
            opt.step()
            if step % max(cfg.expert_steps // 10, 1) == 0:
                history["stage_a"].append(
                    {"expert": name, "step": step, "loss": float(loss.item()), **parts}
                )

    # ---------------- Gate pool & targets ----------------
    gate_rng = np.random.default_rng(cfg.seed + 2000)
    gate_ids = gate_rng.choice(coords_np.shape[0], cfg.gate_pool_size, replace=False)
    gate_coords = coords[gate_ids].to(device)
    gate_values = values[gate_ids].to(device)
    gate_region = region[gate_ids].to(device)
    targets, confidence = make_gate_targets(model, gate_coords, gate_values, gate_region, cfg)

    # ---------------- Main phase ----------------
    pre_metrics = evaluate(model, coords, values)
    started = time.time()
    if mode == "staged":
        for p in model.experts.parameters():
            p.requires_grad_(False)
        opt = torch.optim.Adam(model.gating.parameters(), lr=cfg.gate_lr)
        for step in range(cfg.main_steps):
            opt.zero_grad(set_to_none=True)
            ids = rng.integers(0, coords_np.shape[0], cfg.sup_batch)
            sup_coords = coords[ids].to(device)
            sup_values_t = values[ids].to(device)
            sup_loss = F.mse_loss(model(sup_coords), sup_values_t)
            col_ids = rng.integers(0, coords_np.shape[0], cfg.pde_batch)
            r = pde_residual(model, coords[col_ids].to(device), EPS)
            pde_loss = r.square().mean()
            probs = model.gate_weights(gate_coords)
            gate_match = (confidence * F.kl_div(
                probs.clamp_min(1e-8).log(), targets, reduction="none"
            ).sum(dim=1)).mean()
            balance = model.gating.load_balance_loss(gate_coords)
            loss = (
                cfg.sup_weight * sup_loss
                + cfg.pde_weight * pde_loss
                + cfg.gate_match_weight * gate_match
                + cfg.balance_weight * balance
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.gating.parameters(), 1.0)
            opt.step()
            if step % max(cfg.main_steps // 10, 1) == 0:
                history["main"].append({
                    "step": step, "loss": float(loss.item()),
                    "sup": float(sup_loss.item()), "pde": float(pde_loss.item()),
                    "gate_match": float(gate_match.item()), "balance": float(balance.item()),
                })
    else:
        opt = torch.optim.Adam([
            {"params": [p for e in model.experts for p in e.parameters()], "lr": cfg.expert_lr},
            {"params": model.gating.parameters(), "lr": cfg.gate_lr},
        ])
        for step in range(cfg.main_steps):
            opt.zero_grad(set_to_none=True)
            ids = rng.integers(0, coords_np.shape[0], cfg.sup_batch)
            sup_coords = coords[ids].to(device)
            sup_values_t = values[ids].to(device)
            sup_loss = F.mse_loss(model(sup_coords), sup_values_t)
            col_ids = rng.integers(0, coords_np.shape[0], cfg.pde_batch)
            r = pde_residual(model, coords[col_ids].to(device), EPS)
            pde_loss = r.square().mean()
            if step % cfg.target_refresh == 0:
                targets, confidence = make_gate_targets(
                    model, gate_coords, gate_values, gate_region, cfg
                )
            probs = model.gate_weights(gate_coords)
            gate_match = (confidence * F.kl_div(
                probs.clamp_min(1e-8).log(), targets, reduction="none"
            ).sum(dim=1)).mean()
            balance = model.gating.load_balance_loss(gate_coords)
            loss = (
                cfg.sup_weight * sup_loss
                + cfg.pde_weight * pde_loss
                + cfg.gate_match_weight * gate_match
                + cfg.balance_weight * balance
            )
            loss.backward()
            params = [p for e in model.experts for p in e.parameters()] + list(model.gating.parameters())
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if step % max(cfg.main_steps // 10, 1) == 0:
                history["main"].append({
                    "step": step, "loss": float(loss.item()),
                    "sup": float(sup_loss.item()), "pde": float(pde_loss.item()),
                    "gate_match": float(gate_match.item()), "balance": float(balance.item()),
                })

    metrics = evaluate(model, coords, values)
    metrics["pre_main"] = pre_metrics
    metrics["wall_seconds"] = float(time.time() - started)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    (output_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    torch.save(
        {"model": model.state_dict(), "config": asdict(cfg)},
        output_dir / "checkpoint.pt",
    )
    print(f"[{mode} seed={cfg.seed}] l2_mixed={metrics['l2_mixed']:.4f} "
          f"worst_expert={metrics['worst_expert_l2']:.4f} "
          f"eff_experts={metrics['effective_experts']:.3f} "
          f"cancellation={metrics['error_cancellation']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["staged", "coadapt"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expert-steps", type=int, default=1500)
    parser.add_argument("--main-steps", type=int, default=1500)
    parser.add_argument("--expert-lr", type=float, default=None)
    parser.add_argument("--gate-match", type=float, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = AllenConfig(
        seed=args.seed,
        mode=args.mode,
        expert_steps=args.expert_steps,
        main_steps=args.main_steps,
        smoke=args.smoke,
    )
    if args.expert_lr is not None:
        cfg.expert_lr = args.expert_lr
    if args.gate_match is not None:
        cfg.gate_match_weight = args.gate_match
    if args.smoke:
        cfg.nx, cfg.ny, cfg.nt = 81, 81, 11
        cfg.expert_steps = min(cfg.expert_steps, 120)
        cfg.main_steps = min(cfg.main_steps, 120)
        cfg.gate_pool_size = 2048
        cfg.sup_batch = 512
        cfg.pde_batch = 256
        cfg.log_freq = 20

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR / f"{args.mode}_seed{args.seed}"
    train(args.mode, cfg, device, output_dir)


if __name__ == "__main__":
    main()
