"""
Generic trainer for PINN / MoE-PINN models.

The main change from the previous version is the NTK-inspired weighting rule:
- keep the PDE residual weight anchored instead of letting it collapse to ~0
- adapt IC/BC weights within bounded ranges
- smooth weight updates over time
"""

import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn,
        lr: float = 1e-3,
        n_steps: int = 10000,
        ntk_update_freq: int = 200,
        use_ntk: bool = True,
        device: torch.device = None,
        save_dir: str = "results",
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.n_steps = n_steps
        self.ntk_update_freq = ntk_update_freq
        self.use_ntk = use_ntk
        self.device = device or torch.device("cpu")
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_steps, eta_min=lr * 0.01
        )

        self.history: Dict[str, List[float]] = {
            "total": [],
            "res": [],
            "ic": [],
            "bc": [],
            "sparse": [],
            "balance": [],
            "l2_error": [],
            "ntk_w_res": [],
            "ntk_w_ic": [],
            "ntk_w_bc": [],
            "gate_entropy": [],
            "gate_max": [],
        }

        self._ntk_ema: Dict[str, float] = {}
        self._ntk_weights: Dict[str, float] = {
            "res": float(loss_fn.cfg.w_res),
            "ic": float(loss_fn.cfg.w_ic),
            "bc": float(loss_fn.cfg.w_bc),
            "sparse": float(getattr(model, "sparsity_weight", loss_fn.cfg.w_sparse)),
        }

    def _update_ntk_weights(self, loss_dict: Dict[str, torch.Tensor]):
        alpha = 0.9
        smooth = 0.8
        vals = {k: v.item() for k, v in loss_dict.items() if k not in {"sparse", "balance"}}

        for k, v in vals.items():
            self._ntk_ema[k] = alpha * self._ntk_ema.get(k, v) + (1 - alpha) * v

        ref = self._ntk_ema.get("res", 1.0)
        target_w = {
            "res": float(self.loss_fn.cfg.w_res),
            "sparse": self._ntk_weights.get(
                "sparse",
                float(getattr(self.model, "sparsity_weight", self.loss_fn.cfg.w_sparse)),
            ),
        }

        for key, base in (("ic", self.loss_fn.cfg.w_ic), ("bc", self.loss_fn.cfg.w_bc)):
            current = self._ntk_ema.get(key, ref)
            ratio = ref / (current + 1e-10)
            ratio = float(np.clip(ratio, 0.1, 2.0))
            target_w[key] = float(np.clip(base * ratio, 0.5, 10.0))

        for key, value in target_w.items():
            prev = self._ntk_weights.get(key, value)
            self._ntk_weights[key] = smooth * prev + (1 - smooth) * value

    def train(
        self,
        batch: dict,
        eval_fn=None,
        eval_freq: int = 500,
        log_freq: int = 100,
    ) -> Dict[str, List[float]]:
        self.model.train()
        t_start = time.time()

        pbar = tqdm(range(1, self.n_steps + 1), desc="Training", ncols=100)
        for step in pbar:
            self.optimizer.zero_grad()

            ntk_w = self._ntk_weights if self.use_ntk else None
            total_loss, loss_dict = self.loss_fn.compute(
                self.model, batch, ntk_weights=ntk_w
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.scheduler.step()

            if self.use_ntk and step % self.ntk_update_freq == 0:
                self._update_ntk_weights(loss_dict)

            self.history["total"].append(total_loss.item())
            for key in ["res", "ic", "bc", "sparse", "balance"]:
                if key in loss_dict:
                    self.history[key].append(loss_dict[key].item())
            if self.use_ntk:
                self.history["ntk_w_res"].append(self._ntk_weights.get("res", 1.0))
                self.history["ntk_w_ic"].append(self._ntk_weights.get("ic", 1.0))
                self.history["ntk_w_bc"].append(self._ntk_weights.get("bc", 1.0))

            if eval_fn is not None and step % eval_freq == 0:
                self.model.eval()
                with torch.no_grad():
                    l2_err = eval_fn()
                self.history["l2_error"].append(l2_err)
                self.model.train()

            if step % log_freq == 0:
                gate_info = {}
                if hasattr(self.model, "gating") and "xt_col" in batch:
                    xt_col = batch["xt_col"]
                    n_probe = min(512, xt_col.shape[0])
                    probe_idx = torch.randperm(xt_col.shape[0], device=xt_col.device)[:n_probe]
                    if hasattr(self.model, "load_balance_stats"):
                        stats = self.model.load_balance_stats(xt_col[probe_idx])
                    else:
                        stats = self.model.gating.load_balance_stats(xt_col[probe_idx])
                    gate_info = {
                        "gmax": f"{stats['max_gate_weight']:.2f}",
                        "gent": f"{stats['mean_entropy']:.2f}",
                    }
                    self.history["gate_entropy"].append(stats["mean_entropy"])
                    self.history["gate_max"].append(stats["max_gate_weight"])
                    for i, frac in enumerate(stats["expert_load_frac"]):
                        key = f"gate_load_{i}"
                        self.history.setdefault(key, []).append(float(frac))

                pbar.set_postfix(
                    {
                        "L": f"{total_loss.item():.3e}",
                        "res": f"{loss_dict['res'].item():.3e}",
                        "ic": f"{loss_dict['ic'].item():.3e}",
                        "w_ic": f"{self._ntk_weights.get('ic', 1.0):.2f}",
                        **gate_info,
                    }
                )

        print(
            f"\n[OK] Training done in {time.time() - t_start:.1f}s | "
            f"Final loss: {self.history['total'][-1]:.4e}"
        )
        return self.history

    def save_checkpoint(self, name: str = "model"):
        path = os.path.join(self.save_dir, f"{name}.pt")
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "history": self.history,
                "ntk_weights": self._ntk_weights,
            },
            path,
        )
        print(f"[OK] Saved checkpoint: {path}")

    def load_checkpoint(self, name: str = "model"):
        path = os.path.join(self.save_dir, f"{name}.pt")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.history = ckpt.get("history", self.history)
        self._ntk_weights = ckpt.get("ntk_weights", self._ntk_weights)
        print(f"[OK] Loaded checkpoint: {path}")
