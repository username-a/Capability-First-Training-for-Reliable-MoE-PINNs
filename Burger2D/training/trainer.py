"""
Generic trainer for the Burger2D baseline.
"""

from __future__ import annotations

import os
import time
from typing import Callable, Dict, List, Optional

import torch
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn,
        lr: float = 1e-3,
        n_steps: int = 1500,
        device: Optional[torch.device] = None,
        save_dir: str = "results",
        grad_clip: float = 1.0,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.n_steps = n_steps
        self.device = device or torch.device("cpu")
        self.save_dir = save_dir
        self.grad_clip = grad_clip
        os.makedirs(save_dir, exist_ok=True)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=n_steps,
            eta_min=lr * 0.05,
        )

        self.history: Dict[str, List[float]] = {
            "total": [],
            "l2_error": [],
            "gate_entropy": [],
            "gate_max": [],
        }

    def train(
        self,
        batch: dict[str, torch.Tensor],
        eval_fn: Optional[Callable[[], float]] = None,
        eval_freq: int = 100,
        log_freq: int = 50,
        batch_refresh_fn: Optional[Callable[[], dict[str, torch.Tensor]]] = None,
        batch_refresh_freq: int = 0,
    ) -> Dict[str, List[float]]:
        start = time.time()
        self.model.train()

        pbar = tqdm(range(1, self.n_steps + 1), desc="Training", ncols=100)
        for step in pbar:
            if (
                batch_refresh_fn is not None
                and batch_refresh_freq > 0
                and step > 1
                and (step - 1) % batch_refresh_freq == 0
            ):
                batch = batch_refresh_fn()
            self.optimizer.zero_grad()
            total_loss, loss_dict = self.loss_fn.compute(self.model, batch)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.scheduler.step()

            self.history["total"].append(total_loss.item())
            for key, value in loss_dict.items():
                self.history.setdefault(key, []).append(value.item())

            if eval_fn is not None and step % eval_freq == 0:
                self.model.eval()
                with torch.no_grad():
                    self.history["l2_error"].append(eval_fn())
                self.model.train()

            gate_info = {}
            if hasattr(self.model, "gating") and "xt_col" in batch and step % log_freq == 0:
                xt_col = batch["xt_col"]
                probe_size = min(1024, xt_col.shape[0])
                probe_idx = torch.randperm(xt_col.shape[0], device=xt_col.device)[:probe_size]
                if hasattr(self.model, "load_balance_stats"):
                    stats = self.model.load_balance_stats(xt_col[probe_idx])
                else:
                    stats = self.model.gating.load_balance_stats(xt_col[probe_idx])
                self.history["gate_entropy"].append(stats["mean_entropy"])
                self.history["gate_max"].append(stats["max_gate_weight"])
                for i, frac in enumerate(stats["expert_load_frac"]):
                    self.history.setdefault(f"gate_load_{i}", []).append(float(frac))
                gate_info = {
                    "gmax": f"{stats['max_gate_weight']:.2f}",
                    "gent": f"{stats['mean_entropy']:.2f}",
                }

            if step % log_freq == 0:
                pbar.set_postfix(
                    {
                        "L": f"{total_loss.item():.3e}",
                        "res": f"{loss_dict['res'].item():.3e}",
                        "ic": f"{loss_dict['ic'].item():.3e}",
                        "bc": f"{loss_dict['bc'].item():.3e}",
                        **gate_info,
                    }
                )

        print(f"[OK] Training finished in {time.time() - start:.1f}s")
        return self.history

    def save_checkpoint(self, name: str = "model.pt") -> str:
        path = os.path.join(self.save_dir, name)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "history": self.history,
            },
            path,
        )
        print(f"[OK] Saved checkpoint: {path}")
        return path
