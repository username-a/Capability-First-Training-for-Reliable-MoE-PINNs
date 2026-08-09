"""
Gating network for Burger2D MoE-PINN.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatingNetwork2D(nn.Module):
    def __init__(
        self,
        in_dim: int = 3,
        num_experts: int = 4,
        hidden: int = 48,
        depth: int = 3,
        sparsity_p: float = 0.5,
        temperature: float = 0.8,
        init_gain: float = 0.5,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_experts = num_experts
        self.sparsity_p = sparsity_p
        self.temperature = temperature
        self.feature_dim = 25
        self.requires_expert_context = False
        self.recommended_inference_batch_size = 65536

        layers = [nn.Linear(self.feature_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        layers.append(nn.Linear(hidden, num_experts))
        self.mlp = nn.Sequential(*layers)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=init_gain)
                nn.init.zeros_(module.bias)

    def _expand_features(self, coords: torch.Tensor) -> torch.Tensor:
        x = coords[:, 0:1]
        y = coords[:, 1:2]
        t = coords[:, 2:3]
        r2 = x.square() + y.square()
        r = torch.sqrt(r2 + 1e-8)
        diag_p = x + y
        diag_m = x - y
        base = torch.cat([x, y, t], dim=-1)
        poly = torch.cat([x.square(), y.square(), t.square()], dim=-1)
        cross = torch.cat([x * y, x * t, y * t], dim=-1)
        geometry = torch.cat([diag_p, diag_m, r, r2], dim=-1)
        periodic_inputs = torch.cat([x, y, t, diag_p, diag_m, r2], dim=-1)
        periodic = torch.cat(
            [
                torch.sin(np.pi * periodic_inputs),
                torch.cos(np.pi * periodic_inputs),
            ],
            dim=-1,
        )
        return torch.cat([base, poly, cross, geometry, periodic], dim=-1)

    def forward(self, coords: torch.Tensor, expert_preds: torch.Tensor | None = None) -> torch.Tensor:
        del expert_preds
        feats = self._expand_features(coords)
        logits = self.mlp(feats)
        return F.softmax(logits / self.temperature, dim=-1)

    def sparsity_loss(self, coords: torch.Tensor, expert_preds: torch.Tensor | None = None) -> torch.Tensor:
        weights = self.forward(coords, expert_preds=expert_preds)
        sparse_penalty = torch.pow(weights.clamp(min=1e-8), self.sparsity_p)
        return sparse_penalty.sum(dim=-1).mean()

    def load_balance_loss(self, coords: torch.Tensor, expert_preds: torch.Tensor | None = None) -> torch.Tensor:
        weights = self.forward(coords, expert_preds=expert_preds)
        mean_load = weights.mean(dim=0)
        target = torch.full_like(mean_load, 1.0 / self.num_experts)
        return ((mean_load - target) ** 2).mean()

    def load_balance_stats(self, coords: torch.Tensor, expert_preds: torch.Tensor | None = None) -> dict[str, object]:
        with torch.no_grad():
            weights = self.forward(coords, expert_preds=expert_preds)
            routing = weights.argmax(dim=-1)
            counts = torch.bincount(routing, minlength=self.num_experts)
            entropy = -(weights * (weights + 1e-10).log()).sum(dim=-1).mean().item()
            return {
                "mean_entropy": entropy,
                "expert_load_frac": (counts.float() / counts.sum().clamp_min(1)).tolist(),
                "max_gate_weight": weights.max(dim=-1).values.mean().item(),
            }


class LocalContextGatingNetwork2D(GatingNetwork2D):
    def __init__(
        self,
        in_dim: int = 3,
        num_experts: int = 4,
        hidden: int = 64,
        depth: int = 3,
        sparsity_p: float = 0.5,
        temperature: float = 0.8,
        init_gain: float = 0.5,
        context_k: int = 16,
        time_scale: float = 1.5,
        context_query_batch_size: int = 1024,
    ):
        self.context_k = context_k
        self.time_scale = time_scale
        self.context_query_batch_size = context_query_batch_size
        super().__init__(
            in_dim=in_dim,
            num_experts=num_experts,
            hidden=hidden,
            depth=depth,
            sparsity_p=sparsity_p,
            temperature=temperature,
            init_gain=init_gain,
        )
        self.requires_expert_context = True
        self.recommended_inference_batch_size = 2048
        self.feature_dim = 25 + 5 * num_experts + 5
        layers = [nn.Linear(self.feature_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        layers.append(nn.Linear(hidden, num_experts))
        self.mlp = nn.Sequential(*layers)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=init_gain)
                nn.init.zeros_(module.bias)

    def _normalize_coords(self, coords: torch.Tensor) -> torch.Tensor:
        scale = torch.tensor([1.0, 1.0, self.time_scale], dtype=coords.dtype, device=coords.device)
        return coords * scale

    def _knn_lookup(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n_points = coords.shape[0]
        if n_points <= 1:
            empty_dist = torch.zeros(n_points, 0, dtype=coords.dtype, device=coords.device)
            empty_idx = torch.zeros(n_points, 0, dtype=torch.long, device=coords.device)
            return empty_dist, empty_idx

        normalized = self._normalize_coords(coords)
        k = min(self.context_k, max(1, n_points - 1))
        chunk_size = min(self.context_query_batch_size, n_points)
        all_distances = []
        all_indices = []
        global_idx = torch.arange(n_points, device=coords.device)

        for start in range(0, n_points, chunk_size):
            end = min(start + chunk_size, n_points)
            query = normalized[start:end]
            distances = torch.cdist(query, normalized)
            row_idx = torch.arange(end - start, device=coords.device)
            distances[row_idx, global_idx[start:end]] = 1e6
            knn_dist, knn_idx = torch.topk(distances, k=k, dim=1, largest=False)
            all_distances.append(knn_dist)
            all_indices.append(knn_idx)

        return torch.cat(all_distances, dim=0), torch.cat(all_indices, dim=0)

    def _context_features(self, coords: torch.Tensor, expert_preds: torch.Tensor | None) -> torch.Tensor:
        coords = coords.detach()
        n_points = coords.shape[0]
        if expert_preds is None:
            zeros = torch.zeros(n_points, 5 * self.num_experts + 5, dtype=coords.dtype, device=coords.device)
            return zeros

        expert_values = expert_preds.detach()
        expert_values = expert_values.squeeze(-1) if expert_values.dim() == 3 else expert_values
        if n_points <= 1:
            center_std = expert_values.std(dim=1, unbiased=False, keepdim=True)
            center_range = (expert_values.max(dim=1, keepdim=True).values - expert_values.min(dim=1, keepdim=True).values)
            zeros_neighbors = torch.zeros_like(expert_values)
            zeros_scalar = torch.zeros(n_points, 3, dtype=coords.dtype, device=coords.device)
            return torch.cat(
                [
                    expert_values,
                    zeros_neighbors,
                    zeros_neighbors,
                    zeros_neighbors,
                    zeros_neighbors,
                    center_std,
                    center_range,
                    zeros_scalar,
                ],
                dim=-1,
            )

        knn_dist, knn_idx = self._knn_lookup(coords)
        neighbor_values = expert_values[knn_idx]
        neighbor_mean = neighbor_values.mean(dim=1)
        neighbor_std = neighbor_values.std(dim=1, unbiased=False)
        local_contrast = (neighbor_values - expert_values.unsqueeze(1)).abs().mean(dim=1)
        local_slope = (neighbor_values - expert_values.unsqueeze(1)).abs() / knn_dist.unsqueeze(-1).clamp_min(1e-4)
        local_slope = local_slope.mean(dim=1)

        center_std = expert_values.std(dim=1, unbiased=False, keepdim=True)
        center_range = (
            expert_values.max(dim=1, keepdim=True).values - expert_values.min(dim=1, keepdim=True).values
        )
        neighbor_range = (
            neighbor_values.max(dim=2).values - neighbor_values.min(dim=2).values
        ).mean(dim=1, keepdim=True)
        mean_knn_dist = knn_dist.mean(dim=1, keepdim=True)
        min_knn_dist = knn_dist[:, :1]

        return torch.cat(
            [
                expert_values,
                neighbor_mean,
                neighbor_std,
                local_contrast,
                local_slope,
                center_std,
                center_range,
                neighbor_range,
                mean_knn_dist,
                min_knn_dist,
            ],
            dim=-1,
        )

    def forward(self, coords: torch.Tensor, expert_preds: torch.Tensor | None = None) -> torch.Tensor:
        base_feats = self._expand_features(coords)
        context_feats = self._context_features(coords, expert_preds)
        feats = torch.cat([base_feats, context_feats], dim=-1)
        logits = self.mlp(feats)
        return F.softmax(logits / self.temperature, dim=-1)


class LocalConvContextGatingNetwork2D(LocalContextGatingNetwork2D):
    def __init__(
        self,
        in_dim: int = 3,
        num_experts: int = 4,
        hidden: int = 64,
        depth: int = 3,
        sparsity_p: float = 0.5,
        temperature: float = 0.8,
        init_gain: float = 0.5,
        context_k: int = 16,
        time_scale: float = 1.5,
        context_query_batch_size: int = 1024,
        conv_hidden: int = 24,
    ):
        self.conv_hidden = conv_hidden
        super().__init__(
            in_dim=in_dim,
            num_experts=num_experts,
            hidden=hidden,
            depth=depth,
            sparsity_p=sparsity_p,
            temperature=temperature,
            init_gain=init_gain,
            context_k=context_k,
            time_scale=time_scale,
            context_query_batch_size=context_query_batch_size,
        )
        self.recommended_inference_batch_size = 1536
        token_dim = 4 + 2 * num_experts
        context_dim = num_experts + 2 * conv_hidden + 4
        self.feature_dim = 25 + context_dim
        self.context_proj = nn.Conv1d(token_dim, conv_hidden, kernel_size=1)
        self.context_depthwise = nn.Conv1d(
            conv_hidden,
            conv_hidden,
            kernel_size=3,
            padding=1,
            groups=conv_hidden,
        )
        self.context_pointwise = nn.Conv1d(conv_hidden, conv_hidden, kernel_size=1)
        self.context_norm = nn.LayerNorm(context_dim)
        layers = [nn.Linear(self.feature_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        layers.append(nn.Linear(hidden, num_experts))
        self.mlp = nn.Sequential(*layers)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=init_gain)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_uniform_(module.weight, a=np.sqrt(5))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _context_features(self, coords: torch.Tensor, expert_preds: torch.Tensor | None) -> torch.Tensor:
        coords = coords.detach()
        n_points = coords.shape[0]
        context_dim = self.num_experts + 2 * self.conv_hidden + 4
        if expert_preds is None:
            return torch.zeros(n_points, context_dim, dtype=coords.dtype, device=coords.device)

        expert_values = expert_preds.detach()
        expert_values = expert_values.squeeze(-1) if expert_values.dim() == 3 else expert_values
        if n_points <= 1:
            zeros_conv = torch.zeros(n_points, 2 * self.conv_hidden, dtype=coords.dtype, device=coords.device)
            zeros_stats = torch.zeros(n_points, 4, dtype=coords.dtype, device=coords.device)
            return self.context_norm(torch.cat([expert_values, zeros_conv, zeros_stats], dim=-1))

        knn_dist, knn_idx = self._knn_lookup(coords)
        neighbor_coords = coords[knn_idx]
        rel_coords = self._normalize_coords(neighbor_coords - coords.unsqueeze(1))
        neighbor_values = expert_values[knn_idx]
        delta_values = neighbor_values - expert_values.unsqueeze(1)

        token_features = torch.cat([rel_coords, knn_dist.unsqueeze(-1), neighbor_values, delta_values], dim=-1)
        token_features = token_features.permute(0, 2, 1)

        hidden = F.gelu(self.context_proj(token_features))
        hidden = hidden + F.gelu(self.context_pointwise(self.context_depthwise(hidden)))
        pooled_mean = hidden.mean(dim=-1)
        pooled_max = hidden.max(dim=-1).values

        delta_abs = delta_values.abs()
        local_stats = torch.cat(
            [
                knn_dist.mean(dim=1, keepdim=True),
                knn_dist[:, :1],
                delta_abs.mean(dim=(1, 2)).unsqueeze(-1),
                delta_abs.amax(dim=(1, 2)).unsqueeze(-1),
            ],
            dim=-1,
        )
        context = torch.cat([expert_values, pooled_mean, pooled_max, local_stats], dim=-1)
        return self.context_norm(context)


class RotationLayerGate2D(nn.Module):
    def __init__(
        self,
        in_dim: int = 3,
        hidden: int = 48,
        depth: int = 3,
        context_dim: int = 4,
        num_angles: int = 8,
        confidence_threshold: float = 0.35,
        adaptive_threshold_weight: float = 0.5,
        threshold_slope: float = 20.0,
        activation_floor: float = 0.08,
        context_focus_center: float = 0.70,
        context_focus_slope: float = 5.5,
        context_focus_power: float = 1.4,
        init_gain: float = 0.5,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.feature_dim = 25
        self.context_dim = context_dim
        self.num_angles = num_angles
        self.confidence_threshold = confidence_threshold
        self.adaptive_threshold_weight = adaptive_threshold_weight
        self.threshold_slope = threshold_slope
        self.activation_floor = activation_floor
        self.context_focus_center = context_focus_center
        self.context_focus_slope = context_focus_slope
        self.context_focus_power = context_focus_power
        self.recommended_inference_batch_size = 65536

        self.input_dim = self.feature_dim + self.context_dim
        layers = [nn.Linear(self.input_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        self.backbone = nn.Sequential(*layers)
        self.angle_head = nn.Linear(hidden, num_angles)
        self.threshold_head = nn.Linear(hidden, 1)

        angle_basis = torch.linspace(-np.pi, np.pi, steps=num_angles + 1)[:-1]
        self.register_buffer("angle_basis", angle_basis)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=init_gain)
                nn.init.zeros_(module.bias)

    def _expand_features(self, coords: torch.Tensor) -> torch.Tensor:
        x = coords[:, 0:1]
        y = coords[:, 1:2]
        t = coords[:, 2:3]
        r2 = x.square() + y.square()
        r = torch.sqrt(r2 + 1e-8)
        diag_p = x + y
        diag_m = x - y
        base = torch.cat([x, y, t], dim=-1)
        poly = torch.cat([x.square(), y.square(), t.square()], dim=-1)
        cross = torch.cat([x * y, x * t, y * t], dim=-1)
        geometry = torch.cat([diag_p, diag_m, r, r2], dim=-1)
        periodic_inputs = torch.cat([x, y, t, diag_p, diag_m, r2], dim=-1)
        periodic = torch.cat(
            [
                torch.sin(np.pi * periodic_inputs),
                torch.cos(np.pi * periodic_inputs),
            ],
            dim=-1,
        )
        return torch.cat([base, poly, cross, geometry, periodic], dim=-1)

    def _context_features(self, coords: torch.Tensor, context: torch.Tensor | None) -> torch.Tensor:
        if self.context_dim <= 0:
            return torch.zeros(coords.shape[0], 0, dtype=coords.dtype, device=coords.device)
        if context is None:
            return torch.zeros(coords.shape[0], self.context_dim, dtype=coords.dtype, device=coords.device)
        if context.shape[-1] != self.context_dim:
            raise ValueError(
                f"Rotation context dim mismatch: expected {self.context_dim}, got {context.shape[-1]}"
            )
        return context

    def _forward_logits(
        self,
        coords: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self._expand_features(coords)
        feats = torch.cat([feats, self._context_features(coords, context)], dim=-1)
        hidden = self.backbone(feats)
        return self.angle_head(hidden), self.threshold_head(hidden).squeeze(-1)

    def rotation_state(
        self,
        coords: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        context_feats = self._context_features(coords, context)
        angle_logits, threshold_logits = self._forward_logits(coords, context=context)
        angle_probs = F.softmax(angle_logits, dim=-1)
        cos_mean = torch.sum(angle_probs * torch.cos(self.angle_basis)[None, :], dim=-1)
        sin_mean = torch.sum(angle_probs * torch.sin(self.angle_basis)[None, :], dim=-1)
        rotation_angle = torch.atan2(sin_mean, cos_mean)
        max_prob = angle_probs.max(dim=-1).values
        entropy = -(angle_probs * angle_probs.clamp_min(1e-8).log()).sum(dim=-1)
        entropy = entropy / np.log(max(2, self.num_angles))
        concentration = 1.0 - entropy
        adaptive_threshold = torch.sigmoid(threshold_logits)
        threshold = (1.0 - self.adaptive_threshold_weight) * self.confidence_threshold + (
            self.adaptive_threshold_weight * adaptive_threshold
        )
        raw_activation = torch.sigmoid((concentration - threshold) * self.threshold_slope)
        activation = self.activation_floor + (1.0 - self.activation_floor) * raw_activation
        if self.context_dim >= 3 and context_feats.shape[-1] >= 3:
            grad_mag = context_feats[:, 2].clamp_min(0.0)
            context_focus = torch.sigmoid((grad_mag - self.context_focus_center) * self.context_focus_slope)
            context_focus = context_focus.pow(self.context_focus_power)
            activation = activation * context_focus
        else:
            context_focus = torch.ones_like(raw_activation)
        return {
            "angle_probs": angle_probs,
            "rotation_angle": rotation_angle,
            "activation": activation,
            "raw_activation": raw_activation,
            "context_focus": context_focus,
            "max_prob": max_prob,
            "concentration": concentration,
            "threshold": threshold,
            "entropy": entropy,
        }

    def rotate(
        self,
        coords: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        state = self.rotation_state(coords, context=context)
        theta = state["rotation_angle"]
        activation = state["activation"].unsqueeze(-1)
        x = coords[:, 0:1]
        y = coords[:, 1:2]
        t = coords[:, 2:3]
        cos_theta = torch.cos(theta).unsqueeze(-1)
        sin_theta = torch.sin(theta).unsqueeze(-1)
        rot_x = x * cos_theta + y * sin_theta
        rot_y = -x * sin_theta + y * cos_theta
        rotated = torch.cat([rot_x, rot_y, t], dim=-1)
        blended = activation * rotated + (1.0 - activation) * coords
        state["rotated_coords"] = rotated
        state["blended_coords"] = blended
        return blended, state
