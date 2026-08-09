"""
MoE-PINN containers for Burger2D.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from Burger2D.core.attribute_experts import (
    AnisotropyDirectionalExpert2D,
    CurvatureWaveExpert2D,
    MixedCurvatureWaveExpert2D,
    MixedNormalGradientExpert2D,
    NormalGradientExpert2D,
    SmoothBackgroundExpert2D,
)
from Burger2D.core.complex_experts import ComplexFrameDirectionalShockExpert2D
from Burger2D.core.experts import (
    DirectionalShockExpert2D,
    ExpertNetwork,
    IsoShockExpert2D,
    LegacyDirectionalShockExpert2D,
    SmoothExpert2D,
    VortexExpert2D,
    WavePacketExpert2D,
)
from Burger2D.core.gating import (
    GatingNetwork2D,
    LocalContextGatingNetwork2D,
    LocalConvContextGatingNetwork2D,
    RotationLayerGate2D,
)
from Burger2D.core.models import VanillaPINN


class MoEPINN(nn.Module):
    def __init__(
        self,
        experts: List[ExpertNetwork],
        gating: GatingNetwork2D,
        sparsity_weight: float = 1e-3,
        balance_weight: float = 1e-2,
        expert_names: Optional[List[str]] = None,
    ):
        super().__init__()
        assert len(experts) == gating.num_experts
        self.experts = nn.ModuleList(experts)
        self.gating = gating
        self.sparsity_weight = sparsity_weight
        self.balance_weight = balance_weight
        self.num_experts = len(experts)
        self.expert_names = expert_names or [f"expert_{i}" for i in range(len(experts))]

    def compute_gate_weights(
        self,
        coords: torch.Tensor,
        expert_preds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if expert_preds is None and getattr(self.gating, "requires_expert_context", False):
            expert_preds = self.get_expert_predictions(coords)
        return self.gating(coords, expert_preds=expert_preds)

    def load_balance_loss(self, coords: torch.Tensor) -> torch.Tensor:
        expert_preds = None
        if getattr(self.gating, "requires_expert_context", False):
            expert_preds = self.get_expert_predictions(coords)
        return self.gating.load_balance_loss(coords, expert_preds=expert_preds)

    def load_balance_stats(self, coords: torch.Tensor) -> Dict[str, object]:
        batch_size = int(getattr(self, "inference_batch_size", coords.shape[0]))
        if coords.shape[0] <= batch_size:
            expert_preds = None
            if getattr(self.gating, "requires_expert_context", False):
                expert_preds = self.get_expert_predictions(coords)
            return self.gating.load_balance_stats(coords, expert_preds=expert_preds)

        with torch.no_grad():
            total_counts = torch.zeros(self.num_experts, dtype=torch.float32, device=coords.device)
            entropy_sum = torch.zeros((), dtype=torch.float32, device=coords.device)
            max_weight_sum = torch.zeros((), dtype=torch.float32, device=coords.device)
            total_points = 0
            for start in range(0, coords.shape[0], batch_size):
                chunk = coords[start:start + batch_size]
                expert_preds = None
                if getattr(self.gating, "requires_expert_context", False):
                    expert_preds = self.get_expert_predictions(chunk)
                weights = self.compute_gate_weights(chunk, expert_preds=expert_preds)
                routing = weights.argmax(dim=-1)
                counts = torch.bincount(routing, minlength=self.num_experts).float()
                entropy = -(weights * (weights + 1e-10).log()).sum(dim=-1)
                total_counts += counts
                entropy_sum += entropy.sum()
                max_weight_sum += weights.max(dim=-1).values.sum()
                total_points += chunk.shape[0]
            return {
                "mean_entropy": float((entropy_sum / max(total_points, 1)).item()),
                "expert_load_frac": (total_counts / total_counts.sum().clamp_min(1.0)).tolist(),
                "max_gate_weight": float((max_weight_sum / max(total_points, 1)).item()),
            }

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        expert_preds = torch.stack([expert(coords) for expert in self.experts], dim=1)
        gate_weights = self.compute_gate_weights(coords, expert_preds=expert_preds)
        return torch.einsum("ne,neo->no", gate_weights, expert_preds)

    def forward_with_sparsity(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expert_preds = torch.stack([expert(coords) for expert in self.experts], dim=1)
        gate_weights = self.compute_gate_weights(coords, expert_preds=expert_preds)
        pred = torch.einsum("ne,neo->no", gate_weights, expert_preds)
        sparse_loss = self.gating.sparsity_loss(coords, expert_preds=expert_preds) * self.sparsity_weight
        return pred, sparse_loss

    def get_gate_weights(self, coords: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.compute_gate_weights(coords)

    def get_expert_predictions(self, coords: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.stack([expert(coords) for expert in self.experts], dim=1)

    def count_parameters(self) -> Dict[str, int]:
        gating_params = sum(param.numel() for param in self.gating.parameters())
        expert_params = [sum(param.numel() for param in expert.parameters()) for expert in self.experts]
        return {
            "gating": gating_params,
            "experts": expert_params,
            "experts_total": sum(expert_params),
            "total": gating_params + sum(expert_params),
        }


class ResidualMoEPINN(MoEPINN):
    def __init__(
        self,
        base_model: nn.Module,
        experts: List[ExpertNetwork],
        gating: GatingNetwork2D,
        correction_scale: float = 0.4,
        rotation_layer: nn.Module | None = None,
        rotation_route_adapter: nn.Module | None = None,
        rotation_route_scale: float = 0.0,
        rotation_target_experts: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(experts=experts, gating=gating, **kwargs)
        self.base_model = base_model
        self.correction_scale = correction_scale
        self.rotation_layer = rotation_layer
        self.rotation_route_adapter = rotation_route_adapter
        self.rotation_route_scale = rotation_route_scale
        self.rotation_target_experts = set(rotation_target_experts or [])

    def _rotation_context(self, coords: torch.Tensor) -> torch.Tensor | None:
        if self.rotation_layer is None or self.base_model is None:
            return None
        coords_ctx = coords.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            base_pred = self.base_model(coords_ctx)
            grads = torch.autograd.grad(
                base_pred.sum(),
                coords_ctx,
                create_graph=False,
                retain_graph=False,
            )[0]
        grad_xy = grads[:, :2]
        grad_t = grads[:, 2:3]
        grad_mag = torch.sqrt(grad_xy.square().sum(dim=1, keepdim=True) + 1e-8)
        context = torch.cat([grad_xy, grad_mag, grad_t], dim=1)
        return context.detach()

    def get_rotation_state(self, coords: torch.Tensor) -> dict[str, torch.Tensor] | None:
        if self.rotation_layer is None:
            return None
        rotation_context = self._rotation_context(coords)
        _, state = self.rotation_layer.rotate(coords, context=rotation_context)
        if self.rotation_route_adapter is not None:
            route_feat = torch.cat(
                [
                    torch.cos(state["rotation_angle"]).unsqueeze(-1),
                    torch.sin(state["rotation_angle"]).unsqueeze(-1),
                    state["activation"].unsqueeze(-1),
                    state["concentration"].unsqueeze(-1),
                    state["max_prob"].unsqueeze(-1),
                ],
                dim=-1,
            )
            state["route_bias_logits"] = self.rotation_route_adapter(route_feat)
        return state

    def compute_gate_weights(
        self,
        coords: torch.Tensor,
        expert_preds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if expert_preds is None and getattr(self.gating, "requires_expert_context", False):
            expert_preds = self.get_expert_predictions(coords)
        base_weights = self.gating(coords, expert_preds=expert_preds)
        if self.rotation_layer is None or self.rotation_route_adapter is None:
            return base_weights
        rot_state = self.get_rotation_state(coords)
        if rot_state is None or "route_bias_logits" not in rot_state:
            return base_weights
        adjusted_logits = base_weights.clamp_min(1e-8).log() + self.rotation_route_scale * rot_state["route_bias_logits"]
        return torch.softmax(adjusted_logits, dim=-1)

    def _expert_input_coords(self, coords: torch.Tensor) -> list[torch.Tensor]:
        if self.rotation_layer is None or not self.rotation_target_experts:
            return [coords] * len(self.experts)
        rotation_context = self._rotation_context(coords)
        rotated_coords, _ = self.rotation_layer.rotate(coords, context=rotation_context)
        inputs: list[torch.Tensor] = []
        for expert_name in self.expert_names:
            if expert_name in self.rotation_target_experts:
                inputs.append(rotated_coords)
            else:
                inputs.append(coords)
        return inputs

    def get_expert_corrections(self, coords: torch.Tensor) -> torch.Tensor:
        expert_inputs = self._expert_input_coords(coords)
        return torch.stack([expert(expert_input) for expert, expert_input in zip(self.experts, expert_inputs)], dim=1)

    def get_expert_predictions(self, coords: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            base = self.base_model(coords).unsqueeze(1)
            corr = self.get_expert_corrections(coords)
            return base + self.correction_scale * corr

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        base = self.base_model(coords)
        corr = self.get_expert_corrections(coords)
        branch_preds = base.unsqueeze(1) + self.correction_scale * corr
        gate_weights = self.compute_gate_weights(coords, expert_preds=branch_preds)
        mixed_corr = torch.einsum("ne,neo->no", gate_weights, corr)
        return base + self.correction_scale * mixed_corr

    def count_parameters(self) -> Dict[str, int]:
        counts = super().count_parameters()
        base_params = sum(param.numel() for param in self.base_model.parameters())
        counts["base"] = base_params
        rotation_params = 0
        if self.rotation_layer is not None:
            rotation_params = sum(param.numel() for param in self.rotation_layer.parameters())
        rotation_route_params = 0
        if self.rotation_route_adapter is not None:
            rotation_route_params = sum(param.numel() for param in self.rotation_route_adapter.parameters())
        counts["rotation"] = rotation_params
        counts["rotation_route"] = rotation_route_params
        counts["total"] += base_params
        counts["total"] += rotation_params
        counts["total"] += rotation_route_params
        return counts


def build_burgers2d_moe(
    *,
    include_vortex: bool = False,
    sparsity_weight: float = 1e-3,
    balance_weight: float = 1e-2,
    gate_temperature: float = 0.8,
    directional_expert_variant: str = "hybrid",
    wave_expert_variant: str = "base",
    expert_layout_variant: str = "categorical",
    attribute_expert_variant: str = "base",
    gate_variant: str = "pointwise",
    rotation_variant: str = "none",
    exclude_experts: tuple[str, ...] = (),
    extra_experts: tuple[str, ...] = (),
) -> ResidualMoEPINN:
    base_model = VanillaPINN(
        in_dim=3,
        out_dim=1,
        hidden=96,
        depth=5,
        activation="tanh",
        output_transform="burgers2d_hard_icbc",
    )

    if directional_expert_variant == "hybrid":
        directional_expert = DirectionalShockExpert2D(
            in_dim=3,
            out_dim=1,
            hidden=96,
            depth=5,
            output_transform="burgers2d_residual_icbc",
        )
    elif directional_expert_variant == "legacy":
        directional_expert = LegacyDirectionalShockExpert2D(
            in_dim=3,
            out_dim=1,
            hidden=96,
            depth=5,
            output_transform="burgers2d_residual_icbc",
        )
    elif directional_expert_variant == "complex_frame":
        directional_expert = ComplexFrameDirectionalShockExpert2D(
            in_dim=3,
            out_dim=1,
            hidden=96,
            depth=5,
            output_transform="burgers2d_residual_icbc",
        )
    else:
        raise ValueError(f"Unknown directional expert variant: {directional_expert_variant}")

    if expert_layout_variant == "categorical":
        if wave_expert_variant == "base":
            wave_expert: ExpertNetwork = WavePacketExpert2D(
                in_dim=3,
                out_dim=1,
                hidden=72,
                depth=4,
                output_transform="burgers2d_residual_icbc",
            )
        elif wave_expert_variant == "mixed":
            wave_expert = MixedCurvatureWaveExpert2D(
                in_dim=3,
                out_dim=1,
                hidden=80,
                depth=4,
                output_transform="burgers2d_residual_icbc",
            )
        elif wave_expert_variant == "mixed_lite":
            wave_expert = MixedCurvatureWaveExpert2D(
                in_dim=3,
                out_dim=1,
                hidden=64,
                depth=4,
                output_transform="burgers2d_residual_icbc",
                selector_temperature=0.95,
                shortcut_scale_init=0.08,
                branch_output_scale_init=0.72,
            )
        else:
            raise ValueError(f"Unknown wave expert variant: {wave_expert_variant}")
        experts: List[ExpertNetwork] = [
            SmoothExpert2D(
                in_dim=3,
                out_dim=1,
                hidden=48,
                depth=3,
                output_transform="burgers2d_residual_icbc",
            ),
            IsoShockExpert2D(
                in_dim=3,
                out_dim=1,
                hidden=96,
                depth=5,
                output_transform="burgers2d_residual_icbc",
            ),
            directional_expert,
            wave_expert,
        ]
        expert_names = ["smooth", "iso_shock", "directional_shock", "wave"]
        directional_branch_name = "directional_shock"
    elif expert_layout_variant == "attribute":
        if attribute_expert_variant == "base":
            normal_gradient_expert: ExpertNetwork = NormalGradientExpert2D(
                in_dim=3,
                out_dim=1,
                hidden=88,
                depth=5,
                output_transform="burgers2d_residual_icbc",
            )
            curvature_wave_expert: ExpertNetwork = CurvatureWaveExpert2D(
                in_dim=3,
                out_dim=1,
                hidden=80,
                depth=4,
                output_transform="burgers2d_residual_icbc",
            )
        elif attribute_expert_variant == "normal_mixed":
            normal_gradient_expert = MixedNormalGradientExpert2D(
                in_dim=3,
                out_dim=1,
                hidden=88,
                depth=5,
                output_transform="burgers2d_residual_icbc",
            )
            curvature_wave_expert = CurvatureWaveExpert2D(
                in_dim=3,
                out_dim=1,
                hidden=80,
                depth=4,
                output_transform="burgers2d_residual_icbc",
            )
        elif attribute_expert_variant == "wave_mixed":
            normal_gradient_expert = NormalGradientExpert2D(
                in_dim=3,
                out_dim=1,
                hidden=88,
                depth=5,
                output_transform="burgers2d_residual_icbc",
            )
            curvature_wave_expert = MixedCurvatureWaveExpert2D(
                in_dim=3,
                out_dim=1,
                hidden=80,
                depth=4,
                output_transform="burgers2d_residual_icbc",
            )
        elif attribute_expert_variant == "mixed":
            normal_gradient_expert = MixedNormalGradientExpert2D(
                in_dim=3,
                out_dim=1,
                hidden=88,
                depth=5,
                output_transform="burgers2d_residual_icbc",
            )
            curvature_wave_expert = MixedCurvatureWaveExpert2D(
                in_dim=3,
                out_dim=1,
                hidden=80,
                depth=4,
                output_transform="burgers2d_residual_icbc",
            )
        else:
            raise ValueError(f"Unknown attribute expert variant: {attribute_expert_variant}")
        experts = [
            SmoothBackgroundExpert2D(
                in_dim=3,
                out_dim=1,
                hidden=48,
                depth=3,
                output_transform="burgers2d_residual_icbc",
            ),
            normal_gradient_expert,
            AnisotropyDirectionalExpert2D(
                in_dim=3,
                out_dim=1,
                hidden=96,
                depth=5,
                output_transform="burgers2d_residual_icbc",
            ),
            curvature_wave_expert,
        ]
        expert_names = ["smooth", "normal_gradient", "anisotropy_directional", "curvature_wave"]
        directional_branch_name = "anisotropy_directional"
    else:
        raise ValueError(f"Unknown expert layout variant: {expert_layout_variant}")

    if include_vortex:
        experts.append(
            VortexExpert2D(
                in_dim=3,
                out_dim=1,
                hidden=72,
                depth=4,
                output_transform="burgers2d_residual_icbc",
            )
        )
        expert_names.append("vortex")

    if exclude_experts:
        keep = [(e, n) for e, n in zip(experts, expert_names) if n not in exclude_experts]
        if not keep:
            raise ValueError(f"exclude_experts={exclude_experts} removes all experts")
        experts = [e for e, _ in keep]
        expert_names = [n for _, n in keep]

    if extra_experts:
        extra_map = {
            "vortex": (
                VortexExpert2D,
                dict(in_dim=3, out_dim=1, hidden=72, depth=4, output_transform="burgers2d_residual_icbc"),
            ),
            "wave2": (
                WavePacketExpert2D,
                dict(in_dim=3, out_dim=1, hidden=72, depth=4, output_transform="burgers2d_residual_icbc"),
            ),
            "smooth2": (
                SmoothExpert2D,
                dict(in_dim=3, out_dim=1, hidden=48, depth=3, output_transform="burgers2d_residual_icbc"),
            ),
        }
        for name in extra_experts:
            if name not in extra_map:
                raise ValueError(f"unknown extra expert: {name}")
            cls, kwargs = extra_map[name]
            experts.append(cls(**kwargs))
            expert_names.append(name)

    if gate_variant == "pointwise":
        gating = GatingNetwork2D(
            in_dim=3,
            num_experts=len(experts),
            hidden=48,
            depth=3,
            sparsity_p=0.5,
            temperature=gate_temperature,
        )
    elif gate_variant == "local_knn":
        gating = LocalContextGatingNetwork2D(
            in_dim=3,
            num_experts=len(experts),
            hidden=64,
            depth=3,
            sparsity_p=0.5,
            temperature=gate_temperature,
            context_k=16,
            time_scale=1.5,
        )
    elif gate_variant == "local_conv":
        gating = LocalConvContextGatingNetwork2D(
            in_dim=3,
            num_experts=len(experts),
            hidden=64,
            depth=3,
            sparsity_p=0.5,
            temperature=gate_temperature,
            context_k=16,
            time_scale=1.5,
            context_query_batch_size=1024,
            conv_hidden=24,
        )
    else:
        raise ValueError(f"Unknown gate variant: {gate_variant}")

    rotation_layer = None
    rotation_route_adapter = None
    rotation_route_scale = 0.0
    rotation_target_experts: list[str] = []
    if rotation_variant == "none":
        rotation_layer = None
    elif rotation_variant == "complex_high_threshold":
        rotation_layer = RotationLayerGate2D(
            in_dim=3,
            hidden=48,
            depth=3,
            num_angles=8,
            confidence_threshold=0.35,
            adaptive_threshold_weight=0.5,
            threshold_slope=22.0,
            activation_floor=0.08,
            context_focus_center=0.70,
            context_focus_slope=5.5,
            context_focus_power=1.4,
        )
        rotation_route_adapter = nn.Sequential(
            nn.Linear(5, 24),
            nn.Tanh(),
            nn.Linear(24, len(experts)),
        )
        for module in rotation_route_adapter.modules():
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)
        rotation_route_scale = 0.65
        if expert_layout_variant == "categorical":
            rotation_target_experts = ["directional_shock", "wave"]
        else:
            rotation_target_experts = ["anisotropy_directional", "curvature_wave"]
    elif rotation_variant == "complex_low_threshold_focus":
        rotation_layer = RotationLayerGate2D(
            in_dim=3,
            hidden=48,
            depth=3,
            num_angles=8,
            confidence_threshold=0.05,
            adaptive_threshold_weight=0.15,
            threshold_slope=16.0,
            activation_floor=0.0,
            context_focus_center=0.72,
            context_focus_slope=5.8,
            context_focus_power=1.65,
        )
        rotation_route_adapter = nn.Sequential(
            nn.Linear(5, 24),
            nn.Tanh(),
            nn.Linear(24, len(experts)),
        )
        for module in rotation_route_adapter.modules():
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)
        rotation_route_scale = 0.75
        if expert_layout_variant == "categorical":
            rotation_target_experts = ["directional_shock", "wave"]
        else:
            rotation_target_experts = ["anisotropy_directional", "curvature_wave"]
    elif rotation_variant == "complex_low_threshold_sparse":
        rotation_layer = RotationLayerGate2D(
            in_dim=3,
            hidden=48,
            depth=3,
            num_angles=8,
            confidence_threshold=0.08,
            adaptive_threshold_weight=0.10,
            threshold_slope=14.0,
            activation_floor=0.0,
            context_focus_center=0.76,
            context_focus_slope=6.2,
            context_focus_power=1.90,
        )
        rotation_route_adapter = nn.Sequential(
            nn.Linear(5, 24),
            nn.Tanh(),
            nn.Linear(24, len(experts)),
        )
        for module in rotation_route_adapter.modules():
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)
        rotation_route_scale = 0.55
        if expert_layout_variant == "categorical":
            rotation_target_experts = ["directional_shock", "wave"]
        else:
            rotation_target_experts = ["anisotropy_directional", "curvature_wave"]
    else:
        raise ValueError(f"Unknown rotation variant: {rotation_variant}")
    model = ResidualMoEPINN(
        base_model=base_model,
        experts=experts,
        gating=gating,
        correction_scale=0.35,
        rotation_layer=rotation_layer,
        rotation_route_adapter=rotation_route_adapter,
        rotation_route_scale=rotation_route_scale,
        rotation_target_experts=rotation_target_experts,
        sparsity_weight=sparsity_weight,
        balance_weight=balance_weight,
        expert_names=expert_names,
    )
    model.expert_layout_variant = expert_layout_variant
    model.directional_branch_name = directional_branch_name
    model.directional_expert_variant = directional_expert_variant
    model.wave_expert_variant = wave_expert_variant
    model.attribute_expert_variant = attribute_expert_variant
    model.gate_variant = gate_variant
    model.rotation_variant = rotation_variant
    model.inference_batch_size = getattr(gating, "recommended_inference_batch_size", 65536)
    return model
