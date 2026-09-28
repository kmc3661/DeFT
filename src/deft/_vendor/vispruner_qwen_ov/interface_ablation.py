"""Analysis-only interface/sensitivity component ablations.

This state keeps the confirmed M->K@L9 schedule and changes one selection
component at a time.  It is not a paper method: exact central-difference
sensitivity is intentionally retained as a diagnostic control and its cost is
reported.
"""
from __future__ import annotations

import inspect
import math
import os
from types import MethodType
from typing import Optional

import torch

from .progressive_pruning import ProgressiveConfig, ProgressivePruningState


_STAGE_VARIANTS = {
    "encoder_topm",
    "encoder_feature_diverse",
    "interface_topm",
    "interface_feature_diverse",
    "hybrid_feature_diverse",
    "union_feature_diverse",
    "union_balanced_feature_diverse",
    "mixed_feature_diverse",
    "routed_feature_diverse",
    "interface_sads",
}
_FINAL_VARIANTS = {
    "encoder_only",
    "encoder_pmi",
    "mixed_pmi",
    "interface_only",
    "interface_pmi",
    "interface_pmi_cls_stage",
    "interface_pmi_cls_guarded",
    "encoder_pmi_cls_guarded",
    "encoder_pmi_cls_mix",
    "encoder_pmi_interface_guarded",
    "encoder_pmi_interface_endpoint_guarded",
    "encoder_pmi_functional_aggregate",
    "encoder_pmi_functional_role",
    "encoder_pmi_functional_user",
    "encoder_pmi_functional_user_cls_stage",
    "majority_support_functional_user_cls_stage",
    "encoder_pmi_functional_user_guarded",
    "encoder_pmi_cls_stage",
    "encoder_pmi_cls_final",
    "interface_pmi_cls_quota",
    "interface_pmi_cls_routed",
    "encoder_pmi_cls_quota",
    "encoder_pmi_cls_routed",
    "routed_pmi",
    "text_only",
    "pmi_only",
    "text_pmi_reserve_blend",
}


class InterfaceComponentAblationState(ProgressivePruningState):
    """Orthogonal L0 candidate and L9 prior/query diagnostic controls."""

    def __init__(
        self,
        config: ProgressiveConfig,
        stage_variant: Optional[str] = None,
        final_variant: Optional[str] = None,
    ) -> None:
        self.interface_stage_variant = str(
            stage_variant
            or os.environ.get("VISPRUNER_INTERFACE_STAGE", "encoder_feature_diverse")
        ).strip().lower()
        self.interface_final_variant = str(
            final_variant
            or os.environ.get("VISPRUNER_INTERFACE_FINAL", "encoder_pmi")
        ).strip().lower()
        if self.interface_stage_variant not in _STAGE_VARIANTS:
            raise ValueError(
                f"unknown interface stage variant {self.interface_stage_variant!r}; "
                f"expected {sorted(_STAGE_VARIANTS)}"
            )
        if self.interface_final_variant not in _FINAL_VARIANTS:
            raise ValueError(
                f"unknown interface final variant {self.interface_final_variant!r}; "
                f"expected {sorted(_FINAL_VARIANTS)}"
            )

        # Keep every schedule/cost setting supplied by the confirmed method.
        config.vision_stage_score_mode = (
            "encoder_topk"
            if self.interface_stage_variant in {"encoder_topm", "interface_topm"}
            else "feature_diverse"
        )
        config.vision_score_mode = {
            "encoder_only": "encoder_topk",
            "encoder_pmi": "encoder_pmi_add_topk",
            "mixed_pmi": "encoder_pmi_add_topk",
            "interface_only": "encoder_topk",
            "interface_pmi": "encoder_pmi_add_topk",
            "interface_pmi_cls_stage": "encoder_pmi_add_topk",
            "interface_pmi_cls_guarded": "encoder_pmi_anchor_quota_topk",
            "encoder_pmi_cls_guarded": "encoder_pmi_anchor_quota_topk",
            "encoder_pmi_cls_mix": "encoder_pmi_anchor_quota_topk",
            "encoder_pmi_interface_guarded": "encoder_pmi_anchor_quota_topk",
            "encoder_pmi_interface_endpoint_guarded": "encoder_pmi_anchor_quota_topk",
            "encoder_pmi_functional_aggregate": "functional_coreset_topk",
            "encoder_pmi_functional_role": "functional_role_coreset_topk",
            "encoder_pmi_functional_user": "functional_user_coreset_topk",
            "encoder_pmi_functional_user_cls_stage": "functional_user_coreset_topk",
            "majority_support_functional_user_cls_stage": "majority_support_adaptive_functional_user_topk",
            "encoder_pmi_functional_user_guarded": "functional_user_guarded_topk",
            "encoder_pmi_cls_stage": "encoder_pmi_add_topk",
            "encoder_pmi_cls_final": "encoder_pmi_anchor_quota_topk",
            "interface_pmi_cls_quota": "encoder_pmi_anchor_quota_topk",
            "interface_pmi_cls_routed": "encoder_pmi_anchor_quota_topk",
            "encoder_pmi_cls_quota": "encoder_pmi_anchor_quota_topk",
            "encoder_pmi_cls_routed": "encoder_pmi_anchor_quota_topk",
            "routed_pmi": "encoder_pmi_add_topk",
            "text_only": "text_topk",
            "pmi_only": "pmi_topk",
            "text_pmi_reserve_blend": "text_pmi_reserve_blend_topk",
        }[self.interface_final_variant]
        super().__init__(config)
        self._interface_sensitivity_by_branch: dict[str, torch.Tensor] = {}
        self._interface_projected_by_branch: dict[str, torch.Tensor] = {}
        # Raw projector row counts remain stable even when an adapter later
        # aligns the score to its packed LLM-token order (e.g. LLaVA-NeXT).
        self._interface_raw_target_n_by_branch: dict[str, int] = {}
        self._interface_preactivation_by_branch: dict[str, torch.Tensor] = {}
        self._interface_weight_factor_by_branch: dict[str, torch.Tensor] = {}
        self._interface_zero_output_by_branch: dict[str, torch.Tensor] = {}
        self._interface_stage_encoder_score: Optional[torch.Tensor] = None
        self._interface_stage_aux_score: Optional[torch.Tensor] = None
        self._cls_detail_score: Optional[torch.Tensor] = None
        self._cls_detail_stage_score: Optional[torch.Tensor] = None
        self._cls_detail_candidate_score: Optional[torch.Tensor] = None
        self._interface_candidate_prior: Optional[torch.Tensor] = None
        self._interface_sample_index = -1
        self._input_features_captured_this_generation = False
        self._latest_input_only_features: dict[str, float] = {}
        self._interface_router_use_interface: Optional[bool] = None
        self._cls_detail_router_use_cls: Optional[bool] = None

    def needs_cls_detail_score(self) -> bool:
        return self.interface_final_variant in {
            "interface_pmi_cls_stage", "interface_pmi_cls_guarded",
            "encoder_pmi_cls_guarded", "encoder_pmi_cls_mix", "encoder_pmi_cls_stage",
            "encoder_pmi_cls_final", "interface_pmi_cls_quota",
            "interface_pmi_cls_routed", "encoder_pmi_cls_quota",
            "encoder_pmi_cls_routed", "encoder_pmi_functional_user_cls_stage",
            "majority_support_functional_user_cls_stage"
        }

    def uses_cls_detail_final_quota(self) -> bool:
        if self.interface_final_variant in {
            "interface_pmi_cls_routed", "encoder_pmi_cls_routed"
        }:
            return bool(self._cls_detail_router_use_cls)
        return self.interface_final_variant in {
            "interface_pmi_cls_guarded", "encoder_pmi_cls_guarded",
            "encoder_pmi_cls_mix", "encoder_pmi_cls_final", "interface_pmi_cls_quota",
            "encoder_pmi_cls_quota"
        }

    def uses_cls_detail_stage_protection(self) -> bool:
        """Whether CLS detail may alter the pre-LLM candidate set.

        ``encoder_pmi_cls_guarded`` is the L9-only ablation: it captures the
        same CLS-detail score as the older controls but must preserve the
        frozen I0/E9 candidate construction exactly.
        """
        if self.interface_final_variant in {
            "encoder_pmi_cls_guarded", "encoder_pmi_cls_mix", "encoder_pmi_cls_final"
        }:
            return False
        if self.interface_final_variant in {
            "interface_pmi_cls_routed", "encoder_pmi_cls_routed"
        }:
            return bool(self._cls_detail_router_use_cls)
        return self.needs_cls_detail_score()

    def set_cls_detail_score(self, score: torch.Tensor, source: str = "") -> None:
        if not self.needs_cls_detail_score() or not torch.is_tensor(score) or score.numel() == 0:
            return
        values = torch.nan_to_num(
            score.detach().float().flatten(), nan=0.0, posinf=0.0, neginf=0.0
        ).clamp_min(0.0)
        self._cls_detail_score = values
        self.record("cls_detail_score_events", 1)
        self.record("cls_detail_score_tokens", int(values.numel()))
        if self.config.debug:
            self._debug(f"cls_detail_score source={source} tokens={int(values.numel())}")

    def _aligned_cls_detail_score(
        self, target_len: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        raw = self._cls_detail_score
        if raw is None or target_len <= 0:
            return None
        score = raw.to(device=device).float().flatten()
        if int(score.numel()) == int(target_len):
            return score
        if int(score.numel()) > int(target_len) and int(score.numel()) % int(target_len) == 0:
            group = int(score.numel()) // int(target_len)
            self.record("cls_detail_score_pooled", 1)
            self.record("cls_detail_score_pool_group", group)
            return score.view(int(target_len), group).mean(dim=1)
        raise RuntimeError(
            f"CLS detail score cannot be aligned exactly: "
            f"source={int(score.numel())} target={int(target_len)}"
        )

    def interface_sensitivity_enabled(self) -> bool:
        return bool(
            self.interface_stage_variant.startswith("interface_")
            or self.interface_stage_variant in {"hybrid_feature_diverse", "union_feature_diverse", "union_balanced_feature_diverse", "mixed_feature_diverse", "routed_feature_diverse"}
            or self.interface_final_variant.startswith("interface_")
            or self.interface_final_variant in {
                "mixed_pmi", "routed_pmi", "encoder_pmi_cls_routed",
                "encoder_pmi_interface_guarded",
                "encoder_pmi_interface_endpoint_guarded",
            }
        )

    def install_interface_projector_hooks(self, model) -> None:
        """Patch the native main merger and, optionally, Qwen DeepStack mergers."""
        if not self.interface_sensitivity_enabled():
            return
        visual = getattr(model, "visual", None)
        main = getattr(visual, "merger", None)
        # HF LLaVA exposes a token-wise projector under model.model instead
        # of a Qwen/OneVision visual merger.  The score is aligned to packed
        # LLM-token order by the LLaVA adapter after native AnyRes packing.
        if main is None:
            main = getattr(getattr(model, "model", None), "multi_modal_projector", None)
        if main is None:
            raise RuntimeError(
                "interface ablation requires model.visual.merger or model.model.multi_modal_projector"
            )
        branches = [("main", main)]
        if self.interface_branch_mode() == "all":
            deepstack = getattr(visual, "deepstack_merger_list", None)
            if deepstack is not None:
                branches.extend((f"deepstack{idx}", merger) for idx, merger in enumerate(deepstack))
        owner = self
        for branch_name, merger in branches:
            if getattr(merger, "_vispruner_interface_component_patched", False):
                raise RuntimeError(f"interface merger already patched: {branch_name}")
            original_forward = merger.forward
            activation = getattr(merger, "act_fn", None)
            if activation is None:
                mlp = getattr(merger, "mlp", None)
                if mlp is not None and len(mlp) >= 2:
                    activation = mlp[1]
            if activation is None:
                raise RuntimeError(f"interface merger has no exposed activation: {branch_name}")

            def capture_preactivation(_module, hook_args, __branch=branch_name):
                if hook_args and torch.is_tensor(hook_args[0]):
                    owner._interface_preactivation_by_branch[__branch] = hook_args[0].detach()

            activation.register_forward_pre_hook(capture_preactivation)

            def wrapped_merger(
                merger_self,
                *args,
                __orig=original_forward,
                __branch=branch_name,
                **kwargs,
            ):
                pre_projector = args[0] if args else kwargs.get("x", kwargs.get("hidden_states"))
                projected = __orig(*args, **kwargs)
                if not torch.is_tensor(pre_projector) or not torch.is_tensor(projected):
                    raise RuntimeError(
                        f"interface merger {__branch} did not expose tensor input/output"
                    )
                source = pre_projector.reshape(-1, int(pre_projector.shape[-1]))
                target = projected.reshape(-1, int(projected.shape[-1]))
                owner.capture_interface_projector_sensitivity(
                    __branch,
                    source,
                    target,
                    lambda perturbed: __orig(perturbed),
                    merger_module=merger_self,
                )
                return projected

            wrapped_merger.__signature__ = inspect.signature(original_forward)
            merger.forward = MethodType(wrapped_merger, merger)
            merger._vispruner_interface_component_patched = True
            self.record("interface_projector_hooks_installed", 1)

    def interface_branch_mode(self) -> str:
        mode = str(os.environ.get("VISPRUNER_INTERFACE_BRANCHES", "main") or "main").strip().lower()
        if mode not in {"main", "all"}:
            raise ValueError("VISPRUNER_INTERFACE_BRANCHES must be main or all")
        return mode

    def start_generation(self) -> None:
        super().start_generation()
        self._interface_sample_index += 1
        self._interface_sensitivity_by_branch.clear()
        self._interface_projected_by_branch.clear()
        self._interface_raw_target_n_by_branch.clear()
        self._interface_preactivation_by_branch.clear()
        self._input_features_captured_this_generation = False
        self._latest_input_only_features = {}
        self._interface_router_use_interface = None
        self._cls_detail_router_use_cls = None
        self._cls_detail_score = None
        self._cls_detail_stage_score = None
        self._cls_detail_candidate_score = None
        self.record("interface_component_generation_sessions", 1)

    def captured_input_only_features(self) -> dict[str, float]:
        return dict(self._latest_input_only_features)

    def finish_generation(self) -> None:
        super().finish_generation()
        self._interface_sensitivity_by_branch.clear()
        self._interface_projected_by_branch.clear()
        self._interface_raw_target_n_by_branch.clear()
        self._interface_preactivation_by_branch.clear()

    def begin(self, input_ids, attention_mask, vision_finder) -> None:
        super().begin(input_ids, attention_mask, vision_finder)
        if not self._generation_active:
            self._interface_sensitivity_by_branch.clear()
            self._interface_projected_by_branch.clear()
            self._interface_raw_target_n_by_branch.clear()
            self._interface_preactivation_by_branch.clear()
        self.record(f"interface_stage_variant_{self.interface_stage_variant}", 1)
        self.record(f"interface_final_variant_{self.interface_final_variant}", 1)

    def capture_interface_projector_sensitivity(
        self,
        branch_name: str,
        pre_projector_features: torch.Tensor,
        projected_features: torch.Tensor,
        projector,
        merger_module=None,
    ) -> None:
        """Capture exact ZOO or its forward-native analytic MLP approximation."""
        if not self.interface_sensitivity_enabled():
            return
        name = str(branch_name).strip().lower()
        if not name:
            raise ValueError("interface branch name is empty")
        if self.interface_branch_mode() == "main" and name != "main":
            return
        cached = self._interface_sensitivity_by_branch.get(name)
        if cached is not None:
            raw_target_n = self._interface_raw_target_n_by_branch.get(name)
            if int(cached.numel()) != int(projected_features.shape[0]) and raw_target_n != int(projected_features.shape[0]):
                raise RuntimeError(f"cached interface sensitivity mismatch for {name}")
            self.record("interface_projector_cache_reuse", 1)
            return
        if pre_projector_features.ndim != 2 or projected_features.ndim != 2:
            raise RuntimeError("interface projector hook requires two rank-2 tensors")
        target_n = int(projected_features.shape[0])
        source_n, source_dim = map(int, pre_projector_features.shape)
        if target_n <= 0 or source_n % target_n != 0:
            raise RuntimeError(
                f"interface projector alignment failed for {name}: "
                f"source={source_n}, target={target_n}"
            )
        group = source_n // target_n
        estimator = str(
            os.environ.get("VISPRUNER_INTERFACE_ESTIMATOR", "zoo") or "zoo"
        ).strip().lower()
        if estimator not in {"zoo", "analytic", "contribution"}:
            raise ValueError("VISPRUNER_INTERFACE_ESTIMATOR must be zoo, analytic, or contribution")
        if estimator == "contribution":
            baseline = self._interface_zero_output_by_branch.get(name)
            if baseline is None:
                zero_source = torch.zeros_like(pre_projector_features[:group])
                with torch.no_grad():
                    baseline = projector(zero_source).detach().reshape(1, -1)
                if int(baseline.shape[1]) != int(projected_features.shape[1]):
                    raise RuntimeError(f"interface contribution baseline mismatch for {name}")
                self._interface_zero_output_by_branch[name] = baseline
                self.record("interface_projector_contribution_baseline_queries", 1)
            baseline = baseline.to(device=projected_features.device, dtype=projected_features.dtype)
            sensitivity = (projected_features.detach() - baseline).float().norm(dim=1)
            if (
                not bool(getattr(self, "_runtime_exact_fast_path", False))
                and not bool(torch.isfinite(sensitivity).all().item())
            ):
                raise RuntimeError(f"non-finite interface contribution for {name}")
            self._interface_sensitivity_by_branch[name] = sensitivity.detach()
            if not bool(getattr(self, "_runtime_exact_fast_path", False)):
                self._interface_projected_by_branch[name] = projected_features.detach()
            self._interface_raw_target_n_by_branch[name] = target_n
            self.record("interface_projector_sensitivity_events", 1)
            self.record("interface_projector_contribution_events", 1)
            self.record("interface_projector_tokens", target_n)
            self.record("interface_projector_source_group", group)
            self.record("interface_direction_pair_queries", 0)
            self.record(f"interface_branch_{name}", 1)
            return
        if estimator == "analytic":
            if merger_module is None:
                raise RuntimeError("analytic interface sensitivity requires the merger module")
            preactivation = self._interface_preactivation_by_branch.get(name)
            if preactivation is None:
                raise RuntimeError(f"analytic interface hook missed the GELU input for {name}")
            preactivation = preactivation.detach().reshape(target_n, -1).float()
            fc1 = getattr(merger_module, "linear_fc1", None)
            fc2 = getattr(merger_module, "linear_fc2", None)
            if fc1 is None or fc2 is None:
                mlp = getattr(merger_module, "mlp", None)
                if mlp is not None and len(mlp) >= 3:
                    fc1, fc2 = mlp[0], mlp[2]
            if fc1 is None or fc2 is None:
                raise RuntimeError(f"analytic interface estimator cannot resolve MLP weights for {name}")
            factor = self._interface_weight_factor_by_branch.get(name)
            if factor is None:
                row_energy = fc1.weight.detach().float().square().sum(dim=1)
                column_energy = fc2.weight.detach().float().square().sum(dim=0)
                if int(row_energy.numel()) != int(column_energy.numel()):
                    raise RuntimeError(f"analytic interface MLP width mismatch for {name}")
                factor = (row_energy * column_energy).detach()
                self._interface_weight_factor_by_branch[name] = factor
            factor = factor.to(device=preactivation.device, dtype=torch.float32)
            # Exact GELU derivative.  Ignoring high-dimensional cross terms yields
            # a diagonal Frobenius-Jacobian proxy with one O(tokens*width) pass.
            inv_sqrt_two = 1.0 / math.sqrt(2.0)
            normal_cdf = 0.5 * (1.0 + torch.erf(preactivation * inv_sqrt_two))
            normal_pdf = torch.exp(-0.5 * preactivation.square()) / math.sqrt(2.0 * math.pi)
            derivative = normal_cdf + preactivation * normal_pdf
            sensitivity = (derivative.square() * factor.view(1, -1)).sum(dim=1).clamp_min(0.0).sqrt()
            if not bool(torch.isfinite(sensitivity).all().item()):
                raise RuntimeError(f"non-finite analytic interface sensitivity for {name}")
            self._interface_sensitivity_by_branch[name] = sensitivity.detach()
            self._interface_projected_by_branch[name] = projected_features.detach()
            self._interface_raw_target_n_by_branch[name] = target_n
            self.record("interface_projector_sensitivity_events", 1)
            self.record("interface_projector_analytic_events", 1)
            self.record("interface_projector_tokens", target_n)
            self.record("interface_projector_source_group", group)
            self.record("interface_direction_pair_queries", 0)
            self.record(f"interface_branch_{name}", 1)
            return

        directions = int(os.environ.get("VISPRUNER_INTERFACE_DIRECTIONS", "64") or 64)
        step = float(os.environ.get("VISPRUNER_INTERFACE_STEP", "0.01") or 0.01)
        chunk_size = int(os.environ.get("VISPRUNER_INTERFACE_DIRECTION_CHUNK", "8") or 8)
        seed_base = int(os.environ.get("VISPRUNER_INTERFACE_SEED", "1234") or 1234)
        if directions <= 0 or step <= 0.0 or chunk_size <= 0:
            raise ValueError("interface directions, step, and chunk must be positive")

        grouped = pre_projector_features.detach().reshape(target_n, group, source_dim)
        feature_dim = group * source_dim
        generator = torch.Generator(device=grouped.device)
        # The same seed is intentional across Qwen branches: paired directions
        # remove random-direction variation from main-vs-all branch comparisons.
        generator.manual_seed(seed_base + max(0, int(self._interface_sample_index)))
        random_directions = torch.randn(
            directions,
            feature_dim,
            device=grouped.device,
            dtype=grouped.dtype,
            generator=generator,
        )
        random_directions = random_directions / random_directions.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        coefficient_sum = torch.zeros(target_n, device=grouped.device, dtype=torch.float32)
        with torch.no_grad():
            for start in range(0, directions, chunk_size):
                direction = random_directions[start : start + chunk_size]
                count = int(direction.shape[0])
                direction = direction.view(count, 1, group, source_dim)
                plus = (grouped.unsqueeze(0) + step * direction).reshape(count * source_n, source_dim)
                minus = (grouped.unsqueeze(0) - step * direction).reshape(count * source_n, source_dim)
                projected_plus = projector(plus).reshape(count, target_n, -1)
                projected_minus = projector(minus).reshape(count, target_n, -1)
                coefficient_sum += (
                    (projected_plus - projected_minus).float().norm(dim=-1) / (2.0 * step)
                ).sum(dim=0)
        sensitivity = coefficient_sum / float(directions)
        if not bool(torch.isfinite(sensitivity).all().item()):
            raise RuntimeError(f"non-finite interface sensitivity for {name}")
        self._interface_sensitivity_by_branch[name] = sensitivity.detach()
        self._interface_projected_by_branch[name] = projected_features.detach()
        self._interface_raw_target_n_by_branch[name] = target_n
        self.record("interface_projector_sensitivity_events", 1)
        self.record("interface_projector_tokens", target_n)
        self.record("interface_projector_source_group", group)
        self.record("interface_directions", directions)
        self.record("interface_direction_pair_queries", 2 * directions)
        self.record(f"interface_branch_{name}", 1)

    def align_interface_projector_branch(
        self,
        branch_name: str,
        aligned_sensitivity: torch.Tensor,
        aligned_projected_features: torch.Tensor,
    ) -> None:
        """Replace raw projector rows with native packed LLM-token order."""
        name = str(branch_name).strip().lower()
        if name not in self._interface_sensitivity_by_branch:
            raise RuntimeError(f"cannot align uncaptured interface branch {name}")
        score = aligned_sensitivity.detach().reshape(-1)
        features = aligned_projected_features.detach().reshape(
            -1, int(aligned_projected_features.shape[-1])
        )
        if int(score.numel()) != int(features.shape[0]):
            raise RuntimeError(
                f"aligned interface score/feature mismatch for {name}: "
                f"{int(score.numel())} vs {int(features.shape[0])}"
            )
        self._interface_sensitivity_by_branch[name] = score
        self._interface_projected_by_branch[name] = features
        self.record("interface_projector_packed_alignment_events", 1)
        self.record("interface_projector_packed_alignment_tokens", int(score.numel()))

    def _combined_interface_score(self, target_n: int, device: torch.device) -> torch.Tensor:
        if not self._interface_sensitivity_by_branch:
            raise RuntimeError("interface sensitivity was not captured before L0 selection")
        allowed = ["main"] if self.interface_branch_mode() == "main" else sorted(
            self._interface_sensitivity_by_branch
        )
        pieces = []
        for name in allowed:
            score = self._interface_sensitivity_by_branch.get(name)
            if score is None:
                raise RuntimeError(f"missing requested interface branch {name}")
            score = score.to(device=device, dtype=torch.float32).flatten()
            if int(score.numel()) != int(target_n):
                raise RuntimeError(
                    f"interface score/token mismatch for {name}: {int(score.numel())} vs {target_n}"
                )
            if self.interface_branch_mode() == "all":
                score = score / score.mean().clamp_min(1e-12)
            pieces.append(score)
        combined = pieces[0] if len(pieces) == 1 else torch.stack(pieces, dim=0).mean(dim=0)
        if (
            not bool(getattr(self, "_runtime_exact_fast_path", False))
            and not bool(torch.isfinite(combined).all().item())
        ):
            raise RuntimeError("combined interface score is non-finite")
        self.record("interface_score_combine_events", 1)
        self.record("interface_score_combined_branches", len(pieces))
        return combined

    @staticmethod
    def _interface_sads_select(
        visual_features: torch.Tensor,
        sensitivity: torch.Tensor,
        keep_n: int,
    ) -> torch.Tensor:
        """Official ZOO SADS selection, separated from its sensitivity signal."""
        features = visual_features.float()
        importance = sensitivity.to(device=features.device, dtype=torch.float32).flatten()
        token_n = int(features.shape[0])
        if int(importance.numel()) != token_n:
            raise RuntimeError("SADS feature/sensitivity mismatch")
        keep_n = max(1, min(int(keep_n), token_n))
        normalized = torch.nn.functional.normalize(features, dim=-1)
        distance = 1.0 - normalized @ normalized.T
        minimum, maximum = importance.min(), importance.max()
        weight = (
            (importance - minimum) / (maximum - minimum + 1e-8)
            if bool((maximum > minimum).item())
            else torch.ones_like(importance)
        )
        selected = torch.empty(keep_n, device=features.device, dtype=torch.long)
        available = torch.ones(token_n, device=features.device, dtype=torch.bool)
        for index in range(keep_n):
            if index == 0:
                score = weight
            else:
                min_distance = distance.index_select(0, selected[:index]).min(dim=0).values
                score = min_distance * weight
            selected[index] = torch.argmax(score.masked_fill(~available, float("-inf")))
            available[selected[index]] = False
        return selected

    @staticmethod
    def _percentile_rank(values: torch.Tensor) -> torch.Tensor:
        values = values.detach().float().flatten()
        if int(values.numel()) <= 0:
            raise RuntimeError("percentile rank requires a non-empty score")
        order = torch.argsort(values, descending=False, stable=True)
        sorted_values = values.index_select(0, order)
        _, counts = torch.unique_consecutive(sorted_values, return_counts=True)
        ends = counts.cumsum(dim=0)
        starts = ends - counts
        average_ranks = (starts + ends - 1).to(dtype=torch.float32) / 2.0
        sorted_ranks = torch.repeat_interleave(average_ranks, counts)
        ranks = torch.empty_like(values, dtype=torch.float32)
        ranks[order] = sorted_ranks
        return ranks / max(1, int(values.numel()) - 1)

    def _select_vision_keep_feature_diverse(
        self,
        score: torch.Tensor,
        features: torch.Tensor,
        keep_n: int,
        anchor_n: int,
    ) -> torch.Tensor:
        if self.interface_stage_variant == "interface_sads":
            selected = self._interface_sads_select(features, score, keep_n)
            self.record("interface_sads_stage_events", 1)
            self.record("interface_sads_stage_selected", int(selected.numel()))
            return selected
        if self.interface_stage_variant == "mixed_feature_diverse":
            self.record("interface_mixed_feature_diverse_stage_events", 1)
            return super()._select_vision_keep_feature_diverse(
                score=score, features=features, keep_n=keep_n, anchor_n=anchor_n
            )
        if self.interface_stage_variant in {"union_feature_diverse", "union_balanced_feature_diverse"}:
            encoder = self._interface_stage_encoder_score
            auxiliary = self._interface_stage_aux_score
            if encoder is None or auxiliary is None:
                raise RuntimeError("union interface stage is missing encoder/interface scores")
            encoder = encoder.to(device=features.device, dtype=torch.float32).flatten()
            auxiliary = auxiliary.to(device=features.device, dtype=torch.float32).flatten()
            total = int(features.shape[0])
            if int(encoder.numel()) != total or int(auxiliary.numel()) != total:
                raise RuntimeError("union interface stage score alignment failed")
            keep_n = max(1, min(int(keep_n), total))
            anchor_n = max(1, min(int(anchor_n), keep_n, total))
            encoder_order = torch.argsort(encoder, descending=True, stable=True)
            encoder_anchor = encoder_order[:anchor_n]
            used = torch.zeros(total, device=features.device, dtype=torch.bool)
            used[encoder_anchor] = True
            auxiliary_order = torch.argsort(auxiliary, descending=True, stable=True)
            auxiliary_pool = auxiliary_order[~used.index_select(0, auxiliary_order)]
            surplus_n = keep_n - anchor_n
            auxiliary_capacity = (
                int(math.ceil(surplus_n / 2.0))
                if self.interface_stage_variant == "union_balanced_feature_diverse"
                else surplus_n
            )
            auxiliary_n = min(anchor_n, auxiliary_capacity, int(auxiliary_pool.numel()))
            auxiliary_anchor = auxiliary_pool[:auxiliary_n]
            used[auxiliary_anchor] = True
            anchors = torch.cat((encoder_anchor, auxiliary_anchor), dim=0)
            reserve_n = keep_n - int(anchors.numel())
            if reserve_n > 0:
                residual = torch.arange(total, device=features.device, dtype=torch.long)[~used]
                normalized = torch.nn.functional.normalize(
                    features.detach().float(), dim=-1, eps=1e-6
                )
                similarity = normalized.index_select(0, residual) @ normalized.index_select(0, anchors).T
                novelty = (1.0 - similarity.max(dim=1).values.clamp(-1.0, 1.0)).clamp_min(0.0)
                reserve_score = encoder.index_select(0, residual).clamp_min(0.0) * novelty
                reserve_order = torch.argsort(reserve_score, descending=True, stable=True)
                reserves = residual.index_select(0, reserve_order[:reserve_n])
            else:
                reserves = encoder.new_empty((0,), dtype=torch.long)
            selected = torch.cat((anchors, reserves), dim=0)
            if int(selected.numel()) != keep_n or int(torch.unique(selected).numel()) != keep_n:
                raise RuntimeError("union interface selection is not exact and unique")
            self.record("interface_union_stage_events", 1)
            if self.interface_stage_variant == "union_balanced_feature_diverse":
                self.record("interface_union_balanced_stage_events", 1)
            self.record("interface_union_encoder_anchors", int(encoder_anchor.numel()))
            self.record("interface_union_aux_anchors", int(auxiliary_anchor.numel()))
            self.record("interface_union_total_anchors", int(anchors.numel()))
            self.record("interface_union_diversity_reserves", reserve_n)
            return selected
        if self.interface_stage_variant == "hybrid_feature_diverse":
            encoder = self._interface_stage_encoder_score
            auxiliary = self._interface_stage_aux_score
            if encoder is None or auxiliary is None:
                raise RuntimeError("hybrid interface stage is missing encoder/interface scores")
            encoder = encoder.to(device=features.device, dtype=torch.float32).flatten()
            auxiliary = auxiliary.to(device=features.device, dtype=torch.float32).flatten()
            total = int(features.shape[0])
            if int(encoder.numel()) != total or int(auxiliary.numel()) != total:
                raise RuntimeError("hybrid interface stage score alignment failed")
            keep_n = max(1, min(int(keep_n), total))
            anchor_n = max(1, min(int(anchor_n), keep_n, total))
            fraction = float(os.environ.get("VISPRUNER_INTERFACE_HYBRID_FRACTION", "0.5") or 0.5)
            if not 0.0 <= fraction <= 1.0:
                raise ValueError("VISPRUNER_INTERFACE_HYBRID_FRACTION must be in [0,1]")
            aux_n = max(0, min(anchor_n, int(round(anchor_n * fraction))))
            enc_n = anchor_n - aux_n
            aux_order = torch.argsort(auxiliary, descending=True, stable=True)
            aux_anchor = aux_order[:aux_n]
            used = torch.zeros(total, device=features.device, dtype=torch.bool)
            used[aux_anchor] = True
            enc_pool = torch.argsort(encoder, descending=True, stable=True)
            enc_pool = enc_pool[~used.index_select(0, enc_pool)]
            enc_anchor = enc_pool[:enc_n]
            anchors = torch.cat((aux_anchor, enc_anchor), dim=0)
            used[enc_anchor] = True
            residual = torch.arange(total, device=features.device, dtype=torch.long)[~used]
            reserve_n = keep_n - anchor_n
            if reserve_n > 0:
                normalized = torch.nn.functional.normalize(features.detach().float(), dim=-1, eps=1e-6)
                similarity = normalized.index_select(0, residual) @ normalized.index_select(0, anchors).T
                novelty = (1.0 - similarity.max(dim=1).values.clamp(-1.0, 1.0)).clamp_min(0.0)
                reserve_score = encoder.index_select(0, residual).clamp_min(0.0) * novelty
                reserve_order = torch.argsort(reserve_score, descending=True, stable=True)
                reserves = residual.index_select(0, reserve_order[:reserve_n])
            else:
                reserves = encoder.new_empty((0,), dtype=torch.long)
            selected = torch.cat((anchors, reserves), dim=0)
            if int(selected.numel()) != keep_n or int(torch.unique(selected).numel()) != keep_n:
                raise RuntimeError("hybrid interface selection is not exact and unique")
            self.record("interface_hybrid_stage_events", 1)
            self.record("interface_hybrid_aux_anchors", aux_n)
            self.record("interface_hybrid_encoder_anchors", enc_n)
            self.record("interface_hybrid_diversity_reserves", reserve_n)
            return selected
        selected = super()._select_vision_keep_feature_diverse(
            score=score,
            features=features,
            keep_n=keep_n,
            anchor_n=anchor_n,
        )
        if not self.uses_cls_detail_stage_protection():
            return selected
        detail = self._cls_detail_stage_score
        total = int(features.shape[0])
        if detail is None or int(detail.numel()) != total:
            raise RuntimeError("CLS-detail stage score is missing or misaligned")
        alpha = float(getattr(self.config, "vision_anchor_alpha", 0.0) or 0.0)
        final_n = self.final_vision_budget(total, 0)
        protected_n = min(int(keep_n), int(math.ceil(alpha * float(final_n))))
        if protected_n <= 0:
            return selected
        protected = torch.argsort(
            detail.to(features.device).float(), descending=True, stable=True
        )[:protected_n]
        base_contains = torch.isin(protected, selected)
        protected_mask = torch.zeros(total, device=features.device, dtype=torch.bool)
        protected_mask[protected] = True
        fill = selected[~protected_mask.index_select(0, selected.long())]
        selected = torch.cat((protected, fill), dim=0)[: int(keep_n)]
        if int(selected.numel()) != int(keep_n) or int(torch.unique(selected).numel()) != int(keep_n):
            raise RuntimeError("CLS-detail stage protection is not exact and unique")
        self.record("cls_detail_stage_protect_events", 1)
        self.record("cls_detail_stage_protected", protected_n)
        self.record("cls_detail_stage_replacements", int((~base_contains).sum().item()))
        return selected

    def feature_reserve_pre_llm_compact(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        visual_pos_masks: Optional[torch.Tensor],
        deepstack_visual_embeds,
    ):
        if not self._vision_by_batch:
            raise RuntimeError("interface component ablation has no visual-token span")
        vision_idx = self._vision_by_batch[0].to(inputs_embeds.device)
        initial_n = int(vision_idx.numel())
        encoder_full = self._aligned_encoder_score(initial_n, inputs_embeds.device)
        if encoder_full is None or int(encoder_full.numel()) != initial_n:
            raise RuntimeError("interface component ablation requires aligned encoder saliency")
        detail_full = None
        if self.needs_cls_detail_score():
            detail_full = self._aligned_cls_detail_score(initial_n, inputs_embeds.device)
            if detail_full is None or int(detail_full.numel()) != initial_n:
                raise RuntimeError(
                    "CLS-detail quota requires aligned second-last CLS-query saliency"
                )
            self._cls_detail_stage_score = detail_full.detach().clone()
        need_interface = self.interface_sensitivity_enabled()
        interface_full = (
            self._combined_interface_score(initial_n, inputs_embeds.device)
            if need_interface
            else None
        )
        stage_uses_interface = self.interface_stage_variant.startswith("interface_")
        self._interface_stage_encoder_score = encoder_full.detach().clone()
        self._interface_stage_aux_score = (
            interface_full.detach().clone() if interface_full is not None else None
        )
        if (
            (
                os.environ.get("VISPRUNER_CAPTURE_INPUT_FEATURES", "0").strip().lower()
                in {"1", "true", "yes", "on"}
                or self.interface_stage_variant == "routed_feature_diverse"
                or self.interface_final_variant in {
                    "interface_pmi_cls_routed", "encoder_pmi_cls_routed"
                }
            )
            and not self._input_features_captured_this_generation
        ):
            if interface_full is None:
                raise RuntimeError("input feature capture requires the interface score")
            from scripts.analysis.input_only_alpha_features import input_token_features

            captured = input_token_features(self, inputs_embeds)
            contribution = torch.nan_to_num(interface_full.detach().float().flatten())
            scale = contribution.abs().median().clamp_min(1e-8)
            quantiles = torch.quantile(
                contribution, torch.tensor([0.1, 0.5, 0.9], device=contribution.device)
            )
            captured.update({
                "input_interface_mean_over_abs_median": float((contribution.mean() / scale).item()),
                "input_interface_std_over_abs_median": float((contribution.std(unbiased=False) / scale).item()),
                "input_interface_q10_over_abs_median": float((quantiles[0] / scale).item()),
                "input_interface_q50_over_abs_median": float((quantiles[1] / scale).item()),
                "input_interface_q90_over_abs_median": float((quantiles[2] / scale).item()),
            })
            weights = (contribution - contribution.min()).clamp_min(0.0) + 1e-8
            probability = weights / weights.sum().clamp_min(1e-8)
            entropy = -(probability * probability.log()).sum()
            captured["input_interface_entropy_normalized"] = float(
                (entropy / math.log(max(2, int(probability.numel())))).item()
            )
            captured["input_encoder_interface_pearson"] = float(
                self._pearson_corr(encoder_full, interface_full) or 0.0
            )
            captured["input_encoder_interface_spearman"] = float(
                self._spearman_corr(encoder_full, interface_full) or 0.0
            )
            final_n = self.final_vision_budget(initial_n, 0)
            enc_top = set(torch.topk(encoder_full.float(), k=final_n).indices.cpu().tolist())
            int_top = set(torch.topk(interface_full.float(), k=final_n).indices.cpu().tolist())
            captured["input_encoder_interface_topk_jaccard"] = float(
                len(enc_top & int_top) / max(1, len(enc_top | int_top))
            )
            if detail_full is not None:
                detail = torch.nan_to_num(
                    detail_full.detach().float().flatten(),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).clamp_min(0.0)
                detail_probability = (detail + 1e-12) / (
                    detail.sum() + 1e-12 * max(1, int(detail.numel()))
                )
                detail_entropy = -(detail_probability * detail_probability.log()).sum()
                detail_n = max(1, int(detail.numel()))
                captured["input_cls_detail_entropy_normalized"] = float(
                    (detail_entropy / math.log(max(2, detail_n))).item()
                )
                captured["input_cls_detail_effective_fraction"] = float(
                    (1.0 / detail_probability.square().sum().clamp_min(1e-12) / detail_n).item()
                )
                detail_scale = detail.abs().median().clamp_min(1e-8)
                detail_quantiles = torch.quantile(
                    detail,
                    torch.tensor([0.1, 0.5, 0.9], device=detail.device),
                )
                captured.update({
                    "input_cls_detail_mean_over_abs_median": float((detail.mean() / detail_scale).item()),
                    "input_cls_detail_std_over_abs_median": float((detail.std(unbiased=False) / detail_scale).item()),
                    "input_cls_detail_q10_over_abs_median": float((detail_quantiles[0] / detail_scale).item()),
                    "input_cls_detail_q50_over_abs_median": float((detail_quantiles[1] / detail_scale).item()),
                    "input_cls_detail_q90_over_abs_median": float((detail_quantiles[2] / detail_scale).item()),
                    "input_cls_detail_encoder_pearson": float(self._pearson_corr(detail, encoder_full) or 0.0),
                    "input_cls_detail_encoder_spearman": float(self._spearman_corr(detail, encoder_full) or 0.0),
                    "input_cls_detail_interface_pearson": float(self._pearson_corr(detail, interface_full) or 0.0),
                    "input_cls_detail_interface_spearman": float(self._spearman_corr(detail, interface_full) or 0.0),
                })
                for label, fraction in (("01", 0.01), ("05", 0.05), ("10", 0.10), ("25", 0.25)):
                    count = max(1, min(detail_n, int(math.ceil(fraction * detail_n))))
                    captured[f"input_cls_detail_top{label}_mass"] = float(
                        torch.topk(detail_probability, k=count).values.sum().item()
                    )
                detail_top = set(torch.topk(detail, k=final_n).indices.cpu().tolist())
                captured["input_cls_detail_encoder_topk_jaccard"] = float(
                    len(detail_top & enc_top) / max(1, len(detail_top | enc_top))
                )
                captured["input_cls_detail_interface_topk_jaccard"] = float(
                    len(detail_top & int_top) / max(1, len(detail_top | int_top))
                )
                detail_mean = detail.mean().clamp_min(1e-12)
                for lag in (1, 2, 4, 8):
                    variation = (
                        (detail[lag:] - detail[:-lag]).abs().mean() / detail_mean
                        if detail_n > lag
                        else detail.new_zeros(())
                    )
                    captured[f"input_cls_detail_lag{lag}_variation"] = float(variation.item())
            if not all(math.isfinite(value) for value in captured.values()):
                raise RuntimeError("non-finite input-only interface features")
            self._latest_input_only_features = captured
            self._input_features_captured_this_generation = True
            self.record("interface_input_feature_capture_events", 1)
            self.record("interface_input_feature_count", len(captured))
        if self.interface_final_variant in {
            "interface_pmi_cls_routed", "encoder_pmi_cls_routed"
        }:
            if self._cls_detail_router_use_cls is None:
                decide = getattr(self, "decide_cls_detail_route", None)
                if decide is None:
                    raise RuntimeError("CLS-detail routed final has no decision function")
                self._cls_detail_router_use_cls = bool(
                    decide(self._latest_input_only_features)
                )
                self.record("cls_detail_router_decision_events", 1)
                self.record(
                    "cls_detail_router_cls_decisions"
                    if self._cls_detail_router_use_cls
                    else "cls_detail_router_control_decisions",
                    1,
                )
        if self.interface_stage_variant == "routed_feature_diverse":
            if self._interface_router_use_interface is None:
                decide = getattr(self, "decide_interface_route", None)
                if decide is None:
                    raise RuntimeError("routed interface stage has no router decision function")
                self._interface_router_use_interface = bool(decide(self._latest_input_only_features))
                self.record("interface_router_decision_events", 1)
                self.record(
                    "interface_router_contribution_decisions" if self._interface_router_use_interface
                    else "interface_router_encoder_decisions", 1
                )
            stage_uses_interface = bool(self._interface_router_use_interface)
        if self.interface_stage_variant == "mixed_feature_diverse":
            if interface_full is None:
                raise RuntimeError("mixed interface stage requires an interface score")
            alpha = float(os.environ.get("VISPRUNER_INTERFACE_MIX_ALPHA", "0.5") or 0.5)
            if not 0.0 <= alpha <= 1.0:
                raise ValueError("VISPRUNER_INTERFACE_MIX_ALPHA must be in [0,1]")
            mixed = (1.0 - alpha) * self._percentile_rank(encoder_full) + alpha * self._percentile_rank(interface_full)
            self._vision_encoder_score = mixed.detach().clone()
            self.record("interface_mixed_score_events", 1)
            self.record("interface_mixed_alpha_x1000", int(round(1000.0 * alpha)))
        else:
            self._vision_encoder_score = (
                interface_full if stage_uses_interface else encoder_full
            ).detach().clone()
        output = super().feature_reserve_pre_llm_compact(
            inputs_embeds,
            attention_mask,
            position_ids,
            cache_position,
            visual_pos_masks,
            deepstack_visual_embeds,
        )

        kept_original = self._feature_reserve_original_local_indices
        if kept_original is None:
            kept_original = torch.arange(initial_n, device=inputs_embeds.device, dtype=torch.long)
        else:
            kept_original = kept_original.to(inputs_embeds.device, dtype=torch.long)
        self._interface_candidate_prior = (
            interface_full.index_select(0, kept_original).detach().clone()
            if self.interface_final_variant in {
                "encoder_pmi_interface_guarded",
                "encoder_pmi_interface_endpoint_guarded",
            }
            and interface_full is not None
            else None
        )
        if self.interface_final_variant in {
            "encoder_pmi_interface_guarded",
            "encoder_pmi_interface_endpoint_guarded",
        }:
            if self._interface_candidate_prior is None:
                raise RuntimeError("interface-guarded final is missing its candidate prior")
            self.record("interface_guard_candidate_prior_installed", 1)

        final_uses_interface = (
            self.interface_final_variant.startswith("interface_")
            or (
                self.interface_final_variant == "routed_pmi"
                and bool(self._interface_router_use_interface)
            )
        )
        if self.needs_cls_detail_score():
            if detail_full is None:
                raise RuntimeError("CLS-detail quota lost its full-resolution score")
            self._cls_detail_candidate_score = detail_full.index_select(
                0, kept_original
            ).detach().clone()
            detail_top = torch.argsort(
                detail_full.float(), descending=True, stable=True
            )[: self.final_vision_budget(initial_n, 0)]
            candidate_mask = torch.zeros(
                initial_n, device=inputs_embeds.device, dtype=torch.bool
            )
            candidate_mask[kept_original] = True
            covered = int(candidate_mask.index_select(0, detail_top).sum().item())
            self.record("cls_detail_candidate_coverage_events", 1)
            self.record("cls_detail_candidate_topk_total", int(detail_top.numel()))
            self.record("cls_detail_candidate_topk_covered", covered)
        if self.interface_final_variant == "mixed_pmi":
            if interface_full is None:
                raise RuntimeError("mixed final prior requested without interface sensitivity")
            alpha = float(os.environ.get("VISPRUNER_INTERFACE_FINAL_MIX_ALPHA", "0.5") or 0.5)
            if not 0.0 <= alpha <= 1.0:
                raise ValueError("VISPRUNER_INTERFACE_FINAL_MIX_ALPHA must be in [0,1]")
            encoder_candidate = encoder_full.index_select(0, kept_original)
            interface_candidate = interface_full.index_select(0, kept_original)
            final_prior = (
                (1.0 - alpha) * self._percentile_rank(encoder_candidate)
                + alpha * self._percentile_rank(interface_candidate)
            )
            self.record("interface_final_mixed_prior_installed", 1)
            self.record("interface_final_mixed_alpha_x1000", int(round(1000.0 * alpha)))
        elif final_uses_interface:
            if interface_full is None:
                raise RuntimeError("interface final prior requested without sensitivity")
            final_prior = interface_full.index_select(0, kept_original)
            self.record("interface_final_prior_installed", 1)
        else:
            final_prior = encoder_full.index_select(0, kept_original)
            self.record("encoder_final_prior_restored", 1)
        self._vision_encoder_score = final_prior.detach().clone()
        self.record("interface_component_stage_final_prior_separated", 1)
        return output

    def _select_encoder_anchor_quota(
        self,
        score_encoder: torch.Tensor,
        score_joint: torch.Tensor,
        keep_n: int,
        alpha: float,
        score_text: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, int]:
        if self.interface_final_variant in {
            "encoder_pmi_interface_guarded",
            "encoder_pmi_interface_endpoint_guarded",
        }:
            current_joint = score_joint.detach().float().flatten()
            encoder = score_encoder.detach().float().flatten()
            interface = self._interface_candidate_prior
            if interface is None:
                raise RuntimeError("interface-guarded final lost its candidate prior")
            interface = interface.to(current_joint.device).float().flatten()
            if not (int(current_joint.numel()) == int(encoder.numel()) == int(interface.numel())):
                raise RuntimeError("interface-guarded final score alignment mismatch")
            n = int(current_joint.numel())
            k = max(0, min(int(keep_n), n))
            a = float(alpha)
            if not math.isfinite(a) or not 0.0 <= a <= 1.0:
                raise ValueError(
                    f"vision_anchor_alpha must be finite and in [0, 1], got {alpha!r}"
                )
            if k == 0:
                selected = torch.empty(0, device=current_joint.device, dtype=torch.long)
                locked = 0
                proposed_n = 0
            elif a == 0.0:
                # Exact I0/E9 control, including torch.topk tie behavior.
                selected = torch.topk(current_joint, k=k, largest=True).indices
                locked = 0
                proposed_n = 0
            else:
                lam = float(getattr(self.config, "vision_score_lambda", 1.0) or 0.0)
                if self.interface_final_variant == "encoder_pmi_interface_endpoint_guarded":
                    if score_text is None:
                        raise RuntimeError(
                            "exact interface-endpoint guard requires the aligned L9 text score"
                        )
                    text = score_text.to(current_joint.device).float().flatten()
                    if int(text.numel()) != n:
                        raise RuntimeError(
                            "exact interface-endpoint guard text-score alignment mismatch"
                        )
                    interface_pmi = self._pmi_like_score(text, interface)
                    interface_joint = (
                        self._normalize_score(interface)
                        + lam * self._normalize_score(interface_pmi)
                    )
                    self.record("interface_endpoint_guard_exact_pmi_events", 1)
                else:
                    encoder_norm = self._normalize_score(encoder)
                    if abs(lam) > 1e-12:
                        pmi_norm = (current_joint - encoder_norm) / lam
                    else:
                        pmi_norm = torch.zeros_like(current_joint)
                    interface_joint = self._normalize_score(interface) + lam * pmi_norm
                proposed_n = min(k, int(math.ceil(a * float(k))))
                interface_order = torch.argsort(
                    interface_joint, descending=True, stable=True
                )
                proposed = interface_order[:proposed_n]
                current_order = torch.argsort(
                    current_joint, descending=True, stable=True
                )
                guard_n = min(n, k + proposed_n)
                guard_mask = torch.zeros(n, device=current_joint.device, dtype=torch.bool)
                guard_mask[current_order[:guard_n]] = True
                locked_tokens = proposed[guard_mask.index_select(0, proposed)]
                locked = int(locked_tokens.numel())
                used = torch.zeros(n, device=current_joint.device, dtype=torch.bool)
                used[locked_tokens] = True
                fill = current_order[~used.index_select(0, current_order)][: k - locked]
                selected = torch.cat((locked_tokens, fill), dim=0)
            self.record("interface_guard_select_events", 1)
            self.record("interface_guard_selected", k)
            self.record("interface_guard_proposed", proposed_n)
            self.record("interface_guard_accepted", locked)
            self.record("interface_guard_rejected", proposed_n - locked)
            self.record("interface_guard_alpha_x1000", int(round(1000.0 * a)))
            return selected, locked
        if not self.uses_cls_detail_final_quota():
            effective_alpha = (
                0.0 if self.interface_final_variant in {
                    "interface_pmi_cls_routed", "encoder_pmi_cls_routed"
                } else alpha
            )
            return super()._select_encoder_anchor_quota(
                score_encoder, score_joint, keep_n, effective_alpha, score_text=score_text
            )
        detail = self._cls_detail_candidate_score
        if detail is None or int(detail.numel()) != int(score_joint.numel()):
            raise RuntimeError(
                f"CLS-detail candidate mismatch: "
                f"detail={None if detail is None else int(detail.numel())} "
                f"joint={int(score_joint.numel())}"
            )
        if self.interface_final_variant == "encoder_pmi_cls_mix":
            joint = score_joint.detach().float().flatten()
            n = int(joint.numel())
            k = max(0, min(int(keep_n), n))
            a = float(alpha)
            if not math.isfinite(a) or not 0.0 <= a <= 1.0:
                raise ValueError(
                    f"vision_anchor_alpha must be finite and in [0, 1], got {alpha!r}"
                )
            if k == 0:
                selected = torch.empty(0, device=joint.device, dtype=torch.long)
            elif a == 0.0:
                # Exact C endpoint, including torch.topk tie behavior.
                selected = torch.topk(joint, k=k, largest=True).indices
            else:
                detail_rank = self._percentile_rank(
                    detail.to(joint.device).float().flatten()
                )
                joint_rank = self._percentile_rank(joint)
                mixed = (1.0 - a) * joint_rank + a * detail_rank
                selected = torch.argsort(mixed, descending=True, stable=True)[:k]
            locked = 0
            self.record("cls_detail_mix_select_events", 1)
            self.record("cls_detail_mix_selected", k)
            self.record("cls_detail_mix_alpha_x1000", int(round(1000.0 * a)))
        elif self.interface_final_variant in {
            "interface_pmi_cls_guarded", "encoder_pmi_cls_guarded"
        }:
            joint = score_joint.detach().float().flatten()
            n = int(joint.numel())
            k = max(0, min(int(keep_n), n))
            proposed_n = min(k, int(math.ceil(float(alpha) * float(k))))
            detail_order = torch.argsort(
                detail.to(joint.device).float(), descending=True, stable=True
            )
            proposed = detail_order[:proposed_n]
            joint_order = torch.argsort(joint, descending=True, stable=True)
            guard_n = min(n, k + proposed_n)
            guard_mask = torch.zeros(n, device=joint.device, dtype=torch.bool)
            guard_mask[joint_order[:guard_n]] = True
            locked_tokens = proposed[guard_mask.index_select(0, proposed)]
            locked = int(locked_tokens.numel())
            used = torch.zeros(n, device=joint.device, dtype=torch.bool)
            used[locked_tokens] = True
            fill = joint_order[~used.index_select(0, joint_order)][: k - locked]
            selected = torch.cat((locked_tokens, fill), dim=0)
            self.record("cls_detail_guard_events", 1)
            self.record("cls_detail_guard_proposed", proposed_n)
            self.record("cls_detail_guard_accepted", locked)
            self.record("cls_detail_guard_rejected", proposed_n - locked)
            self.record("cls_detail_guard_rank_span", proposed_n)
        else:
            selected, locked = super()._select_encoder_anchor_quota(
                detail.to(score_joint.device), score_joint, keep_n, alpha, score_text=score_text
            )
        self.record("cls_detail_quota_select_events", 1)
        self.record("cls_detail_quota_locked", locked)
        return selected, locked
