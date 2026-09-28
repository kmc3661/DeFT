from __future__ import annotations

import inspect
import importlib.util
import os
import sys
from types import MethodType
from typing import Optional, Union

import torch
from lmms_eval.api.registry import register_model
from deft._vendor.lmms_llava_onevision1_5 import Llava_OneVision1_5 as BaseLlavaOV15
from lmms_eval.models.simple.qwen2_5_vl import Qwen2_5_VL as BaseQwen25VL

from .progressive_pruning import ProgressiveConfig, ProgressivePruningState, as_bool

from deft._vendor.lmms_qwen3_vl import Qwen3_VL as BaseQwen3VL


def _as_float(x, default: float) -> float:
    if x is None or x == "":
        return default
    return float(x)


def _as_int(x, default: int) -> int:
    if x is None or x == "":
        return default
    return int(x)


def _set_kwarg_or_arg(args, kwargs, index: int, name: str, value):
    if name in kwargs:
        kwargs[name] = value
        return args, kwargs
    if len(args) > index:
        args = list(args)
        args[index] = value
        return tuple(args), kwargs
    kwargs[name] = value
    return args, kwargs


def _get_arg(args, kwargs, index: int, name: str, default=None):
    if name in kwargs:
        return kwargs[name]
    if len(args) > index:
        return args[index]
    return default


def _arg_index(fn, name: str, default: int) -> int:
    try:
        params = [p for p in inspect.signature(fn).parameters if p != "self"]
        return params.index(name)
    except Exception:
        return default


def _merge_key_drop_mask(attention_mask, key_drop: torch.Tensor, q_len: int, dtype: torch.dtype):
    bsz, key_len = key_drop.shape
    device = key_drop.device
    mask_value = torch.finfo(dtype if dtype.is_floating_point else torch.float32).min
    if attention_mask is None:
        out = torch.zeros((bsz, 1, q_len, key_len), device=device, dtype=torch.float32)
        if q_len > 1 and q_len == key_len:
            causal = torch.triu(torch.ones((q_len, key_len), device=device, dtype=torch.bool), diagonal=1)
            out = out.masked_fill(causal[None, None], mask_value)
    elif attention_mask.dtype == torch.bool:
        if attention_mask.ndim == 2:
            keep = attention_mask[:, None, None, :key_len].to(device=device)
        else:
            keep = attention_mask[:, :, :, :key_len].to(device=device)
        out = torch.zeros_like(keep, dtype=torch.float32).masked_fill(~keep, mask_value)
    else:
        out = attention_mask[:, :, :, :key_len].clone().to(device=device)
        if out.shape[-1] < key_len:
            pad = torch.zeros((*out.shape[:-1], key_len - out.shape[-1]), device=device, dtype=out.dtype)
            out = torch.cat([out, pad], dim=-1)

    drop = key_drop[:, None, None, :]
    out = out.masked_fill(drop, mask_value)
    return out.to(dtype=dtype if dtype.is_floating_point else torch.float32)


def _merge_key_drop_padding_mask(attention_mask, key_drop: torch.Tensor, key_len: int):
    """FlashAttention2-compatible token activity mask.

    HF FlashAttention2 accepts only a 2D padding mask here.  When prompt query
    length equals key length, its varlen path uses the same indices to unpad Q,
    K, and V.  Therefore this is query+key token masking, not an arbitrary
    key-only mask.  Physical vision compaction remains exact; reversible text
    key-only masking requires SDPA or a custom kernel and must not be described
    as equivalent to this path.
    """
    bsz = int(key_drop.shape[0])
    key_len = int(key_len)
    device = key_drop.device
    if attention_mask is None:
        keep = torch.ones((bsz, key_len), device=device, dtype=torch.bool)
        out_dtype = torch.long
    elif torch.is_tensor(attention_mask) and attention_mask.ndim == 2:
        keep = attention_mask[:, :key_len].to(device=device).bool().clone()
        if keep.shape[-1] < key_len:
            pad = torch.ones((bsz, key_len - keep.shape[-1]), device=device, dtype=torch.bool)
            keep = torch.cat([keep, pad], dim=-1)
        out_dtype = attention_mask.dtype
    else:
        # If a previous step already produced a non-2D mask, fall back to a
        # dense keep mask.  This function is only used before additive masks
        # are introduced, so this should be rare and is safer than passing 4D
        # masks into flash-attn.
        keep = torch.ones((bsz, key_len), device=device, dtype=torch.bool)
        out_dtype = torch.long
    n = min(int(key_drop.shape[-1]), key_len)
    if n > 0:
        keep[:, :n] &= ~key_drop[:, :n].to(device=device).bool()
    if out_dtype == torch.bool:
        return keep
    return keep.to(dtype=out_dtype)


def _zero_attention_output_queries(result, query_drop: torch.Tensor, q_len: int):
    """Match FA2 varlen query masking for dense attention backends.

    Keys are removed through the additive attention mask.  Dense SDPA still
    computes the selected query rows, so zero their final attention update to
    reproduce FA2's pad-back behavior while preserving the residual/MLP stream.
    """
    if query_drop is None or not torch.is_tensor(query_drop) or int(q_len) != int(query_drop.shape[-1]):
        return result
    drop = query_drop.bool()
    if not bool(drop.any().item()):
        return result
    if isinstance(result, tuple):
        if not result or not torch.is_tensor(result[0]):
            return result
        first = result[0].masked_fill(drop.unsqueeze(-1), 0)
        return (first, *result[1:])
    if isinstance(result, list):
        if not result or not torch.is_tensor(result[0]):
            return result
        out = list(result)
        out[0] = out[0].masked_fill(drop.unsqueeze(-1), 0)
        return out
    if torch.is_tensor(result):
        return result.masked_fill(drop.unsqueeze(-1), 0)
    return result


def _merge_additive_bias_mask(attention_mask, bias: torch.Tensor, q_len: int, dtype: torch.dtype):
    bsz, _, _, key_len = bias.shape
    device = bias.device
    mask_value = torch.finfo(dtype if dtype.is_floating_point else torch.float32).min
    if attention_mask is None:
        out = torch.zeros((bsz, 1, q_len, key_len), device=device, dtype=torch.float32)
        if q_len > 1 and q_len == key_len:
            causal = torch.triu(torch.ones((q_len, key_len), device=device, dtype=torch.bool), diagonal=1)
            out = out.masked_fill(causal[None, None], mask_value)
    elif attention_mask.dtype == torch.bool:
        if attention_mask.ndim == 2:
            keep = attention_mask[:, None, None, :key_len].to(device=device)
        else:
            keep = attention_mask[:, :, :, :key_len].to(device=device)
        out = torch.zeros_like(keep, dtype=torch.float32).masked_fill(~keep, mask_value)
    else:
        out = attention_mask[:, :, :, :key_len].clone().to(device=device)
        if out.shape[-1] < key_len:
            pad = torch.zeros((*out.shape[:-1], key_len - out.shape[-1]), device=device, dtype=out.dtype)
            out = torch.cat([out, pad], dim=-1)
    out = out + bias.to(device=device, dtype=out.dtype)
    return out.to(dtype=dtype if dtype.is_floating_point else torch.float32)


class _ProgressiveMixin:
    def _init_progressive(
        self,
        *,
        progressive_vision_pruning=False,
        progressive_text_masking=False,
        progressive_layer_k=4,
        progressive_vision_layer=0,
        progressive_vision_keep_ratio=1.0 / 9.0,
        progressive_vision_keep_tokens=0,
        progressive_vision_text_aware=False,
        progressive_vision_score_mode="text",
        progressive_vision_score_lambda=1.0,
        progressive_vision_anchor_alpha=0.0,
        progressive_vision_stage_keep_tokens=0,
        progressive_vision_stage_keep_ratio=0.5,
        progressive_vision_stage_budget_mode="ratio",
        progressive_vision_stage_layer=0,
        progressive_vision_stage_score_mode="vispruner",
        progressive_vision_stage_score_lambda=0.0,
        progressive_vision_stage_physical_drop=False,
        progressive_vision_stage_surplus_fraction=-1.0,
        progressive_vision_deferred_drop_layer=-1,
        progressive_vision_deferred_reserve_fraction=0.0,
        progressive_vision_deferred_rescore=False,
        progressive_vision_deferred_rescore_min_swaps=0,
        progressive_vision_auto_rule="corr_pos",
        progressive_vision_auto_min_history=2,
        progressive_vision_encoder_layer=-1,
        progressive_vision_merge=False,
        progressive_vision_merge_placement="legacy",
        progressive_vision_physical_drop=False,
        progressive_vision_merge_mode="weighted",
        progressive_vision_merge_temperature=0.07,
        progressive_important_ratio=0.5,
        progressive_text_threshold=0.12,
        progressive_text_correction="exposure_baseline_ratio",
        progressive_text_mask_mode="threshold",
        progressive_text_mask_ratio=0.0,
        progressive_text_mask_ratio_end=-1.0,
        progressive_text_layer_start=0.0,
        progressive_text_layer_end=1.0,
        progressive_text_piecewise="",
        progressive_text_adaptive_tau=1.0,
        progressive_text_adaptive_alpha=0.45,
        progressive_text_adaptive_floor=0.15,
        progressive_text_adaptive_min_ratio=0.0,
        progressive_text_adaptive_max_ratio=0.2,
        progressive_text_adaptive_floor_mode="warmup_median",
        progressive_text_adaptive_margin=0.0,
        progressive_text_adaptive_ramp_layers=0,
        progressive_text_adaptive_gate_tau=0.0,
        progressive_text_adaptive_gate_min_frac=0.0,
        progressive_text_soft_gamma=2.0,
        progressive_text_grounding_lambda=0.0,
        progressive_text_ema_beta=0.8,
        progressive_analysis_path="",
        progressive_debug=False,
    ) -> None:
        cfg = ProgressiveConfig(
            vision_enabled=as_bool(progressive_vision_pruning),
            text_enabled=as_bool(progressive_text_masking),
            layer_k=_as_int(progressive_layer_k, 4),
            vision_layer=_as_int(progressive_vision_layer, 0),
            vision_keep_ratio=_as_float(progressive_vision_keep_ratio, 1.0 / 9.0),
            vision_keep_tokens=_as_int(progressive_vision_keep_tokens, 0),
            vision_text_aware=as_bool(progressive_vision_text_aware),
            vision_score_mode=str(progressive_vision_score_mode or "text"),
            vision_score_lambda=_as_float(progressive_vision_score_lambda, 1.0),
            vision_anchor_alpha=_as_float(progressive_vision_anchor_alpha, 0.0),
            vision_stage_keep_tokens=_as_int(progressive_vision_stage_keep_tokens, 0),
            vision_stage_keep_ratio=_as_float(progressive_vision_stage_keep_ratio, 0.5),
            vision_stage_budget_mode=str(progressive_vision_stage_budget_mode or "ratio"),
            vision_stage_layer=_as_int(progressive_vision_stage_layer, 0),
            vision_stage_score_mode=str(progressive_vision_stage_score_mode or "vispruner"),
            vision_stage_score_lambda=_as_float(progressive_vision_stage_score_lambda, 0.0),
            vision_stage_physical_drop=as_bool(progressive_vision_stage_physical_drop),
            vision_stage_surplus_fraction=_as_float(progressive_vision_stage_surplus_fraction, -1.0),
            vision_deferred_drop_layer=_as_int(progressive_vision_deferred_drop_layer, -1),
            vision_deferred_reserve_fraction=_as_float(progressive_vision_deferred_reserve_fraction, 0.0),
            vision_deferred_rescore=as_bool(progressive_vision_deferred_rescore),
            vision_deferred_rescore_min_swaps=_as_int(progressive_vision_deferred_rescore_min_swaps, 0),
            vision_auto_rule=str(progressive_vision_auto_rule or "corr_pos"),
            vision_auto_min_history=_as_int(progressive_vision_auto_min_history, 2),
            vision_encoder_layer=_as_int(progressive_vision_encoder_layer, -1),
            vision_merge=as_bool(progressive_vision_merge),
            vision_merge_placement=str(progressive_vision_merge_placement or "legacy"),
            vision_physical_drop=as_bool(progressive_vision_physical_drop),
            vision_merge_mode=str(progressive_vision_merge_mode or "weighted"),
            vision_merge_temperature=_as_float(progressive_vision_merge_temperature, 0.07),
            important_ratio=_as_float(progressive_important_ratio, 0.5),
            text_threshold=_as_float(progressive_text_threshold, 0.12),
            text_correction=str(progressive_text_correction or "exposure_baseline_ratio"),
            text_mask_mode=str(progressive_text_mask_mode or "threshold"),
            text_mask_ratio=_as_float(progressive_text_mask_ratio, 0.0),
            text_mask_ratio_end=_as_float(progressive_text_mask_ratio_end, -1.0),
            text_layer_start=_as_float(progressive_text_layer_start, 0.0),
            text_layer_end=_as_float(progressive_text_layer_end, 1.0),
            text_piecewise=str(progressive_text_piecewise or ""),
            text_adaptive_tau=_as_float(progressive_text_adaptive_tau, 1.0),
            text_adaptive_alpha=_as_float(progressive_text_adaptive_alpha, 0.45),
            text_adaptive_floor=_as_float(progressive_text_adaptive_floor, 0.15),
            text_adaptive_min_ratio=_as_float(progressive_text_adaptive_min_ratio, 0.0),
            text_adaptive_max_ratio=_as_float(progressive_text_adaptive_max_ratio, 0.2),
            text_adaptive_floor_mode=str(progressive_text_adaptive_floor_mode or "warmup_median"),
            text_adaptive_margin=_as_float(progressive_text_adaptive_margin, 0.0),
            text_adaptive_ramp_layers=_as_int(progressive_text_adaptive_ramp_layers, 0),
            text_adaptive_gate_tau=_as_float(progressive_text_adaptive_gate_tau, 0.0),
            text_adaptive_gate_min_frac=_as_float(progressive_text_adaptive_gate_min_frac, 0.0),
            text_soft_gamma=_as_float(progressive_text_soft_gamma, 2.0),
            text_grounding_lambda=_as_float(progressive_text_grounding_lambda, 0.0),
            text_ema_beta=_as_float(progressive_text_ema_beta, 0.8),
            analysis_path=str(progressive_analysis_path or ""),
            num_layers=self._infer_num_layers(),
            debug=as_bool(progressive_debug),
        )
        prior_method = str(os.environ.get("VISPRUNER_PRIOR_METHOD", "") or "").strip().lower()
        interface_component = as_bool(
            os.environ.get("VISPRUNER_INTERFACE_COMPONENT_ABLATION", "0")
        )
        blindspot_component = as_bool(
            os.environ.get("VISPRUNER_BLINDSPOT_ABLATION", "0")
        )
        active_components = int(bool(prior_method)) + int(interface_component) + int(blindspot_component)
        if active_components > 1:
            raise ValueError(
                "paper prior, interface diagnostic, and blind-spot ablation are mutually exclusive"
            )
        if interface_component:
            if as_bool(os.environ.get("VISPRUNER_CLS_DETAIL_ROUTER", "0")):
                from .cls_detail_router import ClsDetailEndpointRouterState

                self._progressive = ClsDetailEndpointRouterState(cfg)
            elif as_bool(os.environ.get("VISPRUNER_INTERFACE_SELECTOR_ROUTER", "0")):
                from .interface_router import InterfaceSelectorRouterState

                self._progressive = InterfaceSelectorRouterState(cfg)
            elif as_bool(os.environ.get("VISPRUNER_INTERFACE_RESIDUAL_ABLATION", "0")):
                from .interface_residual_ablation import InterfaceResidualSummaryAblationState

                self._progressive = InterfaceResidualSummaryAblationState(cfg)
            else:
                from .interface_ablation import InterfaceComponentAblationState

                self._progressive = InterfaceComponentAblationState(cfg)
        elif blindspot_component:
            from .blindspot_ablation import EncoderBlindspotAblationState

            self._progressive = EncoderBlindspotAblationState(cfg)
        elif prior_method:
            from .prior_methods import PriorMethodPruningState

            self._progressive = PriorMethodPruningState(cfg, prior_method)
        else:
            self._progressive = ProgressivePruningState(cfg)
        if bool(getattr(self, "_hf_official_method", "")):
            # Released HF LLaVA baselines prune before decoder layer 0. Their
            # shortened placeholder path supports the native KV cache and must
            # not install the generic mid-decoder physical-compaction wrapper.
            cfg.vision_physical_drop = False
            cfg.vision_stage_physical_drop = False
        if cfg.vision_physical_drop:
            # Physical sequence compaction changes the prompt length mid-forward.
            # Force no-cache both in the lmms wrapper and HF configs so generation
            # re-runs the full prompt instead of mixing compacted/non-compacted KV.
            try:
                self.use_cache = False
            except Exception:
                pass
            for _cfg_obj in (
                getattr(self.model, "config", None),
                getattr(getattr(self.model, "config", None), "text_config", None),
                getattr(getattr(self.model, "model", None), "config", None),
                getattr(getattr(getattr(self.model, "model", None), "language_model", None), "config", None),
            ):
                try:
                    if _cfg_obj is not None:
                        _cfg_obj.use_cache = False
                except Exception:
                    pass
        self._vision_encoder_saliency_parts = {}
        self._vision_encoder_saliency_targets = ()
        if cfg.enabled:
            self._patch_model_forward()
            self._patch_visual_attentions()
            self._patch_official_vispruner_features()
            if hasattr(self._progressive, "install_interface_projector_hooks"):
                self._progressive.install_interface_projector_hooks(self.model)
            if hasattr(self._progressive, "install_blindspot_projector_hooks"):
                self._progressive.install_blindspot_projector_hooks(self.model)
            self._patch_zooprune_projector()
            self._patch_decoder_attentions()
            self._patch_native_text_proxies()
            self._patch_language_model_physical_drop()

    def generate_until(self, requests):
        prog = getattr(self, "_progressive", None)
        if prog is not None:
            prog.stats.clear()
        try:
            return super().generate_until(requests)
        finally:
            prog = getattr(self, "_progressive", None)
            if prog is not None and prog.config.debug:
                summary = ", ".join(f"{k}={v:g}" for k, v in sorted(prog.stats.items()))
                prog._debug(f"summary {summary}")

    def _infer_num_layers(self) -> int:
        cfg = self.model.config
        for obj in (cfg, getattr(cfg, "text_config", None), getattr(cfg, "language_config", None)):
            if obj is not None and getattr(obj, "num_hidden_layers", None) is not None:
                return int(getattr(obj, "num_hidden_layers"))
        layers = [getattr(m, "layer_idx", None) for _, m in self.model.named_modules()]
        layers = [int(x) for x in layers if x is not None]
        return max(layers) + 1 if layers else 32

    def _vision_indices(self, input_ids_1d: torch.Tensor) -> torch.Tensor:
        cfg = self.model.config
        ids = []
        for name in ("image_token_id", "video_token_id", "image_token_index", "video_token_index"):
            value = getattr(cfg, name, None)
            if value is not None:
                ids.append(int(value))
        if not ids:
            return torch.zeros_like(input_ids_1d, dtype=torch.bool)
        out = torch.zeros_like(input_ids_1d, dtype=torch.bool)
        for token_id in sorted(set(ids)):
            out |= input_ids_1d == token_id
        return out

    def _patch_model_forward(self) -> None:
        model = self.model
        if getattr(model, "_vispruner_progressive_forward_patched", False):
            return
        original_forward = model.forward
        owner = self

        def wrapped_forward(model_self, *args, **kwargs):
            if owner._progressive.physical_drop_enabled():
                kwargs["use_cache"] = False
                kwargs["past_key_values"] = None
            input_ids = _get_arg(args, kwargs, 0, "input_ids")
            attention_mask = _get_arg(args, kwargs, 1, "attention_mask")
            began_prompt = bool(
                owner._progressive.enabled
                and torch.is_tensor(input_ids)
                and input_ids.ndim == 2
                and input_ids.shape[1] > 1
            )
            if began_prompt:
                if hasattr(owner._progressive, "set_hawk_special_token_ids"):
                    cfg = getattr(owner.model, "config", None)
                    text_cfg = getattr(cfg, "text_config", None)
                    generation_cfg = getattr(owner.model, "generation_config", None)
                    vision_end_id = getattr(cfg, "vision_end_token_id", None)
                    if vision_end_id is None:
                        # Both evaluated Qwen-family checkpoints use the public
                        # Qwen vision-end ID even when an older config omits it.
                        vision_end_id = 151653
                    im_end_id = getattr(text_cfg, "eos_token_id", None)
                    if im_end_id is None:
                        im_end_id = getattr(generation_cfg, "eos_token_id", None)
                    if isinstance(im_end_id, (list, tuple)):
                        im_end_id = im_end_id[0] if im_end_id else None
                    if im_end_id is None:
                        # Qwen/Qwen2 chat templates define <|im_end|> as 151645.
                        im_end_id = 151645
                    owner._progressive.set_hawk_special_token_ids(vision_end_id, im_end_id)
                owner._progressive.begin(input_ids, attention_mask, owner._vision_indices)
                owner._vision_encoder_saliency_parts = {}
            result = original_forward(*args, **kwargs)
            if began_prompt:
                owner._progressive.finish_forward()
            return result

        wrapped_forward.__signature__ = inspect.signature(original_forward)
        model.forward = MethodType(wrapped_forward, model)
        model._vispruner_progressive_forward_patched = True

        # The physical-drop path deliberately disables the HF KV cache.  Wrap
        # generate itself so repeated full-prefix forwards share one prompt-local
        # text-mask cache, while every new sample starts from a clean state.
        if hasattr(model, "generate") and not getattr(model, "_vispruner_progressive_generate_patched", False):
            original_generate = model.generate

            def wrapped_generate(model_self, *args, __orig=original_generate, **kwargs):
                if bool(getattr(owner._progressive, "_generation_active", False)):
                    return __orig(*args, **kwargs)
                owner._progressive.start_generation()
                try:
                    return __orig(*args, **kwargs)
                finally:
                    owner._progressive.finish_generation()

            wrapped_generate.__signature__ = inspect.signature(original_generate)
            model.generate = MethodType(wrapped_generate, model)
            model._vispruner_progressive_generate_patched = True

    def _uses_encoder_score(self) -> bool:
        modes = {
            str(getattr(self._progressive.config, "vision_score_mode", "") or "").lower(),
            str(getattr(self._progressive.config, "vision_stage_score_mode", "") or "").lower(),
        }
        encoder_modes = {"vispruner", "original_vispruner", "orig_vispruner", "encoder_vispruner", "encoder", "encoder_only", "enc", "anchor_regional", "encoder_anchor_regional", "regional_reserve", "anchor_regional_cls", "encoder_anchor_regional_cls", "feature_diverse", "encoder_feature_diverse", "feature_novelty", "encoder_topk", "encoder_saliency_topk", "encoder_text_add", "enc_text_add", "encoder_add", "encoder_text_mul", "enc_text_mul", "encoder_mul", "encoder_pmi_add", "enc_pmi_add", "encoder_text_pmi_add", "encoder_pmi", "encoder_pmi_add_topk", "encoder_pmi_topk", "encoder_pmi_learned_alpha_topk", "encoder_pmi_multialpha_vote_topk", "encoder_pmi_multialpha_borda_topk", "encoder_pmi_multialpha_minimax_topk", "encoder_pmi_add_entropy", "encoder_pmi_add_adaptive", "encoder_pmi_entropy", "encoder_pmi_add_jsd", "encoder_pmi_jsd", "encoder_pmi_add_jsd_sqrt", "encoder_pmi_jsd_sqrt", "encoder_pmi_add_tv", "encoder_pmi_tv", "encoder_pmi_add_concentration", "encoder_pmi_concentration", "encoder_pmi_add_inv_jsd", "encoder_pmi_inv_jsd", "encoder_pmi_add_inv_jsd_sqrt", "encoder_pmi_inv_jsd_sqrt", "encoder_pmi_add_topk_consensus", "encoder_pmi_topk_consensus", "encoder_pmi_add_rel_conf", "encoder_pmi_rel_conf", "encoder_pmi_add_rel_margin", "encoder_pmi_rel_margin", "encoder_pmi_mul", "enc_pmi_mul", "encoder_text_pmi_mul", "encoder_contribution_add_topk", "functional_coreset_topk", "functional_coreset_merge", "functional_role_coreset_topk", "functional_user_coreset_topk", "text_pmi_reserve_functional_user_topk", "compression_adaptive_functional_user_topk", "competition_adaptive_functional_user_topk", "majority_support_adaptive_functional_user_topk", "functional_user_guarded_topk", "encoder_pmi_functional_alpha_topk"}
        return bool(modes & encoder_modes)

    def _uses_cls_encoder_score(self) -> bool:
        modes = {
            str(getattr(self._progressive.config, "vision_score_mode", "") or "").lower(),
            str(getattr(self._progressive.config, "vision_stage_score_mode", "") or "").lower(),
        }
        return bool(modes & {"anchor_regional_cls", "encoder_anchor_regional_cls"})

    def _vision_encoder_saliency_mode(self) -> str:
        mode = str(os.environ.get("VISPRUNER_ENCODER_SALIENCY_AGGREGATION", "last1") or "last1").strip().lower()
        aliases = {
            "last": "last1", "last_1": "last1",
            "late4_avg": "late4_mean", "late4_average": "late4_mean",
            "late4_geo": "late4_consensus", "late4_geomean": "late4_consensus",
            "late4_max": "late4_any", "all_mean": "all_layer_mean",
        }
        mode = aliases.get(mode, mode)
        allowed = {"last1", "late4_mean", "late4_consensus", "late4_any", "all_layer_mean"}
        if mode not in allowed:
            raise ValueError(f"unknown encoder saliency aggregation: {mode}")
        return mode

    def _vision_encoder_saliency_target_layers(self, n_blocks: int, default_target: int) -> tuple[int, ...]:
        mode = self._vision_encoder_saliency_mode()
        if mode == "last1":
            return (max(0, min(int(n_blocks) - 1, int(default_target))),)
        if mode == "all_layer_mean":
            return tuple(range(max(0, int(n_blocks))))
        start = max(0, int(n_blocks) - 4)
        return tuple(range(start, int(n_blocks)))

    def _accumulate_visual_encoder_saliency(
        self,
        score: torch.Tensor,
        layer_idx: int,
        targets: tuple[int, ...],
    ) -> None:
        if score is None or not torch.is_tensor(score) or score.numel() == 0:
            self._progressive.record("vision_encoder_multilayer_score_missing", 1)
            return
        mode = self._vision_encoder_saliency_mode()
        vals = torch.nan_to_num(score.detach().float().flatten(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        if vals.numel() == 0:
            return
        prob = vals / vals.sum().clamp_min(1e-12)
        self._vision_encoder_saliency_parts[int(layer_idx)] = prob
        self._progressive.record("vision_encoder_multilayer_parts", 1)
        self._progressive.record("vision_encoder_multilayer_layer_sum", int(layer_idx))
        if int(layer_idx) != int(targets[-1]):
            return
        missing = [int(x) for x in targets if int(x) not in self._vision_encoder_saliency_parts]
        if missing:
            raise RuntimeError(f"missing encoder saliency layers for {mode}: {missing}")
        parts = torch.stack([self._vision_encoder_saliency_parts[int(x)] for x in targets], dim=0)
        if mode == "last1":
            aggregate = vals
        elif mode in {"late4_mean", "all_layer_mean"}:
            aggregate = parts.mean(dim=0)
        elif mode == "late4_consensus":
            aggregate = torch.exp(torch.log(parts.clamp_min(1e-12)).mean(dim=0))
        elif mode == "late4_any":
            aggregate = parts.max(dim=0).values
        else:
            raise RuntimeError(mode)
        aggregate = aggregate / aggregate.sum().clamp_min(1e-12)
        last = parts[-1]
        top_n = max(1, min(int(aggregate.numel()), int(getattr(self._progressive.config, "vision_keep_tokens", 0) or 1) * 4))
        top_agg = torch.topk(aggregate, k=top_n, largest=True).indices
        top_last = torch.topk(last, k=top_n, largest=True).indices
        overlap = float(torch.isin(top_agg, top_last).float().mean().item())
        l1 = float((aggregate - last).abs().sum().item())
        entropy = float((-(aggregate * aggregate.clamp_min(1e-12).log()).sum()).item())
        self._progressive.record("vision_encoder_multilayer_finalize", 1)
        self._progressive.record(f"vision_encoder_aggregation_{mode}", 1)
        self._progressive.record("vision_encoder_aggregation_layers", len(targets))
        self._progressive.record("vision_encoder_aggregation_l1_vs_last", l1)
        self._progressive.record("vision_encoder_aggregation_top_overlap_vs_last", overlap)
        self._progressive.record("vision_encoder_aggregation_entropy", entropy)
        self._progressive.set_vision_encoder_score(
            aggregate, source=f"visual_all_query_{mode}_layers={','.join(str(x) for x in targets)}"
        )

    def _patch_official_vispruner_features(self) -> None:
        """Capture pre-projector visual features at native LLM-token granularity.

        Released VisPruner computes diversity before the multimodal projector.
        Qwen3-VL and OneVision-1.5 merge each native 2x2 patch group into one
        decoder token, so one selectable feature is the flattened native merger
        input group. The native merger itself remains unchanged.
        """
        if str(getattr(self._progressive, "prior_method", "") or "").lower() != "official_vispruner":
            return
        visual = getattr(self.model, "visual", None)
        merger = getattr(visual, "merger", None)
        if merger is None or getattr(merger, "_vispruner_official_features_patched", False):
            raise RuntimeError("official VisPruner requires an unpatched native visual merger")
        original_forward = merger.forward
        owner = self

        def wrapped_merger(merger_self, *args, __orig=original_forward, **kwargs):
            pre_projector = _get_arg(args, kwargs, 0, "x")
            if pre_projector is None:
                pre_projector = _get_arg(args, kwargs, 0, "hidden_states")
            projected = __orig(*args, **kwargs)
            if not torch.is_tensor(pre_projector) or not torch.is_tensor(projected):
                raise RuntimeError("official VisPruner native merger did not expose tensor input/output")
            if pre_projector.ndim != 2 or projected.ndim != 2 or int(projected.shape[0]) <= 0:
                raise RuntimeError(
                    f"official VisPruner unsupported merger shapes: pre={tuple(pre_projector.shape)} "
                    f"projected={tuple(projected.shape)}"
                )
            target_n = int(projected.shape[0])
            if int(pre_projector.shape[0]) % target_n != 0:
                raise RuntimeError("official VisPruner merger groups are not integral")
            group = int(pre_projector.shape[0]) // target_n
            grouped = pre_projector.reshape(target_n, group, -1).reshape(target_n, -1)
            owner._progressive.set_official_vispruner_features(grouped, group=group)
            return projected

        wrapped_merger.__signature__ = inspect.signature(original_forward)
        merger.forward = MethodType(wrapped_merger, merger)
        merger._vispruner_official_features_patched = True

    def _patch_zooprune_projector(self) -> None:
        """Query the native visual merger for ZOO-Prune sensitivity."""
        if str(getattr(self._progressive, "prior_method", "") or "").lower() != "zooprune":
            return
        visual = getattr(self.model, "visual", None)
        merger = getattr(visual, "merger", None)
        if merger is None or getattr(merger, "_vispruner_zooprune_patched", False):
            raise RuntimeError("ZOO-Prune requires an unpatched native visual merger")
        original_forward = merger.forward
        owner = self

        def wrapped_merger(merger_self, *args, __orig=original_forward, **kwargs):
            pre_projector = _get_arg(args, kwargs, 0, "x")
            if pre_projector is None:
                pre_projector = _get_arg(args, kwargs, 0, "hidden_states")
            projected = __orig(*args, **kwargs)
            if torch.is_tensor(pre_projector) and torch.is_tensor(projected):
                owner._progressive.capture_zooprune_projector_sensitivity(
                    pre_projector,
                    projected,
                    lambda perturbed: __orig(perturbed),
                )
            else:
                raise RuntimeError("ZOO-Prune native merger did not expose tensor input/output")
            return projected

        wrapped_merger.__signature__ = inspect.signature(original_forward)
        merger.forward = MethodType(wrapped_merger, merger)
        merger._vispruner_zooprune_patched = True

    def _patch_visual_attentions(self) -> None:
        visual = getattr(self.model, "visual", None)
        blocks = getattr(visual, "blocks", None)
        if visual is None or blocks is None:
            return
        n_blocks = len(blocks)
        prior_method = str(getattr(self._progressive, "prior_method", "") or "").lower()
        if prior_method == "visionzip":
            # Official Qwen2.5-VL port uses the last visual block. The original
            # CLIP/LLaVA path uses second-last-layer CLS attention.
            target = n_blocks - 2 if hasattr(visual, "class_embedding") else n_blocks - 1
            if not getattr(visual, "_visionzip_outer_forward_patched", False):
                original_visual_forward = visual.forward
                owner = self
                def wrapped_visual_forward(visual_self, *args, __orig=original_visual_forward, **kwargs):
                    grid_thw = _get_arg(args, kwargs, 1, "grid_thw")
                    owner._visionzip_window_index = None
                    if torch.is_tensor(grid_thw) and hasattr(visual_self, "get_window_index"):
                        try:
                            owner._visionzip_window_index = visual_self.get_window_index(grid_thw)[0].detach()
                        except Exception:
                            owner._visionzip_window_index = None
                    return __orig(*args, **kwargs)
                wrapped_visual_forward.__signature__ = inspect.signature(original_visual_forward)
                visual.forward = MethodType(wrapped_visual_forward, visual)
                visual._visionzip_outer_forward_patched = True
        else:
            target = int(getattr(self._progressive.config, "vision_encoder_layer", -1))
            if target < 0:
                target = n_blocks + target
        target = max(0, min(n_blocks - 1, target)) if n_blocks > 0 else 0
        targets = self._vision_encoder_saliency_target_layers(n_blocks, target)
        self._vision_encoder_saliency_targets = targets
        needs_cls_detail = bool(
            callable(getattr(self._progressive, "needs_cls_detail_score", None))
            and self._progressive.needs_cls_detail_score()
        )
        has_visual_cls = hasattr(visual, "class_embedding")
        # A global-summary path is defined structurally: CLS-query attention
        # when the encoder exposes CLS, otherwise all-query received attention
        # at the native encoder-saliency layer.  This matches VisionZip's
        # architecture-independent definition without branching on model name.
        detail_target = (
            n_blocks - 2 if needs_cls_detail and has_visual_cls and n_blocks > 1
            else target if needs_cls_detail and not has_visual_cls and n_blocks > 0
            else None
        )
        if needs_cls_detail and detail_target is None:
            raise RuntimeError("CLS-detail quota requires a visual CLS token and at least two encoder blocks")
        for idx, block in enumerate(blocks):
            attn = getattr(block, "attn", None)
            if attn is None or getattr(attn, "_vispruner_visual_attn_patched", False):
                continue
            original_forward = attn.forward
            owner = self
            attn._vispruner_visual_layer_idx = idx

            def wrapped_visual_attention(attn_self, *args, __orig=original_forward, **kwargs):
                hidden_states = _get_arg(args, kwargs, 0, "hidden_states")
                cu_seqlens = _get_arg(args, kwargs, 1, "cu_seqlens")
                layer_idx = int(getattr(attn_self, "_vispruner_visual_layer_idx", -1))
                prior_is_visionzip = (
                    str(getattr(owner._progressive, "prior_method", "") or "").lower() == "visionzip"
                )
                primary_layers = (target,) if prior_is_visionzip else targets
                capture_primary = layer_idx in primary_layers
                capture_detail = detail_target is not None and layer_idx == int(detail_target)
                if (
                    owner._progressive.enabled
                    and owner._uses_encoder_score()
                    and (capture_primary or capture_detail)
                    and torch.is_tensor(hidden_states)
                    and torch.is_tensor(cu_seqlens)
                ):
                    if prior_is_visionzip:
                        stats = owner._compute_visionzip_encoder_statistics(
                            attn_self, __orig, hidden_states, cu_seqlens, args, kwargs
                        )
                        if stats is not None:
                            score, metric = stats
                            owner._progressive.set_vision_encoder_score(score, source=f"visionzip_visual_layer={target}")
                            owner._progressive.set_visionzip_encoder_metric(metric, source=f"visionzip_visual_layer={target}")
                        else:
                            owner._progressive.record("visionzip_encoder_statistics_none", 1)
                    else:
                        if capture_detail:
                            detail_score = owner._compute_visual_encoder_score(
                                attn_self, __orig, hidden_states, cu_seqlens, args, kwargs,
                                query_mode="cls" if has_visual_cls else "all",
                            )
                            setter = getattr(owner._progressive, "set_cls_detail_score", None)
                            if detail_score is None or not callable(setter):
                                raise RuntimeError("failed to capture second-last CLS-detail saliency")
                            setter(detail_score, source=f"visual_cls_query_layer={layer_idx}")
                        if capture_primary:
                            query_mode = "cls" if owner._uses_cls_encoder_score() else "all"
                            if (capture_detail and query_mode == ("cls" if has_visual_cls else "all")
                                    and os.environ.get("VISPRUNER_ENCODER_REUSE_IDENTICAL_SCORE", "1") == "1"):
                                # Same layer/input/query rule: the detail result
                                # is also the primary score, not an approximation.
                                score = detail_score
                                owner._progressive.record("vision_encoder_identical_score_reuse", 1)
                            else:
                                score = owner._compute_visual_encoder_score(
                                    attn_self,
                                    __orig,
                                    hidden_states,
                                    cu_seqlens,
                                    args,
                                    kwargs,
                                    query_mode=query_mode,
                                )
                            if score is not None:
                                owner._accumulate_visual_encoder_saliency(score, layer_idx, targets)
                            else:
                                owner._progressive.record("vision_encoder_score_none", 1)
                return __orig(*args, **kwargs)

            wrapped_visual_attention.__signature__ = inspect.signature(original_forward)
            def efficient_visual_attention(attn_self, *args, __wrapped=wrapped_visual_attention,
                                           __score_layer=(idx in targets or idx == detail_target), **kwargs):
                if (__score_layer and owner._progressive.enabled and owner._uses_encoder_score()
                        and not torch.is_grad_enabled() and not attn_self.training
                        and os.environ.get("VISPRUNER_ENCODER_QKV_REUSE", "1") == "1"):
                    from .encoder_exact_reuse import with_qkv_reuse
                    return with_qkv_reuse(attn_self.qkv, lambda: __wrapped(attn_self, *args, **kwargs))
                return __wrapped(attn_self, *args, **kwargs)
            efficient_visual_attention.__signature__ = inspect.signature(original_forward)
            attn.forward = MethodType(efficient_visual_attention, attn)
            attn._vispruner_visual_attn_patched = True

    def _compute_visual_encoder_score(
        self, attn, original_forward, hidden_states, cu_seqlens, args, kwargs, query_mode="all"
    ):
        try:
            seq_length = int(hidden_states.shape[0])
            qkv = attn.qkv(hidden_states).reshape(
                seq_length, 3, int(attn.num_heads), -1
            ).permute(1, 0, 2, 3)
            q, k, value = qkv[0], qkv[1], qkv[2]
            position_embeddings = _get_arg(args, kwargs, 3, "position_embeddings")
            rotary_pos_emb = _get_arg(args, kwargs, 2, "rotary_pos_emb")
            glb = dict(getattr(getattr(original_forward, "__func__", original_forward), "__globals__", {}) or {})
            module = sys.modules.get(attn.__class__.__module__)
            if module is not None:
                glb.update(vars(module))
            if position_embeddings is None and rotary_pos_emb is not None:
                emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
                position_embeddings = (emb.cos(), emb.sin())
            if position_embeddings is not None and "apply_rotary_pos_emb_vision" in glb:
                cos, sin = position_embeddings
                q, k = glb["apply_rotary_pos_emb_vision"](q, k, cos, sin)
            q = q.transpose(0, 1).float()  # heads, seq, dim
            k = k.transpose(0, 1).float()
            scaling = float(getattr(attn, "scaling", q.shape[-1] ** -0.5))
            received = torch.zeros((seq_length,), device=hidden_states.device, dtype=torch.float32)
            blindspot_mode = ""
            mode_getter = getattr(self._progressive, "encoder_blindspot_signal_mode", None)
            if callable(mode_getter):
                blindspot_mode = str(mode_getter() or "").strip().lower()
            needs_head_statistics = blindspot_mode in {"routed_value", "minority_head"}
            head_received = (
                torch.zeros(
                    (int(attn.num_heads), seq_length),
                    device=hidden_states.device,
                    dtype=torch.float32,
                )
                if needs_head_statistics
                else None
            )
            cu = cu_seqlens.detach().to(device=hidden_states.device, dtype=torch.long).flatten()
            # Integer metadata can be read once; avoid a device synchronization
            # for every segment boundary (and again for CLS removal).
            boundaries = cu.detach().cpu().tolist() if os.environ.get("VISPRUNER_ENCODER_BOUNDARY_REUSE", "1") == "1" else cu
            try:
                chunk = int(os.environ.get(
                    "VISPRUNER_ENCODER_QUERY_CHUNK",
                    os.environ.get("VISPRUNER_SCORE_QUERY_CHUNK", "128"),
                ))
            except Exception:
                chunk = 128
            if chunk <= 0:
                chunk = seq_length
            for i in range(1, int(cu.numel())):
                start = int(boundaries[i - 1])
                end = int(boundaries[i])
                if end <= start:
                    continue
                ks = k[:, start:end, :]
                seg_len = end - start
                if query_mode == "cls":
                    visual = getattr(self.model, "visual", None)
                    if not hasattr(visual, "class_embedding"):
                        raise RuntimeError("CLS-query saliency requires a visual encoder CLS token")
                    logits = torch.einsum("hqd,hkd->hqk", q[:, start : start + 1, :], ks) * scaling
                    probs = torch.softmax(logits, dim=-1)
                    per_head = probs.sum(dim=1)
                    received[start:end] = per_head.mean(dim=0)
                    if head_received is not None:
                        head_received[:, start:end] = per_head
                else:
                    seg_received = torch.zeros((seg_len,), device=hidden_states.device, dtype=torch.float32)
                    for qs in range(0, seg_len, max(1, chunk)):
                        qe = min(seg_len, qs + max(1, chunk))
                        logits = torch.einsum(
                            "hqd,hkd->hqk", q[:, start + qs : start + qe, :], ks
                        ) * scaling
                        probs = torch.softmax(logits, dim=-1)
                        per_head = probs.sum(dim=1)
                        seg_received += per_head.mean(dim=0)
                        if head_received is not None:
                            head_received[:, start:end] += per_head
                    received[start:end] = seg_received

            auxiliary = None
            if blindspot_mode == "minority_head":
                if head_received is None:
                    raise RuntimeError("minority-head statistic was not accumulated")
                # Each head has the same total received mass within a segment;
                # max therefore exposes evidence concentrated in one head that
                # mean-head attention suppresses.
                auxiliary = head_received.max(dim=0).values
                self._progressive.record("vision_encoder_minor_head_statistics", 1)
            elif blindspot_mode == "routed_value":
                if not hasattr(attn, "proj"):
                    raise RuntimeError("routed-value statistic requires the native vision attention output projection")
                projected_value = attn.proj(
                    value.reshape(seq_length, -1).to(dtype=attn.proj.weight.dtype)
                ).float()
                value_magnitude = projected_value.norm(dim=-1)
                auxiliary = received * value_magnitude
                self._progressive.record("vision_encoder_routed_value_statistics", 1)
                self._progressive.record(
                    "vision_encoder_routed_value_norm_sum", float(value_magnitude.sum().item())
                )
                self._progressive.record("vision_encoder_extra_projector_calls", 1)

            # LLaVA-OV/Rice inserts one CLS token per visual segment before
            # visual self-attention and removes it before the patch merger.
            visual = getattr(self.model, "visual", None)
            if hasattr(visual, "class_embedding") and cu.numel() > 1:
                keep_pieces = []
                auxiliary_pieces = []
                removed = 0
                for i in range(1, int(cu.numel())):
                    start = int(boundaries[i - 1])
                    end = int(boundaries[i])
                    if end <= start:
                        continue
                    keep_pieces.append(received[start + 1 : end])
                    if auxiliary is not None:
                        auxiliary_pieces.append(auxiliary[start + 1 : end])
                    removed += 1
                if keep_pieces:
                    received = torch.cat(keep_pieces, dim=0)
                    if auxiliary is not None:
                        auxiliary = torch.cat(auxiliary_pieces, dim=0)
                    self._progressive.record("vision_encoder_score_cls_removed", removed)

            if auxiliary is not None:
                setter = getattr(self._progressive, "set_encoder_blindspot_score", None)
                if not callable(setter):
                    raise RuntimeError("blind-spot statistic was computed without an ablation state")
                setter(auxiliary, source=f"visual_last_block_{blindspot_mode}")
            self._progressive.record("vision_encoder_score_hook_calls", 1)
            self._progressive.record(
                "vision_encoder_score_cls_query" if query_mode == "cls" else "vision_encoder_score_all_query", 1
            )
            return received.detach()
        except Exception as exc:
            self._progressive.record("vision_encoder_score_error", 1)
            if self._progressive.config.debug:
                self._progressive._debug(f"vision_encoder_score_error {type(exc).__name__}: {exc}")
            return None

    def _compute_visionzip_encoder_statistics(self, attn, original_forward, hidden_states, cu_seqlens, args, kwargs):
        """Reproduce VisionZip saliency/key metric at the visual encoder."""
        try:
            seq_length = int(hidden_states.shape[0])
            qkv = attn.qkv(hidden_states).reshape(seq_length, 3, int(attn.num_heads), -1).permute(1, 0, 2, 3)
            q, k = qkv[0], qkv[1]
            position_embeddings = _get_arg(args, kwargs, 3, "position_embeddings")
            rotary_pos_emb = _get_arg(args, kwargs, 2, "rotary_pos_emb")
            glb = dict(getattr(getattr(original_forward, "__func__", original_forward), "__globals__", {}) or {})
            module = sys.modules.get(attn.__class__.__module__)
            if module is not None:
                glb.update(vars(module))
            if position_embeddings is None and rotary_pos_emb is not None:
                emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
                position_embeddings = (emb.cos(), emb.sin())
            if position_embeddings is not None and "apply_rotary_pos_emb_vision" in glb:
                cos, sin = position_embeddings
                q, k = glb["apply_rotary_pos_emb_vision"](q, k, cos, sin)
            q = q.transpose(0, 1).float()  # heads, seq, dim
            k = k.transpose(0, 1).float()
            scaling = float(getattr(attn, "scaling", q.shape[-1] ** -0.5))
            cu = cu_seqlens.detach().to(device=hidden_states.device, dtype=torch.long).flatten()
            visual = getattr(self.model, "visual", None)
            has_cls = hasattr(visual, "class_embedding")
            self._progressive.record(
                "visionzip_encoder_second_last_cls_path" if has_cls else "visionzip_encoder_last_navit_path", 1
            )
            score_pieces, metric_pieces = [], []
            for i in range(1, int(cu.numel())):
                start, end = int(cu[i - 1].item()), int(cu[i].item())
                if end <= start:
                    continue
                logits = torch.einsum("hqd,hkd->hqk", q[:, start:end, :], k[:, start:end, :]) * scaling
                probs = torch.softmax(logits, dim=-1)
                if has_cls:
                    # Original CLIP VisionZip: CLS query attention to patches,
                    # summed across heads; CLS itself never enters the LLM.
                    score_pieces.append(probs[:, 0, 1:].sum(dim=0))
                    metric_pieces.append(k[:, start + 1:end, :].mean(dim=0))
                else:
                    # Official Qwen2.5-VL port: mean heads, sum all queries.
                    score_pieces.append(probs.mean(dim=0).sum(dim=0))
                    metric_pieces.append(k[:, start:end, :].mean(dim=0))
            if not score_pieces:
                return None
            score = torch.cat(score_pieces, dim=0)
            metric = torch.cat(metric_pieces, dim=0)
            target_n = 0
            if getattr(self._progressive, "_vision_by_batch", None):
                target_n = int(self._progressive._vision_by_batch[0].numel())
            if target_n > 0 and int(score.numel()) > target_n and int(score.numel()) % target_n == 0:
                group = int(score.numel()) // target_n
                score = score.view(target_n, group).mean(dim=1)
                metric = metric.view(target_n, group, int(metric.shape[-1])).mean(dim=1)
                self._progressive.record("visionzip_patch_merger_group", group)
            window_index = getattr(self, "_visionzip_window_index", None)
            if (
                not has_cls and torch.is_tensor(window_index) and target_n > 0
                and int(window_index.numel()) == target_n and int(score.numel()) == target_n
            ):
                reverse = torch.argsort(window_index.to(score.device))
                score = score.index_select(0, reverse)
                metric = metric.index_select(0, reverse.to(metric.device))
                self._progressive.record("visionzip_window_order_restored", 1)
            elif not has_cls:
                # Qwen3-VL removed Qwen2.5-VL's get_window_index path; its
                # visual outputs are already in LLM token order.
                self._progressive.record("visionzip_window_order_identity", 1)
            self._progressive.record("visionzip_encoder_statistics_calls", 1)
            self._progressive.record("visionzip_encoder_statistics_tokens", int(score.numel()))
            return score.detach(), metric.detach()
        except Exception as exc:
            self._progressive.record("visionzip_encoder_statistics_error", 1)
            if self._progressive.config.debug:
                self._progressive._debug(f"visionzip_encoder_statistics_error {type(exc).__name__}: {exc}")
            return None

    def _patch_decoder_attentions(self) -> None:
        for _, module in self.model.named_modules():
            if getattr(module, "_vispruner_progressive_attn_patched", False):
                continue
            if not all(hasattr(module, attr) for attr in ("q_proj", "k_proj", "v_proj", "o_proj")):
                continue
            layer_idx = getattr(module, "layer_idx", None)
            if layer_idx is None:
                continue
            original_forward = module.forward
            owner = self
            attention_mask_index = _arg_index(original_forward, "attention_mask", 1)
            position_embeddings_index = _arg_index(original_forward, "position_embeddings", 7)
            cache_position_index = _arg_index(original_forward, "cache_position", 6)

            def wrapped_attention(attn_self, *args, __orig=original_forward, **kwargs):
                if getattr(owner._progressive, '_deployment_fixed_decode', False):
                    return __orig(*args, **kwargs)
                hidden_states = _get_arg(args, kwargs, 0, "hidden_states")
                attention_mask = _get_arg(args, kwargs, attention_mask_index, "attention_mask")
                text_query_drop = None
                attn_impl = ""
                q_len = int(hidden_states.shape[1]) if torch.is_tensor(hidden_states) and hidden_states.ndim >= 2 else 0
                key_len = q_len
                if torch.is_tensor(hidden_states) and owner._progressive.enabled:
                    merged_hidden = owner._progressive.merge_hidden_for_layer(int(attn_self.layer_idx), hidden_states)
                    if merged_hidden is not hidden_states:
                        hidden_states = merged_hidden
                        args, kwargs = _set_kwarg_or_arg(args, kwargs, 0, "hidden_states", hidden_states)
                    q_len = int(hidden_states.shape[1])
                    cache_position = _get_arg(args, kwargs, cache_position_index, "cache_position")
                    key_len = owner._infer_key_len(attn_self, hidden_states, attention_mask, q_len, cache_position)
                    key_drop = owner._progressive.mask_for_layer(
                        int(attn_self.layer_idx),
                        int(hidden_states.shape[0]),
                        key_len,
                        hidden_states.device,
                    )
                    text_query_drop = owner._progressive.text_mask_for_layer(
                        int(attn_self.layer_idx),
                        int(hidden_states.shape[0]),
                        key_len,
                        hidden_states.device,
                    )
                    effective_mask = attention_mask
                    attn_impl = str(getattr(getattr(owner.model, "config", None), "_attn_implementation", "") or "").lower()
                    if key_drop is not None and key_drop.any():
                        if attn_impl == "flash_attention_2":
                            effective_mask = _merge_key_drop_padding_mask(attention_mask, key_drop, key_len)
                            owner._progressive.record("flash_key_drop_padding_mask", 1)
                            if q_len == key_len:
                                owner._progressive.record("flash_query_key_drop_padding_mask", 1)
                                if owner._progressive.config.text_enabled:
                                    owner._progressive.record("text_mask_flash_query_key_semantics", 1)
                            if owner._progressive.config.debug:
                                cache_desc = "none"
                                if torch.is_tensor(cache_position) and cache_position.numel() > 0:
                                    cache_desc = f"{tuple(cache_position.shape)}:{int(cache_position.min())}-{int(cache_position.max())}"
                                owner._progressive._debug(
                                    f"flash_mask layer={int(attn_self.layer_idx)} q_len={q_len} key_len={key_len} "
                                    f"hidden={tuple(hidden_states.shape)} mask={tuple(effective_mask.shape)} "
                                    f"cache={cache_desc}"
                                )
                        else:
                            effective_mask = _merge_key_drop_mask(attention_mask, key_drop, q_len, hidden_states.dtype)
                        args, kwargs = _set_kwarg_or_arg(args, kwargs, attention_mask_index, "attention_mask", effective_mask)
                    routing_balance_bias = owner._progressive.routing_balance_bias_for_layer(
                        int(attn_self.layer_idx),
                        int(hidden_states.shape[0]),
                        key_len,
                        hidden_states.device,
                    )
                    text_bias = owner._progressive.text_bias_for_layer(
                        int(attn_self.layer_idx),
                        int(hidden_states.shape[0]),
                        key_len,
                        q_len,
                        hidden_states.device,
                        hidden_states.dtype,
                    )
                    if text_bias is not None:
                        effective_mask = _merge_additive_bias_mask(effective_mask, text_bias, q_len, hidden_states.dtype)
                        args, kwargs = _set_kwarg_or_arg(args, kwargs, attention_mask_index, "attention_mask", effective_mask)
                    head_ablation_bias = owner._progressive.hawk_head_ablation_bias(
                        int(attn_self.layer_idx), q_len, key_len,
                        int(getattr(attn_self, "num_heads", 0) or getattr(getattr(attn_self, "config", None), "num_attention_heads", 0)),
                        hidden_states.device, hidden_states.dtype,
                    ) if hasattr(owner._progressive, "hawk_head_ablation_bias") else None
                    if head_ablation_bias is not None:
                        if effective_mask is None:
                            # SDPA may encode pure causality through is_causal and
                            # return no materialized mask. Once a head-specific
                            # bias is supplied we must materialize that causal part.
                            q_pos = torch.arange(q_len, device=hidden_states.device).view(q_len, 1)
                            k_pos = torch.arange(key_len, device=hidden_states.device).view(1, key_len)
                            causal_offset = max(0, key_len - q_len)
                            blocked = k_pos > (q_pos + causal_offset)
                            effective_mask = torch.zeros(
                                (int(hidden_states.shape[0]), 1, q_len, key_len),
                                device=hidden_states.device, dtype=hidden_states.dtype,
                            )
                            effective_mask.masked_fill_(blocked.view(1, 1, q_len, key_len), torch.finfo(hidden_states.dtype).min)
                            owner._progressive.record("hawk_calibration_materialized_causal_mask", 1)
                        elif not torch.is_tensor(effective_mask) or effective_mask.ndim != 4:
                            raise RuntimeError(f"HAWK calibration expected a 4D attention mask, got {getattr(effective_mask, 'shape', None)}")
                        effective_mask = effective_mask + head_ablation_bias.to(effective_mask.device, dtype=effective_mask.dtype)
                        args, kwargs = _set_kwarg_or_arg(args, kwargs, attention_mask_index, "attention_mask", effective_mask)
                    should_observe = bool(
                        q_len == owner._progressive.prompt_len
                        and owner._progressive.needs_observe(int(attn_self.layer_idx))
                    )
                    if should_observe and str(getattr(owner._progressive, "prior_method", "") or "").lower() == "sparsevlm":
                        owner._progressive.prepare_sparse_text_raters(hidden_states)
                else:
                    should_observe = False
                    routing_balance_bias = None

                should_routing_correct = routing_balance_bias is not None
                captured = {}
                handles = []
                if should_observe or should_routing_correct:
                    handles.append(attn_self.q_proj.register_forward_hook(
                        lambda _module, _inputs, output: captured.__setitem__("q", output)
                    ))
                    handles.append(attn_self.k_proj.register_forward_hook(
                        lambda _module, _inputs, output: captured.__setitem__("k", output)
                    ))
                    text_source, _ = owner._progressive._text_score_source()
                    vision_mode = str(
                        getattr(owner._progressive.config, "vision_score_mode", "") or ""
                    ).lower()
                    deferred_rescore_layer = bool(
                        owner._progressive.deferred_vision_rescore_enabled()
                        and owner._progressive._vision_mid_physical_compacted
                        and not owner._progressive._vision_final_physical_compacted
                        and int(attn_self.layer_idx)
                        == int(getattr(owner._progressive.config, "vision_deferred_drop_layer", -1))
                    )
                    need_vision_value = bool(
                        owner._progressive.config.vision_enabled
                        and (
                            int(attn_self.layer_idx)
                            == int(owner._progressive._effective_vision_layer())
                            or deferred_rescore_layer
                        )
                        and vision_mode in {
                            "last_query_contribution_topk", "encoder_contribution_add_topk",
                            "functional_coreset_topk", "functional_coreset_merge",
                            "functional_role_coreset_topk", "functional_user_coreset_topk",
                            "text_pmi_reserve_functional_user_topk",
                            "compression_adaptive_functional_user_topk",
                            "competition_adaptive_functional_user_topk",
                            "majority_support_adaptive_functional_user_topk",
                            "functional_user_guarded_topk", "encoder_pmi_functional_alpha_topk",
                        }
                    )
                    if text_source == "causal_qkv_contribution" or should_routing_correct or need_vision_value:
                        handles.append(attn_self.v_proj.register_forward_hook(
                            lambda _module, _inputs, output: captured.__setitem__("v", output)
                        ))
                try:
                    result = __orig(*args, **kwargs)
                finally:
                    for handle in handles:
                        handle.remove()

                qk = None
                value_states = None
                if should_observe or should_routing_correct:
                    if torch.is_tensor(captured.get("q")) and torch.is_tensor(captured.get("k")):
                        qk = owner._prepare_qk_for_observe(
                            attn_self,
                            __orig,
                            captured["q"],
                            captured["k"],
                            args,
                            kwargs,
                            position_embeddings_index=position_embeddings_index,
                        )
                        owner._progressive.record("observe_reused_qk", 1)
                    if torch.is_tensor(captured.get("v")):
                        value_states = owner._prepare_v_for_observe(attn_self, captured["v"])
                        owner._progressive.record("observe_reused_v", 1)
                if should_routing_correct:
                    if qk is not None and value_states is not None:
                        result = owner._apply_routing_balance_last_query(
                            attn_self, result, qk[0], qk[1], value_states,
                            attention_mask, routing_balance_bias,
                        )
                    else:
                        owner._progressive.record("routing_balance_correction_qkv_missing", 1)

                if (
                    text_query_drop is not None
                    and bool(text_query_drop.any().item())
                    and attn_impl != "flash_attention_2"
                    and int(q_len) == int(key_len)
                ):
                    result = _zero_attention_output_queries(result, text_query_drop, q_len)
                    owner._progressive.record("dense_text_query_zero_calls", 1)
                    owner._progressive.record("dense_text_query_zero_tokens", int(text_query_drop.sum().item()))

                if (
                    torch.is_tensor(hidden_states)
                    and int(q_len) == int(owner._progressive.prompt_len)
                    and owner._progressive.needs_native_text_observe(
                        int(attn_self.layer_idx), "attention_output_norm"
                    )
                ):
                    native_attention_output = (
                        result[0] if isinstance(result, (tuple, list)) else result
                    )
                    if torch.is_tensor(native_attention_output):
                        owner._progressive.observe_native_text_tensor(
                            int(attn_self.layer_idx),
                            "attention_output_norm",
                            native_attention_output,
                        )
                    else:
                        owner._progressive.record("text_native_attention_output_missing", 1)

                if should_observe:
                    if qk is not None:
                        # Score the next layer from the unmodified causal/padding mask.
                        # The current layer's local text-key mask must not force the same
                        # tokens to remain at zero forever; their Q/K/V states can recover
                        # on the next layer. Physical vision compaction removes old keys.
                        owner._progressive.observe(
                            int(attn_self.layer_idx), qk[0], qk[1], attention_mask,
                            value_states=value_states,
                            output_projection=attn_self.o_proj,
                        )
                    else:
                        owner._progressive.record("observe_qk_none", 1)
                        if owner._progressive.config.debug and (
                            int(attn_self.layer_idx) == owner._progressive.config.layer_k
                            or int(attn_self.layer_idx) == owner._progressive.config.vision_layer
                        ):
                            owner._progressive._debug(f"observe_qk_none layer={int(attn_self.layer_idx)} q_len={q_len}")
                return result

            module.forward = MethodType(wrapped_attention, module)
            module._vispruner_progressive_attn_patched = True


    def _patch_native_text_proxies(self) -> None:
        """Observe already-produced layer tensors without extra model projections."""
        prog = getattr(self, "_progressive", None)
        if prog is None or not prog.config.text_enabled or not prog._text_uses_layer_tensor_source():
            return
        base_model = getattr(self.model, "model", None)
        language_model = getattr(base_model, "language_model", None)
        if language_model is None or not hasattr(language_model, "layers"):
            prog.record("text_native_no_language_model", 1)
            return
        source, _ = prog._text_score_source()
        owner = self
        for layer_idx, decoder_layer in enumerate(language_model.layers):
            # Avoid even a Python wrapper on layers outside the configured mask
            # interval. For 36-layer Qwen3/OV and [0.75,1.0), only L26--L34
            # produce statistics for target layers L27--L35.
            if (
                int(layer_idx) < int(prog.config.layer_k)
                or not prog._text_layer_active(int(layer_idx) + 1)
            ):
                continue
            if source == "mlp_output_norm":
                mlp = getattr(decoder_layer, "mlp", None)
                if mlp is None or getattr(mlp, "_vispruner_native_text_patched", False):
                    continue
                original_mlp_forward = mlp.forward

                def wrapped_mlp(mlp_self, *args, __orig=original_mlp_forward, __layer=layer_idx, **kwargs):
                    result = __orig(*args, **kwargs)
                    if owner._progressive.needs_native_text_observe(__layer, "mlp_output_norm"):
                        tensor = result[0] if isinstance(result, (tuple, list)) else result
                        if torch.is_tensor(tensor):
                            owner._progressive.observe_native_text_tensor(
                                __layer, "mlp_output_norm", tensor
                            )
                        else:
                            owner._progressive.record("text_native_mlp_output_missing", 1)
                    return result

                wrapped_mlp.__signature__ = inspect.signature(original_mlp_forward)
                mlp.forward = MethodType(wrapped_mlp, mlp)
                mlp._vispruner_native_text_patched = True
                continue

            if source not in {"residual_update", "residual_alignment", "local_redundancy"}:
                continue
            if getattr(decoder_layer, "_vispruner_native_text_patched", False):
                continue
            original_layer_forward = decoder_layer.forward

            def wrapped_layer(layer_self, *args, __orig=original_layer_forward, __layer=layer_idx, **kwargs):
                hidden_before = _get_arg(args, kwargs, 0, "hidden_states")
                if (
                    source == "local_redundancy"
                    and torch.is_tensor(hidden_before)
                    and owner._progressive.needs_native_text_observe(__layer, "local_redundancy")
                ):
                    owner._progressive.observe_native_text_tensor(
                        __layer, "local_redundancy", hidden_before
                    )
                result = __orig(*args, **kwargs)
                if source in {"residual_update", "residual_alignment"}:
                    hidden_after = result[0] if isinstance(result, (tuple, list)) else result
                    if torch.is_tensor(hidden_before) and torch.is_tensor(hidden_after):
                        owner._progressive.observe_native_text_residual(
                            __layer, hidden_before, hidden_after
                        )
                    elif owner._progressive.needs_native_text_observe(__layer, source):
                        owner._progressive.record(f"text_native_{source}_output_missing", 1)
                return result

            wrapped_layer.__signature__ = inspect.signature(original_layer_forward)
            decoder_layer.forward = MethodType(wrapped_layer, decoder_layer)
            decoder_layer._vispruner_native_text_patched = True

    def _patch_language_model_physical_drop(self) -> None:
        """Physically compact vision-pruned prompt tokens after selection layer.

        This is intentionally conservative:
        - batch size 1 only for now;
        - disables KV cache in physical mode to avoid cache length mismatches;
        - compacts only after the configured/effective vision-pruning layer has
          produced a keep set, so scoring still sees the original prompt.
        """
        prog = getattr(self, "_progressive", None)
        if prog is None or not prog.physical_drop_enabled():
            return

        base_model = getattr(self.model, "model", None)
        language_model = getattr(base_model, "language_model", None)
        if language_model is None or not hasattr(language_model, "layers"):
            prog.record("physical_drop_no_language_model", 1)
            return
        if getattr(language_model, "_vispruner_physical_drop_patched", False):
            return

        original_forward = language_model.forward
        owner = self

        def wrapped_lm_forward(
            lm_self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            cache_position=None,
            visual_pos_masks=None,
            deepstack_visual_embeds=None,
            **kwargs,
        ):
            from transformers.cache_utils import DynamicCache
            from transformers.masking_utils import create_causal_mask
            from transformers.modeling_outputs import BaseModelOutputWithPast

            # Physical compaction changes sequence length mid-forward.  Caching
            # across the pre/post-compaction boundary is unsafe, so force no-cache.
            use_cache = False
            past_key_values = None

            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

            if use_cache and past_key_values is None and not torch.jit.is_tracing():
                past_key_values = DynamicCache(config=lm_self.config)

            if inputs_embeds is None:
                inputs_embeds = lm_self.embed_tokens(input_ids)

            if cache_position is None:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
                )

            uses_2d_position_ids = (
                "LLaVAOneVision1_5_TextModel" in lm_self.__class__.__name__
                or "LlamaModel" in lm_self.__class__.__name__
            )
            if uses_2d_position_ids:
                if position_ids is None:
                    position_ids = cache_position.unsqueeze(0)
                text_position_ids = position_ids
            else:
                if position_ids is None:
                    position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
                elif position_ids.ndim == 2:
                    position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

                if position_ids.ndim == 3 and position_ids.shape[0] == 4:
                    text_position_ids = position_ids[0]
                    position_ids = position_ids[1:]
                else:
                    text_position_ids = position_ids[0]

            if bool(getattr(owner._progressive, "_hf_official_prepruned", False)) and int(inputs_embeds.shape[1]) > 1:
                (
                    inputs_embeds,
                    attention_mask,
                    position_ids,
                    cache_position,
                    visual_pos_masks,
                    deepstack_visual_embeds,
                ) = owner._progressive.hf_official_pre_llm_compact(
                    inputs_embeds,
                    attention_mask,
                    position_ids,
                    cache_position,
                    visual_pos_masks,
                    deepstack_visual_embeds,
                )
                text_position_ids = position_ids if uses_2d_position_ids else position_ids[0]

            if str(getattr(owner._progressive, "prior_method", "") or "").lower() == "official_vispruner" and int(inputs_embeds.shape[1]) > 1:
                (
                    inputs_embeds,
                    attention_mask,
                    position_ids,
                    cache_position,
                    visual_pos_masks,
                    deepstack_visual_embeds,
                ) = owner._progressive.official_vispruner_pre_llm_compact(
                    inputs_embeds,
                    attention_mask,
                    position_ids,
                    cache_position,
                    visual_pos_masks,
                    deepstack_visual_embeds,
                )
                text_position_ids = position_ids if uses_2d_position_ids else position_ids[0]

            if str(getattr(owner._progressive, "prior_method", "") or "").lower() == "zooprune" and int(inputs_embeds.shape[1]) > 1:
                (
                    inputs_embeds,
                    attention_mask,
                    position_ids,
                    cache_position,
                    visual_pos_masks,
                    deepstack_visual_embeds,
                ) = owner._progressive.zooprune_pre_llm_compact(
                    inputs_embeds,
                    attention_mask,
                    position_ids,
                    cache_position,
                    visual_pos_masks,
                    deepstack_visual_embeds,
                )
                text_position_ids = position_ids if uses_2d_position_ids else position_ids[0]

            if str(getattr(owner._progressive, "prior_method", "") or "").lower() == "visionzip" and int(inputs_embeds.shape[1]) > 1:
                (
                    inputs_embeds,
                    attention_mask,
                    position_ids,
                    cache_position,
                    visual_pos_masks,
                    deepstack_visual_embeds,
                ) = owner._progressive.visionzip_pre_llm_compact(
                    inputs_embeds,
                    attention_mask,
                    position_ids,
                    cache_position,
                    visual_pos_masks,
                    deepstack_visual_embeds,
                )
                text_position_ids = position_ids if uses_2d_position_ids else position_ids[0]

            if str(getattr(owner._progressive, "prior_method", "") or "").lower() == "hawk" and int(inputs_embeds.shape[1]) > 1:
                (
                    inputs_embeds,
                    attention_mask,
                    position_ids,
                    cache_position,
                    visual_pos_masks,
                    deepstack_visual_embeds,
                ) = owner._progressive.hawk_pre_llm_compact(
                    inputs_embeds,
                    attention_mask,
                    position_ids,
                    cache_position,
                    visual_pos_masks,
                    deepstack_visual_embeds,
                    lm_self.layers[0],
                )
                text_position_ids = position_ids if uses_2d_position_ids else position_ids[0]

            if owner._progressive.feature_reserve_pre_llm_enabled() and int(inputs_embeds.shape[1]) > 1:
                (
                    inputs_embeds,
                    attention_mask,
                    position_ids,
                    cache_position,
                    visual_pos_masks,
                    deepstack_visual_embeds,
                ) = owner._progressive.feature_reserve_pre_llm_compact(
                    inputs_embeds,
                    attention_mask,
                    position_ids,
                    cache_position,
                    visual_pos_masks,
                    deepstack_visual_embeds,
                )
                text_position_ids = position_ids if uses_2d_position_ids else position_ids[0]

            if owner._progressive.anchor_regional_pre_llm_enabled() and int(inputs_embeds.shape[1]) > 1:
                (
                    inputs_embeds,
                    attention_mask,
                    position_ids,
                    cache_position,
                    visual_pos_masks,
                    deepstack_visual_embeds,
                ) = owner._progressive.anchor_regional_pre_llm_compact(
                    inputs_embeds,
                    attention_mask,
                    position_ids,
                    cache_position,
                    visual_pos_masks,
                    deepstack_visual_embeds,
                )
                text_position_ids = position_ids if uses_2d_position_ids else position_ids[0]

            def rebuild_causal_mask(_hidden, _attention_mask, _cache_position, _text_position_ids):
                if uses_2d_position_ids and hasattr(lm_self, "_update_causal_mask"):
                    return lm_self._update_causal_mask(
                        _attention_mask, _hidden, _cache_position, past_key_values, False
                    )
                return create_causal_mask(
                    config=lm_self.config,
                    input_embeds=_hidden,
                    attention_mask=_attention_mask,
                    cache_position=_cache_position,
                    past_key_values=past_key_values,
                    position_ids=_text_position_ids,
                )

            attention_mask_current = rebuild_causal_mask(inputs_embeds, attention_mask, cache_position, text_position_ids)
            hidden_states = inputs_embeds
            position_embeddings = lm_self.rotary_emb(hidden_states, position_ids)

            for layer_idx, decoder_layer in enumerate(lm_self.layers):
                layer_kwargs = dict(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask_current,
                    position_ids=position_ids if uses_2d_position_ids else text_position_ids,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
                if uses_2d_position_ids:
                    layer_kwargs.update(
                        past_key_value=past_key_values,
                        output_attentions=False,
                        use_cache=False,
                    )
                else:
                    layer_kwargs.update(past_key_values=past_key_values)
                layer_kwargs.update(kwargs)
                trans_capture, trans_handles = ({}, [])
                if hasattr(owner._progressive, "transprune_capture_layer"):
                    trans_capture, trans_handles = owner._progressive.transprune_capture_layer(
                        int(layer_idx), decoder_layer
                    )
                try:
                    layer_outputs = decoder_layer(**layer_kwargs)
                finally:
                    for handle in trans_handles:
                        handle.remove()
                hidden_states = layer_outputs[0] if isinstance(layer_outputs, (tuple, list)) else layer_outputs

                if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                    hidden_states = lm_self._deepstack_process(
                        hidden_states,
                        visual_pos_masks,
                        deepstack_visual_embeds[layer_idx],
                    )

                if hasattr(owner._progressive, "transprune_accumulate"):
                    owner._progressive.transprune_accumulate(int(layer_idx), trans_capture)
                    trans_locations = tuple(getattr(owner._progressive, "_transprune_locations", ()))
                    if (
                        str(getattr(owner._progressive, "prior_method", "") or "").lower() == "transprune"
                        and int(layer_idx) in trans_locations
                    ):
                        if int(layer_idx) + 1 >= len(lm_self.layers):
                            raise RuntimeError("TransPrune pruning boundary has no following attention layer")
                        next_attn = lm_self.layers[int(layer_idx) + 1].self_attn
                        next_hidden = lm_self.layers[int(layer_idx) + 1].input_layernorm(hidden_states)
                        q_projected = next_attn.q_proj(next_hidden)
                        k_projected = next_attn.k_proj(next_hidden)
                        qk = owner._prepare_qk_for_observe(
                            next_attn,
                            next_attn.forward,
                            q_projected,
                            k_projected,
                            (),
                            {"position_embeddings": position_embeddings},
                        )
                        if qk is None:
                            raise RuntimeError(f"TransPrune could not construct RoPE Q/K after L{int(layer_idx) + 1}")
                        owner._progressive.transprune_select(
                            int(layer_idx), qk[0], qk[1], attention_mask_current
                        )

                if owner._progressive.should_physical_compact_after_layer(int(layer_idx), hidden_states):
                    hidden_states = owner._progressive.merge_hidden_before_physical_drop(hidden_states)
                    vision_local_keep = owner._progressive.physical_vision_local_keep_indices(hidden_states.device)
                    keep_idx = owner._progressive.physical_keep_indices(hidden_states.device)
                    if keep_idx is not None:
                        compaction_kind = owner._progressive._physical_pending_kind
                        hidden_states = hidden_states.index_select(1, keep_idx)
                        if torch.is_tensor(attention_mask) and attention_mask.ndim == 2:
                            attention_mask = attention_mask.index_select(1, keep_idx.to(attention_mask.device))
                        compact_len = int(hidden_states.shape[1])
                        position_policy = getattr(
                            owner._progressive, "should_preserve_final_position_ids", None
                        )
                        position_mode_policy = getattr(
                            owner._progressive, "final_position_id_policy", None
                        )
                        is_final_compaction = str(compaction_kind) == "final"
                        is_fastv = (
                            str(getattr(owner._progressive, "prior_method", "") or "").lower()
                            == "fastv"
                        )
                        position_mode = "original" if is_fastv else "contiguous"
                        if is_final_compaction and not is_fastv:
                            if bool(
                                getattr(
                                    owner._progressive,
                                    "_preserve_final_original_position_ids",
                                    False,
                                )
                            ):
                                position_mode = "original"
                            if callable(position_mode_policy):
                                position_mode = str(
                                    position_mode_policy(uses_2d_position_ids)
                                ).strip().lower()
                            elif callable(position_policy) and bool(
                                position_policy(uses_2d_position_ids)
                            ):
                                position_mode = "original"
                        if position_mode not in {
                            "contiguous",
                            "original",
                            "auxiliary_visual_only",
                        }:
                            raise RuntimeError(
                                f"unsupported final position-id policy: {position_mode}"
                            )
                        old_position_ids = position_ids
                        selected_original_position_ids = old_position_ids.index_select(
                            -1, keep_idx.to(old_position_ids.device)
                        )
                        compacted_visual_pos_masks = None
                        if torch.is_tensor(visual_pos_masks) and visual_pos_masks.ndim == 2:
                            compacted_visual_pos_masks = visual_pos_masks.index_select(
                                1, keep_idx.to(visual_pos_masks.device)
                            )
                        cache_position = torch.arange(
                            compact_len,
                            device=hidden_states.device,
                            dtype=cache_position.dtype if torch.is_tensor(cache_position) else torch.long,
                        )
                        if position_mode == "original":
                            # Keep selected tokens' pre-compaction rotary coordinates
                            # while packing the physical KV sequence.  FastV does
                            # this officially; opt-in analyses may isolate the same
                            # position policy at the final late-pruning boundary.
                            position_ids = selected_original_position_ids
                            text_position_ids = position_ids if uses_2d_position_ids else position_ids[0]
                            if is_fastv:
                                owner._progressive.record("fastv_original_position_ids_preserved", 1)
                            else:
                                owner._progressive.record(
                                    f"physical_compaction_{compaction_kind}_original_position_ids_preserved",
                                    1,
                                )
                                if not bool(getattr(owner._progressive, "_runtime_exact_fast_path", False)):
                                    flat_positions = position_ids.reshape(-1, compact_len)
                                    contiguous_reference = cache_position.to(
                                        device=flat_positions.device, dtype=flat_positions.dtype
                                    ).unsqueeze(0).expand_as(flat_positions)
                                    coordinate_changes = int(
                                        flat_positions.ne(contiguous_reference).sum().item()
                                    )
                                    owner._progressive.record(
                                        f"physical_compaction_{compaction_kind}_preserved_coordinate_changes",
                                        coordinate_changes,
                                    )
                        elif position_mode == "auxiliary_visual_only":
                            if uses_2d_position_ids:
                                raise RuntimeError(
                                    "auxiliary-axis position preservation requires multi-axis position ids"
                                )
                            if (
                                not torch.is_tensor(selected_original_position_ids)
                                or selected_original_position_ids.ndim != 3
                                or int(selected_original_position_ids.shape[0]) < 2
                                or compacted_visual_pos_masks is None
                            ):
                                raise RuntimeError(
                                    "auxiliary-axis position preservation requires [A,B,S] ids and a visual mask"
                                )
                            pos_dtype = selected_original_position_ids.dtype
                            position_ids = cache_position.to(dtype=pos_dtype).view(1, 1, -1).expand(
                                int(selected_original_position_ids.shape[0]),
                                int(hidden_states.shape[0]),
                                -1,
                            ).clone()
                            auxiliary_mask = compacted_visual_pos_masks.to(
                                device=position_ids.device, dtype=torch.bool
                            ).unsqueeze(0).expand(
                                int(position_ids.shape[0]) - 1, -1, -1
                            )
                            original_auxiliary = selected_original_position_ids[1:].to(position_ids.device)
                            candidate_changes = int(
                                (original_auxiliary.ne(position_ids[1:]) & auxiliary_mask).sum().item()
                            )
                            candidate_total = int(auxiliary_mask.sum().item())
                            if candidate_total <= 0:
                                raise RuntimeError("auxiliary-axis position guard has no visual coordinates")
                            threshold = getattr(
                                owner._progressive,
                                "_l9_position_topology_min_displacement_rate",
                                None,
                            )
                            preserve_auxiliary = True
                            if threshold is not None:
                                threshold = float(threshold)
                                if not 0.0 <= threshold <= 1.0:
                                    raise RuntimeError(
                                        "auxiliary-axis displacement threshold must be in [0,1]"
                                    )
                                preserve_auxiliary = candidate_changes / candidate_total >= threshold
                                owner._progressive.record(
                                    f"physical_compaction_{compaction_kind}_auxiliary_visual_guard_events",
                                    1,
                                )
                                owner._progressive.record(
                                    f"physical_compaction_{compaction_kind}_auxiliary_visual_guard_accept_decisions",
                                    int(preserve_auxiliary),
                                )
                                owner._progressive.record(
                                    f"physical_compaction_{compaction_kind}_auxiliary_visual_guard_reject_decisions",
                                    int(not preserve_auxiliary),
                                )
                                owner._progressive.record(
                                    f"physical_compaction_{compaction_kind}_auxiliary_visual_candidate_coordinate_changes",
                                    candidate_changes,
                                )
                                owner._progressive.record(
                                    f"physical_compaction_{compaction_kind}_auxiliary_visual_candidate_coordinate_total",
                                    candidate_total,
                                )
                                decision_name = "accepted" if preserve_auxiliary else "rejected"
                                owner._progressive.record(
                                    f"physical_compaction_{compaction_kind}_auxiliary_visual_{decision_name}_candidate_coordinate_changes",
                                    candidate_changes,
                                )
                                owner._progressive.record(
                                    f"physical_compaction_{compaction_kind}_auxiliary_visual_{decision_name}_candidate_coordinate_total",
                                    candidate_total,
                                )
                            if preserve_auxiliary:
                                position_ids[1:] = torch.where(
                                    auxiliary_mask,
                                    original_auxiliary,
                                    position_ids[1:],
                                )
                                owner._progressive.record(
                                    f"physical_compaction_{compaction_kind}_auxiliary_visual_position_ids_preserved",
                                    1,
                                )
                                owner._progressive.record(
                                    f"physical_compaction_{compaction_kind}_auxiliary_visual_coordinate_changes",
                                    candidate_changes,
                                )
                            text_position_ids = position_ids[0]
                        elif uses_2d_position_ids:
                            pos_dtype = position_ids.dtype if torch.is_tensor(position_ids) else torch.long
                            position_ids = cache_position.to(dtype=pos_dtype).unsqueeze(0).expand(
                                int(hidden_states.shape[0]), -1
                            )
                            text_position_ids = position_ids
                        else:
                            pos_dtype = position_ids.dtype if torch.is_tensor(position_ids) else torch.long
                            position_ids = cache_position.to(dtype=pos_dtype).view(1, 1, -1).expand(
                                3, int(hidden_states.shape[0]), -1
                            )
                            text_position_ids = position_ids[0]
                        if compacted_visual_pos_masks is not None:
                            visual_pos_masks = compacted_visual_pos_masks
                        if deepstack_visual_embeds is not None and vision_local_keep is not None:
                            local_keep_idx, source_vision_tokens = vision_local_keep
                            compacted_deepstack = []
                            for _emb in deepstack_visual_embeds:
                                if (
                                    torch.is_tensor(_emb)
                                    and _emb.ndim >= 2
                                    and int(_emb.shape[0]) == int(source_vision_tokens)
                                ):
                                    merged_deepstack = owner._progressive.merge_vision_local_features_before_physical_drop(
                                        _emb, source="deepstack"
                                    )
                                    compacted_deepstack.append(
                                        merged_deepstack.index_select(0, local_keep_idx.to(_emb.device))
                                    )
                                else:
                                    compacted_deepstack.append(_emb)
                            deepstack_visual_embeds = compacted_deepstack
                        owner._progressive.commit_physical_compaction(keep_idx)
                        if not bool(getattr(owner._progressive, "_runtime_exact_fast_path", False)):
                            expected_position = torch.arange(
                                compact_len, device=cache_position.device, dtype=cache_position.dtype
                            )
                            if torch.equal(cache_position, expected_position):
                                owner._progressive.record(
                                    f"physical_compaction_{compaction_kind}_cache_position_contiguous", 1
                                )
                            position_reference = position_ids.reshape(-1, compact_len)[0]
                            if torch.equal(
                                position_reference,
                                expected_position.to(device=position_reference.device, dtype=position_reference.dtype),
                            ):
                                owner._progressive.record(
                                    f"physical_compaction_{compaction_kind}_position_ids_contiguous", 1
                                )
                        attention_mask_current = rebuild_causal_mask(hidden_states, attention_mask, cache_position, text_position_ids)
                        position_embeddings = lm_self.rotary_emb(hidden_states, position_ids)
                    else:
                        owner._progressive.commit_physical_noop()

            hidden_states = lm_self.norm(hidden_states)
            return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)

        wrapped_lm_forward.__signature__ = inspect.signature(original_forward)
        language_model.forward = MethodType(wrapped_lm_forward, language_model)
        language_model._vispruner_physical_drop_patched = True

    def _infer_key_len(self, attn, hidden_states, attention_mask, q_len: int, cache_position=None) -> int:
        candidates = [q_len, self._progressive.prompt_len]
        if torch.is_tensor(attention_mask) and attention_mask.ndim >= 4:
            candidates.append(int(attention_mask.shape[-1]))
        if torch.is_tensor(cache_position) and cache_position.numel() > 0:
            candidates.append(int(cache_position.max().item()) + 1)
        return max(candidates)

    def _prepare_v_for_observe(self, attn, v_projected):
        if not torch.is_tensor(v_projected) or v_projected.ndim != 3:
            return None
        bsz, q_len, _ = v_projected.shape
        head_dim = int(attn.head_dim)
        return v_projected.view(bsz, q_len, -1, head_dim).transpose(1, 2)

    def _prepare_qk_for_observe(
        self,
        attn,
        original_forward,
        q_projected,
        k_projected,
        args,
        kwargs,
        position_embeddings_index: int = 7,
    ):
        position_embeddings = _get_arg(args, kwargs, position_embeddings_index, "position_embeddings")
        if position_embeddings is None:
            return None
        bsz, q_len, _ = q_projected.shape
        head_dim = int(attn.head_dim)
        q = q_projected.view(bsz, q_len, -1, head_dim)
        k = k_projected.view(bsz, q_len, -1, head_dim)
        if hasattr(attn, "q_norm"):
            q = attn.q_norm(q)
        if hasattr(attn, "k_norm"):
            k = attn.k_norm(k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        glb = dict(getattr(getattr(original_forward, "__func__", original_forward), "__globals__", {}) or {})
        module = sys.modules.get(attn.__class__.__module__)
        if module is not None:
            glb.update(vars(module))
        cos, sin = position_embeddings
        if "apply_multimodal_rotary_pos_emb" in glb:
            q, k = glb["apply_multimodal_rotary_pos_emb"](q, k, cos, sin, attn.rope_scaling["mrope_section"])
        elif "apply_rotary_pos_emb" in glb:
            q, k = glb["apply_rotary_pos_emb"](q, k, cos, sin)
        else:
            return None
        return q, k

    def _apply_routing_balance_last_query(
        self,
        attn,
        result,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        key_bias: torch.Tensor,
    ):
        """Replace only the answer-query attention output using a one-row exact softmax.

        The regular decoder attention remains on FlashAttention2/SDPA.  This
        computes one corrected query row, so a continuous text-key bias does
        not force the full attention matrix onto a dense backend.
        """
        if isinstance(result, (tuple, list)):
            if not result or not torch.is_tensor(result[0]):
                return result
            first = result[0]
        elif torch.is_tensor(result):
            first = result
        else:
            return result
        if query_states.ndim != 4 or key_states.ndim != 4 or value_states.ndim != 4:
            return result
        bsz, n_heads, q_len, head_dim = query_states.shape
        key_len = min(int(key_states.shape[2]), int(value_states.shape[2]), int(key_bias.shape[1]))
        if key_len <= 0 or int(first.shape[0]) != int(bsz) or int(first.shape[1]) != int(q_len):
            return result
        k = key_states[:, :, :key_len, :].float()
        v = value_states[:, :, :key_len, :].float()
        if int(k.shape[1]) != int(n_heads):
            if int(n_heads) % int(k.shape[1]) != 0 or int(v.shape[1]) != int(k.shape[1]):
                raise RuntimeError(
                    f"routing-balance Q/K/V head mismatch: q={int(n_heads)} k={int(k.shape[1])} v={int(v.shape[1])}"
                )
            repeat = int(n_heads) // int(k.shape[1])
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        corrected = first.clone()
        scale = float(getattr(attn, "scaling", float(head_dim) ** -0.5))
        for b in range(int(bsz)):
            text_idx = self._progressive._text_by_batch[b].to(device=query_states.device)
            if text_idx.numel() == 0:
                continue
            query_pos = int(text_idx[-1].item())
            if query_pos < 0 or query_pos >= int(q_len):
                continue
            logits = torch.einsum(
                "hd,hkd->hk", query_states[b, :, query_pos, :].float(), k[b]
            ) * scale
            visible = torch.arange(key_len, device=logits.device) <= query_pos
            if attention_mask is not None:
                if attention_mask.ndim == 4:
                    row = attention_mask[b, 0, query_pos, :key_len]
                    finite = torch.isfinite(row) & (row > torch.finfo(row.dtype).min / 2)
                    visible &= finite
                    logits = logits + row.float().unsqueeze(0)
                elif attention_mask.ndim == 2:
                    visible &= attention_mask[b, :key_len].to(device=logits.device).bool()
            logits = logits + key_bias[b, :key_len].float().unsqueeze(0)
            logits = logits.masked_fill(~visible.unsqueeze(0), torch.finfo(logits.dtype).min)
            probs = torch.softmax(logits, dim=-1).masked_fill(~visible.unsqueeze(0), 0.0)
            context = torch.einsum("hk,hkd->hd", probs, v[b])
            context = context.reshape(1, 1, int(n_heads) * int(head_dim)).to(dtype=first.dtype)
            row_out = attn.o_proj(context)[0, 0].to(dtype=first.dtype)
            corrected[b, query_pos] = row_out
        self._progressive.record("routing_balance_corrected_rows", int(bsz))
        if isinstance(result, tuple):
            return (corrected, *result[1:])
        if isinstance(result, list):
            return [corrected, *result[1:]]
        return corrected

    def _compute_qk_for_observe(self, attn, original_forward, hidden_states, args, kwargs, position_embeddings_index: int = 7):
        """Compatibility helper for tests; runtime observation reuses real projections."""
        return self._prepare_qk_for_observe(
            attn,
            original_forward,
            attn.q_proj(hidden_states),
            attn.k_proj(hidden_states),
            args,
            kwargs,
            position_embeddings_index=position_embeddings_index,
        )


@register_model("vispruner_qwen2_5_vl")
class VisPrunerQwen25VL(_ProgressiveMixin, BaseQwen25VL):
    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = "sdpa",
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,
        max_image_size: Optional[int] = None,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        progressive_vision_pruning=False,
        progressive_text_masking=False,
        progressive_layer_k=4,
        progressive_vision_layer=0,
        progressive_vision_keep_ratio=1.0 / 9.0,
        progressive_vision_keep_tokens=0,
        progressive_vision_text_aware=False,
        progressive_vision_score_mode="text",
        progressive_vision_score_lambda=1.0,
        progressive_vision_anchor_alpha=0.0,
        progressive_vision_stage_keep_tokens=0,
        progressive_vision_stage_keep_ratio=0.5,
        progressive_vision_stage_budget_mode="ratio",
        progressive_vision_stage_layer=0,
        progressive_vision_stage_score_mode="vispruner",
        progressive_vision_stage_score_lambda=0.0,
        progressive_vision_stage_physical_drop=False,
        progressive_vision_stage_surplus_fraction=-1.0,
        progressive_vision_deferred_drop_layer=-1,
        progressive_vision_deferred_reserve_fraction=0.0,
        progressive_vision_deferred_rescore=False,
        progressive_vision_deferred_rescore_min_swaps=0,
        progressive_vision_auto_rule="corr_pos",
        progressive_vision_auto_min_history=2,
        progressive_vision_encoder_layer=-1,
        progressive_vision_merge=False,
        progressive_vision_merge_placement="legacy",
        progressive_vision_physical_drop=False,
        progressive_vision_merge_mode="weighted",
        progressive_vision_merge_temperature=0.07,
        progressive_important_ratio=0.5,
        progressive_text_threshold=0.12,
        progressive_text_correction="exposure_baseline_ratio",
        progressive_text_mask_mode="threshold",
        progressive_text_mask_ratio=0.0,
        progressive_text_mask_ratio_end=-1.0,
        progressive_text_layer_start=0.0,
        progressive_text_layer_end=1.0,
        progressive_text_piecewise="",
        progressive_text_adaptive_tau=1.0,
        progressive_text_adaptive_alpha=0.45,
        progressive_text_adaptive_floor=0.15,
        progressive_text_adaptive_min_ratio=0.0,
        progressive_text_adaptive_max_ratio=0.2,
        progressive_text_adaptive_floor_mode="warmup_median",
        progressive_text_adaptive_margin=0.0,
        progressive_text_adaptive_ramp_layers=0,
        progressive_text_adaptive_gate_tau=0.0,
        progressive_text_adaptive_gate_min_frac=0.0,
        progressive_text_soft_gamma=2.0,
        progressive_text_grounding_lambda=0.0,
        progressive_text_ema_beta=0.8,
        progressive_analysis_path="",
        progressive_debug=False,
        **kwargs,
    ) -> None:
        super().__init__(
            pretrained=pretrained,
            device=device,
            device_map=device_map,
            batch_size=batch_size,
            use_cache=use_cache,
            attn_implementation=attn_implementation,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            max_num_frames=max_num_frames,
            use_custom_video_loader=use_custom_video_loader,
            fps=fps,
            max_image_size=max_image_size,
            system_prompt=system_prompt,
            interleave_visuals=interleave_visuals,
            reasoning_prompt=reasoning_prompt,
            **kwargs,
        )
        self._init_progressive(
            progressive_vision_pruning=progressive_vision_pruning,
            progressive_text_masking=progressive_text_masking,
            progressive_layer_k=progressive_layer_k,
            progressive_vision_layer=progressive_vision_layer,
            progressive_vision_keep_ratio=progressive_vision_keep_ratio,
            progressive_vision_keep_tokens=progressive_vision_keep_tokens,
            progressive_vision_text_aware=progressive_vision_text_aware,
            progressive_vision_score_mode=progressive_vision_score_mode,
            progressive_vision_score_lambda=progressive_vision_score_lambda,
            progressive_vision_anchor_alpha=progressive_vision_anchor_alpha,
            progressive_vision_stage_keep_tokens=progressive_vision_stage_keep_tokens,
            progressive_vision_stage_keep_ratio=progressive_vision_stage_keep_ratio,
            progressive_vision_stage_budget_mode=progressive_vision_stage_budget_mode,
            progressive_vision_stage_layer=progressive_vision_stage_layer,
            progressive_vision_stage_score_mode=progressive_vision_stage_score_mode,
            progressive_vision_stage_score_lambda=progressive_vision_stage_score_lambda,
            progressive_vision_stage_physical_drop=progressive_vision_stage_physical_drop,
            progressive_vision_stage_surplus_fraction=progressive_vision_stage_surplus_fraction,
            progressive_vision_deferred_drop_layer=progressive_vision_deferred_drop_layer,
            progressive_vision_deferred_reserve_fraction=progressive_vision_deferred_reserve_fraction,
            progressive_vision_deferred_rescore=progressive_vision_deferred_rescore,
            progressive_vision_deferred_rescore_min_swaps=progressive_vision_deferred_rescore_min_swaps,
            progressive_vision_auto_rule=progressive_vision_auto_rule,
            progressive_vision_auto_min_history=progressive_vision_auto_min_history,
            progressive_vision_encoder_layer=progressive_vision_encoder_layer,
            progressive_vision_merge=progressive_vision_merge,
            progressive_vision_merge_placement=progressive_vision_merge_placement,
            progressive_vision_physical_drop=progressive_vision_physical_drop,
            progressive_vision_merge_mode=progressive_vision_merge_mode,
            progressive_vision_merge_temperature=progressive_vision_merge_temperature,
            progressive_important_ratio=progressive_important_ratio,
            progressive_text_threshold=progressive_text_threshold,
            progressive_text_correction=progressive_text_correction,
            progressive_text_mask_mode=progressive_text_mask_mode,
            progressive_text_mask_ratio=progressive_text_mask_ratio,
            progressive_text_mask_ratio_end=progressive_text_mask_ratio_end,
            progressive_text_layer_start=progressive_text_layer_start,
            progressive_text_layer_end=progressive_text_layer_end,
            progressive_text_piecewise=progressive_text_piecewise,
            progressive_text_adaptive_tau=progressive_text_adaptive_tau,
            progressive_text_adaptive_alpha=progressive_text_adaptive_alpha,
            progressive_text_adaptive_floor=progressive_text_adaptive_floor,
            progressive_text_adaptive_min_ratio=progressive_text_adaptive_min_ratio,
            progressive_text_adaptive_max_ratio=progressive_text_adaptive_max_ratio,
            progressive_text_adaptive_floor_mode=progressive_text_adaptive_floor_mode,
            progressive_text_adaptive_margin=progressive_text_adaptive_margin,
            progressive_text_adaptive_ramp_layers=progressive_text_adaptive_ramp_layers,
            progressive_text_adaptive_gate_tau=progressive_text_adaptive_gate_tau,
            progressive_text_adaptive_gate_min_frac=progressive_text_adaptive_gate_min_frac,
            progressive_text_soft_gamma=progressive_text_soft_gamma,
            progressive_text_grounding_lambda=progressive_text_grounding_lambda,
            progressive_text_ema_beta=progressive_text_ema_beta,
            progressive_analysis_path=progressive_analysis_path,
            progressive_debug=progressive_debug,
        )


@register_model("vispruner_qwen3_vl")
class VisPrunerQwen3VL(_ProgressiveMixin, BaseQwen3VL):
    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = "sdpa",
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,
        max_image_size: Optional[int] = None,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        progressive_vision_pruning=False,
        progressive_text_masking=False,
        progressive_layer_k=4,
        progressive_vision_layer=0,
        progressive_vision_keep_ratio=1.0 / 9.0,
        progressive_vision_keep_tokens=0,
        progressive_vision_text_aware=False,
        progressive_vision_score_mode="text",
        progressive_vision_score_lambda=1.0,
        progressive_vision_anchor_alpha=0.0,
        progressive_vision_stage_keep_tokens=0,
        progressive_vision_stage_keep_ratio=0.5,
        progressive_vision_stage_budget_mode="ratio",
        progressive_vision_stage_layer=0,
        progressive_vision_stage_score_mode="vispruner",
        progressive_vision_stage_score_lambda=0.0,
        progressive_vision_stage_physical_drop=False,
        progressive_vision_stage_surplus_fraction=-1.0,
        progressive_vision_deferred_drop_layer=-1,
        progressive_vision_deferred_reserve_fraction=0.0,
        progressive_vision_deferred_rescore=False,
        progressive_vision_deferred_rescore_min_swaps=0,
        progressive_vision_auto_rule="corr_pos",
        progressive_vision_auto_min_history=2,
        progressive_vision_encoder_layer=-1,
        progressive_vision_merge=False,
        progressive_vision_merge_placement="legacy",
        progressive_vision_physical_drop=False,
        progressive_vision_merge_mode="weighted",
        progressive_vision_merge_temperature=0.07,
        progressive_important_ratio=0.5,
        progressive_text_threshold=0.12,
        progressive_text_correction="exposure_baseline_ratio",
        progressive_text_mask_mode="threshold",
        progressive_text_mask_ratio=0.0,
        progressive_text_mask_ratio_end=-1.0,
        progressive_text_layer_start=0.0,
        progressive_text_layer_end=1.0,
        progressive_text_piecewise="",
        progressive_text_adaptive_tau=1.0,
        progressive_text_adaptive_alpha=0.45,
        progressive_text_adaptive_floor=0.15,
        progressive_text_adaptive_min_ratio=0.0,
        progressive_text_adaptive_max_ratio=0.2,
        progressive_text_adaptive_floor_mode="warmup_median",
        progressive_text_adaptive_margin=0.0,
        progressive_text_adaptive_ramp_layers=0,
        progressive_text_adaptive_gate_tau=0.0,
        progressive_text_adaptive_gate_min_frac=0.0,
        progressive_text_soft_gamma=2.0,
        progressive_text_grounding_lambda=0.0,
        progressive_text_ema_beta=0.8,
        progressive_analysis_path="",
        progressive_debug=False,
        **kwargs,
    ) -> None:
        if BaseQwen3VL is object:
            raise ImportError(
                "Qwen3_VL wrapper is unavailable. Install a lmms-eval version with "
                "lmms_eval.models.simple.qwen3_vl or set VISPRUNER_QWEN3_LMMS_WRAPPER."
            )
        super().__init__(
            pretrained=pretrained,
            device=device,
            device_map=device_map,
            batch_size=batch_size,
            use_cache=use_cache,
            attn_implementation=attn_implementation,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            max_num_frames=max_num_frames,
            use_custom_video_loader=use_custom_video_loader,
            fps=fps,
            max_image_size=max_image_size,
            system_prompt=system_prompt,
            interleave_visuals=interleave_visuals,
            reasoning_prompt=reasoning_prompt,
            **kwargs,
        )
        self._init_progressive(
            progressive_vision_pruning=progressive_vision_pruning,
            progressive_text_masking=progressive_text_masking,
            progressive_layer_k=progressive_layer_k,
            progressive_vision_layer=progressive_vision_layer,
            progressive_vision_keep_ratio=progressive_vision_keep_ratio,
            progressive_vision_keep_tokens=progressive_vision_keep_tokens,
            progressive_vision_text_aware=progressive_vision_text_aware,
            progressive_vision_score_mode=progressive_vision_score_mode,
            progressive_vision_score_lambda=progressive_vision_score_lambda,
            progressive_vision_anchor_alpha=progressive_vision_anchor_alpha,
            progressive_vision_stage_keep_tokens=progressive_vision_stage_keep_tokens,
            progressive_vision_stage_keep_ratio=progressive_vision_stage_keep_ratio,
            progressive_vision_stage_budget_mode=progressive_vision_stage_budget_mode,
            progressive_vision_stage_layer=progressive_vision_stage_layer,
            progressive_vision_stage_score_mode=progressive_vision_stage_score_mode,
            progressive_vision_stage_score_lambda=progressive_vision_stage_score_lambda,
            progressive_vision_stage_physical_drop=progressive_vision_stage_physical_drop,
            progressive_vision_stage_surplus_fraction=progressive_vision_stage_surplus_fraction,
            progressive_vision_deferred_drop_layer=progressive_vision_deferred_drop_layer,
            progressive_vision_deferred_reserve_fraction=progressive_vision_deferred_reserve_fraction,
            progressive_vision_deferred_rescore=progressive_vision_deferred_rescore,
            progressive_vision_deferred_rescore_min_swaps=progressive_vision_deferred_rescore_min_swaps,
            progressive_vision_auto_rule=progressive_vision_auto_rule,
            progressive_vision_auto_min_history=progressive_vision_auto_min_history,
            progressive_vision_encoder_layer=progressive_vision_encoder_layer,
            progressive_vision_merge=progressive_vision_merge,
            progressive_vision_merge_placement=progressive_vision_merge_placement,
            progressive_vision_physical_drop=progressive_vision_physical_drop,
            progressive_vision_merge_mode=progressive_vision_merge_mode,
            progressive_vision_merge_temperature=progressive_vision_merge_temperature,
            progressive_important_ratio=progressive_important_ratio,
            progressive_text_threshold=progressive_text_threshold,
            progressive_text_correction=progressive_text_correction,
            progressive_text_mask_mode=progressive_text_mask_mode,
            progressive_text_mask_ratio=progressive_text_mask_ratio,
            progressive_text_mask_ratio_end=progressive_text_mask_ratio_end,
            progressive_text_layer_start=progressive_text_layer_start,
            progressive_text_layer_end=progressive_text_layer_end,
            progressive_text_piecewise=progressive_text_piecewise,
            progressive_text_adaptive_tau=progressive_text_adaptive_tau,
            progressive_text_adaptive_alpha=progressive_text_adaptive_alpha,
            progressive_text_adaptive_floor=progressive_text_adaptive_floor,
            progressive_text_adaptive_min_ratio=progressive_text_adaptive_min_ratio,
            progressive_text_adaptive_max_ratio=progressive_text_adaptive_max_ratio,
            progressive_text_adaptive_floor_mode=progressive_text_adaptive_floor_mode,
            progressive_text_adaptive_margin=progressive_text_adaptive_margin,
            progressive_text_adaptive_ramp_layers=progressive_text_adaptive_ramp_layers,
            progressive_text_adaptive_gate_tau=progressive_text_adaptive_gate_tau,
            progressive_text_adaptive_gate_min_frac=progressive_text_adaptive_gate_min_frac,
            progressive_text_soft_gamma=progressive_text_soft_gamma,
            progressive_text_grounding_lambda=progressive_text_grounding_lambda,
            progressive_text_ema_beta=progressive_text_ema_beta,
            progressive_analysis_path=progressive_analysis_path,
            progressive_debug=progressive_debug,
        )


@register_model("vispruner_llava_onevision1_5")
class VisPrunerLlavaOV15(_ProgressiveMixin, BaseLlavaOV15):
    def __init__(
        self,
        pretrained: str = "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = "sdpa",
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,
        max_image_size: Optional[int] = None,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        max_length: int = 2048,
        progressive_vision_pruning=False,
        progressive_text_masking=False,
        progressive_layer_k=4,
        progressive_vision_layer=0,
        progressive_vision_keep_ratio=1.0 / 9.0,
        progressive_vision_keep_tokens=0,
        progressive_vision_text_aware=False,
        progressive_vision_score_mode="text",
        progressive_vision_score_lambda=1.0,
        progressive_vision_anchor_alpha=0.0,
        progressive_vision_stage_keep_tokens=0,
        progressive_vision_stage_keep_ratio=0.5,
        progressive_vision_stage_budget_mode="ratio",
        progressive_vision_stage_layer=0,
        progressive_vision_stage_score_mode="vispruner",
        progressive_vision_stage_score_lambda=0.0,
        progressive_vision_stage_physical_drop=False,
        progressive_vision_stage_surplus_fraction=-1.0,
        progressive_vision_deferred_drop_layer=-1,
        progressive_vision_deferred_reserve_fraction=0.0,
        progressive_vision_deferred_rescore=False,
        progressive_vision_deferred_rescore_min_swaps=0,
        progressive_vision_auto_rule="corr_pos",
        progressive_vision_auto_min_history=2,
        progressive_vision_encoder_layer=-1,
        progressive_vision_merge=False,
        progressive_vision_merge_placement="legacy",
        progressive_vision_physical_drop=False,
        progressive_vision_merge_mode="weighted",
        progressive_vision_merge_temperature=0.07,
        progressive_important_ratio=0.5,
        progressive_text_threshold=0.12,
        progressive_text_correction="exposure_baseline_ratio",
        progressive_text_mask_mode="threshold",
        progressive_text_mask_ratio=0.0,
        progressive_text_mask_ratio_end=-1.0,
        progressive_text_layer_start=0.0,
        progressive_text_layer_end=1.0,
        progressive_text_piecewise="",
        progressive_text_adaptive_tau=1.0,
        progressive_text_adaptive_alpha=0.45,
        progressive_text_adaptive_floor=0.15,
        progressive_text_adaptive_min_ratio=0.0,
        progressive_text_adaptive_max_ratio=0.2,
        progressive_text_adaptive_floor_mode="warmup_median",
        progressive_text_adaptive_margin=0.0,
        progressive_text_adaptive_ramp_layers=0,
        progressive_text_adaptive_gate_tau=0.0,
        progressive_text_adaptive_gate_min_frac=0.0,
        progressive_text_soft_gamma=2.0,
        progressive_text_grounding_lambda=0.0,
        progressive_text_ema_beta=0.8,
        progressive_analysis_path="",
        progressive_debug=False,
        **kwargs,
    ) -> None:
        super().__init__(
            pretrained=pretrained,
            device=device,
            device_map=device_map,
            batch_size=batch_size,
            use_cache=use_cache,
            attn_implementation=attn_implementation,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            max_num_frames=max_num_frames,
            use_custom_video_loader=use_custom_video_loader,
            fps=fps,
            max_image_size=max_image_size,
            system_prompt=system_prompt,
            interleave_visuals=interleave_visuals,
            reasoning_prompt=reasoning_prompt,
            max_length=max_length,
            **kwargs,
        )
        self._init_progressive(
            progressive_vision_pruning=progressive_vision_pruning,
            progressive_text_masking=progressive_text_masking,
            progressive_layer_k=progressive_layer_k,
            progressive_vision_layer=progressive_vision_layer,
            progressive_vision_keep_ratio=progressive_vision_keep_ratio,
            progressive_vision_keep_tokens=progressive_vision_keep_tokens,
            progressive_vision_text_aware=progressive_vision_text_aware,
            progressive_vision_score_mode=progressive_vision_score_mode,
            progressive_vision_score_lambda=progressive_vision_score_lambda,
            progressive_vision_anchor_alpha=progressive_vision_anchor_alpha,
            progressive_vision_stage_keep_tokens=progressive_vision_stage_keep_tokens,
            progressive_vision_stage_keep_ratio=progressive_vision_stage_keep_ratio,
            progressive_vision_stage_budget_mode=progressive_vision_stage_budget_mode,
            progressive_vision_stage_layer=progressive_vision_stage_layer,
            progressive_vision_stage_score_mode=progressive_vision_stage_score_mode,
            progressive_vision_stage_score_lambda=progressive_vision_stage_score_lambda,
            progressive_vision_stage_physical_drop=progressive_vision_stage_physical_drop,
            progressive_vision_stage_surplus_fraction=progressive_vision_stage_surplus_fraction,
            progressive_vision_deferred_drop_layer=progressive_vision_deferred_drop_layer,
            progressive_vision_deferred_reserve_fraction=progressive_vision_deferred_reserve_fraction,
            progressive_vision_deferred_rescore=progressive_vision_deferred_rescore,
            progressive_vision_deferred_rescore_min_swaps=progressive_vision_deferred_rescore_min_swaps,
            progressive_vision_auto_rule=progressive_vision_auto_rule,
            progressive_vision_auto_min_history=progressive_vision_auto_min_history,
            progressive_vision_encoder_layer=progressive_vision_encoder_layer,
            progressive_vision_merge=progressive_vision_merge,
            progressive_vision_merge_placement=progressive_vision_merge_placement,
            progressive_vision_physical_drop=progressive_vision_physical_drop,
            progressive_vision_merge_mode=progressive_vision_merge_mode,
            progressive_vision_merge_temperature=progressive_vision_merge_temperature,
            progressive_important_ratio=progressive_important_ratio,
            progressive_text_threshold=progressive_text_threshold,
            progressive_text_correction=progressive_text_correction,
            progressive_text_mask_mode=progressive_text_mask_mode,
            progressive_text_mask_ratio=progressive_text_mask_ratio,
            progressive_text_mask_ratio_end=progressive_text_mask_ratio_end,
            progressive_text_layer_start=progressive_text_layer_start,
            progressive_text_layer_end=progressive_text_layer_end,
            progressive_text_piecewise=progressive_text_piecewise,
            progressive_text_adaptive_tau=progressive_text_adaptive_tau,
            progressive_text_adaptive_alpha=progressive_text_adaptive_alpha,
            progressive_text_adaptive_floor=progressive_text_adaptive_floor,
            progressive_text_adaptive_min_ratio=progressive_text_adaptive_min_ratio,
            progressive_text_adaptive_max_ratio=progressive_text_adaptive_max_ratio,
            progressive_text_adaptive_floor_mode=progressive_text_adaptive_floor_mode,
            progressive_text_adaptive_margin=progressive_text_adaptive_margin,
            progressive_text_adaptive_ramp_layers=progressive_text_adaptive_ramp_layers,
            progressive_text_adaptive_gate_tau=progressive_text_adaptive_gate_tau,
            progressive_text_adaptive_gate_min_frac=progressive_text_adaptive_gate_min_frac,
            progressive_text_soft_gamma=progressive_text_soft_gamma,
            progressive_text_grounding_lambda=progressive_text_grounding_lambda,
            progressive_text_ema_beta=progressive_text_ema_beta,
            progressive_analysis_path=progressive_analysis_path,
            progressive_debug=progressive_debug,
        )
