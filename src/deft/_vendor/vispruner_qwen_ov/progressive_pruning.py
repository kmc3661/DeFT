from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
import sys
from typing import Callable, Dict, Optional, Tuple

import torch


_HARD_ASSIGN_MERGE_MODES = {
    "mean", "hard", "nearest_mean", "nearest_residual", "visionzip_residual",
    "functional_nearest_mean", "functional_nearest_residual",
    "nearest_saliency_anchor", "segment_saliency_anchor", "segment_saliency_mean",
}
_SALIENCY_CONVEX_MERGE_MODES = {
    "nearest_saliency_anchor", "segment_saliency_anchor", "segment_saliency_mean",
}
_SEGMENT_LOCAL_MERGE_MODES = {"segment_saliency_anchor", "segment_saliency_mean"}
_ANCHOR_PRIOR_MERGE_MODES = {"nearest_saliency_anchor", "segment_saliency_anchor"}
_WEIGHTED_MEAN_PRIOR_MERGE_MODES = {"segment_saliency_mean"}


def as_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    return str(x).strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class ProgressiveConfig:
    vision_enabled: bool = False
    text_enabled: bool = False
    layer_k: int = 4  # text-token masking observation warmup only
    vision_layer: int = 0  # vision pruning is independent of text K
    vision_keep_ratio: float = 1.0 / 9.0
    vision_keep_tokens: int = 0
    vision_text_aware: bool = False
    vision_score_mode: str = "text"
    vision_score_lambda: float = 1.0
    # Opt-in quota of final tokens protected by encoder saliency. This is only
    # consulted by the explicit encoder_pmi_anchor_quota_topk score mode.
    vision_anchor_alpha: float = 0.0
    vision_stage_keep_tokens: int = 0
    vision_stage_keep_ratio: float = 0.5
    vision_stage_budget_mode: str = "ratio"
    vision_stage_layer: int = 0
    vision_stage_score_mode: str = "vispruner"
    vision_stage_score_lambda: float = 0.0
    vision_stage_physical_drop: bool = False
    # Optional analysis-derived three-boundary schedule. A non-negative stage
    # surplus fraction replaces the legacy depth fraction only for the pre-LLM
    # feature-reserve compaction. Deferred dropping scores once at vision_layer,
    # keeps a small support tail, and removes that tail after the configured
    # later layer without re-scoring.
    vision_stage_surplus_fraction: float = -1.0
    vision_deferred_drop_layer: int = -1
    vision_deferred_reserve_fraction: float = 0.0
    # Opt-in boundary-reserve revision: L9 retains the union of three nearby
    # encoder/PMI blends, then the later boundary re-scores that small union.
    # False preserves the established one-score deferred schedule exactly.
    vision_deferred_rescore: bool = False
    # Apply the later candidate set only when it replaces at least this many
    # L9 core tokens. Zero means unconditional re-scoring.
    vision_deferred_rescore_min_swaps: int = 0
    vision_auto_rule: str = "corr_pos"
    vision_auto_min_history: int = 2
    vision_encoder_layer: int = -1
    vision_merge: bool = False
    vision_merge_placement: str = "legacy"
    vision_physical_drop: bool = False
    vision_merge_mode: str = "weighted"
    vision_merge_temperature: float = 0.07
    important_ratio: float = 0.5
    text_threshold: float = 0.12
    text_correction: str = "exposure_baseline_ratio"
    text_mask_mode: str = "threshold"
    text_mask_ratio: float = 0.0
    text_mask_ratio_end: float = -1.0
    text_layer_start: float = 0.0
    text_layer_end: float = 1.0
    text_piecewise: str = ""
    text_adaptive_tau: float = 1.0
    text_adaptive_alpha: float = 0.45
    text_adaptive_floor: float = 0.15
    text_adaptive_min_ratio: float = 0.0
    text_adaptive_max_ratio: float = 0.2
    text_adaptive_floor_mode: str = "warmup_median"
    text_adaptive_margin: float = 0.0
    text_adaptive_ramp_layers: int = 0
    text_adaptive_gate_tau: float = 0.0
    text_adaptive_gate_min_frac: float = 0.0
    text_soft_gamma: float = 2.0
    text_grounding_lambda: float = 0.0
    text_ema_beta: float = 0.8
    analysis_path: str = ""
    num_layers: int = 32
    debug: bool = False

    @property
    def enabled(self) -> bool:
        return self.vision_enabled or self.text_enabled or bool(self.analysis_path)


class ProgressivePruningState:
    """One-forward progressive vision-key pruning + next-layer text-key masking.

    Vision pruning supports persistent masks for controls and packed physical
    sequence compaction for measured pruning runs. Text masking uses text queries
    and text-key received attention, but normalizes attention over every causally
    visible key so text/vision competition matches the decoder attention domain.
    """

    def __init__(self, config: ProgressiveConfig):
        self.config = config
        self._input_ids: Optional[torch.Tensor] = None
        self._active: Optional[torch.Tensor] = None
        self._vision_by_batch: Tuple[torch.Tensor, ...] = ()
        self._text_by_batch: Tuple[torch.Tensor, ...] = ()
        self._vision_drop: Optional[torch.Tensor] = None
        self._vision_keep: Optional[torch.Tensor] = None
        self._vision_stage_drop: Optional[torch.Tensor] = None
        self._vision_stage_keep: Optional[torch.Tensor] = None
        self._vision_mid_drop: Optional[torch.Tensor] = None
        self._vision_mid_keep: Optional[torch.Tensor] = None
        self._vision_deferred_final_drop: Optional[torch.Tensor] = None
        self._vision_deferred_final_keep: Optional[torch.Tensor] = None
        self._vision_alpha_reserve_keep: Optional[torch.Tensor] = None
        self._vision_deferred_core_keep: Optional[torch.Tensor] = None
        self._vision_deferred_core_drop: Optional[torch.Tensor] = None
        self._vision_last_scores: Tuple[Tuple[torch.Tensor, torch.Tensor], ...] = ()
        # Opt-in in-memory selection traces for paired mechanistic experiments.
        # Disabled by default and never consulted by the pruning algorithm.
        self._captured_vision_selections: list[dict] = []
        self._initial_vision_counts: Tuple[int, ...] = ()
        self._vision_encoder_score: Optional[torch.Tensor] = None
        self._vision_pruned_layer: Optional[int] = None
        self._vision_auto_ref_corr: Optional[float] = None
        self._vision_auto_low_frac_history: list[float] = []
        self._vision_merge_applied: bool = False
        self._vision_merge_applied_kinds: set[str] = set()
        # The most recent physical merge plan is shared by hidden states,
        # encoder saliency, and architecture-specific auxiliary vision features
        # (for example Qwen3 DeepStack). It is prompt-local and consumed before
        # the corresponding physical compaction is committed.
        self._vision_merge_plan: Optional[dict] = None
        # Prompt-local L9 visual-to-text update signatures. These are used only
        # by opt-in functional coreset selection/merge modes.
        self._vision_functional_signatures: Dict[int, torch.Tensor] = {}
        self._vision_stage_physical_compacted: bool = False
        self._vision_mid_physical_compacted: bool = False
        self._vision_final_physical_compacted: bool = False
        self._physical_pending_kind: Optional[str] = None
        self._anchor_regional_current_is_anchor: Optional[torch.Tensor] = None
        self._feature_reserve_current_is_anchor: Optional[torch.Tensor] = None
        # Original pre-compaction visual-local indices for opt-in selection
        # traces. This is diagnostic metadata only and never enters scoring.
        self._feature_reserve_original_local_indices: Optional[torch.Tensor] = None
        self._text_drop_by_layer: Dict[int, torch.Tensor] = {}
        self._text_bias_by_layer: Dict[int, torch.Tensor] = {}
        self._text_ema_by_batch: Dict[int, torch.Tensor] = {}
        self._routing_balance_reference_by_batch: Dict[int, float] = {}
        self._adaptive_floor_values: Dict[int, list[float]] = {}
        self._previous_text_mask_by_batch: Dict[int, torch.Tensor] = {}
        # Physical vision compaction currently runs generation without a KV
        # cache.  HF therefore replays the full causal prefix for every newly
        # generated token.  Original prompt-token states are invariant to future
        # suffix tokens, so layer-local text masks selected on the first prefill
        # can be reused exactly for the remaining forwards in that generation.
        self._generation_active: bool = False
        self._generation_forward_index: int = 0
        self._generation_reuse_text_masks: bool = False
        self._generation_text_drop_cache: Dict[int, torch.Tensor] = {}
        self._generation_text_bias_cache: Dict[int, torch.Tensor] = {}
        self._generation_continuous_alpha_cache: Dict[int, float] = {}
        self._continuous_alpha_head_cache: Optional[dict] = None
        self._analysis_prompt_idx = 0
        # Optional prompt-token metadata is supplied by local evaluation runners.
        # It is kept separate from model inputs, so it cannot affect inference.
        self._pending_role_ids: Optional[torch.Tensor] = None
        self._pending_raw_tokens: Tuple[str, ...] = ()
        self._pending_decoded_tokens: Tuple[str, ...] = ()
        self._pending_continuous_alpha_lexical: dict[str, float] = {}
        self._role_ids: Optional[torch.Tensor] = None
        self._raw_tokens: Tuple[str, ...] = ()
        self._decoded_tokens: Tuple[str, ...] = ()
        self._forced_text_ordinals: Tuple[int, ...] = ()
        self._forced_text_layers: Optional[Tuple[int, ...]] = None
        self.stats: Dict[str, float] = {}

    def record(self, name: str, value: float = 1.0) -> None:
        self.stats[name] = self.stats.get(name, 0.0) + float(value)

    def clear_captured_vision_selections(self) -> None:
        self._captured_vision_selections.clear()

    def captured_vision_selections(self) -> tuple[dict, ...]:
        return tuple(self._captured_vision_selections)

    def set_prompt_token_metadata(
        self,
        role_ids: torch.Tensor,
        raw_tokens: list[str] | tuple[str, ...],
        decoded_tokens: list[str] | tuple[str, ...],
    ) -> None:
        """Attach analysis-only token roles/pieces for the next prompt.

        Role IDs are 0=template/special, 1=system content, 2=user content,
        and 3=generated/assistant suffix. Metadata never enters scoring.
        """
        roles = torch.as_tensor(role_ids, dtype=torch.int16).flatten().cpu()
        raw = tuple(str(x) for x in raw_tokens)
        decoded = tuple(str(x) for x in decoded_tokens)
        if int(roles.numel()) != len(raw) or len(raw) != len(decoded):
            raise ValueError(
                f"prompt metadata length mismatch: roles={int(roles.numel())} "
                f"raw={len(raw)} decoded={len(decoded)}"
            )
        self._pending_role_ids = roles
        self._pending_raw_tokens = raw
        self._pending_decoded_tokens = decoded

    def set_continuous_alpha_prompt(self, prompt: str) -> None:
        text=str(prompt);words=text.lower().split();n=max(1,len(words));bins=[0.0]*32
        for word in words:
            digest=hashlib.sha256(word.encode("utf-8")).digest();idx=int.from_bytes(digest[:2],"little")%len(bins);sign=1.0 if digest[2]&1 else -1.0;bins[idx]+=sign/math.sqrt(n)
        self._pending_continuous_alpha_lexical={
            "prompt_char_len":float(len(text)),"prompt_word_count":float(len(words)),
            "prompt_digit_fraction":float(sum(ch.isdigit() for ch in text))/float(max(1,len(text))),
            "prompt_nonascii_fraction":float(sum(ord(ch)>127 for ch in text))/float(max(1,len(text))),
            **{f"prompt_hash_{i:02d}":float(value) for i,value in enumerate(bins)},
        }

    def _activate_prompt_token_metadata(self, ids: torch.Tensor) -> None:
        seqlen = int(ids.shape[1])
        pending = self._pending_role_ids
        if pending is None or int(pending.numel()) > seqlen:
            self._role_ids = torch.zeros(seqlen, device=ids.device, dtype=torch.int16)
            self._raw_tokens = tuple(str(int(x)) for x in ids[0].detach().cpu().tolist())
            self._decoded_tokens = self._raw_tokens
            return
        base_len = int(pending.numel())
        suffix_len = seqlen - base_len
        suffix_roles = torch.full((suffix_len,), 3, dtype=torch.int16)
        self._role_ids = torch.cat((pending, suffix_roles), dim=0).to(ids.device)
        suffix_ids = tuple(str(int(x)) for x in ids[0, base_len:].detach().cpu().tolist())
        self._raw_tokens = self._pending_raw_tokens + suffix_ids
        self._decoded_tokens = self._pending_decoded_tokens + suffix_ids

    def set_forced_text_ordinals(
        self,
        ordinals: list[int] | tuple[int, ...],
        layers: Optional[list[int] | tuple[int, ...]] = None,
    ) -> None:
        """Set analysis-only text-token ordinals and optional exact target layers."""
        self._forced_text_ordinals = tuple(sorted({int(x) for x in ordinals if int(x) >= 0}))
        self._forced_text_layers = (
            None if layers is None else tuple(sorted({int(x) for x in layers if int(x) >= 0}))
        )

    def clear_forced_text_ordinals(self) -> None:
        self._forced_text_ordinals = ()
        self._forced_text_layers = None

    def _install_forced_text_masks(self) -> None:
        if not self._forced_text_ordinals or self._input_ids is None:
            return
        bsz = len(self._text_by_batch)
        configured_layers = self._forced_text_layers
        layer_iter = (
            configured_layers
            if configured_layers is not None
            else tuple(range(max(0, int(self.config.num_layers or 0))))
        )
        for layer_idx in layer_iter:
            if configured_layers is None and not self._text_layer_active(layer_idx):
                continue
            if layer_idx < 0 or layer_idx >= int(self.config.num_layers or 0):
                continue
            mask = torch.zeros(
                (bsz, self.prompt_len), device=self._input_ids.device, dtype=torch.bool
            )
            for b, text_idx in enumerate(self._text_by_batch):
                valid = [x for x in self._forced_text_ordinals if x < int(text_idx.numel())]
                if valid:
                    ord_idx = torch.tensor(valid, device=text_idx.device, dtype=torch.long)
                    mask[b, text_idx.index_select(0, ord_idx).to(mask.device)] = True
            self._text_drop_by_layer[int(layer_idx)] = mask.detach()
        self.record("forced_text_mask_sessions", 1)
        self.record("forced_text_mask_ordinals", len(self._forced_text_ordinals))

    def _debug(self, message: str) -> None:
        if self.config.debug:
            sys.stderr.write(f"[vispruner-debug] {message}\n")
            sys.stderr.flush()

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def prompt_len(self) -> int:
        if self._input_ids is None:
            return 0
        return int(self._input_ids.shape[1])

    def final_vision_budget(self, initial_n: int, batch_idx: int = 0) -> int:
        """Resolve one exact final visual budget from fixed-K or input-relative ratio.

        Ratio mode is always anchored to the prompt's original visual-token
        count, never to an already compacted intermediate sequence.  This
        prevents progressive methods from applying the requested ratio twice.
        """
        initial_n = max(0, int(initial_n))
        if initial_n <= 0:
            return 0
        fixed = int(getattr(self.config, "vision_keep_tokens", 0) or 0)
        if fixed > 0:
            return max(1, min(fixed, initial_n))
        source_n = initial_n
        if 0 <= int(batch_idx) < len(self._initial_vision_counts):
            source_n = max(1, int(self._initial_vision_counts[int(batch_idx)]))
        ratio = max(0.0, min(1.0, float(getattr(self.config, "vision_keep_ratio", 1.0))))
        requested = max(1, int(round(float(source_n) * ratio)))
        return max(1, min(requested, initial_n))

    def start_generation(self) -> None:
        """Open a generation-local cache for reversible text masks."""
        self._generation_active = True
        self._generation_forward_index = 0
        self._generation_reuse_text_masks = False
        self._generation_text_drop_cache.clear()
        self._generation_text_bias_cache.clear()
        self._generation_continuous_alpha_cache.clear()
        self.record("text_mask_generation_cache_sessions", 1)

    def finish_generation(self) -> None:
        """Close the cache so masks can never leak into the next sample."""
        if self._generation_active:
            self.record("text_mask_generation_cache_completed", 1)
        self._generation_active = False
        self._generation_forward_index = 0
        self._generation_reuse_text_masks = False
        self._generation_text_drop_cache.clear()
        self._generation_text_bias_cache.clear()
        self._generation_continuous_alpha_cache.clear()

    def finish_forward(self) -> None:
        """Persist first-prefill masks after all decoder layers have run."""
        if not self._generation_active:
            return
        if self._generation_forward_index == 0 and self.config.text_enabled:
            self._generation_text_drop_cache = {
                int(layer): mask.detach().clone()
                for layer, mask in self._text_drop_by_layer.items()
            }
            self._generation_text_bias_cache = {
                int(layer): bias.detach().clone()
                for layer, bias in self._text_bias_by_layer.items()
            }
            cached_layers = len(self._generation_text_drop_cache) + len(self._generation_text_bias_cache)
            self.record("text_mask_generation_cache_store_events", 1)
            self.record("text_mask_generation_cache_store_layers", cached_layers)
        elif self._generation_forward_index > 0 and self._generation_reuse_text_masks:
            self.record("text_mask_generation_cache_reuse_forwards", 1)
        self._generation_forward_index += 1

    def _restore_generation_text_masks(self) -> None:
        if not self._generation_reuse_text_masks:
            return
        self._text_drop_by_layer = {
            int(layer): mask.detach()
            for layer, mask in self._generation_text_drop_cache.items()
        }
        self._text_bias_by_layer = {
            int(layer): bias.detach()
            for layer, bias in self._generation_text_bias_cache.items()
        }
        restored_layers = len(self._text_drop_by_layer) + len(self._text_bias_by_layer)
        self.record("text_mask_generation_cache_restore_events", 1)
        self.record("text_mask_generation_cache_restore_layers", restored_layers)

    def hf_official_pre_llm_compact(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        visual_pos_masks: Optional[torch.Tensor],
        deepstack_visual_embeds,
    ):
        """Replace one contiguous HF LLaVA vision span with official reduced features.

        HF validates and fills the original image placeholders before entering
        the language model. Released LLaVA pruning methods instead shorten the
        projected feature sequence before decoder layer 0. We reproduce that
        boundary here, after placeholder validation but before any LLM layer.
        """
        reduced = getattr(self, "_hf_official_reduced_features", None)
        if isinstance(reduced, (tuple, list)):
            if len(reduced) != 1:
                raise RuntimeError("diagnostic no-cache compaction supports one image; use cached generation for multi-image inputs")
            reduced = reduced[0]
        if not bool(getattr(self, "_hf_official_prepruned", False)) or reduced is None:
            return (
                inputs_embeds, attention_mask, position_ids, cache_position,
                visual_pos_masks, deepstack_visual_embeds,
            )
        if inputs_embeds.ndim != 3 or int(inputs_embeds.shape[0]) != 1 or len(self._vision_by_batch) != 1:
            raise RuntimeError("official HF LLaVA compaction requires batch size 1")
        if deepstack_visual_embeds is not None:
            raise RuntimeError("official HF LLaVA compaction does not support DeepStack features")
        vision_idx = self._vision_by_batch[0].to(inputs_embeds.device)
        initial_n = int(vision_idx.numel())
        if initial_n <= 0:
            raise RuntimeError("official HF LLaVA compaction found no vision tokens")
        start = int(vision_idx[0].item())
        end = int(vision_idx[-1].item()) + 1
        expected = torch.arange(start, end, device=vision_idx.device, dtype=vision_idx.dtype)
        if int(expected.numel()) != initial_n or not torch.equal(vision_idx, expected):
            raise RuntimeError("official HF LLaVA compaction requires one contiguous vision span")
        reduced = reduced.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        if reduced.ndim != 2 or int(reduced.shape[-1]) != int(inputs_embeds.shape[-1]):
            raise RuntimeError("official reduced feature shape does not match LLM embeddings")
        final_n = int(reduced.shape[0])
        if final_n <= 0:
            raise RuntimeError("official HF LLaVA selector returned no tokens")
        compacted = torch.cat(
            (inputs_embeds[:, :start], reduced.unsqueeze(0), inputs_embeds[:, end:]), dim=1
        )
        new_len = int(compacted.shape[1])
        if torch.is_tensor(attention_mask) and attention_mask.ndim == 2:
            middle = torch.ones(
                (1, final_n), device=attention_mask.device, dtype=attention_mask.dtype
            )
            attention_mask = torch.cat(
                (attention_mask[:, :start], middle, attention_mask[:, end:]), dim=1
            )
        position_dtype = position_ids.dtype if torch.is_tensor(position_ids) else torch.long
        position_ids = torch.arange(
            new_len, device=compacted.device, dtype=position_dtype
        ).unsqueeze(0)
        cache_dtype = cache_position.dtype if torch.is_tensor(cache_position) else torch.long
        cache_position = torch.arange(new_len, device=compacted.device, dtype=cache_dtype)
        if torch.is_tensor(visual_pos_masks) and visual_pos_masks.ndim == 2:
            middle = torch.ones(
                (1, final_n), device=visual_pos_masks.device, dtype=visual_pos_masks.dtype
            )
            visual_pos_masks = torch.cat(
                (visual_pos_masks[:, :start], middle, visual_pos_masks[:, end:]), dim=1
            )

        # Keep controller coordinates aligned with the shortened decoder input.
        # The synthetic IDs are bookkeeping only; model embeddings above remain
        # the exact official selected/merged features.
        if torch.is_tensor(self._input_ids) and int(self._input_ids.shape[1]) == int(inputs_embeds.shape[1]):
            vision_id = self._input_ids[:, start : start + 1]
            repeated = vision_id.expand(1, final_n)
            self._input_ids = torch.cat(
                (self._input_ids[:, :start], repeated, self._input_ids[:, end:]), dim=1
            ).detach()
        if torch.is_tensor(attention_mask) and attention_mask.ndim == 2:
            self._active = attention_mask.bool().detach()
        else:
            self._active = torch.ones((1, new_len), device=compacted.device, dtype=torch.bool)
        self._vision_by_batch = (
            torch.arange(start, start + final_n, device=compacted.device, dtype=torch.long),
        )
        active_idx = torch.arange(new_len, device=compacted.device)[self._active[0].to(compacted.device)]
        is_vision = torch.zeros(new_len, device=compacted.device, dtype=torch.bool)
        is_vision[start : start + final_n] = True
        self._text_by_batch = (active_idx[~is_vision[active_idx]],)
        self._vision_final_physical_compacted = True
        self._vision_drop = None
        self._vision_keep = None
        self.record("hf_official_embedding_replacement_events", 1)
        self.record("hf_official_embedding_replacement_initial_tokens", initial_n)
        self.record("hf_official_embedding_replacement_final_tokens", final_n)
        self.record("hf_official_position_ids_contiguous", 1)
        return (
            compacted, attention_mask, position_ids, cache_position,
            visual_pos_masks, deepstack_visual_embeds,
        )

    def begin(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        vision_finder: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        if not self.enabled or input_ids is None or input_ids.ndim != 2:
            return
        ids = input_ids.detach()
        bsz, seqlen = ids.shape
        self._generation_reuse_text_masks = bool(
            self._generation_active
            and self._generation_forward_index > 0
            and (
                self._generation_text_drop_cache
                or self._generation_text_bias_cache
            )
        )
        if attention_mask is None:
            active = torch.ones((bsz, seqlen), device=ids.device, dtype=torch.bool)
        elif attention_mask.ndim == 2:
            active = attention_mask.to(device=ids.device).bool()
        else:
            active = torch.ones((bsz, seqlen), device=ids.device, dtype=torch.bool)

        vision, text = [], []
        arange = torch.arange(seqlen, device=ids.device)
        for b in range(bsz):
            vis = vision_finder(ids[b]).to(device=ids.device).bool() & active[b]
            txt = (~vis) & active[b]
            vision.append(arange[vis].detach())
            text.append(arange[txt].detach())

        self._input_ids = ids
        self._active = active.detach()
        self._activate_prompt_token_metadata(ids)
        self._vision_by_batch = tuple(vision)
        self._text_by_batch = tuple(text)
        self._vision_drop = None
        self._vision_keep = None
        self._vision_stage_drop = None
        self._vision_stage_keep = None
        self._vision_mid_drop = None
        self._vision_mid_keep = None
        self._vision_deferred_final_drop = None
        self._vision_deferred_final_keep = None
        self._vision_alpha_reserve_keep = None
        self._vision_deferred_core_keep = None
        self._vision_deferred_core_drop = None
        self._vision_last_scores = ()
        self._initial_vision_counts = tuple(int(x.numel()) for x in vision)
        if int(getattr(self.config, "vision_keep_tokens", 0) or 0) <= 0:
            ratio = max(0.0, min(1.0, float(getattr(self.config, "vision_keep_ratio", 1.0))))
            for count in self._initial_vision_counts:
                if count <= 0:
                    continue
                target = max(1, min(int(round(float(count) * ratio)), count))
                self.record("vision_ratio_budget_events", 1)
                self.record("vision_ratio_budget_original_tokens", count)
                self.record("vision_ratio_budget_target_tokens", target)
                self.record("vision_ratio_budget_retention_x1000000", int(round(ratio * 1_000_000.0)))
        self._vision_encoder_score = None
        # Auto-layer decisions are prompt-local. Carrying these values across
        # prompts makes a sample depend on the previous sample's utilization
        # trajectory and can select the wrong final-pruning layer.
        self._vision_pruned_layer = None
        self._vision_auto_ref_corr = None
        self._vision_auto_low_frac_history.clear()
        self._vision_merge_applied = False
        self._vision_merge_applied_kinds.clear()
        self._vision_merge_plan = None
        self._vision_functional_signatures.clear()
        self._vision_stage_physical_compacted = False
        self._vision_mid_physical_compacted = False
        self._vision_final_physical_compacted = False
        self._physical_pending_kind = None
        self._anchor_regional_current_is_anchor = None
        self._feature_reserve_current_is_anchor = None
        self._feature_reserve_original_local_indices = None
        self._text_drop_by_layer.clear()
        self._text_bias_by_layer.clear()
        self._text_ema_by_batch.clear()
        self._routing_balance_reference_by_batch.clear()
        self._adaptive_floor_values.clear()
        self._previous_text_mask_by_batch.clear()
        self._install_forced_text_masks()
        if self._generation_reuse_text_masks and not self.physical_drop_enabled():
            self._restore_generation_text_masks()
        self._analysis_prompt_idx += 1
        self.record("prompts", bsz)
        if not bool(getattr(self, "_runtime_exact_fast_path", False)):
            self.record("prompt_tokens", int(active.sum().item()))
        self.record("vision_candidates", sum(int(x.numel()) for x in vision))
        self.record("text_candidates", sum(int(x.numel()) for x in text))
        if self.config.debug:
            v_counts = [int(x.numel()) for x in vision]
            t_counts = [int(x.numel()) for x in text]
            self._debug(
                f"begin bsz={bsz} seqlen={seqlen} "
                f"vision_candidates={v_counts} text_candidates={t_counts} "
                f"enabled=(vision={self.config.vision_enabled}, text={self.config.text_enabled})"
            )

    def _auto_vision_enabled(self) -> bool:
        return bool(self.config.vision_enabled and int(getattr(self.config, "vision_layer", 0) or 0) < 0)

    def _auto_vision_max_layer(self) -> int:
        # Negative vision_layer activates auto selection.  The magnitude is an
        # optional cap; a large value such as -999 means "search all layers".
        raw = abs(int(getattr(self.config, "vision_layer", -999) or -999))
        n = max(1, int(getattr(self.config, "num_layers", 32) or 32))
        if raw <= 0:
            raw = n - 1
        return max(0, min(raw, n - 1))

    def _effective_vision_layer(self) -> int:
        if self._vision_pruned_layer is not None:
            return int(self._vision_pruned_layer)
        if self._auto_vision_enabled():
            return self._auto_vision_max_layer()
        return int(getattr(self.config, "vision_layer", 0) or 0)

    def needs_observe(self, layer_idx: int) -> bool:
        """Return whether this layer needs Q/K-derived pruning statistics.

        This guard must run before any observation projection or score work.  In
        particular, an analysis output path alone must never turn a no-mask
        control into an all-layer attention recomputation.
        """
        if not self.enabled or self._input_ids is None:
            return False
        layer_idx = int(layer_idx)
        if self.config.vision_enabled and not bool(getattr(self, "_hf_official_prepruned", False)):
            if self._auto_vision_enabled():
                if self._vision_drop is None and layer_idx <= self._auto_vision_max_layer():
                    return True
            elif self._vision_drop is None and layer_idx == int(self.config.vision_layer):
                return True
            if (
                self.deferred_vision_rescore_enabled()
                and self._vision_mid_physical_compacted
                and not self._vision_final_physical_compacted
                and self._vision_deferred_final_drop is None
                and layer_idx == int(getattr(self.config, "vision_deferred_drop_layer", -1))
            ):
                return True
            stage_requested = bool(
                int(getattr(self.config, "vision_stage_keep_tokens", 0) or 0) > 0
                or float(getattr(self.config, "vision_stage_keep_ratio", 1.0) or 1.0) < 1.0
                or str(getattr(self.config, "vision_stage_budget_mode", "ratio") or "ratio").lower()
                not in {"ratio", "fixed_ratio", "tokens", "fixed_tokens"}
            )
            if (
                stage_requested
                and self._vision_stage_drop is None
                and self._vision_drop is None
                and not self._vision_stage_physical_compacted
                and layer_idx == int(getattr(self.config, "vision_stage_layer", 0) or 0)
            ):
                return True
        if self.config.text_enabled:
            target_layer = layer_idx + 1
            text_mask_cached = bool(
                target_layer in self._text_drop_by_layer
                or target_layer in self._text_bias_by_layer
            )
            if self._uses_routing_balance_text_mode():
                reference_layer = int(self._effective_vision_layer())
                if (
                    self._routing_balance_source_active(layer_idx)
                    and self._text_layer_active(target_layer)
                    and not text_mask_cached
                ):
                    return True
            elif (
                not self._text_uses_layer_tensor_source()
                and layer_idx >= int(self.config.layer_k)
                and self._text_layer_active(target_layer)
                and not text_mask_cached
            ):
                return True
            if self._adaptive_text_needs_warmup() and layer_idx < int(self.config.layer_k):
                return True
        return False

    def deferred_vision_drop_enabled(self) -> bool:
        return bool(
            self.config.vision_enabled
            and self.physical_drop_enabled()
            and int(getattr(self.config, "vision_deferred_drop_layer", -1) or -1)
            > int(self._effective_vision_layer())
            and (
                float(getattr(self.config, "vision_deferred_reserve_fraction", 0.0) or 0.0) > 0.0
                or bool(getattr(self.config, "vision_deferred_rescore", False))
            )
        )

    def deferred_vision_rescore_enabled(self) -> bool:
        return bool(
            self.deferred_vision_drop_enabled()
            and bool(getattr(self.config, "vision_deferred_rescore", False))
        )

    def physical_drop_enabled(self) -> bool:
        return bool(self.config.vision_enabled and getattr(self.config, "vision_physical_drop", False))

    @property
    def _vision_physical_compacted(self) -> bool:
        """Backward-compatible indicator for the final physical compaction."""
        return self._vision_final_physical_compacted

    def stage_physical_drop_enabled(self) -> bool:
        return bool(
            self.physical_drop_enabled()
            and getattr(self.config, "vision_stage_physical_drop", False)
        )

    def should_physical_compact_after_layer(self, layer_idx: int, hidden_states: torch.Tensor) -> bool:
        self._physical_pending_kind = None
        if (
            not self.physical_drop_enabled()
            or not torch.is_tensor(hidden_states)
            or hidden_states.ndim != 3
            or int(hidden_states.shape[1]) != self.prompt_len
        ):
            return False
        if int(hidden_states.shape[0]) != 1:
            self.record("physical_drop_skip_batch_gt1", 1)
            return False
        if (
            self.stage_physical_drop_enabled()
            and self._vision_stage_drop is not None
            and not self._vision_stage_physical_compacted
            and int(layer_idx) >= int(getattr(self.config, "vision_stage_layer", 0) or 0)
        ):
            self._physical_pending_kind = "stage"
            return True
        if (
            self.deferred_vision_drop_enabled()
            and self._vision_mid_drop is not None
            and not self._vision_mid_physical_compacted
            and int(layer_idx) >= int(self._effective_vision_layer())
        ):
            self._physical_pending_kind = "mid"
            self.record("physical_compaction_mid_trigger_layer_sum", int(layer_idx))
            return True
        if (
            self.deferred_vision_drop_enabled()
            and self._vision_deferred_final_drop is not None
            and self._vision_mid_physical_compacted
            and not self._vision_final_physical_compacted
            and int(layer_idx) >= int(getattr(self.config, "vision_deferred_drop_layer", -1))
        ):
            self._physical_pending_kind = "final"
            self.record("physical_compaction_final_trigger_layer_sum", int(layer_idx))
            return True
        if (
            self._vision_drop is not None
            and not self._vision_final_physical_compacted
            and int(layer_idx) >= int(self._effective_vision_layer())
        ):
            self._physical_pending_kind = "final"
            return True
        return False

    def physical_keep_indices(self, device: torch.device) -> Optional[torch.Tensor]:
        kind = self._physical_pending_kind
        if kind == "stage":
            drop_mask = self._vision_stage_drop
        elif kind == "mid":
            drop_mask = self._vision_mid_drop
        elif kind == "final" and self.deferred_vision_drop_enabled():
            drop_mask = self._vision_deferred_final_drop
        else:
            drop_mask = self._vision_drop
        if kind not in {"stage", "mid", "final"} or drop_mask is None or self._input_ids is None:
            return None
        keep = torch.ones((self.prompt_len,), device=device, dtype=torch.bool)
        drop = drop_mask[0, : self.prompt_len].to(device=device).bool()
        keep[: drop.numel()] &= ~drop
        # Always keep the final prompt token/generation anchor.
        if keep.numel() > 0:
            keep[-1] = True
        idx = torch.nonzero(keep, as_tuple=False).flatten().long()
        if idx.numel() <= 0 or idx.numel() == self.prompt_len:
            return None
        self.record("physical_drop_events", 1)
        self.record("physical_drop_removed", int(self.prompt_len - idx.numel()))
        self.record("physical_drop_kept", int(idx.numel()))
        self.record(f"physical_drop_{kind}_events", 1)
        self.record(f"physical_drop_{kind}_removed", int(self.prompt_len - idx.numel()))
        self._debug(
            f"physical_drop kind={kind} kept={int(idx.numel())} "
            f"removed={int(self.prompt_len - idx.numel())}"
        )
        return idx

    def physical_vision_local_keep_indices(self, device: torch.device) -> Optional[Tuple[torch.Tensor, int]]:
        kind = self._physical_pending_kind
        if kind == "stage":
            vision_keep = self._vision_stage_keep
        elif kind == "mid":
            vision_keep = self._vision_mid_keep
        elif kind == "final" and self.deferred_vision_drop_enabled():
            vision_keep = self._vision_deferred_final_keep
        else:
            vision_keep = self._vision_keep
        if vision_keep is None or not self._vision_by_batch:
            return None
        vision_idx = self._vision_by_batch[0]
        if vision_idx.numel() <= 0:
            return None
        keep_mask = vision_keep[0, vision_idx].bool()
        local_keep = torch.nonzero(keep_mask, as_tuple=False).flatten().long().to(device=device)
        if local_keep.numel() <= 0:
            return None
        return local_keep, int(vision_idx.numel())

    def commit_physical_compaction(self, keep_idx: torch.Tensor) -> None:
        """Remap all prompt-local state after a stage or final compaction."""
        kind = self._physical_pending_kind
        if kind not in {"stage", "mid", "final"} or self._input_ids is None:
            raise RuntimeError("physical compaction committed without a pending stage/mid/final plan")
        old_len = self.prompt_len
        keep_idx = keep_idx.to(device=self._input_ids.device, dtype=torch.long)
        fast_no_diagnostics = bool(getattr(self, "_runtime_exact_fast_path", False))
        if keep_idx.numel() <= 0 or (
            not fast_no_diagnostics and int(keep_idx.max().item()) >= old_len
        ):
            raise RuntimeError(f"invalid physical keep indices for prompt_len={old_len}")

        old_vision = tuple(x.to(device=keep_idx.device) for x in self._vision_by_batch)
        aligned_encoder = None
        if old_vision:
            aligned_encoder = self._aligned_encoder_score(int(old_vision[0].numel()), keep_idx.device)

        old_to_new = torch.full((old_len,), -1, device=keep_idx.device, dtype=torch.long)
        old_to_new[keep_idx] = torch.arange(keep_idx.numel(), device=keep_idx.device)

        def remap(groups: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
            out = []
            for group in groups:
                group = group.to(device=keep_idx.device, dtype=torch.long)
                mapped = old_to_new[group]
                out.append(mapped[mapped >= 0].detach())
            return tuple(out)

        def remap_prompt_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if mask is None:
                return None
            if mask.ndim != 2 or int(mask.shape[1]) != old_len:
                raise RuntimeError(
                    f"cannot remap prompt mask of shape={tuple(mask.shape)} for prompt_len={old_len}"
                )
            return mask.index_select(1, keep_idx.to(mask.device)).detach()

        if kind == "mid":
            self._vision_deferred_final_drop = remap_prompt_mask(
                self._vision_deferred_final_drop
            )
            self._vision_deferred_final_keep = remap_prompt_mask(
                self._vision_deferred_final_keep
            )
            self._vision_deferred_core_drop = remap_prompt_mask(
                self._vision_deferred_core_drop
            )
            self._vision_deferred_core_keep = remap_prompt_mask(
                self._vision_deferred_core_keep
            )

        self._input_ids = self._input_ids.index_select(1, keep_idx.to(self._input_ids.device)).detach()
        if self._active is not None:
            self._active = self._active.index_select(1, keep_idx.to(self._active.device)).detach()
        if self._role_ids is not None and int(self._role_ids.numel()) == old_len:
            self._role_ids = self._role_ids.index_select(0, keep_idx.to(self._role_ids.device)).detach()
        if fast_no_diagnostics:
            # The deployed vision selector consumes tensor role ids, never the
            # human-readable token strings.  Avoid a full GPU-to-CPU index copy
            # at each physical compaction.
            self._raw_tokens = ()
            self._decoded_tokens = ()
        else:
            keep_cpu = keep_idx.detach().cpu().tolist()
            if len(self._raw_tokens) == old_len:
                self._raw_tokens = tuple(self._raw_tokens[int(i)] for i in keep_cpu)
            if len(self._decoded_tokens) == old_len:
                self._decoded_tokens = tuple(self._decoded_tokens[int(i)] for i in keep_cpu)
        self._vision_by_batch = remap(self._vision_by_batch)
        self._text_by_batch = remap(self._text_by_batch)

        if aligned_encoder is not None and old_vision:
            kept_old_vision = old_to_new[old_vision[0]] >= 0
            detail_score = getattr(self, "_cls_detail_candidate_score", None)
            if (
                torch.is_tensor(detail_score)
                and int(detail_score.numel()) == int(old_vision[0].numel())
            ):
                self._cls_detail_candidate_score = detail_score[
                    kept_old_vision.to(detail_score.device)
                ].detach()
                self.record("cls_detail_candidate_score_physical_remap", 1)
                self.record(
                    f"cls_detail_candidate_score_physical_{kind}_tokens",
                    int(self._cls_detail_candidate_score.numel()),
                )
            if (
                self._feature_reserve_original_local_indices is not None
                and int(self._feature_reserve_original_local_indices.numel()) == int(old_vision[0].numel())
            ):
                self._feature_reserve_original_local_indices = self._feature_reserve_original_local_indices[
                    kept_old_vision.to(self._feature_reserve_original_local_indices.device)
                ].detach()
                self.record("feature_reserve_original_indices_physical_remap", 1)
            if (
                self._feature_reserve_current_is_anchor is not None
                and int(self._feature_reserve_current_is_anchor.numel()) == int(old_vision[0].numel())
            ):
                self._feature_reserve_current_is_anchor = self._feature_reserve_current_is_anchor[
                    kept_old_vision.to(self._feature_reserve_current_is_anchor.device)
                ].detach()
                self.record("feature_reserve_anchor_labels_physical_remap", 1)
            if fast_no_diagnostics or int(kept_old_vision.sum().item()) == int(self._vision_by_batch[0].numel()):
                self._vision_encoder_score = aligned_encoder[
                    kept_old_vision.to(aligned_encoder.device)
                ].detach()
                self.record("vision_encoder_score_physical_remap", 1)
                self.record("vision_encoder_score_physical_tokens", int(self._vision_encoder_score.numel()))
                self.record(
                    f"vision_encoder_score_physical_{kind}_tokens",
                    int(self._vision_encoder_score.numel()),
                )
                merge_plan = self._vision_merge_plan
                if merge_plan is not None and str(merge_plan.get("kind")) == kind:
                    expected = int(merge_plan["keep_local"].numel())
                    if int(self._vision_encoder_score.numel()) != expected:
                        raise RuntimeError(
                            f"merged encoder-score length mismatch: expected={expected} "
                            f"actual={int(self._vision_encoder_score.numel())}"
                        )
                    expected_mass = merge_plan.get("encoder_mass_before")
                    if expected_mass is not None:
                        actual_mass = float(self._vision_encoder_score.float().sum().item())
                        tolerance = 1e-4 * max(1.0, abs(float(expected_mass)))
                        if abs(actual_mass - float(expected_mass)) > tolerance:
                            raise RuntimeError(
                                f"merged encoder-score mass mismatch: expected={float(expected_mass):.8g} "
                                f"actual={actual_mass:.8g}"
                            )
                        self.record("vision_merge_encoder_mass_commit_conserved", 1)

        # Preserve a text mask selected at the same source layer as final
        # vision pruning (e.g. source layer 9 -> target layer 10). Vision
        # compaction changes absolute prompt coordinates but never removes text,
        # so remap pending layer-local masks instead of silently dropping one
        # masking layer at the physical-compaction boundary.
        remapped_text_drop = {}
        for target_layer, mask in self._text_drop_by_layer.items():
            if mask.ndim == 2 and int(mask.shape[1]) == old_len:
                remapped_text_drop[target_layer] = mask.index_select(
                    1, keep_idx.to(mask.device)
                ).detach()
        self._text_drop_by_layer = remapped_text_drop
        remapped_text_bias = {}
        for target_layer, bias in self._text_bias_by_layer.items():
            if bias.ndim == 2 and int(bias.shape[1]) == old_len:
                remapped_text_bias[target_layer] = bias.index_select(
                    1, keep_idx.to(bias.device)
                ).detach()
        self._text_bias_by_layer = remapped_text_bias
        if kind == "stage":
            self._vision_stage_physical_compacted = True
            # Stage masks use the old absolute prompt coordinates. The selected
            # tokens are now the entire current vision candidate set.
            self._vision_stage_drop = None
            self._vision_stage_keep = None
            self._vision_keep = None
        elif kind == "mid":
            self._vision_mid_physical_compacted = True
            self._vision_mid_drop = None
            self._vision_mid_keep = None
        else:
            self._vision_final_physical_compacted = True
            self._vision_drop = None
            self._vision_keep = None
            self._vision_deferred_final_drop = None
            self._vision_deferred_final_keep = None
            self._vision_deferred_core_drop = None
            self._vision_deferred_core_keep = None
        if self._vision_merge_plan is not None and str(self._vision_merge_plan.get("kind")) == kind:
            self._vision_merge_plan = None
        self._physical_pending_kind = None
        if kind == "final":
            self._restore_generation_text_masks()
        self.record("physical_compaction_commits", 1)
        self.record(f"physical_compaction_{kind}_commits", 1)
        self.record(f"physical_compaction_{kind}_prompt_len_before", old_len)
        self.record(f"physical_compaction_{kind}_prompt_len_after", self.prompt_len)
        self.record(
            f"physical_compaction_{kind}_vision_after",
            sum(int(x.numel()) for x in self._vision_by_batch),
        )

    def commit_physical_noop(self) -> None:
        """Finish a pending compaction whose keep set removes no token."""
        kind = self._physical_pending_kind
        if kind not in {"stage", "mid", "final"}:
            return
        if kind == "stage":
            self._vision_stage_physical_compacted = True
            self._vision_stage_drop = None
            self._vision_stage_keep = None
            self._vision_keep = None
        elif kind == "mid":
            self._vision_mid_physical_compacted = True
            self._vision_mid_drop = None
            self._vision_mid_keep = None
        else:
            self._vision_final_physical_compacted = True
            self._vision_drop = None
            self._vision_keep = None
            self._vision_deferred_final_drop = None
            self._vision_deferred_final_keep = None
            self._vision_deferred_core_drop = None
            self._vision_deferred_core_keep = None
        if self._vision_merge_plan is not None and str(self._vision_merge_plan.get("kind")) == kind:
            self._vision_merge_plan = None
        self._physical_pending_kind = None
        if kind == "final":
            self._restore_generation_text_masks()
        self.record("physical_compaction_noop", 1)
        self.record(f"physical_compaction_{kind}_noop", 1)
        self.record(
            f"physical_compaction_{kind}_vision_after",
            sum(int(x.numel()) for x in self._vision_by_batch),
        )

    def anchor_regional_pre_llm_enabled(self) -> bool:
        mode = str(getattr(self.config, "vision_stage_score_mode", "") or "").lower()
        return bool(
            self.stage_physical_drop_enabled()
            and mode in {
                "anchor_regional", "encoder_anchor_regional", "regional_reserve",
                "anchor_regional_cls", "encoder_anchor_regional_cls",
            }
            and int(getattr(self.config, "vision_stage_layer", 0) or 0) == 0
            and str(getattr(self.config, "vision_stage_budget_mode", "ratio") or "ratio").lower()
            == "depth_coupled"
        )

    def feature_reserve_pre_llm_enabled(self) -> bool:
        if not self.stage_physical_drop_enabled():
            return False
        mode = str(getattr(self.config, "vision_stage_score_mode", "") or "").lower()
        return bool(
            mode in {
                "feature_diverse", "encoder_feature_diverse", "feature_novelty",
                "encoder_topk", "encoder_saliency_topk",
                "prellm_vispruner", "original_vispruner_prellm",
            }
            and int(getattr(self.config, "vision_stage_layer", 0) or 0) == 0
            and str(getattr(self.config, "vision_stage_budget_mode", "ratio") or "ratio").lower()
            in {"depth_coupled", "surplus_fraction"}
        )

    def anchor_regional_pre_llm_compact(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        visual_pos_masks: Optional[torch.Tensor],
        deepstack_visual_embeds,
    ):
        """Apply the encoder-only anchor/regional stage before decoder layer 0."""
        if not self.anchor_regional_pre_llm_enabled():
            return inputs_embeds, attention_mask, position_ids, cache_position, visual_pos_masks, deepstack_visual_embeds
        if self._vision_stage_physical_compacted:
            return inputs_embeds, attention_mask, position_ids, cache_position, visual_pos_masks, deepstack_visual_embeds
        if inputs_embeds.ndim != 3 or int(inputs_embeds.shape[0]) != 1:
            raise RuntimeError("anchor_regional pre-LLM compaction requires batch_size=1")
        if not self._vision_by_batch:
            raise RuntimeError("anchor_regional pre-LLM compaction has no visual-token span")

        vision_idx = self._vision_by_batch[0].to(inputs_embeds.device)
        initial_n = int(vision_idx.numel())
        final_n = self.final_vision_budget(initial_n, 0)
        final_layer = max(0, int(self._effective_vision_layer()))
        num_layers = max(1, int(getattr(self.config, "num_layers", 1) or 1))
        depth_fraction = max(0.0, min(1.0, float(final_layer) / float(num_layers)))
        reserve_n = int(math.ceil(depth_fraction * float(max(0, initial_n - final_n))))
        keep_n = max(final_n, min(final_n + reserve_n, initial_n))

        score = self._aligned_encoder_score(initial_n, inputs_embeds.device)
        if score is None:
            raise RuntimeError("anchor_regional requires visual-encoder attention saliency")
        keep_local = self._select_vision_keep_anchor_regional(score, keep_n, final_n)
        local_keep_mask = torch.zeros(initial_n, device=inputs_embeds.device, dtype=torch.bool)
        local_keep_mask[keep_local] = True
        self._vision_stage_keep = torch.zeros((1, self.prompt_len), device=inputs_embeds.device, dtype=torch.bool)
        self._vision_stage_keep[0, vision_idx[local_keep_mask]] = True
        self._vision_stage_drop = torch.zeros_like(self._vision_stage_keep)
        self._vision_stage_drop[0, vision_idx[~local_keep_mask]] = True
        self._physical_pending_kind = "stage"
        keep_idx = self.physical_keep_indices(inputs_embeds.device)
        if keep_idx is None:
            self.commit_physical_noop()
            self.record("anchor_regional_pre_llm_noop", 1)
            return inputs_embeds, attention_mask, position_ids, cache_position, visual_pos_masks, deepstack_visual_embeds

        compacted = inputs_embeds.index_select(1, keep_idx)
        if torch.is_tensor(attention_mask) and attention_mask.ndim == 2:
            attention_mask = attention_mask.index_select(1, keep_idx.to(attention_mask.device))
        compact_len = int(compacted.shape[1])
        cache_position = torch.arange(
            compact_len,
            device=compacted.device,
            dtype=cache_position.dtype if torch.is_tensor(cache_position) else torch.long,
        )
        if torch.is_tensor(position_ids):
            pos_dtype = position_ids.dtype
            if position_ids.ndim == 2:
                position_ids = cache_position.to(dtype=pos_dtype).unsqueeze(0).expand(
                    int(compacted.shape[0]), -1
                )
            elif position_ids.ndim == 3:
                position_ids = cache_position.to(dtype=pos_dtype).view(1, 1, -1).expand(
                    int(position_ids.shape[0]), int(compacted.shape[0]), -1
                )
            else:
                raise RuntimeError(f"unsupported position_ids rank for anchor_regional: {position_ids.ndim}")
        if torch.is_tensor(visual_pos_masks) and visual_pos_masks.ndim == 2:
            visual_pos_masks = visual_pos_masks.index_select(1, keep_idx.to(visual_pos_masks.device))

        kept_local = torch.nonzero(local_keep_mask, as_tuple=False).flatten().long()
        anchor_local_mask = torch.zeros(initial_n, device=inputs_embeds.device, dtype=torch.bool)
        anchor_local_mask[keep_local[:final_n]] = True
        current_is_anchor = anchor_local_mask[kept_local].detach()
        if deepstack_visual_embeds is not None:
            compacted_deepstack = []
            for emb in deepstack_visual_embeds:
                if torch.is_tensor(emb) and emb.ndim >= 2 and int(emb.shape[0]) == initial_n:
                    compacted_deepstack.append(emb.index_select(0, kept_local.to(emb.device)))
                else:
                    compacted_deepstack.append(emb)
            deepstack_visual_embeds = compacted_deepstack

        self.record("vision_stage_events", 1)
        self.record("vision_stage_layer_sum", 0)
        self.record("vision_stage_dropped", initial_n - keep_n)
        self.record("vision_stage_budget_depth_coupled", 1)
        self.record("vision_stage_depth_fraction_x1000", int(round(depth_fraction * 1000.0)))
        self.record("vision_stage_depth_coupled_candidates", initial_n)
        self.record("vision_stage_depth_coupled_final_tokens", final_n)
        self.record("vision_stage_depth_coupled_keep_tokens", keep_n)
        self.record("vision_stage_depth_coupled_reserved", reserve_n)
        stage_keep_sum = int(keep_local.long().sum().item())
        stage_keep_sqsum = int((keep_local.long() * keep_local.long()).sum().item())
        self.record("anchor_regional_pre_llm_compaction", 1)
        self.record("anchor_regional_decoder_layer0_vision_tokens", keep_n)
        self.record("anchor_regional_stage_keep_local_sum", stage_keep_sum)
        self.record("anchor_regional_stage_keep_local_sqsum", stage_keep_sqsum)
        self.record("physical_compaction_stage_cache_position_contiguous", 1)
        self._debug(
            f"anchor_regional_pre_llm initial={initial_n} anchors={final_n} "
            f"reserves={reserve_n} layer0_vision={keep_n} depth_fraction={depth_fraction:.4f} "
            f"keep_local_sum={stage_keep_sum} keep_local_sqsum={stage_keep_sqsum}"
        )
        self.commit_physical_compaction(keep_idx)
        self._anchor_regional_current_is_anchor = current_is_anchor
        return compacted, attention_mask, position_ids, cache_position, visual_pos_masks, deepstack_visual_embeds

    def feature_reserve_pre_llm_compact(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        visual_pos_masks: Optional[torch.Tensor],
        deepstack_visual_embeds,
    ):
        """Pre-LLM depth-coupled selection using aligned visual embeddings.

        The final-K most salient encoder tokens are fixed as anchors. The
        depth-coupled surplus is selected from real residual tokens by the
        parameter-free product of encoder saliency and feature novelty relative
        to those anchors. ``encoder_topk`` is the saliency-only control. No
        token merging or model-specific positional heuristic is used.
        """
        if not self.feature_reserve_pre_llm_enabled():
            return inputs_embeds, attention_mask, position_ids, cache_position, visual_pos_masks, deepstack_visual_embeds
        if self._vision_stage_physical_compacted:
            return inputs_embeds, attention_mask, position_ids, cache_position, visual_pos_masks, deepstack_visual_embeds
        if self._vision_merge_enabled_for("stage"):
            raise RuntimeError("feature-reserve early selection does not support stage merging")
        if inputs_embeds.ndim != 3 or int(inputs_embeds.shape[0]) != 1:
            raise RuntimeError("feature-reserve pre-LLM compaction requires batch_size=1")
        if not self._vision_by_batch:
            raise RuntimeError("feature-reserve pre-LLM compaction has no visual-token span")

        vision_idx = self._vision_by_batch[0].to(inputs_embeds.device)
        initial_n = int(vision_idx.numel())
        final_n = self.final_vision_budget(initial_n, 0)
        final_layer = max(0, int(self._effective_vision_layer()))
        num_layers = max(1, int(getattr(self.config, "num_layers", 1) or 1))
        budget_mode = str(
            getattr(self.config, "vision_stage_budget_mode", "depth_coupled") or "depth_coupled"
        ).lower()
        explicit_stage_tokens = int(getattr(self.config, "vision_stage_keep_tokens", 0) or 0)
        if explicit_stage_tokens > 0:
            keep_n = max(final_n, min(explicit_stage_tokens, initial_n))
            reserve_n = keep_n - final_n
            surplus_fraction = reserve_n / float(max(1, initial_n - final_n))
            self.record("vision_stage_budget_absolute_cap", 1)
            self.record("vision_stage_budget_absolute_cap_tokens", explicit_stage_tokens)
        else:
            if budget_mode == "surplus_fraction":
                surplus_fraction = max(
                    0.0,
                    min(1.0, float(getattr(self.config, "vision_stage_surplus_fraction", -1.0))),
                )
            else:
                surplus_fraction = max(0.0, min(1.0, float(final_layer) / float(num_layers)))
            reserve_n = int(math.ceil(surplus_fraction * float(max(0, initial_n - final_n))))
            keep_n = max(final_n, min(final_n + reserve_n, initial_n))

        score = self._aligned_encoder_score(initial_n, inputs_embeds.device)
        if score is None or int(score.numel()) != initial_n:
            raise RuntimeError("feature-reserve selection requires aligned visual-encoder saliency")
        skip_runtime_diagnostics = bool(getattr(self, "_runtime_exact_fast_path", False))
        if not skip_runtime_diagnostics and not bool(torch.isfinite(score).all().item()):
            raise RuntimeError("feature-reserve encoder saliency contains non-finite values")
        visual_features = inputs_embeds[0].index_select(0, vision_idx).detach()
        if int(visual_features.shape[0]) != initial_n:
            raise RuntimeError("feature-reserve visual embedding alignment failed")

        mode = str(getattr(self.config, "vision_stage_score_mode", "") or "").lower()
        if mode in {"encoder_topk", "encoder_saliency_topk"}:
            order = torch.argsort(score.float(), descending=True, stable=True)
            keep_local = order[:keep_n]
            anchors = order[:final_n]
            self.record("vision_encoder_topk_stage_events", 1)
        elif mode in {"prellm_vispruner", "original_vispruner_prellm"}:
            keep_local = self._select_vision_keep_vispruner_features(
                features=visual_features,
                score=score,
                keep_n=keep_n,
                important_ratio=float(getattr(self.config, "important_ratio", 0.5)),
            )
            important_n = min(
                keep_n,
                max(0, int(round(float(keep_n) * float(getattr(self.config, "important_ratio", 0.5))))),
            )
            anchors = keep_local[:important_n]
            self.record("vision_prellm_vispruner_stage_events", 1)
        else:
            keep_local = self._select_vision_keep_feature_diverse(
                score=score,
                features=visual_features,
                keep_n=keep_n,
                anchor_n=final_n,
            )
            anchors = keep_local[:final_n]

        local_keep_mask = torch.zeros(initial_n, device=inputs_embeds.device, dtype=torch.bool)
        local_keep_mask[keep_local] = True
        self._vision_stage_keep = torch.zeros((1, self.prompt_len), device=inputs_embeds.device, dtype=torch.bool)
        self._vision_stage_keep[0, vision_idx[local_keep_mask]] = True
        self._vision_stage_drop = torch.zeros_like(self._vision_stage_keep)
        self._vision_stage_drop[0, vision_idx[~local_keep_mask]] = True
        self._physical_pending_kind = "stage"
        keep_idx = self.physical_keep_indices(inputs_embeds.device)
        if keep_idx is None:
            self.commit_physical_noop()
            self.record("feature_reserve_pre_llm_noop", 1)
            return inputs_embeds, attention_mask, position_ids, cache_position, visual_pos_masks, deepstack_visual_embeds

        compacted = inputs_embeds.index_select(1, keep_idx)
        if torch.is_tensor(attention_mask) and attention_mask.ndim == 2:
            attention_mask = attention_mask.index_select(1, keep_idx.to(attention_mask.device))
        compact_len = int(compacted.shape[1])
        selected_original_position_ids = (
            position_ids.index_select(-1, keep_idx.to(position_ids.device))
            if torch.is_tensor(position_ids)
            else None
        )
        cache_position = torch.arange(
            compact_len,
            device=compacted.device,
            dtype=cache_position.dtype if torch.is_tensor(cache_position) else torch.long,
        )
        if torch.is_tensor(position_ids):
            preserve_stage_positions = bool(
                getattr(self, "_preserve_stage_original_position_ids", False)
            )
            if preserve_stage_positions:
                position_ids = selected_original_position_ids
                self.record("physical_compaction_stage_original_position_ids_preserved", 1)
                if not skip_runtime_diagnostics:
                    flat_positions = position_ids.reshape(-1, compact_len)
                    contiguous_reference = cache_position.to(
                        device=flat_positions.device, dtype=flat_positions.dtype
                    ).unsqueeze(0).expand_as(flat_positions)
                    self.record(
                        "physical_compaction_stage_preserved_coordinate_changes",
                        int(flat_positions.ne(contiguous_reference).sum().item()),
                    )
            else:
                self.record("physical_compaction_stage_position_ids_contiguous", 1)
                pos_dtype = position_ids.dtype
                if position_ids.ndim == 2:
                    position_ids = cache_position.to(dtype=pos_dtype).unsqueeze(0).expand(
                        int(compacted.shape[0]), -1
                    )
                elif position_ids.ndim == 3:
                    position_ids = cache_position.to(dtype=pos_dtype).view(1, 1, -1).expand(
                        int(position_ids.shape[0]), int(compacted.shape[0]), -1
                    )
                else:
                    raise RuntimeError(f"unsupported position_ids rank for feature-reserve: {position_ids.ndim}")
        if torch.is_tensor(visual_pos_masks) and visual_pos_masks.ndim == 2:
            visual_pos_masks = visual_pos_masks.index_select(1, keep_idx.to(visual_pos_masks.device))

        kept_local = torch.nonzero(local_keep_mask, as_tuple=False).flatten().long()
        anchor_local_mask = torch.zeros(initial_n, device=inputs_embeds.device, dtype=torch.bool)
        anchor_local_mask[anchors] = True
        current_is_anchor = anchor_local_mask[kept_local].detach()
        if deepstack_visual_embeds is not None:
            compacted_deepstack = []
            for emb in deepstack_visual_embeds:
                if torch.is_tensor(emb) and emb.ndim >= 2 and int(emb.shape[0]) == initial_n:
                    compacted_deepstack.append(emb.index_select(0, kept_local.to(emb.device)))
                else:
                    compacted_deepstack.append(emb)
            deepstack_visual_embeds = compacted_deepstack

        self.record("vision_stage_events", 1)
        self.record("vision_stage_layer_sum", 0)
        self.record("vision_stage_dropped", initial_n - keep_n)
        if budget_mode == "surplus_fraction":
            self.record("vision_stage_budget_surplus_fraction", 1)
            self.record(
                "vision_stage_surplus_fraction_x1000000",
                int(round(surplus_fraction * 1_000_000.0)),
            )
        else:
            self.record("vision_stage_budget_depth_coupled", 1)
        self.record("vision_stage_depth_fraction_x1000", int(round(surplus_fraction * 1000.0)))
        self.record("vision_stage_depth_coupled_candidates", initial_n)
        self.record("vision_stage_depth_coupled_final_tokens", final_n)
        self.record("vision_stage_depth_coupled_keep_tokens", keep_n)
        self.record("vision_stage_depth_coupled_reserved", reserve_n)
        stage_keep_sum = 0
        stage_keep_sqsum = 0
        if not skip_runtime_diagnostics:
            stage_keep_sum = int(keep_local.long().sum().item())
            stage_keep_sqsum = int((keep_local.long() * keep_local.long()).sum().item())
        self.record("feature_reserve_pre_llm_compaction", 1)
        self.record("feature_reserve_decoder_layer0_vision_tokens", keep_n)
        self.record("feature_reserve_stage_anchors", final_n)
        self.record("feature_reserve_stage_reserves", keep_n - final_n)
        if not skip_runtime_diagnostics:
            self.record("feature_reserve_stage_keep_local_sum", stage_keep_sum)
            self.record("feature_reserve_stage_keep_local_sqsum", stage_keep_sqsum)
        self.record("physical_compaction_stage_cache_position_contiguous", 1)
        if not skip_runtime_diagnostics:
            self._debug(
                f"feature_reserve_pre_llm mode={mode} initial={initial_n} anchors={final_n} "
                f"reserves={keep_n - final_n} layer0_vision={keep_n} "
                f"surplus_fraction={surplus_fraction:.6f} budget_mode={budget_mode} "
                f"keep_local_sum={stage_keep_sum} keep_local_sqsum={stage_keep_sqsum}"
            )
        self.commit_physical_compaction(keep_idx)
        self._feature_reserve_current_is_anchor = current_is_anchor
        self._feature_reserve_original_local_indices = kept_local.detach().clone()
        return compacted, attention_mask, position_ids, cache_position, visual_pos_masks, deepstack_visual_embeds

    def mask_for_layer(
        self,
        layer_idx: int,
        batch_size: int,
        key_len: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if not self.enabled or key_len <= 0:
            return None
        pieces = []
        # In physical-drop mode, once the prompt has been compacted, the pruned
        # vision tokens are no longer present in the sequence.  Re-applying the
        # old key mask would either be redundant or misaligned with the shorter
        # key length, especially under FlashAttention2.
        vision_already_compacted = self.physical_drop_enabled() and self._vision_physical_compacted
        if (
            self._vision_stage_drop is not None
            and self._vision_drop is None
            and layer_idx > int(getattr(self.config, "vision_stage_layer", 0) or 0)
            and not vision_already_compacted
        ):
            pieces.append(self._fit_key_len(self._vision_stage_drop, batch_size, key_len, device))
        if (
            self._vision_drop is not None
            and layer_idx > self._effective_vision_layer()
            and not vision_already_compacted
        ):
            pieces.append(self._fit_key_len(self._vision_drop, batch_size, key_len, device))
        text_drop = self._text_drop_by_layer.get(layer_idx)
        if text_drop is not None:
            pieces.append(self._fit_key_len(text_drop, batch_size, key_len, device))
        if not pieces:
            return None
        out = pieces[0]
        for p in pieces[1:]:
            out = out | p
        dropped = int(out.sum().item())
        if dropped > 0:
            self.record("mask_apply_calls", 1)
            self.record("mask_apply_dropped_keys", dropped)
            self._debug(f"apply layer={layer_idx} key_len={key_len} dropped_keys={dropped}")
        return out

    def text_mask_for_layer(
        self,
        layer_idx: int,
        batch_size: int,
        key_len: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Return only the layer-local text activity mask, excluding vision."""
        if not self.enabled or key_len <= 0:
            return None
        text_drop = self._text_drop_by_layer.get(int(layer_idx))
        if text_drop is None:
            return None
        return self._fit_key_len(text_drop, batch_size, key_len, device)

    def routing_balance_bias_for_layer(
        self,
        layer_idx: int,
        batch_size: int,
        key_len: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Return a key bias for last-query correction without leaving FA2/SDPA."""
        if not self.enabled or key_len <= 0 or not self._uses_routing_balance_text_mode():
            return None
        bias = self._text_bias_by_layer.get(int(layer_idx))
        if bias is None:
            return None
        fitted = self._fit_key_len(bias, batch_size, key_len, device).to(dtype=torch.float32)
        if not bool((fitted.abs() > 0).any().item()):
            return None
        self.record("routing_balance_apply_calls", 1)
        self.record("routing_balance_apply_biased_keys", int((fitted.abs() > 0).sum().item()))
        self.record("routing_balance_apply_abs_bias", float(fitted.abs().sum().item()))
        return fitted

    def text_bias_for_layer(
        self,
        layer_idx: int,
        batch_size: int,
        key_len: int,
        q_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not self.enabled or key_len <= 0 or self._uses_routing_balance_text_mode():
            return None
        bias = self._text_bias_by_layer.get(layer_idx)
        if bias is None:
            return None
        fitted = self._fit_key_len(bias, batch_size, key_len, device).to(dtype=torch.float32)
        if fitted.abs().sum().item() <= 0:
            return None
        out = fitted[:, None, None, :].expand(batch_size, 1, q_len, key_len).clone()
        self.record("soft_mask_apply_calls", 1)
        self.record("soft_mask_apply_abs_bias", float(fitted.abs().sum().item()))
        self._debug(
            f"soft_apply layer={layer_idx} key_len={key_len} "
            f"biased_keys={int((fitted.abs() > 0).sum().item())} "
            f"gamma={float(getattr(self.config, 'text_soft_gamma', 0.0) or 0.0):.4f}"
        )
        return out.to(dtype=dtype if dtype.is_floating_point else torch.float32)

    def set_vision_encoder_score(self, score: torch.Tensor, source: str = "") -> None:
        if not self.enabled or score is None or not torch.is_tensor(score) or score.numel() == 0:
            return
        vals = score.detach().float().flatten()
        vals = vals.clamp_min(0.0)
        self._vision_encoder_score = vals
        self.record("vision_encoder_score_events", 1)
        self.record("vision_encoder_score_tokens", int(vals.numel()))
        if self.config.debug:
            self._debug(
                f"vision_encoder_score source={source} tokens={int(vals.numel())} "
                f"mean={float(vals.mean().item()):.6g} max={float(vals.max().item()):.6g}"
            )

    def _normalized_vision_merge_placement(self) -> str:
        raw = str(getattr(self.config, "vision_merge_placement", "legacy") or "legacy").strip().lower()
        aliases = {"auto": "legacy", "first": "legacy", "stage": "early"}
        placement = aliases.get(raw, raw)
        allowed = {"legacy", "none", "early", "final", "both"}
        if placement not in allowed:
            raise ValueError(
                f"invalid vision_merge_placement={raw!r}; expected one of {sorted(allowed)}"
            )
        return placement

    def _vision_merge_enabled_for(self, kind: str) -> bool:
        if not self.enabled or not bool(getattr(self.config, "vision_merge", False)):
            return False
        if kind not in {"stage", "final"}:
            return False
        placement = self._normalized_vision_merge_placement()
        if placement == "none":
            return False
        if placement == "legacy":
            # Historical behavior only had a final-mask merge. Keeping this as
            # the default preserves old experiment scripts and prior results.
            return kind == "final" and not self._vision_merge_applied
        requested = kind == "final" if placement == "final" else kind == "stage" if placement == "early" else True
        return requested and kind not in self._vision_merge_applied_kinds

    def _vision_merge_masks(self, kind: str):
        if kind == "stage":
            return self._vision_stage_drop, self._vision_stage_keep
        if kind == "final":
            return self._vision_drop, self._vision_keep
        return None, None

    @staticmethod
    def _apply_vision_merge_plan_to_local_features(
        features: torch.Tensor,
        plan: dict,
    ) -> torch.Tensor:
        """Apply one physical merge assignment to an aligned vision feature tensor."""
        source_tokens = int(plan["source_tokens"])
        if not torch.is_tensor(features) or features.ndim < 1 or int(features.shape[0]) != source_tokens:
            raise RuntimeError(
                f"merge-plan feature alignment mismatch: expected first dim={source_tokens}, "
                f"got={tuple(features.shape) if torch.is_tensor(features) else type(features).__name__}"
            )
        original_shape = tuple(features.shape)
        work = features.float().reshape(source_tokens, -1)
        keep_local = plan["keep_local"].to(device=work.device, dtype=torch.long)
        drop_local = plan["drop_local"].to(device=work.device, dtype=torch.long)
        keep_feat = work.index_select(0, keep_local)
        drop_feat = work.index_select(0, drop_local)
        mode = str(plan["mode"])
        if mode in _SALIENCY_CONVEX_MERGE_MODES:
            assignment = plan["assignment"].to(device=work.device, dtype=torch.long)
            keep_weight = plan["keep_weight"].to(device=work.device, dtype=torch.float32).view(-1, 1)
            drop_weight = plan["drop_weight"].to(device=work.device, dtype=torch.float32).view(-1, 1)
            numerator = keep_feat * keep_weight
            numerator.index_add_(0, assignment, drop_feat * drop_weight)
            denom = keep_weight.clone()
            denom.index_add_(0, assignment, drop_weight)
            merged = numerator / denom.clamp_min(1e-8)
        elif mode in {"mean", "hard", "nearest_mean", "functional_nearest_mean"}:
            assignment = plan["assignment"].to(device=work.device, dtype=torch.long)
            contribution = torch.zeros_like(keep_feat)
            contribution.index_add_(0, assignment, drop_feat)
            counts = torch.zeros((keep_local.numel(), 1), device=work.device, dtype=torch.float32)
            counts.index_add_(0, assignment, torch.ones((drop_local.numel(), 1), device=work.device))
            merged = (keep_feat + contribution) / (1.0 + counts).clamp_min(1.0)
        elif mode in {"nearest_residual", "visionzip_residual", "functional_nearest_residual"}:
            assignment = plan["assignment"].to(device=work.device, dtype=torch.long)
            contribution = torch.zeros_like(keep_feat)
            contribution.index_add_(0, assignment, drop_feat)
            counts = torch.zeros((keep_local.numel(), 1), device=work.device, dtype=torch.float32)
            counts.index_add_(0, assignment, torch.ones((drop_local.numel(), 1), device=work.device))
            source_mean = contribution / counts.clamp_min(1.0)
            has_source = counts.squeeze(-1) > 0
            merged = keep_feat.clone()
            merged[has_source] = keep_feat[has_source] + source_mean[has_source]
        else:
            weights = plan["weights"].to(device=work.device, dtype=torch.float32)
            contribution = weights @ drop_feat
            denom = 1.0 + weights.sum(dim=1, keepdim=True).clamp_min(0.0)
            merged = (keep_feat + contribution) / denom.clamp_min(1e-6)
        out = work.clone()
        out.index_copy_(0, keep_local, merged)
        return out.reshape(original_shape).to(dtype=features.dtype)

    def _merge_encoder_score_with_plan(self, plan: dict, device: torch.device) -> None:
        """Update encoder saliency using the merge mode's explicit prior policy."""
        source_tokens = int(plan["source_tokens"])
        score = self._aligned_encoder_score(source_tokens, device)
        if score is None:
            self.record("vision_merge_encoder_score_missing", 1)
            return
        score = score.float().flatten()
        keep_local = plan["keep_local"].to(device=score.device, dtype=torch.long)
        drop_local = plan["drop_local"].to(device=score.device, dtype=torch.long)
        merged_keep = score.index_select(0, keep_local).clone()
        mode = str(plan["mode"])
        if mode in _ANCHOR_PRIOR_MERGE_MODES:
            # The slot was selected by its anchor saliency. Recycled content
            # enriches the representation but must not inflate its PMI rank by
            # the number of discarded tokens assigned to it.
            self.record("vision_merge_encoder_policy_anchor", 1)
        elif mode in _WEIGHTED_MEAN_PRIOR_MERGE_MODES:
            assignment = plan["assignment"].to(device=score.device, dtype=torch.long)
            keep_weight = plan["keep_weight"].to(device=score.device, dtype=torch.float32)
            drop_weight = plan["drop_weight"].to(device=score.device, dtype=torch.float32)
            numerator = merged_keep * keep_weight
            numerator.index_add_(0, assignment, score.index_select(0, drop_local) * drop_weight)
            denom = keep_weight.clone()
            denom.index_add_(0, assignment, drop_weight)
            merged_keep = numerator / denom.clamp_min(1e-8)
            self.record("vision_merge_encoder_policy_weighted_mean", 1)
        elif mode in {
            "mean", "hard", "nearest_mean", "nearest_residual", "visionzip_residual",
            "functional_nearest_mean", "functional_nearest_residual",
        }:
            assignment = plan["assignment"].to(device=score.device, dtype=torch.long)
            merged_keep.index_add_(0, assignment, score.index_select(0, drop_local))
            self.record("vision_merge_encoder_policy_sum", 1)
        else:
            weights = plan["weights"].to(device=score.device, dtype=torch.float32)
            merged_keep = merged_keep + weights @ score.index_select(0, drop_local)
            self.record("vision_merge_encoder_policy_sum", 1)
        merged_score = torch.zeros_like(score)
        merged_score.index_copy_(0, keep_local, merged_keep)
        before = float(score.sum().item())
        after = float(merged_score.sum().item())
        plan["encoder_score_before"] = before
        plan["encoder_score_after"] = after
        if mode not in (_ANCHOR_PRIOR_MERGE_MODES | _WEIGHTED_MEAN_PRIOR_MERGE_MODES):
            tolerance = 1e-4 * max(1.0, abs(before))
            if abs(after - before) > tolerance:
                raise RuntimeError(
                    f"encoder saliency mass was not conserved by merge: before={before:.8g} after={after:.8g}"
                )
            plan["encoder_mass_before"] = before
            self.record("vision_merge_encoder_mass_conserved", 1)
        self._vision_encoder_score = merged_score.detach()
        self.record("vision_merge_encoder_score_events", 1)
        self.record("vision_merge_encoder_score_sources", int(drop_local.numel()))
        self.record("vision_merge_encoder_score_targets", int(keep_local.numel()))

    def merge_vision_local_features_before_physical_drop(
        self,
        features: torch.Tensor,
        *,
        source: str = "aux",
    ) -> torch.Tensor:
        """Merge aligned auxiliary vision features with the pending hidden-state plan."""
        kind = str(self._physical_pending_kind or "")
        plan = self._vision_merge_plan
        if plan is None or kind not in {"stage", "final"} or str(plan.get("kind")) != kind:
            return features
        merged = self._apply_vision_merge_plan_to_local_features(features, plan)
        self.record("vision_merge_aux_feature_events", 1)
        self.record(f"vision_merge_aux_{source}_events", 1)
        return merged

    def _merge_hidden_with_plan(
        self,
        hidden_states: torch.Tensor,
        *,
        kind: str,
        layer_label: str,
    ) -> torch.Tensor:
        if not self._vision_merge_enabled_for(kind):
            return hidden_states
        drop_plan, keep_plan = self._vision_merge_masks(kind)
        if (
            drop_plan is None
            or keep_plan is None
            or not torch.is_tensor(hidden_states)
            or hidden_states.ndim != 3
            or int(hidden_states.shape[1]) != self.prompt_len
        ):
            return hidden_states

        out = hidden_states.clone()
        total_assigned = 0
        merge_mode = str(getattr(self.config, "vision_merge_mode", "weighted") or "weighted").lower()
        if int(hidden_states.shape[0]) == 1:
            self._vision_merge_plan = None
        for batch_idx in range(min(int(hidden_states.shape[0]), len(self._vision_by_batch))):
            vision_idx = self._vision_by_batch[batch_idx].to(device=hidden_states.device)
            if vision_idx.numel() <= 1:
                continue
            drop_mask = drop_plan[batch_idx, vision_idx].to(device=hidden_states.device).bool()
            keep_mask = keep_plan[batch_idx, vision_idx].to(device=hidden_states.device).bool()
            if int(drop_mask.sum().item()) == 0 or int(keep_mask.sum().item()) == 0:
                continue
            if bool((drop_mask & keep_mask).any().item()) or int((drop_mask | keep_mask).sum().item()) != int(vision_idx.numel()):
                raise RuntimeError("vision merge keep/drop masks do not partition the current vision tokens")
            keep_local = torch.nonzero(keep_mask, as_tuple=False).flatten().long()
            drop_local = torch.nonzero(drop_mask, as_tuple=False).flatten().long()
            local_features = hidden_states[batch_idx, vision_idx, :]
            keep_feat = local_features.index_select(0, keep_local).float()
            drop_feat = local_features.index_select(0, drop_local).float()
            similarity = torch.nn.functional.normalize(drop_feat, dim=-1) @ torch.nn.functional.normalize(
                keep_feat, dim=-1
            ).T
            if merge_mode in {"functional_nearest_mean", "functional_nearest_residual"}:
                signatures = self._vision_functional_signatures.get(batch_idx)
                if signatures is None or int(signatures.shape[0]) != int(vision_idx.numel()):
                    raise RuntimeError(
                        "functional merge requires signatures aligned with current vision tokens"
                    )
                signatures = signatures.to(device=hidden_states.device).float()
                keep_signature = torch.nn.functional.normalize(
                    signatures.index_select(0, keep_local), dim=-1
                )
                drop_signature = torch.nn.functional.normalize(
                    signatures.index_select(0, drop_local), dim=-1
                )
                similarity = drop_signature @ keep_signature.T
                self.record("vision_merge_functional_similarity_events", 1)
                self.record("vision_merge_functional_similarity_sources", int(drop_local.numel()))
            plan = {
                "kind": kind,
                "mode": merge_mode,
                "source_tokens": int(vision_idx.numel()),
                "keep_local": keep_local.detach(),
                "drop_local": drop_local.detach(),
            }
            if merge_mode in _HARD_ASSIGN_MERGE_MODES:
                assignment_similarity = similarity
                if merge_mode in _SEGMENT_LOCAL_MERGE_MODES:
                    starts = torch.ones((vision_idx.numel(),), device=vision_idx.device, dtype=torch.bool)
                    if vision_idx.numel() > 1:
                        starts[1:] = vision_idx[1:] != (vision_idx[:-1] + 1)
                    segment_id = torch.cumsum(starts.long(), dim=0) - 1
                    allowed = segment_id.index_select(0, drop_local)[:, None] == segment_id.index_select(0, keep_local)[None, :]
                    missing = ~allowed.any(dim=1)
                    if bool(missing.any().item()):
                        # A globally-selected keep set can theoretically leave a
                        # visual block without an anchor. Keep the assignment
                        # total and record the rare fallback instead of silently
                        # discarding its recycled information.
                        allowed[missing] = True
                        self.record("vision_merge_segment_fallback_sources", int(missing.sum().item()))
                    assignment_similarity = similarity.masked_fill(~allowed, float("-inf"))
                    self.record("vision_merge_segment_local_events", 1)
                    self.record("vision_merge_segment_count", int(starts.sum().item()))
                assignment = torch.argmax(assignment_similarity, dim=-1)
                if int(assignment.numel()) != int(drop_local.numel()):
                    raise RuntimeError("each dropped vision token must have exactly one merge assignment")
                plan["assignment"] = assignment.detach()
                if merge_mode in _SALIENCY_CONVEX_MERGE_MODES:
                    aligned = self._aligned_encoder_score(int(vision_idx.numel()), hidden_states.device)
                    if aligned is None:
                        token_weight = torch.ones((vision_idx.numel(),), device=hidden_states.device, dtype=torch.float32)
                        self.record("vision_merge_saliency_weight_fallback", 1)
                    else:
                        token_weight = aligned.float().clamp_min(0.0)
                        if not bool((token_weight > 0).any().item()):
                            token_weight = torch.ones_like(token_weight)
                            self.record("vision_merge_saliency_weight_fallback", 1)
                    keep_weight = token_weight.index_select(0, keep_local).clone()
                    drop_weight = token_weight.index_select(0, drop_local).clone()
                    cluster_denom = keep_weight.clone()
                    cluster_denom.index_add_(0, assignment, drop_weight)
                    zero_cluster = cluster_denom <= 0
                    if bool(zero_cluster.any().item()):
                        keep_weight[zero_cluster] = 1.0
                    plan["keep_weight"] = keep_weight.detach()
                    plan["drop_weight"] = drop_weight.detach()
                    self.record(f"vision_merge_mode_{merge_mode}", 1)
                    self.record("vision_merge_mode_saliency_convex", 1)
                elif merge_mode in {"mean", "hard", "nearest_mean", "functional_nearest_mean"}:
                    self.record("vision_merge_mode_mean", 1)
                    self.record("vision_merge_mode_nearest_mean", 1)
                else:
                    self.record("vision_merge_mode_nearest_residual", 1)
                    self.record("vision_merge_mode_visionzip_residual", 1)
            else:
                tau = max(
                    1e-6,
                    float(getattr(self.config, "vision_merge_temperature", 0.07) or 0.07),
                )
                plan["weights"] = torch.softmax(similarity / tau, dim=-1).T.detach()
                self.record("vision_merge_mode_weighted", 1)
            merged_local = self._apply_vision_merge_plan_to_local_features(local_features, plan)
            out[batch_idx, vision_idx, :] = merged_local.to(dtype=out.dtype)
            total_assigned += int(drop_local.numel())
            if int(hidden_states.shape[0]) == 1 and batch_idx == 0:
                self._vision_merge_plan = plan
                self._merge_encoder_score_with_plan(plan, hidden_states.device)

        self._vision_merge_applied = True
        self._vision_merge_applied_kinds.add(kind)
        placement = self._normalized_vision_merge_placement()
        self.record(f"vision_merge_placement_{placement}", 1)
        if total_assigned > 0:
            self.record("vision_merge_events", 1)
            self.record("vision_merge_assigned", total_assigned)
            self.record(f"vision_merge_{kind}_events", 1)
            self.record(f"vision_merge_{kind}_assigned", total_assigned)
            self._debug(
                f"vision_merge kind={kind} layer={layer_label} assigned={total_assigned} "
                f"mode={merge_mode} placement={placement}"
            )
        return out

    def merge_hidden_for_layer(self, layer_idx: int, hidden_states: torch.Tensor) -> torch.Tensor:
        if int(layer_idx) != int(self._effective_vision_layer()) + 1:
            return hidden_states
        return self._merge_hidden_with_plan(
            hidden_states,
            kind="final",
            layer_label=str(layer_idx),
        )

    def merge_hidden_before_physical_drop(self, hidden_states: torch.Tensor) -> torch.Tensor:
        kind = str(self._physical_pending_kind or "")
        if kind not in {"stage", "final"}:
            return hidden_states
        return self._merge_hidden_with_plan(
            hidden_states,
            kind=kind,
            layer_label="physical",
        )

    def _uses_routing_balance_text_mode(self) -> bool:
        return str(self.config.text_mask_mode or "").lower().startswith("routing_balance")

    def _routing_balance_source_active(self, layer_idx: int) -> bool:
        reference_layer = int(self._effective_vision_layer())
        mode = str(self.config.text_mask_mode or "").lower()
        if mode.startswith("routing_balance_prune_delta"):
            return int(layer_idx) == reference_layer
        if mode.startswith("routing_balance_once"):
            return int(layer_idx) in {reference_layer, reference_layer + 1}
        return reference_layer <= int(layer_idx) < max(0, int(self.config.num_layers) - 1)

    def _routing_balance_target_indices(
        self,
        batch_idx: int,
        text_idx: torch.Tensor,
    ) -> torch.Tensor:
        mode = str(self.config.text_mask_mode or "").lower()
        if "option_label" not in mode:
            return text_idx
        if self._role_ids is None or int(self._role_ids.numel()) != self.prompt_len:
            self.record("routing_balance_option_metadata_missing", 1)
            return text_idx[:0]
        selected = []
        for pos in text_idx.detach().long().cpu().tolist():
            role = int(self._role_ids[int(pos)].item())
            token = self._decoded_tokens[int(pos)].strip() if int(pos) < len(self._decoded_tokens) else ""
            if role == 2 and len(token) == 1 and token in "ABCDEFGH":
                selected.append(int(pos))
        if not selected:
            self.record("routing_balance_option_labels_missing", 1)
            return text_idx[:0]
        self.record("routing_balance_option_labels_found", len(selected))
        return torch.tensor(selected, device=text_idx.device, dtype=torch.long)

    def _last_query_modality_mass(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        text_idx: torch.Tensor,
        vision_idx: torch.Tensor,
        key_drop: Optional[torch.Tensor] = None,
    ) -> tuple[float, float]:
        device = query_states.device
        key_len = int(key_states.shape[2])
        text_idx = text_idx.to(device=device, dtype=torch.long)
        vision_idx = vision_idx.to(device=device, dtype=torch.long)
        text_idx = text_idx[text_idx < key_len]
        vision_idx = vision_idx[vision_idx < key_len]
        if text_idx.numel() == 0 or vision_idx.numel() == 0:
            return float("nan"), float("nan")
        query_pos = int(text_idx[-1].item())
        q = query_states[0, :, query_pos, :].float()
        k = key_states[0, :, :key_len, :].float()
        if int(k.shape[0]) != int(q.shape[0]):
            if int(q.shape[0]) % int(k.shape[0]) != 0:
                raise RuntimeError(
                    f"routing-balance query/key head mismatch: q={int(q.shape[0])} kv={int(k.shape[0])}"
                )
            k = k.repeat_interleave(int(q.shape[0]) // int(k.shape[0]), dim=0)
        logits = torch.einsum("hd,hkd->hk", q, k) / (float(q.shape[-1]) ** 0.5)
        visible = torch.arange(key_len, device=device) <= query_pos
        if key_drop is not None:
            drop = key_drop.to(device=device).bool().flatten()[:key_len]
            if int(drop.numel()) < key_len:
                drop = torch.cat((drop, torch.zeros(key_len - int(drop.numel()), device=device, dtype=torch.bool)))
            visible &= ~drop
        if attention_mask is not None:
            if attention_mask.ndim == 4:
                row = attention_mask[0, 0, query_pos, :key_len]
                finite = torch.isfinite(row) & (row > torch.finfo(row.dtype).min / 2)
                visible &= finite
                logits = logits + row.float().unsqueeze(0)
            elif attention_mask.ndim == 2:
                visible &= attention_mask[0, :key_len].to(device=device).bool()
        logits = logits.masked_fill(~visible.unsqueeze(0), torch.finfo(logits.dtype).min)
        probs = torch.softmax(logits, dim=-1).masked_fill(~visible.unsqueeze(0), 0.0)
        text_mass = float(probs.index_select(-1, text_idx).sum(dim=-1).mean().item())
        vision_mass = float(probs.index_select(-1, vision_idx).sum(dim=-1).mean().item())
        return text_mass, vision_mass

    def _observe_routing_balance(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> None:
        reference_layer = int(self._effective_vision_layer())
        target_layer = int(layer_idx) + 1
        eps = 1e-6
        for b, text_idx_raw in enumerate(self._text_by_batch):
            text_idx = text_idx_raw.to(device=query_states.device)
            vision_idx = self._vision_by_batch[b].to(device=query_states.device)
            if text_idx.numel() == 0 or vision_idx.numel() == 0:
                self.record("routing_balance_missing_modality", 1)
                continue
            mask_b = (
                attention_mask[b : b + 1]
                if torch.is_tensor(attention_mask) and int(attention_mask.shape[0]) == int(query_states.shape[0])
                else attention_mask
            )
            current, vision = self._last_query_modality_mass(
                query_states[b : b + 1], key_states[b : b + 1], mask_b, text_idx, vision_idx
            )
            if not math.isfinite(current):
                self.record("routing_balance_nonfinite_mass", 1)
                continue
            if int(layer_idx) == reference_layer:
                self._routing_balance_reference_by_batch[int(b)] = current
                self.record("routing_balance_reference_events", 1)
                self.record("routing_balance_reference_text_mass", current)
                if str(getattr(self.config, "analysis_path", "") or "").strip():
                    self._write_analysis_record({
                        "record_type": "routing_balance", "prompt_idx": int(self._analysis_prompt_idx),
                        "batch_idx": int(b), "source_layer": int(layer_idx), "target_layer": target_layer,
                        "is_reference": 1, "reference_text_mass": current,
                        "current_text_mass": current, "current_vision_mass": vision,
                        "applied_bias": 0.0, "target_key_count": 0,
                    })
                continue
            reference = self._routing_balance_reference_by_batch.get(int(b))
            if reference is None:
                self.record("routing_balance_reference_missing", 1)
                continue
            raw_bias = 0.0
            target_idx = text_idx[:0]
            if current > reference + 1e-7 and self._text_layer_active(target_layer):
                reference_c = min(1.0 - eps, max(eps, float(reference)))
                current_c = min(1.0 - eps, max(eps, float(current)))
                raw_bias = math.log(reference_c / (1.0 - reference_c)) - math.log(current_c / (1.0 - current_c))
                strength = max(0.0, float(getattr(self.config, "text_soft_gamma", 1.0) or 0.0))
                raw_bias = max(-4.0, min(0.0, strength * raw_bias))
                target_idx = self._routing_balance_target_indices(b, text_idx)
                if target_idx.numel() > 0 and raw_bias < 0.0:
                    bias = torch.zeros((int(query_states.shape[0]), self.prompt_len), device=query_states.device, dtype=torch.float32)
                    bias[b, target_idx] = float(raw_bias)
                    self._text_bias_by_layer[target_layer] = bias.detach()
                    self.record("routing_balance_bias_events", 1)
                    self.record("routing_balance_biased_keys", int(target_idx.numel()))
                    self.record("routing_balance_abs_bias", abs(float(raw_bias)))
            self.record("routing_balance_observation_events", 1)
            self.record("routing_balance_current_text_mass", current)
            self.record("routing_balance_current_vision_mass", vision)
            self.record("routing_balance_excess_text_mass", max(0.0, current - float(reference)))
            if str(getattr(self.config, "analysis_path", "") or "").strip():
                self._write_analysis_record({
                    "record_type": "routing_balance", "prompt_idx": int(self._analysis_prompt_idx),
                    "batch_idx": int(b), "source_layer": int(layer_idx), "target_layer": target_layer,
                    "is_reference": 0, "reference_text_mass": float(reference),
                    "current_text_mass": current, "current_vision_mass": vision,
                    "applied_bias": float(raw_bias), "target_key_count": int(target_idx.numel()),
                })

    def _finalize_routing_balance_prune_delta(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> None:
        mode = str(self.config.text_mask_mode or "").lower()
        if not mode.startswith("routing_balance_prune_delta") or self._vision_drop is None:
            return
        target_layer = int(layer_idx) + 1
        eps = 1e-6
        for b, text_idx_raw in enumerate(self._text_by_batch):
            reference = self._routing_balance_reference_by_batch.get(int(b))
            if reference is None:
                self.record("routing_balance_reference_missing", 1)
                continue
            text_idx = text_idx_raw.to(device=query_states.device)
            vision_idx = self._vision_by_batch[b].to(device=query_states.device)
            mask_b = (
                attention_mask[b : b + 1]
                if torch.is_tensor(attention_mask) and int(attention_mask.shape[0]) == int(query_states.shape[0])
                else attention_mask
            )
            current, vision = self._last_query_modality_mass(
                query_states[b : b + 1], key_states[b : b + 1], mask_b,
                text_idx, vision_idx, key_drop=self._vision_drop[b],
            )
            if not math.isfinite(current):
                self.record("routing_balance_nonfinite_mass", 1)
                continue
            raw_bias = 0.0
            target_idx = text_idx[:0]
            if current > float(reference) + 1e-7 and self._text_layer_active(target_layer):
                reference_c = min(1.0 - eps, max(eps, float(reference)))
                current_c = min(1.0 - eps, max(eps, float(current)))
                raw_bias = math.log(reference_c / (1.0 - reference_c)) - math.log(current_c / (1.0 - current_c))
                strength = max(0.0, float(getattr(self.config, "text_soft_gamma", 1.0) or 0.0))
                raw_bias = max(-4.0, min(0.0, strength * raw_bias))
                target_idx = self._routing_balance_target_indices(b, text_idx)
                if target_idx.numel() > 0 and raw_bias < 0.0:
                    bias = torch.zeros((int(query_states.shape[0]), self.prompt_len), device=query_states.device, dtype=torch.float32)
                    bias[b, target_idx] = float(raw_bias)
                    self._text_bias_by_layer[target_layer] = bias.detach()
                    self.record("routing_balance_bias_events", 1)
                    self.record("routing_balance_biased_keys", int(target_idx.numel()))
                    self.record("routing_balance_abs_bias", abs(float(raw_bias)))
            self.record("routing_balance_observation_events", 1)
            self.record("routing_balance_current_text_mass", current)
            self.record("routing_balance_current_vision_mass", vision)
            self.record("routing_balance_excess_text_mass", max(0.0, current - float(reference)))

    def observe(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        value_states: Optional[torch.Tensor] = None,
        output_projection=None,
    ) -> None:
        if not self.enabled or self._input_ids is None:
            return
        self.record("observe_calls", 1)
        target_layer = layer_idx + 1
        auto_vision = self._auto_vision_enabled()
        auto_vision_max = self._auto_vision_max_layer() if auto_vision else int(getattr(self.config, "vision_layer", 0) or 0)
        need_stage_vision = bool(
            self.config.vision_enabled
            and (
                int(getattr(self.config, "vision_stage_keep_tokens", 0) or 0) > 0
                or float(getattr(self.config, "vision_stage_keep_ratio", 1.0) or 1.0) < 1.0
                or str(getattr(self.config, "vision_stage_budget_mode", "ratio") or "ratio").lower()
                not in {"ratio", "fixed_ratio", "tokens", "fixed_tokens"}
            )
            and self._vision_stage_drop is None
            and self._vision_drop is None
            and not self._vision_stage_physical_compacted
            and int(layer_idx) == int(getattr(self.config, "vision_stage_layer", 0) or 0)
        )
        need_vision_probe = bool(
            self.config.vision_enabled
            and auto_vision
            and self._vision_drop is None
            and int(layer_idx) <= int(auto_vision_max)
        )
        need_deferred_rescore = bool(
            self.deferred_vision_rescore_enabled()
            and self._vision_mid_physical_compacted
            and not self._vision_final_physical_compacted
            and self._vision_deferred_final_drop is None
            and int(layer_idx) == int(getattr(self.config, "vision_deferred_drop_layer", -1))
        )
        need_vision = bool(
            self.config.vision_enabled
            and self._vision_drop is None
            and not auto_vision
            and (
                int(layer_idx) == int(self.config.vision_layer)
                or need_deferred_rescore
            )
        )
        routing_balance_mode = bool(
            self.config.text_enabled and self._uses_routing_balance_text_mode()
        )
        need_routing_balance = bool(
            routing_balance_mode
            and self._routing_balance_source_active(layer_idx)
            and self._text_layer_active(target_layer)
        )
        need_text = (
            self.config.text_enabled
            and not routing_balance_mode
            and layer_idx >= self.config.layer_k
            and self._text_layer_active(target_layer)
        )
        need_text_for_vision = bool(
            need_vision
            and getattr(self.config, "vision_text_aware", False)
            and self._text_layer_active(target_layer)
        )
        need_adaptive_warmup = (
            self.config.text_enabled
            and self._adaptive_text_needs_warmup()
            and layer_idx < self.config.layer_k
        )
        need_analysis = bool(getattr(self.config, "analysis_path", "")) and not routing_balance_mode
        need_text_scores = need_text or need_text_for_vision or need_adaptive_warmup or need_analysis
        if not need_stage_vision and not need_vision and not need_vision_probe and not need_text_scores and not need_routing_balance:
            if layer_idx < self.config.layer_k:
                self.record("observe_before_text_k", 1)
            return
        if query_states.ndim != 4 or key_states.ndim != 4:
            self.record("observe_bad_rank", 1)
            return
        if int(query_states.shape[2]) != self.prompt_len:
            self.record("observe_non_prompt_q_len", 1)
            return

        bsz, _, q_len, _ = query_states.shape
        key_len = int(key_states.shape[2])
        if key_len < self.prompt_len:
            self.record("observe_short_key_len", 1)
            return

        if need_routing_balance:
            self._observe_routing_balance(layer_idx, query_states, key_states, attention_mask)

        text_drop = None
        selectable_total = 0
        if need_text_scores:
            # Text masking is decided first from text->text attention only.  When
            # vision_text_aware is enabled, this same candidate mask is used to
            # filter noisy text queries before computing text->vision importance.
            text_drop = torch.zeros((bsz, self.prompt_len), device=query_states.device, dtype=torch.bool)
            text_bias = torch.zeros((bsz, self.prompt_len), device=query_states.device, dtype=torch.float32)
            for b in range(bsz):
                text_idx = self._text_by_batch[b]
                if text_idx.numel() <= 2:
                    continue
                attn_mask_b = attention_mask[b : b + 1] if attention_mask is not None and attention_mask.shape[0] == bsz else attention_mask
                value_b = (
                    value_states[b : b + 1]
                    if torch.is_tensor(value_states) and int(value_states.shape[0]) == bsz
                    else None
                )
                score_self = self._text_self_scores(
                    query_states[b : b + 1],
                    key_states[b : b + 1],
                    value_b,
                    attn_mask_b,
                    text_idx=text_idx,
                    batch_idx=b,
                )
                need_grounding = self._uses_veto_text_mode() or self._uses_grounding_text_score()
                grounding = self._text_grounding_scores(
                    query_states[b : b + 1],
                    key_states[b : b + 1],
                    attn_mask_b,
                    text_idx,
                    b,
                ) if need_grounding else None
                score = self._text_importance_score(
                    query_states[b : b + 1],
                    key_states[b : b + 1],
                    attn_mask_b,
                    text_idx=text_idx,
                    score_self=score_self,
                    batch_idx=b,
                    grounding=grounding,
                )
                selectable = torch.ones_like(score, dtype=torch.bool)
                selectable[0] = False
                selectable[-1] = False
                if selectable.sum() <= 0:
                    continue
                selectable_total += int(selectable.sum().item())
                if need_adaptive_warmup:
                    self._update_adaptive_floor(batch_idx=b, score=score, selectable=selectable)
                if need_analysis:
                    self._record_text_score_analysis(layer_idx, target_layer, b, score, selectable)
                if need_text or need_text_for_vision:
                    masked = self._select_text(score, selectable, target_layer, batch_idx=b)
                    veto_removed = 0
                    if self._uses_veto_text_mode() and masked.numel() > 0 and grounding is not None:
                        grounding_mean = grounding[selectable].mean()
                        before = int(masked.numel())
                        masked = masked[grounding[masked] < grounding_mean]
                        veto_removed = before - int(masked.numel())
                        self.record("text_vision_veto_candidates", before)
                        self.record("text_vision_veto_removed", veto_removed)
                    if need_text:
                        self._record_text_selection_analysis(
                            layer_idx, target_layer, b, text_idx, score,
                            selectable, masked, grounding, veto_removed,
                        )
                    if masked.numel() > 0:
                        if self._is_soft_text_mode() and need_text:
                            text_bias[b, text_idx[masked]] = -float(getattr(self.config, "text_soft_gamma", 2.0) or 0.0)
                        else:
                            text_drop[b, text_idx[masked]] = True

            if need_text_for_vision:
                dropped_for_vision = int(text_drop.sum().item()) if text_drop is not None else 0
                self.record("vision_text_aware_events", 1)
                self.record("vision_text_aware_text_dropped", dropped_for_vision)
                self._debug(
                    f"vision_text_aware source_layer={layer_idx} target_layer={target_layer} "
                    f"selectable={selectable_total} dropped_text_queries={dropped_for_vision} "
                    f"mode={self.config.text_mask_mode}"
                )

            if need_text:
                if self._is_soft_text_mode():
                    self._text_bias_by_layer[layer_idx + 1] = text_bias.detach()
                    dropped = int((text_bias.abs() > 0).sum().item())
                    self.record("soft_text_select_events", 1)
                    self.record("soft_text_biased", dropped)
                    self.record("soft_text_abs_bias", float(text_bias.abs().sum().item()))
                else:
                    self._text_drop_by_layer[layer_idx + 1] = text_drop.detach()
                    dropped = int(text_drop.sum().item())
                self.record("text_select_events", 1)
                self.record("text_selectable", selectable_total)
                self.record("text_dropped", dropped)
                ratio_dbg = (float(dropped) / float(selectable_total)) if selectable_total > 0 else 0.0
                self._debug(
                    f"text_select source_layer={layer_idx} target_layer={target_layer} "
                    f"selectable={selectable_total} dropped={dropped} "
                    f"mode={self.config.text_mask_mode} ratio={ratio_dbg:.4f}"
                )

        if need_stage_vision:
            old_keep_tokens = self.config.vision_keep_tokens
            old_keep_ratio = self.config.vision_keep_ratio
            old_score_mode = self.config.vision_score_mode
            old_score_lambda = self.config.vision_score_lambda
            old_keep = self._vision_keep
            try:
                stage_tokens = int(getattr(self.config, "vision_stage_keep_tokens", 0) or 0)
                stage_ratio = max(0.0, min(1.0, float(getattr(self.config, "vision_stage_keep_ratio", 0.5) or 0.5)))
                stage_budget_mode = str(
                    getattr(self.config, "vision_stage_budget_mode", "ratio") or "ratio"
                ).lower()
                max_vis = max((int(x.numel()) for x in self._vision_by_batch), default=0)
                final_tokens = self.final_vision_budget(max_vis, 0) if max_vis > 0 else 0
                if (
                    stage_budget_mode in {"ratio", "fixed_ratio", "tokens", "fixed_tokens"}
                    and stage_tokens <= 0
                    and final_tokens > 0
                    and self._vision_by_batch
                ):
                    ratio_tokens = max(1, int(math.ceil(float(max_vis) * stage_ratio))) if max_vis > 0 else 0
                    if ratio_tokens < final_tokens:
                        stage_tokens = final_tokens
                        stage_ratio = 1.0
                        self.record("vision_stage_keep_clamped_to_final", 1)
                self.config.vision_keep_tokens = stage_tokens
                self.config.vision_keep_ratio = stage_ratio
                self.config.vision_score_mode = str(getattr(self.config, "vision_stage_score_mode", "vispruner") or "vispruner")
                self.config.vision_score_lambda = float(getattr(self.config, "vision_stage_score_lambda", 0.0) or 0.0)
                self._vision_stage_drop = self._select_vision_drop(
                    query_states,
                    key_states,
                    attention_mask,
                    text_query_drop=None,
                    source_layer=layer_idx,
                    stage_budget_mode=stage_budget_mode,
                    final_keep_tokens=final_tokens,
                    value_states=value_states,
                    output_projection=output_projection,
                )
                self._vision_stage_keep = self._vision_keep
            finally:
                self.config.vision_keep_tokens = old_keep_tokens
                self.config.vision_keep_ratio = old_keep_ratio
                self.config.vision_score_mode = old_score_mode
                self.config.vision_score_lambda = old_score_lambda
                self._vision_keep = old_keep
            stage_dropped = int(self._vision_stage_drop.sum().item()) if self._vision_stage_drop is not None else 0
            self.record("vision_stage_events", 1)
            self.record("vision_stage_layer_sum", int(layer_idx))
            self.record("vision_stage_dropped", stage_dropped)
            self.record("vision_stage_keep_tokens", int(getattr(self.config, "vision_stage_keep_tokens", 0) or 0))
            self.record("vision_stage_keep_ratio_x1000", int(round(float(getattr(self.config, "vision_stage_keep_ratio", 0.5) or 0.5) * 1000.0)))
            if stage_budget_mode == "depth_coupled":
                self.record("vision_stage_budget_depth_coupled", 1)
            self._debug(
                f"vision_stage_select layer={layer_idx} dropped={stage_dropped} "
                f"stage_keep={int(getattr(self.config, 'vision_stage_keep_tokens', 0) or 0)} "
                f"stage_ratio={float(getattr(self.config, 'vision_stage_keep_ratio', 0.5) or 0.5):.3f} "
                f"budget_mode={stage_budget_mode}"
            )

        if need_vision_probe:
            auto_ready, auto_corr = self._auto_vision_ready(
                layer_idx=layer_idx,
                query_states=query_states,
                key_states=key_states,
                attention_mask=attention_mask,
            )
            self.record("vision_auto_probe_events", 1)
            self.record("vision_auto_probe_layer_sum", int(layer_idx))
            if auto_corr is not None:
                self.record("vision_auto_probe_corr_x1000", int(round(float(auto_corr) * 1000.0)))
            if auto_ready:
                need_vision = True
                self._vision_pruned_layer = int(layer_idx)
                self.record("vision_auto_selected_events", 1)
                self.record("vision_auto_selected_layer_sum", int(layer_idx))
                if int(layer_idx) >= int(auto_vision_max) and (auto_corr is None or float(auto_corr) <= 0.0):
                    self.record("vision_auto_fallback_max_layer", 1)
                self._debug(
                    f"vision_auto_select layer={layer_idx} max={auto_vision_max} "
                    f"corr={auto_corr if auto_corr is not None else 'NA'}"
                )

        if need_vision:
            if self._vision_pruned_layer is None:
                self._vision_pruned_layer = int(layer_idx)
            self._vision_drop = self._select_vision_drop(
                query_states,
                key_states,
                attention_mask,
                text_query_drop=text_drop if need_text_for_vision else None,
                source_layer=layer_idx,
                value_states=value_states,
                output_projection=output_projection,
            )
            if self._vision_stage_drop is not None:
                self._vision_drop = (self._vision_drop | self._vision_stage_drop.to(device=self._vision_drop.device)).detach()
            self.record("vision_drop_events", 1)
            if not bool(getattr(self, "_runtime_exact_fast_path", False)):
                dropped = int(self._vision_drop.sum().item())
                self.record("vision_dropped", dropped)
                self._debug(
                    f"vision_select layer={layer_idx} q_len={q_len} key_len={key_len} "
                    f"dropped={dropped} keep_tokens={self.config.vision_keep_tokens} "
                    f"keep_ratio={self.config.vision_keep_ratio} "
                    f"text_aware={bool(getattr(self.config, 'vision_text_aware', False))}"
                )
            if need_deferred_rescore:
                candidate_keep = self._vision_keep.detach().clone()
                candidate_drop = self._vision_drop.detach().clone()
                core_keep = self._vision_deferred_core_keep
                core_drop = self._vision_deferred_core_drop
                if core_keep is None or core_drop is None or core_keep.shape != candidate_keep.shape:
                    raise RuntimeError("deferred rescore lost the remapped L9 core")
                vision_idx = self._vision_by_batch[0]
                candidate_local = candidate_keep[0, vision_idx].bool()
                core_local = core_keep[0, vision_idx].bool()
                swaps = int(core_local.sum().item()) - int((candidate_local & core_local).sum().item())
                min_swaps = max(0, int(getattr(self.config, "vision_deferred_rescore_min_swaps", 0) or 0))
                accepted = bool(swaps >= min_swaps)
                self._vision_deferred_final_keep = candidate_keep if accepted else core_keep.detach().clone()
                self._vision_deferred_final_drop = candidate_drop if accepted else core_drop.detach().clone()
                self._vision_keep = None
                self._vision_drop = None
                self.record("vision_deferred_rescore_events", 1)
                self.record("vision_deferred_rescore_layer_sum", int(layer_idx))
                self.record("vision_deferred_rescore_candidate_swaps", swaps)
                self.record("vision_deferred_rescore_gate_accepted", int(accepted))
                self.record("vision_deferred_rescore_gate_rejected", int(not accepted))
                self.record("vision_deferred_rescore_min_swaps", min_swaps)
                self.record(
                    "vision_deferred_rescore_final_tokens",
                    int(self._vision_deferred_final_keep.sum().item()),
                )
            elif self.deferred_vision_drop_enabled():
                self._prepare_deferred_vision_drop()
            if routing_balance_mode:
                self._finalize_routing_balance_prune_delta(
                    layer_idx, query_states, key_states, attention_mask
                )

    def _prepare_deferred_vision_drop(self) -> None:
        """Convert a one-shot final-K ranking into nested M1 then K drops.

        The score is computed exactly once at ``vision_layer``. The K core is
        the ordinary selector output; the highest-scoring non-core tokens form
        a temporary support tail. No additional decoder attention or learned
        component is introduced at the later drop boundary.
        """
        if not self.deferred_vision_drop_enabled():
            return
        if bool(getattr(self.config, "vision_merge", False)):
            raise RuntimeError("deferred vision dropping requires vision_merge=false")
        if len(self._vision_by_batch) != 1 or len(self._vision_last_scores) != 1:
            raise RuntimeError("deferred vision dropping currently requires batch_size=1")
        if self._vision_drop is None or self._vision_keep is None:
            raise RuntimeError("deferred vision dropping requires a completed final-K selection")

        vision_idx = self._vision_by_batch[0]
        score_idx, score = self._vision_last_scores[0]
        if not torch.equal(score_idx.to(vision_idx.device).long(), vision_idx.long()):
            raise RuntimeError("deferred vision score/token alignment mismatch")
        current_n = int(vision_idx.numel())
        initial_n = int(self._initial_vision_counts[0]) if self._initial_vision_counts else current_n
        core_local = self._vision_keep[0, vision_idx].to(device=vision_idx.device).bool()
        final_n = int(core_local.sum().item())
        if final_n <= 0:
            raise RuntimeError("deferred vision dropping produced an empty final core")

        reserve_fraction = max(
            0.0,
            min(1.0, float(getattr(self.config, "vision_deferred_reserve_fraction", 0.0))),
        )
        if self.deferred_vision_rescore_enabled():
            reserve = self._vision_alpha_reserve_keep
            if reserve is None or reserve.shape != self._vision_keep.shape:
                raise RuntimeError("deferred rescore requires an aligned alpha-union reserve")
            mid_local = reserve[0, vision_idx].to(device=vision_idx.device).bool() | core_local
            target_n = int(mid_local.sum().item())
            support_n = target_n - final_n
            # Preserve the L9 core in remappable prompt coordinates. The
            # later gate can therefore reject a weak revision without a second
            # forward, while the small reserve has still participated up to L17.
            self._vision_deferred_core_keep = self._vision_keep.detach().clone()
            self._vision_deferred_core_drop = self._vision_drop.detach().clone()
            self._vision_deferred_final_keep = None
            self._vision_deferred_final_drop = None
            self.record("vision_deferred_alpha_union_events", 1)
        else:
            target_n = final_n + int(
                math.ceil(reserve_fraction * float(max(0, initial_n - final_n)))
            )
            target_n = max(final_n, min(target_n, current_n))
            support_n = target_n - final_n

            mid_local = core_local.clone()
            if support_n > 0:
                noncore_local = torch.nonzero(~core_local, as_tuple=False).flatten().long()
                noncore_score = score.to(device=vision_idx.device).float()[noncore_local]
                support_order = torch.argsort(noncore_score, descending=True, stable=True)[:support_n]
                mid_local[noncore_local[support_order]] = True

            self._vision_deferred_final_keep = self._vision_keep.detach().clone()
            self._vision_deferred_final_drop = self._vision_drop.detach().clone()
        self._vision_mid_keep = torch.zeros_like(self._vision_keep)
        self._vision_mid_drop = torch.zeros_like(self._vision_drop)
        self._vision_mid_keep[0, vision_idx[mid_local]] = True
        self._vision_mid_drop[0, vision_idx[~mid_local]] = True

        self.record("vision_deferred_plan_events", 1)
        self.record("vision_deferred_initial_tokens", initial_n)
        self.record("vision_deferred_layer9_candidates", current_n)
        self.record("vision_deferred_mid_tokens", target_n)
        self.record("vision_deferred_final_tokens", final_n)
        self.record("vision_deferred_support_tokens", support_n)
        self.record(
            "vision_deferred_reserve_fraction_x1000000",
            int(round(reserve_fraction * 1_000_000.0)),
        )
        self._debug(
            f"vision_deferred_plan initial={initial_n} layer9_candidates={current_n} "
            f"mid={target_n} core={final_n} support={support_n} "
            f"drop_layer={int(getattr(self.config, 'vision_deferred_drop_layer', -1))}"
        )
        # Planned masks live separately and therefore cannot affect attention
        # before their physical boundaries.
        self._vision_drop = None
        self._vision_keep = None
        self._vision_alpha_reserve_keep = None

    def _fit_key_len(self, mask: torch.Tensor, batch_size: int, key_len: int, device: torch.device) -> torch.Tensor:
        mask = mask.to(device=device)
        if mask.shape[0] != batch_size:
            mask = mask[:batch_size]
        if mask.shape[1] == key_len:
            return mask
        if mask.shape[1] > key_len:
            return mask[:, :key_len]
        pad = torch.zeros((mask.shape[0], key_len - mask.shape[1]), device=device, dtype=torch.bool)
        return torch.cat([mask, pad], dim=1)

    def _layer_bound(self, value: float, default: int) -> int:
        try:
            v = float(value)
        except Exception:
            return default
        n = max(1, int(self.config.num_layers or 1))
        if 0.0 <= v <= 1.0:
            return int(round(v * n))
        return int(round(v))

    def _piecewise_segments(self) -> Tuple[Tuple[int, int, float], ...]:
        spec = str(getattr(self.config, "text_piecewise", "") or "").strip()
        if not spec:
            return ()
        n = max(1, int(self.config.num_layers or 1))
        out = []
        for raw in spec.replace("|", ";").split(";"):
            item = raw.strip()
            if not item:
                continue
            parts = [x.strip() for x in item.split(":")]
            if len(parts) != 3:
                continue
            try:
                start = max(0, min(n, self._layer_bound(float(parts[0]), 0)))
                end = max(start + 1, min(n, self._layer_bound(float(parts[1]), n)))
                ratio = max(0.0, min(1.0, float(parts[2])))
            except Exception:
                continue
            out.append((start, end, ratio))
        return tuple(out)

    def _piecewise_ratio(self, target_layer: int) -> Optional[float]:
        for start, end, ratio in self._piecewise_segments():
            if start <= int(target_layer) < end:
                return ratio
        return None

    def _text_layer_active(self, target_layer: int) -> bool:
        piece = self._piecewise_ratio(target_layer)
        if piece is not None:
            return piece > 0.0
        n = max(1, int(self.config.num_layers or 1))
        start = max(0, min(n, self._layer_bound(self.config.text_layer_start, 0)))
        end = max(start + 1, min(n, self._layer_bound(self.config.text_layer_end, n)))
        return start <= int(target_layer) < end

    def _scheduled_ratio(self, target_layer: int) -> float:
        piece = self._piecewise_ratio(target_layer)
        if piece is not None:
            return piece
        start = self._layer_bound(self.config.text_layer_start, 0)
        end = self._layer_bound(self.config.text_layer_end, int(self.config.num_layers or 1))
        base = max(0.0, min(1.0, float(self.config.text_mask_ratio or 0.0)))
        ratio_end = float(self.config.text_mask_ratio_end)
        if ratio_end < 0:
            return base
        ratio_end = max(0.0, min(1.0, ratio_end))
        denom = max(1, end - start - 1)
        t = max(0.0, min(1.0, (float(target_layer) - float(start)) / float(denom)))
        return base + (ratio_end - base) * t

    def _record_text_score_analysis(
        self,
        source_layer: int,
        target_layer: int,
        batch_idx: int,
        score: torch.Tensor,
        selectable: torch.Tensor,
    ) -> None:
        path = str(getattr(self.config, "analysis_path", "") or "").strip()
        if not path:
            return
        vals = score[selectable].detach().float().cpu()
        n = int(vals.numel())
        if n <= 0:
            return
        vals_sorted = torch.sort(vals).values
        qs = torch.tensor([0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
        qv = torch.quantile(vals_sorted, qs) if n > 1 else vals_sorted.repeat(qs.numel())
        thresholds = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00]
        hist_edges = [0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 4.00, float("inf")]
        hist = []
        for lo, hi in zip(hist_edges[:-1], hist_edges[1:]):
            if hi == float("inf"):
                hist.append(int((vals >= lo).sum().item()))
            else:
                hist.append(int(((vals >= lo) & (vals < hi)).sum().item()))
        total = vals.sum().item()
        vals_nonneg = vals.clamp_min(0.0)
        mass = vals_nonneg.sum().item()
        if mass > 0 and n > 1:
            prob = vals_nonneg / mass
            entropy = float((-(prob * (prob + 1e-12).log()).sum() / torch.tensor(float(n)).log()).item())
        else:
            entropy = 0.0
        if n > 1 and vals_sorted.abs().sum().item() > 0:
            idx = torch.arange(1, n + 1, dtype=vals_sorted.dtype)
            gini = float(((2 * idx - n - 1) * vals_sorted).sum().item() / (n * vals_sorted.sum().item() + 1e-12))
        else:
            gini = 0.0
        rec = {
            "record_type": "text",
            "prompt_idx": int(self._analysis_prompt_idx),
            "batch_idx": int(batch_idx),
            "source_layer": int(source_layer),
            "target_layer": int(target_layer),
            "num_layers": int(self.config.num_layers or 0),
            "n_text_selectable": n,
            "scheduled_ratio": float(self._scheduled_ratio(target_layer)),
            "text_correction": str(self.config.text_correction),
            "score_sum": float(total),
            "score_sumsq": float((vals * vals).sum().item()),
            "score_mean": float(vals.mean().item()),
            "score_std": float(vals.std(unbiased=False).item()) if n > 1 else 0.0,
            "score_min": float(vals.min().item()),
            "score_max": float(vals.max().item()),
            "score_entropy_norm": entropy,
            "score_concentration": float(1.0 - entropy),
            "score_gini": gini,
            "concentration": float(1.0 - entropy),
            "gini": gini,
            "hist_edges": hist_edges,
            "hist_counts": hist,
        }
        for q, v in zip(qs.tolist(), qv.tolist()):
            rec[f"q{int(round(q * 100)):02d}"] = float(v)
        for t in thresholds:
            rec[f"frac_lt_{str(t).replace('.', 'p')}"] = float((vals < t).float().mean().item())
        self._write_analysis_record(rec)

    def _record_text_selection_analysis(
        self,
        source_layer: int,
        target_layer: int,
        batch_idx: int,
        text_idx: torch.Tensor,
        score: torch.Tensor,
        selectable: torch.Tensor,
        masked: torch.Tensor,
        grounding: Optional[torch.Tensor],
        veto_removed: int,
    ) -> None:
        selected = torch.zeros_like(selectable, dtype=torch.bool)
        if masked.numel() > 0:
            selected[masked] = True
        selectable_n = int(selectable.sum().item())
        masked_n = int(selected.sum().item())
        actual_ratio = float(masked_n) / float(selectable_n) if selectable_n else 0.0
        prev = self._previous_text_mask_by_batch.get(int(batch_idx))
        jaccard = None
        recovery = None
        if prev is not None and prev.numel() == selected.numel():
            union = int((prev | selected).sum().item())
            if union > 0:
                jaccard = float((prev & selected).sum().item()) / float(union)
            prev_n = int(prev.sum().item())
            if prev_n > 0:
                recovery = float((prev & ~selected).sum().item()) / float(prev_n)
        self._previous_text_mask_by_batch[int(batch_idx)] = selected.detach()

        abs_masked = text_idx[selected].detach().long()
        active_edges = 0
        for pos in abs_masked.tolist():
            active_edges += max(0, self.prompt_len - int(pos))
        scoring_visible_pairs = sum(
            min(self.prompt_len, int(pos) + 1) for pos in text_idx.detach().long().tolist()
        )
        concentration = self._score_concentration(score, selectable)
        self.record("text_actual_mask_ratio_sum", actual_ratio)
        self.record("text_actual_mask_ratio_records", 1)
        self.record("text_masked_attention_edges", active_edges)
        self.record("text_scoring_visible_pairs", scoring_visible_pairs)
        self.record(f"text_layer_{target_layer}_selectable", selectable_n)
        self.record(f"text_layer_{target_layer}_masked", masked_n)
        self.record(f"text_layer_{target_layer}_actual_ratio_sum", actual_ratio)
        self.record(f"text_layer_{target_layer}_concentration_sum", concentration)
        if jaccard is not None:
            self.record("text_mask_jaccard_sum", jaccard)
            self.record("text_mask_jaccard_records", 1)
        if recovery is not None:
            self.record("text_mask_recovery_sum", recovery)
            self.record("text_mask_recovery_records", 1)
        if not str(getattr(self.config, "analysis_path", "") or "").strip():
            return
        rec = {
            "record_type": "text_selection",
            "prompt_idx": int(self._analysis_prompt_idx),
            "batch_idx": int(batch_idx),
            "source_layer": int(source_layer),
            "target_layer": int(target_layer),
            "num_layers": int(self.config.num_layers or 0),
            "text_mask_mode": str(self.config.text_mask_mode),
            "text_score_scope": (
                "linearized_causal_utilization"
                if str(self.config.text_correction or "").lower()
                in {
                    "linearized_causal_utilization",
                    "linearized_utilization",
                    "first_order_utilization",
                }
                else "linear_causal_query"
                if str(self.config.text_correction or "").lower()
                in {"linear_causal_query", "linear_received", "mean_query_alignment"}
                else "random_control"
                if str(self.config.text_correction or "").lower()
                in {"random_control", "deterministic_random"}
                else "full_causal_keys"
            ),
            "n_text_selectable": selectable_n,
            "n_text_masked": masked_n,
            "actual_mask_ratio": actual_ratio,
            "score_concentration": concentration,
            "masked_attention_edges": int(active_edges),
            "scoring_visible_pairs": int(scoring_visible_pairs),
            "vision_veto_removed": int(veto_removed),
        }
        if jaccard is not None:
            rec["mask_jaccard_vs_previous_layer"] = jaccard
        if recovery is not None:
            rec["recovery_fraction_vs_previous_layer"] = recovery
        if as_bool(os.environ.get("VISPRUNER_ANALYSIS_TOKEN_DETAILS", "0")):
            positions = text_idx.detach().long().cpu()
            ids = (
                self._input_ids[int(batch_idx), text_idx.to(self._input_ids.device)].detach().long().cpu()
                if self._input_ids is not None else torch.full_like(positions, -1)
            )
            roles = (
                self._role_ids.index_select(0, text_idx.to(self._role_ids.device)).detach().long().cpu()
                if self._role_ids is not None and int(self._role_ids.numel()) == self.prompt_len
                else torch.zeros_like(positions)
            )
            pos_list = [int(x) for x in positions.tolist()]
            rec.update({
                "candidate_positions": pos_list,
                "candidate_token_ids": [int(x) for x in ids.tolist()],
                "candidate_role_ids": [int(x) for x in roles.tolist()],
                "candidate_raw_tokens": [
                    self._raw_tokens[p] if 0 <= p < len(self._raw_tokens) else str(int(ids[i]))
                    for i, p in enumerate(pos_list)
                ],
                "candidate_decoded_tokens": [
                    self._decoded_tokens[p] if 0 <= p < len(self._decoded_tokens) else str(int(ids[i]))
                    for i, p in enumerate(pos_list)
                ],
                "candidate_scores": [float(x) for x in score.detach().float().cpu().tolist()],
                "candidate_selectable": [bool(x) for x in selectable.detach().bool().cpu().tolist()],
                "candidate_selected": [bool(x) for x in selected.detach().bool().cpu().tolist()],
            })
        if grounding is not None and grounding.numel() == score.numel():
            vals = grounding[selectable].detach().float()
            if vals.numel() > 0:
                rec.update({
                    "vision_grounding_mean": float(vals.mean().item()),
                    "vision_grounding_std": float(vals.std(unbiased=False).item()),
                    "vision_grounding_min": float(vals.min().item()),
                    "vision_grounding_max": float(vals.max().item()),
                })
        self._write_analysis_record(rec)

    def _write_analysis_record(self, rec: dict) -> None:
        path = str(getattr(self.config, "analysis_path", "") or "").strip()
        if not path:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.record("analysis_records", 1)

    @staticmethod
    def _hashed_projection_features(prefix: str, value: torch.Tensor, bins: int = 16) -> dict[str, float]:
        """Deterministic low-dimensional projection for opt-in predictor analysis."""
        vals = value.detach().float().flatten()
        if vals.numel() == 0 or bins <= 0:
            return {}
        vals = vals / vals.norm().clamp_min(1e-8)
        indices = torch.arange(vals.numel(), device=vals.device, dtype=torch.long)
        bucket = torch.remainder(indices, int(bins))
        sign_bits = torch.remainder(indices * 1103515245 + 12345, 2)
        signs = torch.where(sign_bits == 0, torch.ones_like(vals), -torch.ones_like(vals))
        projected = torch.zeros(int(bins), device=vals.device, dtype=torch.float32)
        projected.scatter_add_(0, bucket, vals * signs)
        counts = torch.bincount(bucket, minlength=int(bins)).float().clamp_min(1.0)
        projected = projected / counts.sqrt()
        return {f"{prefix}_{i:02d}": float(x) for i, x in enumerate(projected.cpu().tolist())}

    def _tensor_score_stats(self, prefix: str, score: Optional[torch.Tensor]) -> dict:
        if score is None or not torch.is_tensor(score) or score.numel() == 0:
            return {f"{prefix}_available": 0}
        vals = score.detach().float().flatten().cpu()
        n = int(vals.numel())
        out = {f"{prefix}_available": 1, f"{prefix}_n": n}
        out[f"{prefix}_sum"] = float(vals.sum().item())
        out[f"{prefix}_mean"] = float(vals.mean().item())
        out[f"{prefix}_std"] = float(vals.std(unbiased=False).item()) if n > 1 else 0.0
        out[f"{prefix}_min"] = float(vals.min().item())
        out[f"{prefix}_max"] = float(vals.max().item())
        vals_nonneg = vals.clamp_min(0.0)
        mass = vals_nonneg.sum().item()
        if mass > 0 and n > 1:
            prob = vals_nonneg / mass
            entropy = float((-(prob * (prob + 1e-12).log()).sum() / torch.tensor(float(n)).log()).item())
        else:
            entropy = 0.0
        out[f"{prefix}_entropy_norm"] = entropy
        out[f"{prefix}_concentration"] = float(1.0 - entropy)
        vals_sorted = torch.sort(vals).values
        if n > 1 and abs(float(vals_sorted.sum().item())) > 0:
            idx = torch.arange(1, n + 1, dtype=vals_sorted.dtype)
            out[f"{prefix}_gini"] = float(((2 * idx - n - 1) * vals_sorted).sum().item() / (n * vals_sorted.sum().item() + 1e-12))
        else:
            out[f"{prefix}_gini"] = 0.0
        qs = torch.tensor([0.10, 0.20, 0.50, 0.80, 0.90])
        qv = torch.quantile(vals_sorted, qs) if n > 1 else vals_sorted.repeat(qs.numel())
        for q, v in zip(qs.tolist(), qv.tolist()):
            out[f"{prefix}_q{int(round(q * 100)):02d}"] = float(v)
        return out

    def _pearson_corr(self, a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> Optional[float]:
        if a is None or b is None or not torch.is_tensor(a) or not torch.is_tensor(b):
            return None
        x = a.detach().float().flatten()
        y = b.detach().float().flatten().to(device=x.device)
        if x.numel() != y.numel() or x.numel() <= 1:
            return None
        x = x - x.mean()
        y = y - y.mean()
        den = x.norm() * y.norm()
        if float(den.item()) <= 1e-12:
            return None
        return float((x * y).sum().div(den).item())

    def _rank_tensor(self, x: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(x.detach().float().flatten())
        ranks = torch.empty_like(order, dtype=torch.float32)
        ranks[order] = torch.arange(order.numel(), device=order.device, dtype=torch.float32)
        return ranks

    def _spearman_corr(self, a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> Optional[float]:
        if a is None or b is None or not torch.is_tensor(a) or not torch.is_tensor(b):
            return None
        x = a.detach().float().flatten()
        y = b.detach().float().flatten().to(device=x.device)
        if x.numel() != y.numel() or x.numel() <= 1:
            return None
        return self._pearson_corr(self._rank_tensor(x), self._rank_tensor(y))

    def _topk_mask(self, score: Optional[torch.Tensor], k: int) -> Optional[torch.Tensor]:
        if score is None or not torch.is_tensor(score) or score.numel() == 0 or k <= 0:
            return None
        vals = score.detach().float().flatten()
        kk = min(int(k), int(vals.numel()))
        mask = torch.zeros(vals.numel(), device=vals.device, dtype=torch.bool)
        mask[torch.topk(vals, k=kk, largest=True).indices] = True
        return mask

    def _mask_overlap(self, a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> Optional[float]:
        if a is None or b is None or not torch.is_tensor(a) or not torch.is_tensor(b):
            return None
        aa = a.detach().bool().flatten()
        bb = b.detach().bool().flatten().to(device=aa.device)
        if aa.numel() != bb.numel() or aa.numel() == 0:
            return None
        denom = int((aa | bb).sum().item())
        if denom <= 0:
            return None
        return float((aa & bb).sum().item()) / float(denom)


    def _anchor_quota_token_features(
        self,
        encoder_score: torch.Tensor,
        joint_score: torch.Tensor,
        pmi_score: torch.Tensor,
        text_score: torch.Tensor,
        text_queries: torch.Tensor,
        vision_keys: torch.Tensor,
        keep_n: int,
        original_map: Optional[torch.Tensor] = None,
        anchor_mask: Optional[torch.Tensor] = None,
        text_positions: Optional[torch.Tensor] = None,
    ) -> dict:
        """Capture identity-free L9 token features for exact anchor-quota sets.

        This is opt-in analysis support. It performs no additional model forward
        and does not alter the selected set. Token indices are retained only for
        exact signature audits and must be excluded from predictor inputs.
        """
        count = int(encoder_score.numel())
        keep_n = max(1, min(int(keep_n), count))
        for name, value in (("joint", joint_score), ("pmi", pmi_score), ("text", text_score)):
            if int(value.numel()) != count:
                raise RuntimeError(f"anchor-quota {name} score alignment mismatch")
        if vision_keys.ndim != 3 or int(vision_keys.shape[1]) != count:
            raise RuntimeError(f"anchor-quota vision-key alignment mismatch: {tuple(vision_keys.shape)} vs {count}")
        if text_queries.ndim != 3 or int(text_queries.shape[-1]) != int(vision_keys.shape[-1]):
            raise RuntimeError("anchor-quota text-query/key dimension mismatch")
        query_heads = int(text_queries.shape[0])
        key_heads = int(vision_keys.shape[0])
        if query_heads != key_heads:
            if key_heads <= 0 or query_heads % key_heads != 0:
                raise RuntimeError(
                    f"anchor-quota incompatible query/KV heads: {query_heads} vs {key_heads}"
                )
            vision_keys = vision_keys.repeat_interleave(query_heads // key_heads, dim=0)

        enc_raw = encoder_score.detach().float()
        enc = self._normalize_score(enc_raw).detach().float()
        joint = joint_score.detach().float()
        pmi = self._normalize_score(pmi_score).detach().float()
        text = self._normalize_score(text_score).detach().float()
        original = (
            original_map.to(device=enc.device, dtype=torch.long)
            if original_map is not None and int(original_map.numel()) == count
            else torch.arange(count, device=enc.device, dtype=torch.long)
        )
        early_anchor = (
            anchor_mask.to(device=enc.device).bool()
            if anchor_mask is not None and int(anchor_mask.numel()) == count
            else torch.zeros(count, device=enc.device, dtype=torch.bool)
        )

        dim = int(vision_keys.shape[-1])
        last_q = text_queries[:, -1, :]
        mean_q = text_queries.mean(dim=1)
        last_qk = (vision_keys * last_q[:, None, :]).sum(dim=-1) / math.sqrt(max(1, dim))
        mean_qk = (vision_keys * mean_q[:, None, :]).sum(dim=-1) / math.sqrt(max(1, dim))
        # Opt-in predictor diagnostics: retain head disagreement rather than only
        # the head mean. No attention matrix or additional forward is computed.
        head_keep = min(keep_n, count)
        last_vote = torch.zeros(count, device=last_qk.device, dtype=torch.float32)
        mean_vote = torch.zeros(count, device=mean_qk.device, dtype=torch.float32)
        for head in range(query_heads):
            last_vote[torch.topk(last_qk[head].float(), k=head_keep, largest=True).indices] += 1.0
            mean_vote[torch.topk(mean_qk[head].float(), k=head_keep, largest=True).indices] += 1.0
        last_vote = last_vote / float(max(1, query_heads))
        mean_vote = mean_vote / float(max(1, query_heads))
        sequence_queries = text_queries
        sequence_positions = torch.arange(
            int(text_queries.shape[1]), device=text_queries.device, dtype=torch.long
        )
        if (
            text_positions is not None
            and self._role_ids is not None
            and int(text_positions.numel()) == int(text_queries.shape[1])
        ):
            absolute = text_positions.to(device=text_queries.device, dtype=torch.long)
            valid = (absolute >= 0) & (absolute < int(self._role_ids.numel()))
            roles = torch.zeros_like(absolute)
            roles[valid] = self._role_ids.to(device=absolute.device).index_select(0, absolute[valid]).long()
            user = valid & roles.eq(2)
            if bool(user.any().item()):
                sequence_queries = text_queries[:, user, :]
                sequence_positions = absolute[user]

        key_rows = vision_keys.permute(1, 0, 2).reshape(count, -1).detach().float()
        key_rows = key_rows / key_rows.norm(dim=1, keepdim=True).clamp_min(1e-8)
        bins = int(os.environ.get("VISPRUNER_ANCHOR_QUOTA_KEY_BINS", "16") or 16)
        if bins < 16 or bins > 512:
            raise RuntimeError(f"anchor-quota key projection bins must be in [16,512], got {bins}")
        coordinate = torch.arange(key_rows.shape[1], device=key_rows.device, dtype=torch.long)
        bucket = torch.remainder(coordinate, bins)
        sign_bits = torch.remainder(coordinate * 1103515245 + 12345, 2)
        signs = torch.where(sign_bits == 0, torch.ones_like(coordinate, dtype=torch.float32), -torch.ones_like(coordinate, dtype=torch.float32))
        key_projection = torch.zeros((count, bins), device=key_rows.device, dtype=torch.float32)
        key_projection.scatter_add_(1, bucket[None, :].expand(count, -1), key_rows * signs[None, :])
        counts = torch.bincount(bucket, minlength=bins).float().clamp_min(1.0)
        key_projection = key_projection / counts.sqrt()[None, :]
        context_rows = torch.stack((last_q.reshape(-1), mean_q.reshape(-1)), dim=0).detach().float()
        context_rows = context_rows / context_rows.norm(dim=1, keepdim=True).clamp_min(1e-8)
        context_projection = torch.zeros((2, bins), device=key_rows.device, dtype=torch.float32)
        context_projection.scatter_add_(
            1, bucket[None, :].expand(2, -1), context_rows * signs[None, :]
        )
        context_projection = context_projection / counts.sqrt()[None, :]

        text_rows = sequence_queries.permute(1, 0, 2).reshape(int(sequence_queries.shape[1]), -1).detach().float()
        text_rows = text_rows / text_rows.norm(dim=1, keepdim=True).clamp_min(1e-8)
        text_projection = torch.zeros((text_rows.shape[0], bins), device=key_rows.device, dtype=torch.float32)
        text_projection.scatter_add_(1, bucket[None, :].expand(text_rows.shape[0], -1), text_rows * signs[None, :])
        text_projection = text_projection / counts.sqrt()[None, :]
        cross_logits = torch.einsum("htd,hvd->htv", sequence_queries.float(), vision_keys.float()) / math.sqrt(max(1, dim))
        cross_prob = torch.softmax(cross_logits, dim=-1)
        cross_entropy = -(cross_prob.clamp_min(1e-12) * cross_prob.clamp_min(1e-12).log()).sum(dim=-1)
        cross_entropy = cross_entropy / math.log(max(2, count))
        cross_top_n = min(8, count)
        text_base = torch.stack((
            cross_logits.mean(dim=(0, 2)),
            cross_logits.amax(dim=(0, 2)),
            cross_logits.std(dim=(0, 2), unbiased=False),
            cross_prob.max(dim=-1).values.mean(dim=0),
            cross_entropy.mean(dim=0),
            torch.topk(cross_prob, k=cross_top_n, dim=-1).values.sum(dim=-1).mean(dim=0),
            sequence_positions.float() / float(max(1, self.prompt_len - 1)),
        ), dim=1)
        text_token_matrix = torch.cat((text_base, text_projection.to(text_base.device)), dim=1)
        if not bool(torch.isfinite(text_token_matrix).all().item()):
            raise RuntimeError("non-finite anchor-quota text-token features")
        text_token_names = [
            "vision_qk_mean", "vision_qk_max", "vision_qk_std",
            "vision_attention_max", "vision_attention_entropy_norm",
            "vision_attention_top8_mass", "prompt_position_fraction",
        ] + [f"text_query_hash_{idx:02d}" for idx in range(bins)]
        position_scale = float(max(1, int(original.max().item())))

        base_columns = torch.stack((
            enc, pmi, text, joint,
            last_qk.mean(dim=0), last_qk.max(dim=0).values,
            last_qk.std(dim=0, unbiased=False), last_vote,
            mean_qk.mean(dim=0), mean_qk.std(dim=0, unbiased=False), mean_vote,
            early_anchor.float(), original.float() / position_scale,
        ), dim=1)
        token_matrix = torch.cat((base_columns, key_projection.to(base_columns.device)), dim=1)
        if not bool(torch.isfinite(token_matrix).all().item()):
            raise RuntimeError("non-finite anchor-quota token features")
        feature_names = [
            "encoder_score", "pmi_score", "text_score", "joint_score",
            "last_query_qk_mean", "last_query_qk_max", "last_query_qk_std",
            "last_query_head_topk_vote", "mean_query_qk_mean", "mean_query_qk_std",
            "mean_query_head_topk_vote", "early_anchor", "original_position_fraction",
        ] + [f"vision_key_hash_{idx:02d}" for idx in range(bins)]

        encoder_top, _ = self._select_encoder_anchor_quota(enc_raw, joint, keep_n, 1.0)
        joint_top, _ = self._select_encoder_anchor_quota(enc_raw, joint, keep_n, 0.0)
        encoder_mask = torch.zeros(count, device=enc.device, dtype=torch.bool); encoder_mask[encoder_top] = True
        joint_mask = torch.zeros(count, device=enc.device, dtype=torch.bool); joint_mask[joint_top] = True
        quota_sets = []
        previous = None
        for revision in (0.0, 0.25, 0.5, 0.75, 1.0):
            anchor_alpha = 1.0 - revision
            selected, locked = self._select_encoder_anchor_quota(enc_raw, joint, keep_n, anchor_alpha)
            mask = torch.zeros(count, device=enc.device, dtype=torch.bool); mask[selected] = True
            union_previous = int((mask | previous).sum().item()) if previous is not None else keep_n
            quota_sets.append({
                "revision_fraction": revision,
                "anchor_alpha": anchor_alpha,
                "anchor_tokens_locked": int(locked),
                "encoder_topk_overlap": int((mask & encoder_mask).sum().item()),
                "joint_topk_overlap": int((mask & joint_mask).sum().item()),
                "jaccard_previous": (
                    float((mask & previous).sum().item()) / float(max(1, union_previous))
                    if previous is not None else 1.0
                ),
                "keep_local_indices": [int(x) for x in selected.detach().cpu().tolist()],
                "keep_original_local_indices": [int(x) for x in original.index_select(0, selected).detach().cpu().tolist()],
            })
            previous = mask
        context_names = (
            [f"text_last_query_hash_{idx:03d}" for idx in range(bins)]
            + [f"text_mean_query_hash_{idx:03d}" for idx in range(bins)]
        )
        return {
            "token_feature_names": feature_names,
            "token_features": [[float(x) for x in row] for row in token_matrix.detach().cpu().tolist()],
            "context_feature_names": context_names,
            "context_features": [float(x) for x in context_projection.reshape(-1).detach().cpu().tolist()],
            "text_token_feature_names": text_token_names,
            "text_token_features": [[float(x) for x in row] for row in text_token_matrix.detach().cpu().tolist()],
            "text_token_scope": "user_role_if_available_else_all_nonvision",
            "key_projection_bins": bins,
            "quota_sets": quota_sets,
        }

    def _continuous_alpha_set_features(
        self,
        encoder_score: torch.Tensor,
        pmi_score: torch.Tensor,
        text_score: torch.Tensor,
        vision_keys: torch.Tensor,
        keep_n: int,
        original_map: Optional[torch.Tensor] = None,
        anchor_mask: Optional[torch.Tensor] = None,
    ) -> list[dict]:
        """Describe coherent top-K sets across a dense alpha response.

        This is opt-in analysis support for a set-utility distillation screen.
        It performs no model forward and does not affect the selected tokens.
        """
        count = int(encoder_score.numel())
        keep_n = max(1, min(int(keep_n), count))
        if int(pmi_score.numel()) != count or int(text_score.numel()) != count:
            raise RuntimeError("alpha-set feature score alignment mismatch")
        if vision_keys.ndim != 3 or int(vision_keys.shape[1]) != count:
            raise RuntimeError(f"alpha-set vision-key alignment mismatch: {tuple(vision_keys.shape)} vs {count}")
        enc = self._normalize_score(encoder_score).detach().float()
        pmi = self._normalize_score(pmi_score).detach().float()
        text = self._normalize_score(text_score).detach().float()
        ref = {
            "encoder": self._topk_mask(enc, keep_n),
            "pmi": self._topk_mask(pmi, keep_n),
            "text": self._topk_mask(text, keep_n),
        }
        if original_map is None or int(original_map.numel()) != count:
            original = torch.arange(count, device=enc.device, dtype=torch.long)
        else:
            original = original_map.to(device=enc.device, dtype=torch.long)
        anchors = None
        if anchor_mask is not None and int(anchor_mask.numel()) == count:
            anchors = anchor_mask.to(device=enc.device).bool()
        rows = []
        previous = None
        for step in range(17):
            alpha = float(step) / 16.0
            joint = enc + (1.0 - alpha) * pmi
            order = torch.argsort(joint, descending=True, stable=True)
            selected = order[:keep_n]
            mask = torch.zeros(count, device=enc.device, dtype=torch.bool)
            mask[selected] = True
            sorted_joint = joint.index_select(0, order)
            boundary = float((sorted_joint[keep_n - 1] - sorted_joint[keep_n]).item()) if keep_n < count else 0.0
            positions = original.index_select(0, selected).float()
            row = {
                "alpha": alpha,
                "score_lambda": 1.0 - alpha,
                "score_boundary_margin": boundary,
                "selected_encoder_mean": float(enc.index_select(0, selected).mean().item()),
                "selected_encoder_std": float(enc.index_select(0, selected).std(unbiased=False).item()),
                "selected_pmi_mean": float(pmi.index_select(0, selected).mean().item()),
                "selected_pmi_std": float(pmi.index_select(0, selected).std(unbiased=False).item()),
                "selected_text_mean": float(text.index_select(0, selected).mean().item()),
                "selected_text_std": float(text.index_select(0, selected).std(unbiased=False).item()),
                "selected_joint_mean": float(joint.index_select(0, selected).mean().item()),
                "selected_joint_std": float(joint.index_select(0, selected).std(unbiased=False).item()),
                "selected_position_mean": float(positions.mean().item()),
                "selected_position_std": float(positions.std(unbiased=False).item()),
                "selected_position_min": float(positions.min().item()),
                "selected_position_max": float(positions.max().item()),
                "selected_local_sum": int(selected.sum().item()),
                "selected_local_sqsum": int((selected.long().square()).sum().item()),
                "selected_original_sum": int(positions.long().sum().item()),
                "selected_original_sqsum": int(positions.long().square().sum().item()),
                "overlap_encoder_topk": float(self._mask_overlap(mask, ref["encoder"]) or 0.0),
                "overlap_pmi_topk": float(self._mask_overlap(mask, ref["pmi"]) or 0.0),
                "overlap_text_topk": float(self._mask_overlap(mask, ref["text"]) or 0.0),
                "jaccard_previous_alpha": float(self._mask_overlap(mask, previous) or 0.0) if previous is not None else 1.0,
                "selected_anchor_fraction": float(anchors.index_select(0, selected).float().mean().item()) if anchors is not None else 0.0,
                "keep_local_indices": [int(x) for x in selected.detach().cpu().tolist()],
                "keep_original_local_indices": [int(x) for x in original.index_select(0, selected).detach().cpu().tolist()],
            }
            selected_keys = vision_keys.index_select(1, selected.to(device=vision_keys.device)).mean(dim=1)
            row.update(self._hashed_projection_features("selected_vision_key_proj_b016", selected_keys, bins=16))
            rows.append(row)
            previous = mask
        return rows

    def _continuous_alpha_head_features(
        self, batch_idx: int, query_states: torch.Tensor, key_states: torch.Tensor,
        vision_idx: torch.Tensor, text_idx: torch.Tensor, keep_n: int,
        score_joint: torch.Tensor, score_text: torch.Tensor,
        score_encoder: torch.Tensor, score_pmi: torch.Tensor,
    ) -> dict[str, float]:
        sorted_score = torch.sort(score_joint.detach().float(), descending=True).values
        margin = float((sorted_score[keep_n - 1] - sorted_score[keep_n]).item()) if keep_n < int(sorted_score.numel()) else float("nan")
        labels = self._feature_reserve_current_is_anchor
        anchor_n = int(labels.detach().bool().sum().item()) if labels is not None and int(labels.numel()) == int(vision_idx.numel()) else 0
        out = {
            "prompt_len": float(self.prompt_len), "n_vision_candidates": float(vision_idx.numel()),
            "n_text_queries": float(text_idx.numel()), "keep_n": float(keep_n),
            "candidate_anchor_fraction": float(anchor_n) / float(max(1, int(vision_idx.numel()))),
            "score_boundary_margin": margin,
        }
        for prefix, values in (("joint", score_joint), ("text", score_text), ("encoder", score_encoder), ("pmi", score_pmi)):
            out.update(self._tensor_score_stats(prefix, values))
        for name, left, right in (("encoder_text", score_encoder, score_text), ("encoder_pmi", score_encoder, score_pmi), ("text_pmi", score_text, score_pmi)):
            pearson=self._pearson_corr(left,right);spearman=self._spearman_corr(left,right)
            if pearson is not None: out[f"corr_{name}_pearson"]=pearson
            if spearman is not None: out[f"corr_{name}_spearman"]=spearman
        provisional = self._topk_mask(score_joint, keep_n)
        for name, values in (("joint",score_joint),("text",score_text),("encoder",score_encoder),("pmi",score_pmi)):
            overlap=self._mask_overlap(provisional,self._topk_mask(values,keep_n))
            if overlap is not None: out[f"selected_overlap_{name}_topk_jaccard"]=overlap
        text_queries=query_states[batch_idx,:,text_idx.long(),:];vision_keys=key_states[batch_idx,:,vision_idx.long(),:]
        out.update(self._hashed_projection_features("l9_text_last_proj_b016",text_queries[:,-1,:],bins=16))
        out.update(self._hashed_projection_features("l9_text_mean_proj_b016",text_queries.mean(dim=1),bins=16))
        out.update(self._hashed_projection_features("l9_vision_mean_proj_b016",vision_keys.mean(dim=1),bins=16))
        semantic_top=torch.argsort(score_encoder.detach().float(),descending=True,stable=True)[:min(8,int(vision_idx.numel()))]
        out.update(self._hashed_projection_features("l9_vision_salient_proj_b016",vision_keys.index_select(1,semantic_top).mean(dim=1),bins=16))
        out.update(self._pending_continuous_alpha_lexical)
        return out

    def _predict_continuous_alpha(self, features: dict[str, float]) -> float:
        path=str(os.environ.get("VISPRUNER_CONTINUOUS_ALPHA_HEAD","") or "").strip()
        if not path: raise RuntimeError("learned continuous-alpha mode requires VISPRUNER_CONTINUOUS_ALPHA_HEAD")
        cache=self._continuous_alpha_head_cache
        if cache is None or cache.get("_path") != path:
            with open(path,"r",encoding="utf-8") as f: cache=json.load(f)
            if cache.get("kind") != "linear_tanh_continuous_alpha": raise RuntimeError("unsupported continuous-alpha head")
            cache["_path"]=path;self._continuous_alpha_head_cache=cache
        cols=cache["feature_columns"];med=cache["median"];mask=cache["kept_mask"];vals=[]
        for i,name in enumerate(cols):
            value=float(features.get(name,med[i]));vals.append(value if math.isfinite(value) else float(med[i]))
        kept=[vals[i] for i,k in enumerate(mask) if k];mean=cache["mean"];std=cache["std"];weight=cache["weight"]
        logit=float(cache["bias"])+sum(((x-m)/d)*w for x,m,d,w in zip(kept,mean,std,weight))
        alpha=max(0.0,min(1.0,float(cache["base_alpha"])+0.5*math.tanh(logit)))
        dump_path=str(os.environ.get("VISPRUNER_CONTINUOUS_ALPHA_FEATURE_DUMP","") or "").strip()
        if dump_path:
            with open(dump_path,"a",encoding="utf-8") as f: f.write(json.dumps({"features":features,"alpha":alpha},sort_keys=True)+"\n")
        return alpha

    def _record_vision_score_analysis(
        self,
        source_layer: int,
        batch_idx: int,
        mode: str,
        vision_idx: torch.Tensor,
        text_idx: torch.Tensor,
        keep_n: int,
        score_final: torch.Tensor,
        score_text: torch.Tensor,
        score_encoder: Optional[torch.Tensor],
        score_pmi: Optional[torch.Tensor],
        keep_mask: torch.Tensor,
        effective_lambda: float,
        adaptive_metrics: dict[str, float],
    ) -> None:
        if not str(getattr(self.config, "analysis_path", "") or "").strip():
            return
        rec = {
            "record_type": "vision",
            "prompt_idx": int(self._analysis_prompt_idx),
            "batch_idx": int(batch_idx),
            "source_layer": int(source_layer),
            "target_layer": int(source_layer + 1),
            "num_layers": int(self.config.num_layers or 0),
            "vision_score_mode": str(mode),
            "vision_score_lambda": float(getattr(self.config, "vision_score_lambda", 1.0) or 0.0),
            "vision_score_lambda_effective": float(effective_lambda),
            "vision_anchor_alpha": float(getattr(self.config, "vision_anchor_alpha", 0.0) or 0.0),
            "vision_encoder_layer": int(getattr(self.config, "vision_encoder_layer", -1) or -1),
            "n_vision": int(vision_idx.numel()),
            "n_text_queries": int(text_idx.numel()),
            "keep_n": int(keep_n),
            "retain_ratio": float(keep_n) / float(max(1, int(vision_idx.numel()))),
        }
        rec.update({str(k): float(v) for k, v in adaptive_metrics.items()})
        rec.update(self._tensor_score_stats("final", score_final))
        rec.update(self._tensor_score_stats("text", score_text))
        rec.update(self._tensor_score_stats("encoder", score_encoder))
        rec.update(self._tensor_score_stats("pmi", score_pmi))
        pairs = [("encoder_text", score_encoder, score_text), ("encoder_pmi", score_encoder, score_pmi), ("text_pmi", score_text, score_pmi)]
        for name, a, b in pairs:
            pc = self._pearson_corr(a, b)
            sc = self._spearman_corr(a, b)
            if pc is not None:
                rec[f"corr_{name}_pearson"] = pc
            if sc is not None:
                rec[f"corr_{name}_spearman"] = sc
        top_final = self._topk_mask(score_final, keep_n)
        top_text = self._topk_mask(score_text, keep_n)
        top_encoder = self._topk_mask(score_encoder, keep_n)
        top_pmi = self._topk_mask(score_pmi, keep_n)
        keep_mask = keep_mask.detach().bool().flatten()
        for name, mask in [("final", top_final), ("text", top_text), ("encoder", top_encoder), ("pmi", top_pmi)]:
            ov = self._mask_overlap(keep_mask, mask)
            if ov is not None:
                rec[f"selected_overlap_{name}_topk_jaccard"] = ov
        for name, a, b in [("encoder_text", top_encoder, top_text), ("encoder_pmi", top_encoder, top_pmi), ("text_pmi", top_text, top_pmi)]:
            ov = self._mask_overlap(a, b)
            if ov is not None:
                rec[f"topk_overlap_{name}_jaccard"] = ov
        keep_local_abs = torch.nonzero(keep_mask, as_tuple=False).flatten().long()
        drop_local_abs = torch.nonzero(~keep_mask, as_tuple=False).flatten().long()
        rec["keep_local_sum"] = int(keep_local_abs.sum().item()) if keep_local_abs.numel() else 0
        rec["keep_local_sqsum"] = int((keep_local_abs * keep_local_abs).sum().item()) if keep_local_abs.numel() else 0
        rec["drop_local_sum"] = int(drop_local_abs.sum().item()) if drop_local_abs.numel() else 0
        rec["drop_local_sqsum"] = int((drop_local_abs * drop_local_abs).sum().item()) if drop_local_abs.numel() else 0
        self._write_analysis_record(rec)

    def _is_soft_text_mode(self) -> bool:
        mode = str(self.config.text_mask_mode or "").lower()
        return "soft" in mode or "atten" in mode or mode.startswith("routing_balance")

    def _uses_veto_text_mode(self) -> bool:
        return "veto" in str(self.config.text_mask_mode or "").lower()

    def _uses_grounding_text_score(self) -> bool:
        mode = str(self.config.text_mask_mode or "").lower()
        return not self._uses_veto_text_mode() and ("ground" in mode or "visual" in mode or "_add" in mode)

    def _uses_ema_text_score(self) -> bool:
        mode = str(self.config.text_mask_mode or "").lower()
        return "ema" in mode or "persist" in mode

    def _text_grounding_scores(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        text_idx: torch.Tensor,
        batch_idx: int,
    ) -> Optional[torch.Tensor]:
        vision_idx = self._vision_by_batch[batch_idx].to(device=text_idx.device)
        if self._vision_keep is not None and vision_idx.numel() > 0:
            keep_mask = self._vision_keep[batch_idx, vision_idx].to(device=text_idx.device).bool()
            if keep_mask.any():
                vision_idx = vision_idx[keep_mask]
        if vision_idx.numel() == 0:
            return None
        return self._query_to_key_mass_scores(
            query_states,
            key_states,
            attention_mask,
            query_idx=text_idx,
            target_key_idx=vision_idx,
        )

    def _rank_preserving_positive_score(self, score: torch.Tensor, inverse: bool) -> torch.Tensor:
        vals = torch.nan_to_num(score.detach().float(), nan=0.0, posinf=1e4, neginf=-1e4)
        if vals.numel() == 0:
            return vals
        if inverse:
            vals = vals.max() - vals
        else:
            vals = vals - vals.min()
        return vals + 1e-6

    def _last_query_qk_scores(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        text_idx: torch.Tensor,
    ) -> torch.Tensor:
        q = query_states[0, :, text_idx, :].float()
        k = key_states[0, :, text_idx, :].float()
        if int(k.shape[0]) != int(q.shape[0]):
            if int(q.shape[0]) % int(k.shape[0]) != 0:
                raise RuntimeError(f"query/key head mismatch: q={int(q.shape[0])} kv={int(k.shape[0])}")
            k = k.repeat_interleave(int(q.shape[0]) // int(k.shape[0]), dim=0)
        q_last = q[:, -1:, :]
        logits = (q_last * k).sum(dim=-1).mean(dim=0) / (float(q.shape[-1]) ** 0.5)
        logits = (logits - logits.mean()).clamp(min=-12.0, max=12.0)
        score = torch.exp(logits)
        self.record("text_score_last_query_qk_calls", 1)
        self.record("text_score_last_query_qk_tokens", int(text_idx.numel()))
        return score / score.mean().clamp_min(1e-8)

    def _causal_qkv_contribution_scores(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: Optional[torch.Tensor],
        text_idx: torch.Tensor,
    ) -> torch.Tensor:
        alignment = self._linear_causal_query_scores(
            query_states, key_states, text_idx, text_idx
        )
        if value_states is None:
            self.record("text_score_qkv_missing_value", 1)
            return alignment
        v = value_states[0, :, text_idx, :].float()
        value_norm = v.norm(dim=-1).mean(dim=0)
        value_norm = value_norm / value_norm.mean().clamp_min(1e-8)
        score = alignment * value_norm
        self.record("text_score_causal_qkv_contribution_calls", 1)
        self.record("text_score_causal_qkv_contribution_tokens", int(text_idx.numel()))
        return score / score.mean().clamp_min(1e-8)

    def _text_self_scores(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        text_idx: torch.Tensor,
        batch_idx: int = 0,
    ) -> torch.Tensor:
        source, inverse = self._text_score_source()
        if source == "linearized_causal_utilization":
            score = self._linearized_causal_utilization_scores(
                query_states,
                key_states,
                text_idx,
                text_idx,
                batch_idx=batch_idx,
            )
        elif source == "linear_causal_query":
            score = self._linear_causal_query_scores(
                query_states, key_states, text_idx, text_idx
            )
        elif source == "causal_qkv_contribution":
            score = self._causal_qkv_contribution_scores(
                query_states, key_states, value_states, text_idx
            )
        elif source == "last_query_qk":
            score = self._last_query_qk_scores(query_states, key_states, text_idx)
        elif source in {"random_control", "deterministic_random"}:
            score = self._deterministic_random_text_scores(text_idx)
        else:
            score = self._received_scores(
                query_states, key_states, attention_mask,
                query_idx=text_idx, key_idx=text_idx,
            )
        if inverse:
            self.record(f"text_score_{source}_inverse_calls", 1)
            return self._rank_preserving_positive_score(score, inverse=True)
        return score

    def _store_native_text_score(
        self,
        source_layer: int,
        source: str,
        score_by_batch: list[torch.Tensor],
    ) -> None:
        target_layer = int(source_layer) + 1
        if target_layer in self._text_drop_by_layer or target_layer in self._text_bias_by_layer:
            return
        bsz = len(self._text_by_batch)
        if len(score_by_batch) != bsz:
            raise RuntimeError(f"native text score batch mismatch: scores={len(score_by_batch)} batches={bsz}")
        device = score_by_batch[0].device if score_by_batch else self._input_ids.device
        text_drop = torch.zeros((bsz, self.prompt_len), device=device, dtype=torch.bool)
        selectable_total = 0
        dropped_total = 0
        configured, inverse = self._text_score_source()
        for b, text_idx_raw in enumerate(self._text_by_batch):
            text_idx = text_idx_raw.to(device=device)
            score_raw = score_by_batch[b]
            if int(score_raw.numel()) != int(text_idx.numel()) or text_idx.numel() <= 2:
                continue
            score = self._rank_preserving_positive_score(score_raw, inverse=inverse)
            selectable = torch.ones_like(score, dtype=torch.bool)
            selectable[0] = False
            selectable[-1] = False
            selectable_total += int(selectable.sum().item())
            masked = self._select_text(score, selectable, target_layer, batch_idx=b)
            if masked.numel() > 0:
                text_drop[b, text_idx[masked]] = True
            dropped_total += int(masked.numel())
            self._record_text_selection_analysis(
                int(source_layer), target_layer, b, text_idx, score,
                selectable, masked, None, 0,
            )
        self._text_drop_by_layer[target_layer] = text_drop.detach()
        self.record("text_select_events", 1)
        self.record("text_selectable", selectable_total)
        self.record("text_dropped", dropped_total)
        self.record("text_native_score_events", 1)
        self.record(f"text_native_source_{configured}_events", 1)
        if inverse:
            self.record("text_native_inverse_events", 1)

    def observe_native_text_tensor(
        self,
        layer_idx: int,
        source: str,
        tensor: torch.Tensor,
    ) -> None:
        if not self.needs_native_text_observe(layer_idx, source):
            return
        if not torch.is_tensor(tensor) or tensor.ndim != 3 or int(tensor.shape[1]) != self.prompt_len:
            self.record(f"text_native_{source}_bad_shape", 1)
            return
        scores = []
        for b, text_idx in enumerate(self._text_by_batch):
            values = tensor[b, text_idx.to(tensor.device), :].detach().float()
            if source in {"attention_output_norm", "mlp_output_norm"}:
                score = values.norm(dim=-1)
            elif source == "local_redundancy":
                unit = torch.nn.functional.normalize(values, dim=-1, eps=1e-6)
                prev = torch.full((values.shape[0],), -1.0, device=values.device)
                nxt = torch.full_like(prev, -1.0)
                if values.shape[0] > 1:
                    sim = (unit[:-1] * unit[1:]).sum(dim=-1)
                    prev[1:] = sim
                    nxt[:-1] = sim
                score = 1.0 - torch.maximum(prev, nxt)
            else:
                raise RuntimeError(f"unsupported native tensor source: {source}")
            scores.append(score)
        self.record(f"text_native_{source}_calls", 1)
        self.record(f"text_native_{source}_tokens", sum(int(x.numel()) for x in scores))
        self._store_native_text_score(layer_idx, source, scores)

    def observe_native_text_residual(
        self,
        layer_idx: int,
        hidden_before: torch.Tensor,
        hidden_after: torch.Tensor,
    ) -> None:
        source, _ = self._text_score_source()
        if source not in {"residual_update", "residual_alignment"}:
            return
        if not self.needs_native_text_observe(layer_idx, source):
            return
        if (
            not torch.is_tensor(hidden_before) or not torch.is_tensor(hidden_after)
            or hidden_before.ndim != 3 or hidden_after.ndim != 3
            or tuple(hidden_before.shape) != tuple(hidden_after.shape)
            or int(hidden_after.shape[1]) != self.prompt_len
        ):
            self.record(f"text_native_{source}_bad_shape", 1)
            return
        scores = []
        for b, text_idx in enumerate(self._text_by_batch):
            idx = text_idx.to(hidden_after.device)
            before = hidden_before[b, idx, :].detach().float()
            after = hidden_after[b, idx, :].detach().float()
            delta = after - before
            if source == "residual_update":
                score = delta.norm(dim=-1) / before.norm(dim=-1).clamp_min(1e-6)
            else:
                score = torch.nn.functional.cosine_similarity(before, delta, dim=-1, eps=1e-6)
            scores.append(score)
        self.record(f"text_native_{source}_calls", 1)
        self.record(f"text_native_{source}_tokens", sum(int(x.numel()) for x in scores))
        self._store_native_text_score(layer_idx, source, scores)

    def _text_importance_score(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        text_idx: torch.Tensor,
        score_self: torch.Tensor,
        batch_idx: int,
        grounding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        score = score_self
        if self._uses_grounding_text_score():
            if grounding is None:
                grounding = self._text_grounding_scores(
                    query_states, key_states, attention_mask, text_idx, batch_idx
                )
            lam = max(0.0, float(getattr(self.config, "text_grounding_lambda", 0.0) or 0.0))
            if grounding is not None and lam > 0.0:
                score = self._normalize_score(
                    self._normalize_score(score_self) + lam * self._normalize_score(grounding)
                )
                self.record("text_grounding_used", 1)
                self.record("text_grounding_lambda_x1000", int(round(lam * 1000.0)))
        if self._uses_ema_text_score():
            beta = max(0.0, min(0.999, float(getattr(self.config, "text_ema_beta", 0.8) or 0.0)))
            prev = self._text_ema_by_batch.get(int(batch_idx))
            cur = score.detach().float()
            if prev is None or prev.numel() != cur.numel():
                ema = cur
            else:
                ema = beta * prev.to(device=cur.device) + (1.0 - beta) * cur
            self._text_ema_by_batch[int(batch_idx)] = ema.detach()
            score = ema
            self.record("text_ema_used", 1)
            self.record("text_ema_beta_x1000", int(round(beta * 1000.0)))
        return score

    def _text_score_source(self) -> tuple[str, bool]:
        mode = str(self.config.text_correction or "exposure_baseline_ratio").lower().strip()
        inverse = mode.endswith("_inverse")
        if inverse:
            mode = mode[: -len("_inverse")]
        aliases = {
            "linear_received": "linear_causal_query",
            "mean_query_alignment": "linear_causal_query",
            "linearized_utilization": "linearized_causal_utilization",
            "first_order_utilization": "linearized_causal_utilization",
            "first_order_causal_utilization": "linearized_causal_utilization",
            "causal_qkv": "causal_qkv_contribution",
            "last_qk": "last_query_qk",
        }
        return aliases.get(mode, mode), inverse

    def _text_uses_layer_tensor_source(self) -> bool:
        source, _ = self._text_score_source()
        return source in {
            "residual_update", "residual_alignment", "attention_output_norm",
            "mlp_output_norm", "local_redundancy",
        }

    def needs_native_text_observe(self, layer_idx: int, source: str) -> bool:
        if not self.enabled or self._input_ids is None or not self.config.text_enabled:
            return False
        configured, _ = self._text_score_source()
        if configured != str(source).lower():
            return False
        target_layer = int(layer_idx) + 1
        if int(layer_idx) < int(self.config.layer_k) or not self._text_layer_active(target_layer):
            return False
        return bool(
            target_layer not in self._text_drop_by_layer
            and target_layer not in self._text_bias_by_layer
        )

    def _is_adaptive_text_mode(self) -> bool:
        return str(self.config.text_mask_mode or "").lower().startswith("adaptive")

    def _adaptive_text_needs_warmup(self) -> bool:
        """Whether an adaptive rule needs pre-mask-layer score observations.

        Entropy concentration is prompt- and layer-local, so it needs no warmup
        baseline.  Skipping warmup is essential when masking is restricted to a
        late layer range: no Q/K score work should be introduced before it.
        """
        mode = str(self.config.text_mask_mode or "").lower()
        if not mode.startswith("adaptive"):
            return False
        entropy_only = (
            "entropy" in mode
            and "underutil" not in mode
            and "excess" not in mode
            and "autostart" not in mode
        )
        parameter_free_deficit = "deficit" in mode
        return not (entropy_only or parameter_free_deficit)

    def _adaptive_tau(self) -> float:
        return max(0.0, float(getattr(self.config, "text_adaptive_tau", 1.0) or 0.0))

    def _low_utility_fraction(self, score: torch.Tensor, selectable: torch.Tensor) -> float:
        vals = score[selectable]
        if vals.numel() == 0:
            return 0.0
        return float((vals < self._adaptive_tau()).float().mean().item())

    def _update_adaptive_floor(self, batch_idx: int, score: torch.Tensor, selectable: torch.Tensor) -> None:
        frac = self._low_utility_fraction(score, selectable)
        self._adaptive_floor_values.setdefault(int(batch_idx), []).append(frac)
        self.record("adaptive_warmup_records", 1)
        self.record("adaptive_warmup_frac_sum", frac)

    def _adaptive_floor(self, batch_idx: int) -> float:
        mode = str(getattr(self.config, "text_adaptive_floor_mode", "warmup_median") or "warmup_median").lower()
        fixed = max(0.0, min(1.0, float(getattr(self.config, "text_adaptive_floor", 0.15) or 0.0)))
        if "fixed" in mode:
            return fixed
        vals = self._adaptive_floor_values.get(int(batch_idx), [])
        if not vals:
            return fixed
        t = torch.tensor(vals, dtype=torch.float32)
        if "max" in mode:
            return float(t.max().item())
        if "q75" in mode or "p75" in mode or "upper" in mode:
            return float(torch.quantile(t, 0.75).item())
        if "mean" in mode:
            return float(t.mean().item())
        return float(t.median().item())

    def _adaptive_ratio(self, score: torch.Tensor, selectable: torch.Tensor, batch_idx: int, target_layer: int) -> float:
        mode = str(getattr(self.config, "text_mask_mode", "") or "").lower()
        alpha = max(0.0, float(getattr(self.config, "text_adaptive_alpha", 0.45) or 0.0))
        min_ratio = max(0.0, min(1.0, float(getattr(self.config, "text_adaptive_min_ratio", 0.0) or 0.0)))
        max_ratio = max(0.0, min(1.0, float(getattr(self.config, "text_adaptive_max_ratio", 0.2) or 0.0)))
        if "deficit" in mode:
            vals = score[selectable].detach().float()
            if vals.numel() == 0:
                ratio = 0.0
                deficit = 0.0
            else:
                # Parameter-free adaptive budget. A baseline-normalized
                # utilization of one is uniform attention; each token
                # contributes only its missing mass in [0, 1]. Thus
                # N * ratio is the equivalent number of fully unused tokens.
                deficit_values = (1.0 - vals.clamp_min(0.0)).clamp(min=0.0, max=1.0)
                deficit = float(deficit_values.mean().item())
                ratio = deficit
            self.record("adaptive_ratio_sum", ratio)
            self.record("adaptive_deficit_sum", deficit)
            self.record("adaptive_deficit_calls", 1)
            self.record("adaptive_ratio_calls", 1)
            return ratio
        if "entropy" in mode and "underutil" not in mode and "excess" not in mode:
            concentration = self._score_concentration(score, selectable)
            ratio = max(min_ratio, min(max_ratio, alpha * concentration))
            self.record("adaptive_entropy_concentration_sum", concentration)
            self.record("adaptive_ratio_sum", ratio)
            self.record("adaptive_ratio_calls", 1)
            return ratio
        ramp_layers = max(0, int(getattr(self.config, "text_adaptive_ramp_layers", 0) or 0))
        if ramp_layers > 0:
            # K is only a warmup boundary. Ramp just opens the cap gradually
            # after K to avoid abrupt masking in the first post-warmup layers.
            step = max(0, int(target_layer) - int(self.config.layer_k) + 1)
            ramp_scale = max(0.0, min(1.0, float(step) / float(ramp_layers)))
            max_ratio = max_ratio * ramp_scale

        if "excess" in mode or "autostart" in mode:
            # Auto-start text masking from a prompt-local utilization baseline.
            # Scores are normalized so S_j < tau(=1 by default) means a text token
            # receives less attention than the causal/uniform expectation.  The
            # first observed layer becomes the baseline; later layers mask only
            # the *excess* low-utility mass above that baseline.  No hand-picked
            # start layer is required: if low-utility tokens do not increase,
            # the selected ratio is exactly zero and the layer is skipped.
            low_frac = self._low_utility_fraction(score, selectable)
            vals = self._adaptive_floor_values.setdefault(int(batch_idx), [])
            if not vals:
                vals.append(low_frac)
                self.record("adaptive_excess_ref_frac_sum", low_frac)
                self.record("adaptive_excess_ref_records", 1)
                self.record("adaptive_ratio_calls", 1)
                return 0.0
            floor = float(vals[0])
            delta = max(0.0, low_frac - floor)
            ratio = alpha * delta
            if "entropy" in mode or "ent" in mode:
                concentration = self._score_concentration(score, selectable)
                ratio = ratio * concentration
                self.record("adaptive_entropy_gate_used", 1)
            ratio = max(min_ratio, min(max_ratio, ratio))
            self.record("adaptive_ratio_sum", ratio)
            self.record("adaptive_low_frac_sum", low_frac)
            self.record("adaptive_floor_sum", floor)
            self.record("adaptive_excess_delta_sum", delta)
            self.record("adaptive_ratio_calls", 1)
            return ratio

        if "underutil" in mode or "utilization" in mode:
            vals = score[selectable]
            tau = self._adaptive_tau()
            if vals.numel() == 0 or tau <= 0.0:
                ratio = 0.0
                underutil = 0.0
            else:
                # Layer-wise Under-utilization Adaptive Masking:
                # U_l = mean_j max(0, tau - S_j), with tau=1 for baseline-normalized scores.
                # The selected masking ratio is beta * U_l, clipped by r_max.
                underutil = float((tau - vals).clamp_min(0.0).mean().item())
                ratio = alpha * underutil
                if "entropy" in mode or "ent" in mode:
                    concentration = self._score_concentration(score, selectable)
                    ratio = ratio * concentration
                    self.record("adaptive_entropy_gate_used", 1)
            ratio = max(min_ratio, min(max_ratio, ratio))
            self.record("adaptive_ratio_sum", ratio)
            self.record("adaptive_underutil_sum", underutil)
            self.record("adaptive_ratio_calls", 1)
            return ratio

        low_frac = self._low_utility_fraction(score, selectable)
        floor = self._adaptive_floor(batch_idx)
        margin = max(0.0, min(1.0, float(getattr(self.config, "text_adaptive_margin", 0.0) or 0.0)))
        gate_tau = max(0.0, float(getattr(self.config, "text_adaptive_gate_tau", 0.0) or 0.0))
        gate_min_frac = max(0.0, min(1.0, float(getattr(self.config, "text_adaptive_gate_min_frac", 0.0) or 0.0)))
        gate_frac = 1.0
        if gate_tau > 0.0 and gate_min_frac > 0.0:
            vals = score[selectable]
            gate_frac = float((vals < gate_tau).float().mean().item()) if vals.numel() > 0 else 0.0
            if gate_frac < gate_min_frac:
                self.record("adaptive_gate_skip", 1)
                self.record("adaptive_gate_frac_sum", gate_frac)
                self.record("adaptive_ratio_calls", 1)
                return 0.0
        delta = max(0.0, low_frac - floor - margin)
        ratio = alpha * delta
        ratio = max(min_ratio, min(max_ratio, ratio))
        self.record("adaptive_ratio_sum", ratio)
        self.record("adaptive_low_frac_sum", low_frac)
        self.record("adaptive_floor_sum", floor)
        self.record("adaptive_margin_sum", margin)
        self.record("adaptive_gate_frac_sum", gate_frac)
        self.record("adaptive_ratio_calls", 1)
        return ratio

    def _select_text(
        self,
        score: torch.Tensor,
        selectable: torch.Tensor,
        target_layer: int,
        batch_idx: int = 0,
    ) -> torch.Tensor:
        candidates = torch.nonzero(selectable, as_tuple=False).flatten()
        if candidates.numel() == 0:
            return candidates
        cand_score = score[candidates]
        mode = str(self.config.text_mask_mode or "threshold").lower()
        if mode.startswith("adaptive"):
            ratio = self._adaptive_ratio(score, selectable, batch_idx, target_layer)
            k = int(round(float(candidates.numel()) * ratio))
            if k <= 0:
                return candidates[:0]
            k = min(k, int(candidates.numel()))
            return candidates[torch.topk(cand_score, k=k, largest=False).indices]
        if mode in {"ratio", "fixed", "topk"}:
            ratio = self._scheduled_ratio(target_layer)
            k = int(round(float(candidates.numel()) * ratio))
            if k <= 0:
                return candidates[:0]
            k = min(k, int(candidates.numel()))
            return candidates[torch.topk(cand_score, k=k, largest=False).indices]

        threshold = float(self.config.text_threshold)
        if threshold > 0:
            cutoff = threshold * cand_score.mean().clamp_min(1e-8)
            return candidates[cand_score < cutoff]
        return candidates[:0]

    def _auto_vision_ready(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[bool, Optional[float]]:
        """Return whether auto vision pruning should fire at this layer.

        Supported rules:
        - ``corr_pos``: legacy probe; fires when encoder/text-to-vision rank
          correlation becomes positive.
        - ``text_util_dip_jump``: fires when the text-token low-utility
          fraction shows the transition pattern observed in analysis: early
          gradual change, a local dip, then a sharp rebound that exceeds all
          previous layers.  This uses the same exposure-normalized text->text
          score S_j as the text masking analysis, and only statistics from
          earlier layers in the current forward pass.
        """
        rule = str(getattr(self.config, "vision_auto_rule", "corr_pos") or "corr_pos").lower()
        max_layer = self._auto_vision_max_layer()

        if rule in {
            "text_util_dip_jump",
            "dip_jump",
            "lowfrac_dip_jump",
            "text_lowfrac_dip_jump",
            "text_util_jump_mean_std",
            "text_util_jump_mean_std2",
            "text_util_jump_mean_2std",
            "mean_std_2p0",
            "mean_std2",
            "jump_mean_std2",
        }:
            bsz = int(query_states.shape[0])
            vals = []
            for b in range(min(bsz, len(self._text_by_batch))):
                text_idx = self._text_by_batch[b]
                if text_idx.numel() <= 2:
                    continue
                attn_mask_b = attention_mask[b : b + 1] if attention_mask is not None and attention_mask.shape[0] == bsz else attention_mask
                score_self = self._received_scores(
                    query_states[b : b + 1],
                    key_states[b : b + 1],
                    attn_mask_b,
                    query_idx=text_idx,
                    key_idx=text_idx,
                )
                selectable = torch.ones_like(score_self, dtype=torch.bool)
                selectable[0] = False
                selectable[-1] = False
                if selectable.sum() <= 0:
                    continue
                vals.append(self._low_utility_fraction(score_self, selectable))
            low_frac = (sum(vals) / float(len(vals))) if vals else None
            hist = self._vision_auto_low_frac_history
            ready = False
            fallback = False
            min_history = max(2, int(getattr(self.config, "vision_auto_min_history", 2) or 2))
            if low_frac is not None and len(hist) >= min_history:
                prev = torch.tensor(hist, dtype=torch.float32)
                prev_max = float(prev.max().item())
                prev_mean = float(prev.mean().item())
                last = float(hist[-1])
                deltas = [hist[i] - hist[i - 1] for i in range(1, len(hist)) if hist[i] > hist[i - 1]] if len(hist) >= 2 else []
                mean_pos_delta = (sum(deltas) / float(len(deltas))) if deltas else 0.0
                jump = float(low_frac) - last
                new_peak = float(low_frac) > prev_max
                if rule in {"text_util_dip_jump", "dip_jump", "lowfrac_dip_jump", "text_lowfrac_dip_jump"}:
                    local_dip = last <= prev_mean
                    sharp_rebound = jump > max(0.0, mean_pos_delta)
                    ready = bool(local_dip and new_peak and sharp_rebound)
                    self.record("vision_auto_rule_text_util_dip_jump_checked", 1)
                else:
                    # Mean+2std jump rule: trigger only when the low-utility
                    # fraction reaches a new peak and its layer-to-layer jump is
                    # unusually large relative to previous positive transitions.
                    # Requiring at least three previous positive deltas prevents
                    # trivial L1/L2 triggers from short histories.
                    mult = 2.0
                    if "1p5" in rule:
                        mult = 1.5
                    elif "2p5" in rule:
                        mult = 2.5
                    elif "3p0" in rule or "3std" in rule:
                        mult = 3.0
                    min_pos_deltas = 3
                    if deltas:
                        dt = torch.tensor(deltas, dtype=torch.float32)
                        pos_mean = float(dt.mean().item())
                        pos_std = float(dt.std(unbiased=False).item()) if dt.numel() > 1 else 0.0
                    else:
                        pos_mean = 0.0
                        pos_std = 0.0
                    threshold = pos_mean + mult * pos_std
                    sharp_rebound = jump > max(0.0, threshold)
                    enough_history = len(deltas) >= min_pos_deltas
                    ready = bool(new_peak and enough_history and sharp_rebound)
                    self.record("vision_auto_rule_mean_std_jump_checked", 1)
                    self.record("vision_auto_low_frac_pos_delta_mean_x1000", int(round(pos_mean * 1000.0)))
                    self.record("vision_auto_low_frac_pos_delta_std_x1000", int(round(pos_std * 1000.0)))
                    self.record("vision_auto_low_frac_jump_threshold_x1000", int(round(threshold * 1000.0)))
                    self.record("vision_auto_low_frac_pos_delta_count", int(len(deltas)))
                self.record("vision_auto_low_frac_prev_max_x1000", int(round(prev_max * 1000.0)))
                self.record("vision_auto_low_frac_prev_mean_x1000", int(round(prev_mean * 1000.0)))
                self.record("vision_auto_low_frac_jump_x1000", int(round(jump * 1000.0)))
            if int(layer_idx) >= int(max_layer):
                fallback = not ready
                ready = True
            if low_frac is not None:
                hist.append(float(low_frac))
                self.record("vision_auto_low_frac_x1000", int(round(float(low_frac) * 1000.0)))
                self.record("vision_auto_low_frac_records", 1)
            if ready:
                if rule in {"text_util_dip_jump", "dip_jump", "lowfrac_dip_jump", "text_lowfrac_dip_jump"}:
                    self.record("vision_auto_rule_text_util_dip_jump", 1)
                else:
                    self.record("vision_auto_rule_mean_std_jump", 1)
                if fallback:
                    self.record("vision_auto_fallback_max_layer", 1)
            return ready, low_frac

        # Legacy auto criterion retained for ablations/backward compatibility.
        bsz = int(query_states.shape[0])
        corrs = []
        for b in range(min(bsz, len(self._vision_by_batch))):
            vision_idx = self._vision_by_batch[b]
            if self._vision_stage_drop is not None:
                stage_mask = self._vision_stage_drop[b, vision_idx].to(device=vision_idx.device).bool()
                if stage_mask.any():
                    vision_idx = vision_idx[~stage_mask]
                    self.record("vision_stage_active_candidates", int(vision_idx.numel()))
            text_idx = self._text_by_batch[b]
            if vision_idx.numel() <= 1 or text_idx.numel() == 0:
                continue
            score_encoder = self._aligned_encoder_score_for_vision_idx(b, vision_idx, query_states.device)
            if score_encoder is None:
                continue
            attn_mask_b = attention_mask[b : b + 1] if attention_mask is not None and attention_mask.shape[0] == bsz else attention_mask
            score_text = self._received_scores(
                query_states[b : b + 1],
                key_states[b : b + 1],
                attn_mask_b,
                query_idx=text_idx,
                key_idx=vision_idx,
            )
            corr = self._spearman_corr(score_encoder, score_text)
            if corr is not None and torch.isfinite(torch.tensor(float(corr))):
                corrs.append(float(corr))
        mean_corr = (sum(corrs) / float(len(corrs))) if corrs else None
        if int(layer_idx) == 0 and mean_corr is not None and self._vision_auto_ref_corr is None:
            self._vision_auto_ref_corr = float(mean_corr)
            self.record("vision_auto_ref_corr_x1000", int(round(float(mean_corr) * 1000.0)))
            return False, mean_corr
        threshold = 0.0
        ready = bool(mean_corr is not None and mean_corr > threshold)
        if int(layer_idx) >= int(max_layer):
            ready = True
        if ready and mean_corr is not None:
            self.record("vision_auto_selected_corr_x1000", int(round(float(mean_corr) * 1000.0)))
            self.record("vision_auto_threshold_x1000", int(round(float(threshold) * 1000.0)))
        return ready, mean_corr

    def _select_vision_drop(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        text_query_drop: Optional[torch.Tensor] = None,
        source_layer: Optional[int] = None,
        stage_budget_mode: Optional[str] = None,
        final_keep_tokens: int = 0,
        value_states: Optional[torch.Tensor] = None,
        output_projection=None,
    ) -> torch.Tensor:
        bsz = query_states.shape[0]
        drop = torch.zeros((bsz, self.prompt_len), device=query_states.device, dtype=torch.bool)
        keep_ratio = max(0.0, min(1.0, float(self.config.vision_keep_ratio)))
        keep_tokens = max(0, int(self.config.vision_keep_tokens or 0))
        important_ratio = max(0.0, min(1.0, float(self.config.important_ratio)))
        last_scores = []
        for b in range(bsz):
            vision_idx = self._vision_by_batch[b]
            if self._vision_stage_drop is not None:
                stage_mask = self._vision_stage_drop[b, vision_idx].to(device=vision_idx.device).bool()
                if stage_mask.any():
                    vision_idx = vision_idx[~stage_mask]
                    self.record("vision_stage_active_candidates", int(vision_idx.numel()))
            text_idx = self._text_by_batch[b]
            if (
                os.environ.get("VISPRUNER_EXCLUDE_ASSISTANT_TARGET_QUERIES", "0") == "1"
                and text_idx.numel() > 0
            ):
                if self._role_ids is None or int(self._role_ids.numel()) != self.prompt_len:
                    raise RuntimeError("assistant-target exclusion requires aligned prompt roles")
                roles = self._role_ids.to(device=text_idx.device).index_select(0, text_idx)
                filtered = text_idx[roles.ne(3)]
                if filtered.numel() == 0:
                    raise RuntimeError("assistant-target exclusion removed every text query")
                self.record("vision_assistant_target_queries_excluded", int(text_idx.numel() - filtered.numel()))
                text_idx = filtered
            if text_query_drop is not None and text_idx.numel() > 0:
                text_drop_b = text_query_drop[b, text_idx].to(device=text_idx.device).bool()
                filtered = text_idx[~text_drop_b]
                if filtered.numel() > 0:
                    self.record("vision_text_aware_queries_removed", int(text_idx.numel() - filtered.numel()))
                    self.record("vision_text_aware_queries_kept", int(filtered.numel()))
                    text_idx = filtered
                else:
                    self.record("vision_text_aware_all_queries_dropped_fallback", 1)
            if vision_idx.numel() <= 1 or text_idx.numel() == 0:
                continue
            score_query_idx = text_idx
            if bool(getattr(self, "_vision_score_last_user_query_only", False)):
                if self._role_ids is None or int(self._role_ids.numel()) != self.prompt_len:
                    raise RuntimeError("last-user-query scoring requires aligned prompt roles")
                roles = self._role_ids.to(device=text_idx.device).index_select(0, text_idx)
                user_idx = text_idx[roles.eq(2)]
                if user_idx.numel() == 0:
                    raise RuntimeError("last-user-query scoring found no user-content query")
                score_query_idx = user_idx[-1:]
                self.record("vision_score_last_user_query_only_events", 1)
            if stage_budget_mode == "depth_coupled":
                n_vision = int(vision_idx.numel())
                final_n = max(1, min(int(final_keep_tokens or 0), n_vision))
                final_layer = max(0, int(self._effective_vision_layer()))
                num_layers = max(1, int(getattr(self.config, "num_layers", 1) or 1))
                depth_fraction = max(0.0, min(1.0, float(final_layer) / float(num_layers)))
                reserve_n = int(math.ceil(depth_fraction * float(max(0, n_vision - final_n))))
                keep_n = max(final_n, min(final_n + reserve_n, n_vision))
                self.record("vision_stage_depth_fraction_x1000", int(round(depth_fraction * 1000.0)))
                self.record("vision_stage_depth_coupled_candidates", n_vision)
                self.record("vision_stage_depth_coupled_final_tokens", final_n)
                self.record("vision_stage_depth_coupled_keep_tokens", keep_n)
                self.record("vision_stage_depth_coupled_reserved", keep_n - final_n)
            elif keep_tokens > 0:
                keep_n = max(1, min(keep_tokens, int(vision_idx.numel())))
            else:
                keep_n = self.final_vision_budget(int(vision_idx.numel()), b)
            self.record("vision_kept", keep_n)
            attn_mask_b = attention_mask[b : b + 1] if attention_mask is not None and attention_mask.shape[0] == bsz else attention_mask
            score_text = self._received_scores(
                query_states[b : b + 1],
                key_states[b : b + 1],
                attn_mask_b,
                query_idx=score_query_idx,
                key_idx=vision_idx,
            )
            score_encoder_for_analysis = self._aligned_encoder_score_for_vision_idx(b, vision_idx, query_states.device)
            score_pmi_for_analysis = None
            if score_encoder_for_analysis is not None:
                score_pmi_for_analysis = self._pmi_like_score(score_text, score_encoder_for_analysis)
            mode = str(getattr(self.config, "vision_score_mode", "text") or "text").lower()
            effective_lambda = float(getattr(self.config, "vision_score_lambda", 1.0) or 0.0)
            adaptive_metrics: dict[str, float] = {}
            functional_signatures = None
            if mode in {
                "functional_coreset_topk", "functional_coreset_merge",
                "functional_role_coreset_topk", "functional_user_coreset_topk",
                "text_pmi_reserve_functional_user_topk",
                "compression_adaptive_functional_user_topk",
                "competition_adaptive_functional_user_topk",
                "majority_support_adaptive_functional_user_topk",
                "functional_user_guarded_topk", "encoder_pmi_functional_alpha_topk",
            }:
                # This exact aligned tensor was already obtained above to
                # construct PMI; do not repeat the alignment/remap checks.
                score_encoder = score_encoder_for_analysis
                if score_encoder is None or score_pmi_for_analysis is None:
                    raise RuntimeError("functional coreset requires aligned encoder saliency")
                lam = float(getattr(self.config, "vision_score_lambda", 1.0) or 0.0)
                if mode in {"text_pmi_reserve_functional_user_topk", "compression_adaptive_functional_user_topk", "competition_adaptive_functional_user_topk", "majority_support_adaptive_functional_user_topk"}:
                    candidate_count = max(1, int(vision_idx.numel()))
                    reserve_fraction = 1.0 - float(keep_n) / float(candidate_count)
                    reserve_fraction = max(0.0, min(1.0, reserve_fraction))
                    reserve_score = (
                        (1.0 - reserve_fraction) * self._normalize_score(score_text)
                        + reserve_fraction * self._normalize_score(score_pmi_for_analysis)
                    )
                    if mode in {"compression_adaptive_functional_user_topk", "competition_adaptive_functional_user_topk", "majority_support_adaptive_functional_user_topk"}:
                        competition = float(max(0, candidate_count - keep_n)) / float(max(1, keep_n))
                        if mode == "competition_adaptive_functional_user_topk":
                            prior_weight = max(0.0, min(1.0, competition))
                        elif mode == "majority_support_adaptive_functional_user_topk":
                            question_dominant = 2 * keep_n > candidate_count
                            if bool(getattr(self, "_majority_support_require_path_complete", False)):
                                from .l9_path_complete_majority_support import resolve_question_dominant
                                question_dominant = resolve_question_dominant(
                                    keep_n=keep_n,
                                    candidate_count=candidate_count,
                                    topology_family=getattr(
                                        self, "_majority_support_topology_family", None
                                    ),
                                    require_path_complete=True,
                                )
                                self.record("vision_majority_support_path_gate_events", 1)
                                self.record(
                                    "vision_majority_support_path_complete",
                                    int(
                                        str(getattr(self, "_majority_support_topology_family", ""))
                                        in {"single_merger", "single_projector"}
                                    ),
                                )
                            prior_weight = 0.0 if question_dominant else 1.0
                        else:
                            prior_weight = max(0.0, min(1.0, competition - 1.0))
                        current_score = self._normalize_score(score_encoder) + lam * self._normalize_score(score_pmi_for_analysis)
                        score = (
                            (1.0 - prior_weight) * self._normalize_score(reserve_score)
                            + prior_weight * self._normalize_score(current_score)
                        )
                        self.record("vision_score_compression_adaptive_functional_used", 1)
                        self.record("vision_compression_competition_x1000000", int(round(competition * 1_000_000.0)))
                        self.record("vision_compression_prior_weight_x1000000", int(round(prior_weight * 1_000_000.0)))
                        if mode == "competition_adaptive_functional_user_topk":
                            self.record("vision_score_competition_adaptive_functional_used", 1)
                        elif mode == "majority_support_adaptive_functional_user_topk":
                            self.record("vision_score_majority_support_adaptive_functional_used", 1)
                            self.record("vision_majority_support_question_dominant", int(prior_weight == 0.0))
                    else:
                        score = reserve_score
                        self.record("vision_score_text_pmi_reserve_functional_used", 1)
                        self.record("vision_text_pmi_reserve_functional_x1000000", int(round(reserve_fraction * 1_000_000.0)))
                else:
                    score = self._normalize_score(score_encoder) + lam * self._normalize_score(
                        score_pmi_for_analysis
                    )
                if mode == "functional_role_coreset_topk":
                    signature_fn = self._text_update_role_resolved_functional_signatures
                elif mode in {"functional_user_coreset_topk", "text_pmi_reserve_functional_user_topk", "compression_adaptive_functional_user_topk", "competition_adaptive_functional_user_topk", "majority_support_adaptive_functional_user_topk", "functional_user_guarded_topk"}:
                    signature_fn = self._text_update_user_functional_signatures
                else:
                    signature_fn = self._text_update_functional_signatures
                functional_signatures = signature_fn(
                    query_states[b : b + 1], key_states[b : b + 1],
                    value_states[b : b + 1] if torch.is_tensor(value_states) else None,
                    attn_mask_b, text_idx, vision_idx, output_projection,
                )
                if int(functional_signatures.shape[0]) != int(vision_idx.numel()):
                    raise RuntimeError(
                        f"functional signature alignment failed: signatures={int(functional_signatures.shape[0])} "
                        f"vision={int(vision_idx.numel())}"
                    )
                self._vision_functional_signatures[b] = functional_signatures.detach()
                self.record("vision_score_encoder_used", 1)
                self.record("vision_score_pmi_add_used", 1)
                if mode in {
                    "functional_coreset_topk", "functional_coreset_merge",
                    "functional_role_coreset_topk", "functional_user_coreset_topk",
                    "text_pmi_reserve_functional_user_topk", "compression_adaptive_functional_user_topk", "competition_adaptive_functional_user_topk", "majority_support_adaptive_functional_user_topk", "functional_user_guarded_topk",
                }:
                    self.record("vision_score_functional_coreset_used", 1)
                    if mode == "functional_role_coreset_topk":
                        self.record("vision_score_functional_role_used", 1)
                    elif mode in {"functional_user_coreset_topk", "text_pmi_reserve_functional_user_topk", "compression_adaptive_functional_user_topk", "competition_adaptive_functional_user_topk", "majority_support_adaptive_functional_user_topk", "functional_user_guarded_topk"}:
                        self.record("vision_score_functional_user_used", 1)
                else:
                    self.record("vision_score_functional_alpha_used", 1)
            elif mode in {"last_query_contribution_topk", "encoder_contribution_add_topk"}:
                contribution = self._last_query_value_contribution_scores(
                    query_states[b : b + 1], key_states[b : b + 1],
                    value_states[b : b + 1] if torch.is_tensor(value_states) else None,
                    attn_mask_b, int(text_idx[-1]), vision_idx,
                )
                if mode == "last_query_contribution_topk":
                    score = contribution
                else:
                    score_encoder = self._aligned_encoder_score_for_vision_idx(
                        b, vision_idx, query_states.device
                    )
                    if score_encoder is None:
                        raise RuntimeError("encoder-contribution scoring requires encoder saliency")
                    lam = float(getattr(self.config, "vision_score_lambda", 1.0) or 0.0)
                    score = self._normalize_score(score_encoder) + lam * self._normalize_score(contribution)
                    self.record("vision_score_encoder_used", 1)
                    self.record("vision_score_contribution_add_used", 1)
            elif mode in {"text", "text_only", "text_topk"}:
                score = score_text
            elif mode in {"random_topk", "deterministic_random_topk"}:
                score = torch.zeros_like(score_text.float())
                self.record("vision_score_random_control", 1)
            elif mode in {"text_vision_sum", "sum", "textvision_sum"}:
                score_vision = self._received_scores(
                    query_states[b : b + 1],
                    key_states[b : b + 1],
                    attn_mask_b,
                    query_idx=vision_idx,
                    key_idx=vision_idx,
                )
                score = score_text + score_vision
                self.record("vision_score_vision_self_used", 1)
            elif mode in {"text_vision_norm", "norm", "normalized_sum", "textvision_norm"}:
                score_vision = self._received_scores(
                    query_states[b : b + 1],
                    key_states[b : b + 1],
                    attn_mask_b,
                    query_idx=vision_idx,
                    key_idx=vision_idx,
                )
                score = self._normalize_score(score_text) + self._normalize_score(score_vision)
                self.record("vision_score_vision_self_used", 1)
            elif mode in {"text_vision_max", "max", "textvision_max"}:
                score_vision = self._received_scores(
                    query_states[b : b + 1],
                    key_states[b : b + 1],
                    attn_mask_b,
                    query_idx=vision_idx,
                    key_idx=vision_idx,
                )
                score = torch.maximum(self._normalize_score(score_text), self._normalize_score(score_vision))
                self.record("vision_score_vision_self_used", 1)
            elif mode in {"vision", "vision_only", "vonly"}:
                score_vision = self._received_scores(
                    query_states[b : b + 1],
                    key_states[b : b + 1],
                    attn_mask_b,
                    query_idx=vision_idx,
                    key_idx=vision_idx,
                )
                score = score_vision
                self.record("vision_score_vision_self_used", 1)
            elif mode in {"text_vision_add", "add", "weighted_add", "textvision_add"}:
                lam = float(getattr(self.config, "vision_score_lambda", 1.0) or 0.0)
                score_vision = self._received_scores(
                    query_states[b : b + 1],
                    key_states[b : b + 1],
                    attn_mask_b,
                    query_idx=vision_idx,
                    key_idx=vision_idx,
                )
                text_scale = score_text.float().mean().clamp_min(1e-8)
                score = score_text + lam * text_scale * self._normalize_score(score_vision)
                self.record("vision_score_vision_self_used", 1)
                self.record("vision_score_lambda_x1000", int(round(lam * 1000.0)))
            elif mode in {"text_vision_mul", "mul", "weighted_mul", "textvision_mul"}:
                lam = float(getattr(self.config, "vision_score_lambda", 1.0) or 0.0)
                score_vision = self._received_scores(
                    query_states[b : b + 1],
                    key_states[b : b + 1],
                    attn_mask_b,
                    query_idx=vision_idx,
                    key_idx=vision_idx,
                )
                score = score_text * (1.0 + lam * self._normalize_score(score_vision))
                self.record("vision_score_vision_self_used", 1)
                self.record("vision_score_lambda_x1000", int(round(lam * 1000.0)))
            elif mode in {"vispruner", "original_vispruner", "orig_vispruner", "encoder_vispruner"}:
                score_encoder = self._aligned_encoder_score_for_vision_idx(b, vision_idx, query_states.device)
                if score_encoder is None:
                    self.record("vision_encoder_score_missing_fallback", 1)
                    score = score_text
                else:
                    score = score_encoder
                    self.record("vision_score_encoder_used", 1)
                    self.record("vision_score_vispruner_original", 1)
            elif mode in {
                "encoder", "encoder_only", "enc",
                "anchor_regional", "encoder_anchor_regional", "regional_reserve",
                "anchor_regional_cls", "encoder_anchor_regional_cls",
                "feature_diverse", "encoder_feature_diverse", "feature_novelty",
                "encoder_topk", "encoder_saliency_topk",
            }:
                score_encoder = self._aligned_encoder_score_for_vision_idx(b, vision_idx, query_states.device)
                if score_encoder is None:
                    self.record("vision_encoder_score_missing_fallback", 1)
                    score = score_text
                else:
                    score = score_encoder
                    self.record("vision_score_encoder_used", 1)
            elif mode in {"pmi_only", "pmi_topk", "positive_information_gain_topk"}:
                score_encoder = self._aligned_encoder_score_for_vision_idx(b, vision_idx, query_states.device)
                if score_encoder is None:
                    raise RuntimeError("PMI-only selection requires aligned encoder saliency")
                score_pmi = self._pmi_like_score(score_text, score_encoder)
                score = score_pmi
                self.record("vision_score_pmi_only_used", 1)
            elif mode in {"text_pmi_reserve_blend", "text_pmi_reserve_blend_topk"}:
                score_encoder = self._aligned_encoder_score_for_vision_idx(b, vision_idx, query_states.device)
                if score_encoder is None:
                    raise RuntimeError("text-PMI reserve blend requires aligned encoder saliency")
                score_pmi = self._pmi_like_score(score_text, score_encoder)
                reserve_fraction = 1.0 - float(keep_n) / float(max(1, int(vision_idx.numel())))
                reserve_fraction = max(0.0, min(1.0, reserve_fraction))
                score = (
                    (1.0 - reserve_fraction) * self._normalize_score(score_text)
                    + reserve_fraction * self._normalize_score(score_pmi)
                )
                self.record("vision_score_text_pmi_reserve_blend_used", 1)
                self.record("vision_text_pmi_reserve_blend_x1000000", int(round(reserve_fraction * 1_000_000.0)))
            elif mode in {"encoder_text_add", "enc_text_add", "encoder_add"}:
                lam = float(getattr(self.config, "vision_score_lambda", 1.0) or 0.0)
                score_encoder = self._aligned_encoder_score_for_vision_idx(b, vision_idx, query_states.device)
                if score_encoder is None:
                    self.record("vision_encoder_score_missing_fallback", 1)
                    score = score_text
                else:
                    score = self._normalize_score(score_encoder) + lam * self._normalize_score(score_text)
                    self.record("vision_score_encoder_used", 1)
                    self.record("vision_score_lambda_x1000", int(round(lam * 1000.0)))
            elif mode in {
                "encoder_pmi_add", "enc_pmi_add", "encoder_text_pmi_add", "encoder_pmi",
                "encoder_pmi_add_topk", "encoder_pmi_topk", "encoder_pmi_learned_alpha_topk",
                "encoder_pmi_multialpha_vote_topk", "encoder_pmi_multialpha_borda_topk",
                "encoder_pmi_multialpha_minimax_topk",
                "encoder_pmi_anchor_quota_topk",
                "encoder_pmi_add_entropy", "encoder_pmi_add_adaptive", "encoder_pmi_entropy",
                "encoder_pmi_add_jsd", "encoder_pmi_jsd",
                "encoder_pmi_add_jsd_sqrt", "encoder_pmi_jsd_sqrt",
                "encoder_pmi_add_tv", "encoder_pmi_tv",
                "encoder_pmi_add_concentration", "encoder_pmi_concentration",
                "encoder_pmi_add_inv_jsd", "encoder_pmi_inv_jsd",
                "encoder_pmi_add_inv_jsd_sqrt", "encoder_pmi_inv_jsd_sqrt",
                "encoder_pmi_add_topk_consensus", "encoder_pmi_topk_consensus",
                "encoder_pmi_add_rel_conf", "encoder_pmi_rel_conf",
                "encoder_pmi_add_rel_margin", "encoder_pmi_rel_margin",
            }:
                lam = float(getattr(self.config, "vision_score_lambda", 1.0) or 0.0)
                score_encoder = self._aligned_encoder_score_for_vision_idx(b, vision_idx, query_states.device)
                if score_encoder is None:
                    if mode == "encoder_pmi_anchor_quota_topk":
                        raise RuntimeError("encoder-anchor quota selection requires aligned encoder saliency")
                    self.record("vision_encoder_score_missing_fallback", 1)
                    score = score_text
                else:
                    score_pmi = self._pmi_like_score(score_text, score_encoder)
                    if mode in {"encoder_pmi_add_entropy", "encoder_pmi_add_adaptive", "encoder_pmi_entropy"}:
                        selectable = torch.ones_like(score_pmi, dtype=torch.bool)
                        conc = self._score_concentration(score_pmi, selectable)
                        lam = lam * conc
                        self.record("vision_score_lambda_adaptive_entropy", 1)
                        self.record("vision_score_lambda_effective_x1000", int(round(lam * 1000.0)))
                        self.record("vision_pmi_concentration_x1000", int(round(conc * 1000.0)))
                    elif mode in {
                        "encoder_pmi_add_jsd", "encoder_pmi_jsd",
                        "encoder_pmi_add_jsd_sqrt", "encoder_pmi_jsd_sqrt",
                        "encoder_pmi_add_tv", "encoder_pmi_tv",
                        "encoder_pmi_add_concentration", "encoder_pmi_concentration",
                        "encoder_pmi_add_inv_jsd", "encoder_pmi_inv_jsd",
                        "encoder_pmi_add_inv_jsd_sqrt", "encoder_pmi_inv_jsd_sqrt",
                        "encoder_pmi_add_topk_consensus", "encoder_pmi_topk_consensus",
                        "encoder_pmi_add_rel_conf", "encoder_pmi_rel_conf",
                        "encoder_pmi_add_rel_margin", "encoder_pmi_rel_margin",
                    }:
                        lam, adaptive_metrics = self._vision_adaptive_lambda(
                            mode=mode,
                            score_text=score_text,
                            score_encoder=score_encoder,
                            score_pmi=score_pmi,
                            keep_n=keep_n,
                        )
                    effective_lambda = lam
                    score = self._normalize_score(score_encoder) + lam * self._normalize_score(score_pmi)
                    self.record("vision_score_encoder_used", 1)
                    self.record("vision_score_pmi_add_used", 1)
                    self.record("vision_score_lambda_x1000", int(round(float(getattr(self.config, "vision_score_lambda", 1.0) or 0.0) * 1000.0)))
            elif mode in {"encoder_pmi_mul", "enc_pmi_mul", "encoder_text_pmi_mul"}:
                lam = float(getattr(self.config, "vision_score_lambda", 1.0) or 0.0)
                score_encoder = self._aligned_encoder_score_for_vision_idx(b, vision_idx, query_states.device)
                if score_encoder is None:
                    self.record("vision_encoder_score_missing_fallback", 1)
                    score = score_text
                else:
                    score_pmi = self._pmi_like_score(score_text, score_encoder)
                    score = self._normalize_score(score_encoder) * (1.0 + lam * self._normalize_score(score_pmi))
                    self.record("vision_score_encoder_used", 1)
                    self.record("vision_score_pmi_mul_used", 1)
                    self.record("vision_score_lambda_x1000", int(round(lam * 1000.0)))
            elif mode in {"encoder_text_mul", "enc_text_mul", "encoder_mul"}:
                lam = float(getattr(self.config, "vision_score_lambda", 1.0) or 0.0)
                score_encoder = self._aligned_encoder_score_for_vision_idx(b, vision_idx, query_states.device)
                if score_encoder is None:
                    self.record("vision_encoder_score_missing_fallback", 1)
                    score = score_text
                else:
                    score = self._normalize_score(score_encoder) * (1.0 + lam * self._normalize_score(score_text))
                    self.record("vision_score_encoder_used", 1)
                    self.record("vision_score_lambda_x1000", int(round(lam * 1000.0)))
            else:
                self.record("vision_score_unknown_mode_fallback", 1)
                score = score_text
            self.record(f"vision_score_mode_{mode}", 1)
            if stage_budget_mode is None and self.deferred_vision_drop_enabled():
                last_scores.append((vision_idx.detach().clone(), score.detach().float().clone()))
            if mode in {
                "functional_coreset_topk", "functional_coreset_merge",
                "functional_role_coreset_topk", "functional_user_coreset_topk",
                "text_pmi_reserve_functional_user_topk", "compression_adaptive_functional_user_topk", "competition_adaptive_functional_user_topk", "majority_support_adaptive_functional_user_topk", "functional_user_guarded_topk",
            }:
                if functional_signatures is None:
                    raise RuntimeError("functional selector reached selection without signatures")
                if mode == "functional_user_guarded_topk":
                    keep_local = self._select_functional_guarded_coreset(
                        functional_signatures, score, keep_n
                    )
                else:
                    keep_local = self._select_functional_coreset(
                        functional_signatures, score, keep_n
                    )
                if bool(getattr(self, "_skip_functional_runtime_diagnostics", False)):
                    # These coverage/Jaccard values are analysis telemetry only;
                    # neither score construction nor selected indices consume
                    # them.  Paper/runtime mode therefore avoids two large
                    # M0-by-K similarity matrices and GPU-to-host list copies.
                    self.record("vision_functional_runtime_diagnostics_skipped", 1)
                else:
                    current_topk = torch.topk(score.float(), k=keep_n, largest=True).indices
                    functional_coverage = self._functional_coverage_score(
                        functional_signatures, keep_local
                    )
                    current_coverage = self._functional_coverage_score(
                        functional_signatures, current_topk
                    )
                    self.record("vision_functional_coverage_sum", functional_coverage)
                    self.record("vision_functional_current_topk_coverage_sum", current_coverage)
                    self.record(
                        "vision_functional_coverage_gain_sum",
                        functional_coverage - current_coverage,
                    )
                    self.record(
                        "vision_functional_vs_current_jaccard_sum",
                        float(len(set(keep_local.tolist()) & set(current_topk.tolist())))
                        / float(len(set(keep_local.tolist()) | set(current_topk.tolist()))),
                    )
            elif mode == "encoder_pmi_functional_alpha_topk":
                if (
                    score_encoder_for_analysis is None
                    or score_pmi_for_analysis is None
                    or functional_signatures is None
                ):
                    raise RuntimeError("functional continuous-alpha selection requires encoder, PMI, and value-write signatures")
                alpha, functional_target, fit_gain = self._functional_continuous_alpha(
                    functional_signatures,
                    score_encoder_for_analysis,
                    score_pmi_for_analysis,
                )
                effective_lambda = 1.0 - alpha
                score = self._normalize_score(score_encoder_for_analysis) + effective_lambda * self._normalize_score(score_pmi_for_analysis)
                keep_local = torch.argsort(score.float(), descending=True, stable=True)[:keep_n]
                self.record("vision_functional_alpha_select_events", 1)
                self.record("vision_functional_alpha_selected", keep_n)
                self.record("vision_functional_alpha_x1000000", int(round(alpha * 1000000.0)))
                self.record("vision_functional_alpha_fit_gain_x1000000", int(round(fit_gain * 1000000.0)))
                self.record("vision_functional_alpha_target_norm_sum", float(functional_target.norm().item()))
            elif mode == "encoder_pmi_anchor_quota_topk":
                if score_encoder_for_analysis is None:
                    raise RuntimeError("encoder-anchor quota selection reached selection without encoder saliency")
                alpha = float(getattr(self.config, "vision_anchor_alpha", 0.0) or 0.0)
                keep_local, anchor_n = self._select_encoder_anchor_quota(
                    score_encoder_for_analysis, score, keep_n, alpha, score_text=score_text
                )
                self.record("vision_anchor_quota_select_events", 1)
                self.record("vision_anchor_quota_alpha_x1000", int(round(alpha * 1000.0)))
                self.record("vision_anchor_quota_locked", anchor_n)
                self.record("vision_anchor_quota_joint_filled", keep_n - anchor_n)
                encoder_topk = torch.argsort(
                    score_encoder_for_analysis.detach().float(), descending=True, stable=True
                )[:keep_n]
                encoder_mask = torch.zeros(
                    int(vision_idx.numel()), device=keep_local.device, dtype=torch.bool
                )
                encoder_mask[encoder_topk] = True
                self.record(
                    "vision_anchor_quota_encoder_topk_kept",
                    int(encoder_mask.index_select(0, keep_local.long()).sum().item()),
                )
            elif mode == "encoder_pmi_learned_alpha_topk":
                if score_encoder_for_analysis is None or score_pmi_for_analysis is None:
                    raise RuntimeError("learned continuous-alpha selection requires aligned encoder and PMI scores")
                if self._generation_active and self._generation_forward_index > 0 and b in self._generation_continuous_alpha_cache:
                    alpha=float(self._generation_continuous_alpha_cache[b]);self.record("vision_continuous_alpha_cache_hits",1)
                else:
                    head_features=self._continuous_alpha_head_features(b,query_states,key_states,vision_idx,text_idx,keep_n,score,score_text,score_encoder_for_analysis,score_pmi_for_analysis)
                    alpha=self._predict_continuous_alpha(head_features)
                    if self._generation_active:self._generation_continuous_alpha_cache[b]=alpha
                    self.record("vision_continuous_alpha_predictions",1)
                effective_lambda=1.0-alpha
                score=self._normalize_score(score_encoder_for_analysis)+(1.0-alpha)*self._normalize_score(score_pmi_for_analysis)
                keep_local=torch.argsort(score.float(),descending=True,stable=True)[:keep_n]
                self.record("vision_continuous_alpha_select_events",1);self.record("vision_continuous_alpha_x1000000",int(round(alpha*1000000.0)))
            elif mode in {
                "encoder_pmi_multialpha_vote_topk", "encoder_pmi_multialpha_borda_topk",
                "encoder_pmi_multialpha_minimax_topk",
            }:
                if score_encoder_for_analysis is None or score_pmi_for_analysis is None:
                    raise RuntimeError("multi-alpha consensus requires aligned encoder and PMI scores")
                enc = self._normalize_score(score_encoder_for_analysis).float()
                pmi = self._normalize_score(score_pmi_for_analysis).float()
                n_candidates = int(enc.numel())
                rank_rows = []
                votes = torch.zeros_like(enc)
                for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
                    candidate = enc + (1.0 - alpha) * pmi
                    order = torch.argsort(candidate, descending=True, stable=True)
                    ranks = torch.empty_like(candidate)
                    ranks[order] = torch.arange(n_candidates, device=order.device, dtype=candidate.dtype)
                    rank_rows.append(ranks)
                    votes[order[:keep_n]] += 1.0
                rank_stack = torch.stack(rank_rows, dim=0)
                mean_rank = rank_stack.mean(dim=0)
                if mode == "encoder_pmi_multialpha_vote_topk":
                    score = votes * float(n_candidates + 1) - mean_rank
                elif mode == "encoder_pmi_multialpha_borda_topk":
                    score = -mean_rank
                else:
                    score = -rank_stack.max(dim=0).values
                keep_local = torch.argsort(score, descending=True, stable=True)[:keep_n]
                self.record("vision_multialpha_consensus_select_events", 1)
                self.record("vision_multialpha_consensus_selected", keep_n)
                self.record("vision_multialpha_grid_points", 5)
            elif mode in {
                "encoder_pmi_add_topk", "encoder_pmi_topk", "text_topk", "pmi_topk",
                "text_pmi_reserve_blend_topk",
                "last_query_contribution_topk", "encoder_contribution_add_topk",
            }:
                keep_local = torch.topk(score.float(), k=keep_n, largest=True).indices
                if mode == "text_topk":
                    self.record("vision_text_topk_select_events", 1)
                    self.record("vision_text_topk_selected", keep_n)
                elif mode == "pmi_topk":
                    self.record("vision_pmi_only_topk_select_events", 1)
                    self.record("vision_pmi_only_topk_selected", keep_n)
                elif mode == "text_pmi_reserve_blend_topk":
                    self.record("vision_text_pmi_reserve_blend_topk_select_events", 1)
                    self.record("vision_text_pmi_reserve_blend_topk_selected", keep_n)
                elif mode in {"last_query_contribution_topk", "encoder_contribution_add_topk"}:
                    self.record("vision_contribution_topk_select_events", 1)
                    self.record("vision_contribution_topk_selected", keep_n)
                else:
                    self.record("vision_pmi_topk_select_events", 1)
                    self.record("vision_pmi_topk_selected", keep_n)
            elif mode in {"random_topk", "deterministic_random_topk"}:
                local = torch.arange(int(vision_idx.numel()), device=vision_idx.device, dtype=torch.long)
                pseudo = torch.remainder((local + 1) * 1103515245 + 12345, 2147483647)
                keep_local = torch.argsort(pseudo, descending=False, stable=True)[:keep_n]
                self.record("vision_random_topk_select_events", 1)
                self.record("vision_random_topk_selected", keep_n)
            elif mode in {"feature_diverse", "encoder_feature_diverse", "feature_novelty"}:
                raise RuntimeError("feature-diverse selection must execute in the pre-LLM compaction path")
            elif mode in {"encoder_topk", "encoder_saliency_topk"}:
                keep_local = torch.argsort(score.float(), descending=True, stable=True)[:keep_n]
                self.record("vision_encoder_topk_select_events", 1)
            elif mode in {
                "anchor_regional", "encoder_anchor_regional", "regional_reserve",
                "anchor_regional_cls", "encoder_anchor_regional_cls",
            }:
                anchor_n = int(final_keep_tokens or 0)
                if anchor_n <= 0:
                    raise RuntimeError(
                        "anchor_regional selection requires final_keep_tokens > 0; "
                        "use it as the depth-coupled stage selector"
                    )
                keep_local = self._select_vision_keep_anchor_regional(
                    score=score,
                    keep_n=keep_n,
                    anchor_n=anchor_n,
                )
            elif mode in {
                "vispruner", "original_vispruner", "orig_vispruner", "encoder_vispruner",
                "encoder", "encoder_only", "enc", "encoder_text_add", "enc_text_add", "encoder_add",
                "encoder_text_mul", "enc_text_mul", "encoder_mul", "encoder_pmi_add", "enc_pmi_add",
                "encoder_text_pmi_add", "encoder_pmi", "encoder_pmi_add_entropy",
                "encoder_pmi_add_adaptive", "encoder_pmi_entropy", "encoder_pmi_add_jsd",
                "encoder_pmi_jsd", "encoder_pmi_add_jsd_sqrt", "encoder_pmi_jsd_sqrt",
                "encoder_pmi_add_tv", "encoder_pmi_tv", "encoder_pmi_add_concentration",
                "encoder_pmi_concentration", "encoder_pmi_mul", "enc_pmi_mul",
                "encoder_pmi_add_inv_jsd", "encoder_pmi_inv_jsd",
                "encoder_pmi_add_inv_jsd_sqrt", "encoder_pmi_inv_jsd_sqrt",
                "encoder_pmi_add_topk_consensus", "encoder_pmi_topk_consensus",
                "encoder_text_pmi_mul",
            }:
                keep_local = self._select_vision_keep_vispruner(key_states[b], vision_idx, score, keep_n, important_ratio)
            else:
                keep_local = self._select_vision_keep(key_states[b], vision_idx, score, keep_n, important_ratio)
            keep = torch.zeros(vision_idx.numel(), device=query_states.device, dtype=torch.bool)
            keep[keep_local] = True
            if bool(getattr(self.config, "capture_vision_selection", False)):
                def _top_indices(values: Optional[torch.Tensor]) -> list[int]:
                    if values is None or not torch.is_tensor(values) or int(values.numel()) != int(vision_idx.numel()):
                        return []
                    return [int(x) for x in torch.topk(values.detach().float(), k=keep_n, largest=True).indices.cpu().tolist()]

                sorted_score = torch.sort(score.detach().float(), descending=True).values
                margin = float((sorted_score[keep_n - 1] - sorted_score[keep_n]).item()) if keep_n < int(sorted_score.numel()) else float("nan")
                anchor_local = []
                if self._feature_reserve_current_is_anchor is not None and int(self._feature_reserve_current_is_anchor.numel()) == int(vision_idx.numel()):
                    anchor_local = [int(x) for x in torch.nonzero(self._feature_reserve_current_is_anchor.detach().bool(), as_tuple=False).flatten().cpu().tolist()]
                original_candidate_local = []
                original_keep_local = []
                original_anchor_local = []
                original_map = self._feature_reserve_original_local_indices
                if original_map is not None and int(original_map.numel()) == int(vision_idx.numel()):
                    original_map = original_map.to(device=keep_local.device, dtype=torch.long)
                    original_candidate_local = [int(x) for x in original_map.detach().cpu().tolist()]
                    original_keep_local = [int(x) for x in original_map.index_select(0, keep_local.long()).detach().cpu().tolist()]
                    if anchor_local:
                        anchor_tensor = torch.tensor(anchor_local, device=original_map.device, dtype=torch.long)
                        original_anchor_local = [int(x) for x in original_map.index_select(0, anchor_tensor).detach().cpu().tolist()]
                feature_summary = {
                    "prompt_len": int(self.prompt_len),
                    "n_vision_candidates": int(vision_idx.numel()),
                    "n_text_queries": int(text_idx.numel()),
                    "keep_n": int(keep_n),
                    "candidate_anchor_fraction": float(len(anchor_local)) / float(max(1, int(vision_idx.numel()))),
                    "score_boundary_margin": margin,
                }
                for prefix, values in (
                    ("joint", score),
                    ("text", score_text),
                    ("encoder", score_encoder_for_analysis),
                    ("pmi", score_pmi_for_analysis),
                ):
                    feature_summary.update(self._tensor_score_stats(prefix, values))
                for name, left, right in (
                    ("encoder_text", score_encoder_for_analysis, score_text),
                    ("encoder_pmi", score_encoder_for_analysis, score_pmi_for_analysis),
                    ("text_pmi", score_text, score_pmi_for_analysis),
                ):
                    pearson = self._pearson_corr(left, right)
                    spearman = self._spearman_corr(left, right)
                    if pearson is not None:
                        feature_summary[f"corr_{name}_pearson"] = pearson
                    if spearman is not None:
                        feature_summary[f"corr_{name}_spearman"] = spearman
                selected_mask = keep.detach().bool().flatten()
                for name, values in (
                    ("joint", score),
                    ("text", score_text),
                    ("encoder", score_encoder_for_analysis),
                    ("pmi", score_pmi_for_analysis),
                ):
                    overlap = self._mask_overlap(selected_mask, self._topk_mask(values, keep_n))
                    if overlap is not None:
                        feature_summary[f"selected_overlap_{name}_topk_jaccard"] = overlap
                text_queries = query_states[b, :, text_idx.long(), :]
                vision_keys = key_states[b, :, vision_idx.long(), :]
                feature_summary.update(self._hashed_projection_features(
                    "l9_text_last_proj", text_queries[:, -1, :], bins=16
                ))
                feature_summary.update(self._hashed_projection_features(
                    "l9_text_mean_proj", text_queries.mean(dim=1), bins=16
                ))
                feature_summary.update(self._hashed_projection_features(
                    "l9_vision_mean_proj", vision_keys.mean(dim=1), bins=16
                ))
                if score_encoder_for_analysis is not None:
                    semantic_top_n = min(8, int(vision_idx.numel()))
                    semantic_top = torch.argsort(
                        score_encoder_for_analysis.detach().float(), descending=True, stable=True
                    )[:semantic_top_n]
                    feature_summary.update(self._hashed_projection_features(
                        "l9_vision_salient_proj", vision_keys.index_select(1, semantic_top).mean(dim=1), bins=16
                    ))
                alpha_set_features = []
                if as_bool(os.environ.get("VISPRUNER_CAPTURE_ALPHA_SET_FEATURES", "0")):
                    alpha_set_features = self._continuous_alpha_set_features(
                        score_encoder_for_analysis,
                        score_pmi_for_analysis,
                        score_text,
                        vision_keys,
                        keep_n,
                        original_map=original_map,
                        anchor_mask=self._feature_reserve_current_is_anchor,
                    )
                    self.record("vision_alpha_set_feature_events", 1)
                    self.record("vision_alpha_set_feature_rows", len(alpha_set_features))
                anchor_quota_token_features = {}
                if as_bool(os.environ.get("VISPRUNER_CAPTURE_ANCHOR_QUOTA_TOKEN_FEATURES", "0")):
                    # Analysis capture is mode-name agnostic; aligned encoder and PMI
                    # tensors below are the actual mechanism requirement.
                    if score_encoder_for_analysis is None or score_pmi_for_analysis is None:
                        raise RuntimeError("anchor-quota token capture requires aligned encoder and PMI scores")
                    anchor_quota_token_features = self._anchor_quota_token_features(
                        score_encoder_for_analysis,
                        score,
                        score_pmi_for_analysis,
                        score_text,
                        text_queries,
                        vision_keys,
                        keep_n,
                        original_map=original_map,
                        anchor_mask=self._feature_reserve_current_is_anchor,
                        text_positions=text_idx,
                    )
                    self.record("vision_anchor_quota_token_feature_events", 1)
                    self.record("vision_anchor_quota_token_feature_rows", len(anchor_quota_token_features["token_features"]))
                raw_semantic_features = {}
                if as_bool(os.environ.get("VISPRUNER_CAPTURE_RAW_L9_SEMANTIC_FEATURES", "0")):
                    semantic_keys = vision_keys
                    if int(semantic_keys.shape[0]) != int(text_queries.shape[0]):
                        if int(semantic_keys.shape[0]) <= 0 or int(text_queries.shape[0]) % int(semantic_keys.shape[0]) != 0:
                            raise RuntimeError("raw L9 semantic capture has incompatible query/KV heads")
                        semantic_keys = semantic_keys.repeat_interleave(
                            int(text_queries.shape[0]) // int(semantic_keys.shape[0]), dim=0
                        )
                    semantic_rows = semantic_keys.permute(1, 0, 2).reshape(int(vision_idx.numel()), -1).detach().float()
                    semantic_rows = semantic_rows / semantic_rows.norm(dim=1, keepdim=True).clamp_min(1e-8)
                    semantic_context = torch.stack((
                        text_queries[:, -1, :].reshape(-1),
                        text_queries.mean(dim=1).reshape(-1),
                    ), dim=0).detach().float()
                    semantic_context = semantic_context / semantic_context.norm(dim=1, keepdim=True).clamp_min(1e-8)
                    raw_semantic_features = {
                        "vision_key_rows": semantic_rows.to(dtype=torch.float16, device="cpu"),
                        "text_last_mean_rows": semantic_context.to(dtype=torch.float16, device="cpu"),
                    }
                    self.record("vision_raw_l9_semantic_feature_events", 1)
                    self.record("vision_raw_l9_semantic_feature_rows", int(semantic_rows.shape[0]))
                self._captured_vision_selections.append({
                    "mode": mode,
                    "source_layer": int(self._effective_vision_layer() if source_layer is None else source_layer),
                    "candidate_positions": [int(x) for x in vision_idx.detach().cpu().tolist()],
                    "keep_local_indices": [int(x) for x in keep_local.detach().cpu().tolist()],
                    "anchor_local_indices": anchor_local,
                    "original_candidate_local_indices": original_candidate_local,
                    "keep_original_local_indices": original_keep_local,
                    "anchor_original_local_indices": original_anchor_local,
                    "encoder_topk_local_indices": _top_indices(score_encoder_for_analysis),
                    "text_topk_local_indices": _top_indices(score_text),
                    "pmi_topk_local_indices": _top_indices(score_pmi_for_analysis),
                    "score_boundary_margin": margin,
                    "features": feature_summary,
                    "alpha_set_features": alpha_set_features,
                    "anchor_quota_token_features": anchor_quota_token_features,
                    "raw_l9_semantic_features": raw_semantic_features,
                })
            if self._vision_keep is None:
                self._vision_keep = torch.zeros_like(drop)
            self._vision_keep[b, vision_idx[keep]] = True
            keep_abs = vision_idx[keep].long()
            drop_abs = vision_idx[~keep].long()
            skip_runtime_diagnostics = bool(getattr(self, "_runtime_exact_fast_path", False))
            if (
                not skip_runtime_diagnostics
                and self._anchor_regional_current_is_anchor is not None
                and int(self._anchor_regional_current_is_anchor.numel()) == int(keep.numel())
                and mode not in {
                    "anchor_regional", "encoder_anchor_regional", "regional_reserve",
                    "anchor_regional_cls", "encoder_anchor_regional_cls",
                }
            ):
                labels = self._anchor_regional_current_is_anchor.to(device=keep.device).bool()
                self.record("vision_anchor_regional_final_events", 1)
                self.record("vision_anchor_regional_final_anchors_kept", int((keep & labels).sum().item()))
                self.record("vision_anchor_regional_final_reserves_kept", int((keep & ~labels).sum().item()))
                self.record("vision_anchor_regional_final_anchors_dropped", int((~keep & labels).sum().item()))
                self.record("vision_anchor_regional_final_reserves_dropped", int((~keep & ~labels).sum().item()))
                self._debug(
                    f"anchor_regional_final mode={mode} "
                    f"anchors_kept={int((keep & labels).sum().item())} "
                    f"reserves_kept={int((keep & ~labels).sum().item())} "
                    f"anchors_dropped={int((~keep & labels).sum().item())} "
                    f"reserves_dropped={int((~keep & ~labels).sum().item())}"
                )
            if (
                not skip_runtime_diagnostics
                and self._feature_reserve_current_is_anchor is not None
                and int(self._feature_reserve_current_is_anchor.numel()) == int(keep.numel())
            ):
                labels = self._feature_reserve_current_is_anchor.to(device=keep.device).bool()
                self.record("vision_feature_reserve_final_events", 1)
                self.record("vision_feature_reserve_final_anchors_kept", int((keep & labels).sum().item()))
                self.record("vision_feature_reserve_final_reserves_kept", int((keep & ~labels).sum().item()))
                self.record("vision_feature_reserve_final_anchors_dropped", int((~keep & labels).sum().item()))
                self.record("vision_feature_reserve_final_reserves_dropped", int((~keep & ~labels).sum().item()))
            if (
                self.deferred_vision_rescore_enabled()
                and not self._vision_mid_physical_compacted
                and score_encoder_for_analysis is not None
                and score_pmi_for_analysis is not None
            ):
                enc_reserve = self._normalize_score(score_encoder_for_analysis).float()
                pmi_reserve = self._normalize_score(score_pmi_for_analysis).float()
                union_local = keep.clone()
                base_alpha = max(
                    0.0,
                    min(1.0, 1.0 - float(getattr(self.config, "vision_score_lambda", 1.0) or 0.0)),
                )
                reserve_lo = max(0.0, base_alpha - 0.25)
                reserve_hi = min(1.0, base_alpha + 0.25)
                reserve_grid = (reserve_lo, 0.5 * (reserve_lo + reserve_hi), reserve_hi)
                for reserve_alpha in reserve_grid:
                    reserve_score = enc_reserve + (1.0 - reserve_alpha) * pmi_reserve
                    reserve_order = torch.argsort(reserve_score, descending=True, stable=True)[:keep_n]
                    union_local[reserve_order] = True
                reserve_prompt = torch.zeros_like(drop)
                reserve_prompt[b, vision_idx[union_local]] = True
                self._vision_alpha_reserve_keep = reserve_prompt.detach()
                self.record("vision_deferred_alpha_union_tokens", int(union_local.sum().item()))
                self.record("vision_deferred_alpha_union_support", int(union_local.sum().item()) - keep_n)
                self.record("vision_deferred_alpha_union_base_x1000", int(round(base_alpha * 1000.0)))
                self.record("vision_deferred_alpha_union_span_x1000", int(round((reserve_hi - reserve_lo) * 1000.0)))
            # Numeric signatures are audit telemetry and force device syncs.
            # Selection capture already records exact token sets when requested.
            keep_sum = keep_sqsum = drop_sum = drop_sqsum = 0
            if not skip_runtime_diagnostics:
                keep_sum = int(keep_abs.sum().item()) if keep_abs.numel() > 0 else 0
                keep_sqsum = int((keep_abs * keep_abs).sum().item()) if keep_abs.numel() > 0 else 0
                drop_sum = int(drop_abs.sum().item()) if drop_abs.numel() > 0 else 0
                drop_sqsum = int((drop_abs * drop_abs).sum().item()) if drop_abs.numel() > 0 else 0
                self.record("vision_keep_signature_sum", keep_sum)
                self.record("vision_keep_signature_sqsum", keep_sqsum)
                self.record("vision_drop_signature_sum", drop_sum)
                self.record("vision_drop_signature_sqsum", drop_sqsum)
            self._record_vision_score_analysis(
                source_layer=int(self._effective_vision_layer() if source_layer is None else source_layer),
                batch_idx=b,
                mode=mode,
                vision_idx=vision_idx,
                text_idx=text_idx,
                keep_n=keep_n,
                score_final=score,
                score_text=score_text,
                score_encoder=score_encoder_for_analysis,
                score_pmi=score_pmi_for_analysis,
                keep_mask=keep,
                effective_lambda=effective_lambda,
                adaptive_metrics=adaptive_metrics,
            )
            if self.config.debug:
                first_keep = keep_abs[:8].detach().cpu().tolist()
                first_drop = drop_abs[:8].detach().cpu().tolist()
                self._debug(
                    f"vision_keep_set b={b} mode={mode} text_queries={int(text_idx.numel())} "
                    f"keep_n={keep_n} keep_sum={keep_sum} keep_sqsum={keep_sqsum} "
                    f"drop_sum={drop_sum} drop_sqsum={drop_sqsum} "
                    f"first_keep={first_keep} first_drop={first_drop}"
                )
            drop[b, drop_abs] = True
        if stage_budget_mode is None and self.deferred_vision_drop_enabled():
            self._vision_last_scores = tuple(last_scores)
        return drop.detach()


    def _aligned_encoder_score_for_vision_idx(self, batch_idx: int, vision_idx: torch.Tensor, device: torch.device) -> Optional[torch.Tensor]:
        if batch_idx >= len(self._vision_by_batch):
            return self._aligned_encoder_score(int(vision_idx.numel()), device)
        orig_idx = self._vision_by_batch[batch_idx].to(device=vision_idx.device)
        full = self._aligned_encoder_score(int(orig_idx.numel()), device)
        if full is None:
            return None
        same_storage = (
            int(orig_idx.numel()) == int(vision_idx.numel())
            and orig_idx.device == vision_idx.device
            and orig_idx.data_ptr() == vision_idx.data_ptr()
            and orig_idx.storage_offset() == vision_idx.storage_offset()
        )
        if same_storage or (
            not bool(getattr(self, "_runtime_exact_fast_path", False))
            and int(orig_idx.numel()) == int(vision_idx.numel())
            and torch.equal(orig_idx.long(), vision_idx.long())
        ):
            return full
        local = torch.searchsorted(orig_idx.long(), vision_idx.long()).clamp(0, max(0, int(orig_idx.numel()) - 1))
        valid = orig_idx[local].long() == vision_idx.long()
        if bool(valid.all().item()):
            self.record("vision_encoder_score_subset", 1)
            return full[local.to(device=full.device)]
        self.record("vision_encoder_score_subset_fallback", 1)
        return self._aligned_encoder_score(int(vision_idx.numel()), device)

    def _aligned_encoder_score(self, target_len: int, device: torch.device) -> Optional[torch.Tensor]:
        if self._vision_encoder_score is None or target_len <= 0:
            return None
        score = self._vision_encoder_score.to(device=device).float().flatten()
        if score.numel() == target_len:
            return score
        if score.numel() <= 1:
            return score.repeat(target_len)[:target_len]
        # Qwen2.5-VL and LLaVA-OV/Rice both merge 2x2 visual patches into one
        # LLM-side vision token.  If the encoder score length is an exact
        # multiple of the LLM vision-token length, preserve the real grouping
        # by pooling consecutive patch scores instead of interpolating over the
        # whole sequence.  Interpolation shifts group boundaries and is
        # especially harmful for OV, which inserts/removes a visual CLS token.
        if score.numel() > target_len and score.numel() % int(target_len) == 0:
            group = int(score.numel() // int(target_len))
            pooled = score.view(int(target_len), group).mean(dim=1)
            self.record("vision_encoder_score_pooled", 1)
            self.record("vision_encoder_score_pool_group", group)
            self.record("vision_encoder_score_source_tokens", int(score.numel()))
            self.record("vision_encoder_score_target_tokens", int(target_len))
            return pooled
        x = score.view(1, 1, -1)
        y = torch.nn.functional.interpolate(x, size=int(target_len), mode="linear", align_corners=False).view(-1)
        self.record("vision_encoder_score_resampled", 1)
        self.record("vision_encoder_score_source_tokens", int(score.numel()))
        self.record("vision_encoder_score_target_tokens", int(target_len))
        return y

    @staticmethod
    def _select_encoder_anchor_quota(
        score_encoder: torch.Tensor,
        score_joint: torch.Tensor,
        keep_n: int,
        alpha: float,
        score_text: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, int]:
        """Protect ceil(alpha*K) encoder anchors, then fill by joint score.

        The alpha=0 branch intentionally calls the same torch.topk expression as
        the frozen joint top-K path, preserving exact endpoint behavior.
        """
        encoder = score_encoder.detach().float().flatten()
        joint = score_joint.detach().float().flatten()
        if encoder.numel() != joint.numel():
            raise ValueError(
                f"anchor quota score length mismatch: encoder={encoder.numel()} joint={joint.numel()}"
            )
        n = int(joint.numel())
        k = max(0, min(int(keep_n), n))
        a = float(alpha)
        if not math.isfinite(a) or a < 0.0 or a > 1.0:
            raise ValueError(f"vision_anchor_alpha must be finite and in [0, 1], got {alpha!r}")
        if k == 0:
            return torch.empty(0, device=joint.device, dtype=torch.long), 0
        if a == 0.0:
            return torch.topk(joint, k=k, largest=True).indices, 0
        anchor_n = min(k, int(math.ceil(a * float(k))))
        encoder_order = torch.argsort(encoder, descending=True, stable=True)
        locked = encoder_order[:anchor_n]
        if anchor_n == k:
            return locked, anchor_n
        available = torch.ones(n, device=joint.device, dtype=torch.bool)
        available[locked] = False
        remaining = torch.nonzero(available, as_tuple=False).flatten()
        fill_order = torch.argsort(joint.index_select(0, remaining), descending=True, stable=True)
        filled = remaining.index_select(0, fill_order[: k - anchor_n])
        return torch.cat([locked, filled]), anchor_n

    def _normalize_score(self, score: torch.Tensor) -> torch.Tensor:
        vals = score.float()
        mean = vals.mean().clamp_min(1e-8) if vals.numel() > 0 else torch.tensor(1.0, device=score.device)
        return vals / mean

    def _prob_dist(self, score: torch.Tensor) -> torch.Tensor:
        vals = score.detach().float().clamp_min(0.0)
        if vals.numel() == 0:
            return vals
        total = vals.sum()
        # Keep the zero/non-finite fallback on device.  The previous Python
        # branch forced a GPU synchronization for every normalization.
        normalized = vals / total.clamp_min(1e-12)
        uniform = torch.full_like(vals, 1.0 / float(max(1, vals.numel())))
        valid = torch.isfinite(total) & total.gt(0.0)
        return torch.where(valid, normalized, uniform)

    def _pmi_like_score(self, text_score: torch.Tensor, prior_score: torch.Tensor) -> torch.Tensor:
        # Lightweight text-conditioned informativeness proxy:
        # p(i|T) log p(i|T)/p0(i), where p0 is the encoder visual prior.
        # Negative PMI is clipped because we use this term only as a boost for
        # tokens that are more text-relevant than their image-intrinsic prior.
        p_text = self._prob_dist(text_score)
        p0 = self._prob_dist(prior_score).to(device=p_text.device)
        eps = 1e-12
        if p_text.numel() != p0.numel() or p_text.numel() == 0:
            self.record("vision_pmi_shape_mismatch", 1)
            return torch.zeros_like(text_score.float())
        pmi = p_text * ((p_text + eps) / (p0 + eps)).log()
        pmi = pmi.clamp_min(0.0)
        if not bool(getattr(self, "_runtime_exact_fast_path", False)):
            if pmi.sum().item() <= 0.0:
                self.record("vision_pmi_all_nonpositive", 1)
            self.record("vision_pmi_positive", int((pmi > 0).sum().item()))
        self.record("vision_pmi_used", 1)
        return pmi

    def _vision_adaptive_lambda(
        self,
        mode: str,
        score_text: torch.Tensor,
        score_encoder: torch.Tensor,
        score_pmi: torch.Tensor,
        keep_n: Optional[int] = None,
    ) -> Tuple[float, dict[str, float]]:
        """Return a parameter-free PMI coefficient from canonical distances."""
        q = self._prob_dist(score_text)
        p = self._prob_dist(score_encoder).to(device=q.device)
        if p.numel() != q.numel() or p.numel() == 0:
            self.record("vision_adaptive_shape_mismatch", 1)
            return 0.0, {}

        eps = 1e-12
        m = 0.5 * (p + q)
        kl_pm = (p * ((p + eps) / (m + eps)).log()).sum()
        kl_qm = (q * ((q + eps) / (m + eps)).log()).sum()
        jsd = 0.5 * (kl_pm + kl_qm)
        jsd_norm = float((jsd / math.log(2.0)).clamp(0.0, 1.0).item())
        js_distance = math.sqrt(jsd_norm)
        tv = float((0.5 * (p - q).abs().sum()).clamp(0.0, 1.0).item())
        selectable = torch.ones_like(score_pmi, dtype=torch.bool)
        concentration = self._score_concentration(score_pmi, selectable)
        encoder_concentration = self._score_concentration(score_encoder, selectable)

        mode = str(mode).lower()
        topk_consensus = None
        if "topk_consensus" in mode:
            n = int(score_text.numel())
            k = max(1, min(int(keep_n if keep_n is not None else n), n))
            top_text = torch.topk(score_text.float(), k=k, largest=True).indices
            top_pmi = torch.topk(score_pmi.float(), k=k, largest=True).indices
            top_text_mask = torch.zeros(n, dtype=torch.bool, device=score_text.device)
            top_text_mask[top_text] = True
            intersection = int(top_text_mask[top_pmi].sum().item())
            union = 2 * k - intersection
            topk_consensus = float(intersection / union) if union > 0 else 1.0

        relative_margin = None
        encoder_margin = None
        pmi_margin = None
        if "rel_margin" in mode:
            n = int(score_pmi.numel())
            k = max(1, min(int(keep_n if keep_n is not None else n), n))

            def probability_margin(score: torch.Tensor) -> float:
                prob = self._prob_dist(score)
                if k >= n or n <= 1:
                    return 0.0
                top_result = torch.topk(prob, k=k, largest=True)
                rest_mask = torch.ones(n, dtype=torch.bool, device=prob.device)
                rest_mask[top_result.indices] = False
                rest = prob[rest_mask]
                if rest.numel() == 0:
                    return 0.0
                return max(0.0, float((top_result.values.mean() - rest.mean()).item()))

            encoder_margin = probability_margin(score_encoder)
            pmi_margin = probability_margin(score_pmi)
            margin_total = pmi_margin + encoder_margin
            relative_margin = pmi_margin / margin_total if margin_total > eps else 0.0

        if "topk_consensus" in mode:
            rule = "topk_consensus"
            lam = topk_consensus
        elif "rel_conf" in mode:
            rule = "rel_conf"
            confidence_total = concentration + encoder_concentration
            lam = concentration / confidence_total if confidence_total > eps else 0.0
        elif "rel_margin" in mode:
            rule = "rel_margin"
            lam = relative_margin
        elif "inv_jsd_sqrt" in mode:
            rule = "inv_jsd_sqrt"
            lam = 1.0 - js_distance
        elif "inv_jsd" in mode:
            rule = "inv_jsd"
            lam = 1.0 - jsd_norm
        elif "jsd_sqrt" in mode:
            rule = "jsd_sqrt"
            lam = js_distance
        elif "jsd" in mode:
            rule = "jsd"
            lam = jsd_norm
        elif mode.endswith("_tv") or "_add_tv" in mode:
            rule = "tv"
            lam = tv
        elif "concentration" in mode:
            rule = "concentration"
            lam = concentration
        else:
            raise ValueError(f"unsupported adaptive vision mode: {mode}")

        lam = max(0.0, min(1.0, float(lam)))
        metrics = {
            "vision_adaptive_jsd_norm": jsd_norm,
            "vision_adaptive_js_distance": js_distance,
            "vision_adaptive_tv": tv,
            "vision_adaptive_pmi_concentration": concentration,
            "vision_adaptive_encoder_concentration": encoder_concentration,
        }
        if topk_consensus is not None:
            metrics["vision_adaptive_topk_consensus"] = topk_consensus
        if relative_margin is not None:
            metrics["vision_adaptive_relative_margin"] = relative_margin
            metrics["vision_adaptive_encoder_margin"] = encoder_margin
            metrics["vision_adaptive_pmi_margin"] = pmi_margin
        self.record("vision_adaptive_lambda_calls", 1)
        self.record("vision_adaptive_lambda_sum", lam)
        self.record("vision_adaptive_lambda_sqsum", lam * lam)
        self.record(f"vision_adaptive_rule_{rule}", 1)
        self.record("vision_adaptive_jsd_norm_sum", jsd_norm)
        self.record("vision_adaptive_js_distance_sum", js_distance)
        self.record("vision_adaptive_tv_sum", tv)
        self.record("vision_adaptive_pmi_concentration_sum", concentration)
        if topk_consensus is not None:
            self.record("vision_adaptive_topk_consensus_sum", topk_consensus)
        self.record("vision_score_lambda_effective_x1000", int(round(lam * 1000.0)))
        self.record("vision_pmi_concentration_x1000", int(round(concentration * 1000.0)))
        return lam, metrics

    def _score_concentration(self, score: torch.Tensor, selectable: torch.Tensor) -> float:
        vals = score[selectable].detach().float().clamp_min(0.0)
        n = int(vals.numel())
        if n <= 1:
            return 0.0
        total = vals.sum()
        if not torch.isfinite(total) or float(total.item()) <= 0.0:
            return 0.0
        prob = vals / total.clamp_min(1e-12)
        entropy = -(prob * (prob + 1e-12).log()).sum() / torch.tensor(float(n), device=vals.device).log().clamp_min(1e-12)
        conc = float((1.0 - entropy).clamp(0.0, 1.0).item())
        self.record("text_entropy_gate_entropy_sum", float(entropy.item()))
        self.record("text_entropy_gate_concentration_sum", conc)
        self.record("text_entropy_gate_calls", 1)
        return conc

    def _select_vision_keep_feature_diverse(
        self,
        score: torch.Tensor,
        features: torch.Tensor,
        keep_n: int,
        anchor_n: int,
    ) -> torch.Tensor:
        """Select salient anchors plus parameter-free feature-novel reserves."""
        values = score.detach().float().flatten()
        if features.ndim != 2 or int(features.shape[0]) != int(values.numel()):
            raise RuntimeError(
                f"feature-diverse shape mismatch: score={tuple(values.shape)} features={tuple(features.shape)}"
            )
        total = int(values.numel())
        if total <= 0:
            return torch.empty((0,), device=score.device, dtype=torch.long)
        keep_n = max(1, min(int(keep_n), total))
        anchor_n = max(1, min(int(anchor_n), keep_n, total))
        reserve_n = keep_n - anchor_n

        if bool(getattr(self, "_feature_diverse_fast_topk", False)):
            anchors = torch.topk(values, k=anchor_n, largest=True, sorted=False).indices
        else:
            order = torch.argsort(values, descending=True, stable=True)
            anchors = order[:anchor_n]
        if reserve_n <= 0:
            self.record("vision_feature_diverse_events", 1)
            self.record("vision_feature_diverse_anchors", anchor_n)
            self.record("vision_feature_diverse_reserves", 0)
            return anchors

        anchor_mask = torch.zeros(total, device=values.device, dtype=torch.bool)
        anchor_mask[anchors] = True
        residual = torch.arange(total, device=values.device, dtype=torch.long)[~anchor_mask]
        if int(residual.numel()) < reserve_n:
            raise RuntimeError(
                f"feature-diverse budget is impossible: residual={int(residual.numel())}, reserve={reserve_n}"
            )

        normalized = torch.nn.functional.normalize(
            features.detach().to(device=values.device, dtype=torch.float32), dim=-1, eps=1e-6
        )
        similarity = normalized.index_select(0, residual) @ normalized.index_select(0, anchors).T
        max_similarity = similarity.max(dim=1).values.clamp(min=-1.0, max=1.0)
        novelty = (1.0 - max_similarity).clamp_min(0.0)
        reserve_score = values.index_select(0, residual).clamp_min(0.0) * novelty
        skip_diagnostics = bool(getattr(self, "_skip_selection_runtime_diagnostics", False))
        if not skip_diagnostics and not bool(torch.isfinite(reserve_score).all().item()):
            raise RuntimeError("feature-diverse reserve score contains non-finite values")
        if bool(getattr(self, "_feature_diverse_fast_topk", False)):
            reserve_local = torch.topk(reserve_score, k=reserve_n, largest=True, sorted=False).indices
        else:
            reserve_local = torch.argsort(reserve_score, descending=True, stable=True)[:reserve_n]
        reserves = residual.index_select(0, reserve_local)
        selected = torch.cat([anchors, reserves], dim=0)
        if not skip_diagnostics:
            if int(selected.numel()) != keep_n or int(torch.unique(selected).numel()) != keep_n:
                raise RuntimeError(
                    f"feature-diverse selection is not exact/unique: expected={keep_n}, "
                    f"actual={int(selected.numel())}, unique={int(torch.unique(selected).numel())}"
                )

        self.record("vision_feature_diverse_events", 1)
        self.record("vision_feature_diverse_candidates", total)
        self.record("vision_feature_diverse_anchors", anchor_n)
        self.record("vision_feature_diverse_reserves", reserve_n)
        self.record("vision_feature_diverse_feature_dim", int(features.shape[-1]))
        self.record("vision_feature_diverse_similarity_pairs", int(residual.numel()) * anchor_n)
        if not skip_diagnostics:
            self.record("vision_feature_diverse_novelty_sum", float(novelty.sum().item()))
            self.record("vision_feature_diverse_reserve_score_sum", float(reserve_score.sum().item()))
            self.record("vision_feature_diverse_anchor_local_sum", int(anchors.sum().item()))
            self.record("vision_feature_diverse_reserve_local_sum", int(reserves.sum().item()))
        return selected

    def _select_vision_keep_anchor_regional(
        self,
        score: torch.Tensor,
        keep_n: int,
        anchor_n: int,
    ) -> torch.Tensor:
        """Keep global anchors plus saliency maxima from ordered residual strata.

        The final target ``anchor_n`` determines the number of globally salient
        anchors.  The depth-coupled surplus ``keep_n - anchor_n`` is filled by
        splitting the non-anchor visual-token sequence into equal-population,
        contiguous strata and taking the highest-saliency token from each.  The
        visual token order is the model's native raster/packed order, so this
        adds broad coverage without pairwise similarity, clustering, merging,
        or a tunable anchor ratio.
        """
        values = score.detach().float().flatten()
        v = int(values.numel())
        if v <= 0:
            return torch.empty((0,), device=score.device, dtype=torch.long)
        keep_n = max(1, min(int(keep_n), v))
        anchor_n = max(1, min(int(anchor_n), keep_n, v))
        reserve_n = keep_n - anchor_n

        order = torch.argsort(values, descending=True, stable=True)
        anchors = order[:anchor_n]
        if reserve_n <= 0:
            self.record("vision_anchor_regional_events", 1)
            self.record("vision_anchor_regional_anchors", int(anchors.numel()))
            self.record("vision_anchor_regional_reserves", 0)
            return anchors

        anchor_mask = torch.zeros((v,), device=values.device, dtype=torch.bool)
        anchor_mask[anchors] = True
        residual = torch.arange(v, device=values.device, dtype=torch.long)[~anchor_mask]
        residual_n = int(residual.numel())
        if residual_n < reserve_n:
            raise RuntimeError(
                f"anchor_regional budget is impossible: residual={residual_n}, reserve={reserve_n}"
            )

        reserves = []
        min_width = residual_n
        max_width = 0
        for region_idx in range(reserve_n):
            start = (region_idx * residual_n) // reserve_n
            end = ((region_idx + 1) * residual_n) // reserve_n
            candidates = residual[start:end]
            width = int(candidates.numel())
            if width <= 0:
                raise RuntimeError(
                    f"anchor_regional produced an empty stratum: region={region_idx}/{reserve_n}"
                )
            local_best = torch.argmax(values[candidates])
            reserves.append(candidates[local_best])
            min_width = min(min_width, width)
            max_width = max(max_width, width)
        reserve = torch.stack(reserves).long()
        selected = torch.cat([anchors, reserve], dim=0)
        if int(selected.numel()) != keep_n or int(torch.unique(selected).numel()) != keep_n:
            raise RuntimeError(
                f"anchor_regional selection is not exact/unique: expected={keep_n}, "
                f"actual={int(selected.numel())}, unique={int(torch.unique(selected).numel())}"
            )

        self.record("vision_anchor_regional_events", 1)
        self.record("vision_anchor_regional_candidates", v)
        self.record("vision_anchor_regional_anchors", anchor_n)
        self.record("vision_anchor_regional_reserves", reserve_n)
        self.record("vision_anchor_regional_region_min_width", min_width)
        self.record("vision_anchor_regional_region_max_width", max_width)
        self.record("vision_anchor_regional_anchor_local_sum", int(anchors.sum().item()))
        self.record("vision_anchor_regional_reserve_local_sum", int(reserve.sum().item()))
        return selected

    def _select_vision_keep_vispruner_features(
        self,
        features: torch.Tensor,
        score: torch.Tensor,
        keep_n: int,
        important_ratio: float,
    ) -> torch.Tensor:
        """Original VisPruner important/diverse rule on pre-LLM visual features.

        This is an analysis-only backbone adaptation of the published selector:
        encoder attention supplies importance and projected visual embeddings
        supply pairwise similarity. It performs no decoder layer and no merge.
        """
        v = int(features.shape[0])
        if int(score.numel()) != v:
            raise RuntimeError(f"pre-LLM VisPruner alignment failed: score={score.numel()} features={v}")
        if keep_n >= v:
            return torch.arange(v, device=features.device)
        keep_n = max(1, min(int(keep_n), v))
        important_n = min(keep_n, max(0, int(float(keep_n) * float(important_ratio))))
        diverse_n = keep_n - important_n
        order = torch.argsort(score.float(), descending=True, stable=True)
        important = order[:important_n]
        residual_indices = order[important_n:].clone()
        if diverse_n <= 0:
            return important
        if int(residual_indices.numel()) <= diverse_n:
            return torch.cat([important, residual_indices], dim=0)[:keep_n]

        feats = features.float()
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        while diverse_n > 0:
            residual_count = int(residual_indices.numel())
            remove_n = min(8, residual_count - diverse_n)
            if remove_n <= 0:
                break
            even = residual_indices[::2]
            odd = residual_indices[1::2]
            pair_n = min(int(even.numel()), int(odd.numel()))
            if pair_n <= 0:
                break
            even = even[:pair_n]
            odd = odd[:pair_n]
            duplicate_score = (feats[even] @ feats[odd].transpose(0, 1)).max(dim=-1).values
            remove_n = min(remove_n, int(duplicate_score.numel()))
            kept_even = even[duplicate_score.argsort(descending=True)[remove_n:]]
            tail = residual_indices[2 * pair_n :]
            residual_indices = torch.cat([kept_even, odd, tail], dim=0)
            if int(residual_indices.numel()) <= diverse_n:
                break
        selected = torch.cat([important, residual_indices[:diverse_n]], dim=0)
        if int(selected.numel()) != keep_n or int(torch.unique(selected).numel()) != keep_n:
            raise RuntimeError(
                f"pre-LLM VisPruner selection is not exact/unique: expected={keep_n}, "
                f"actual={selected.numel()}, unique={torch.unique(selected).numel()}"
            )
        return selected

    def _select_vision_keep_vispruner(
        self,
        key_states_b: torch.Tensor,
        vision_idx: torch.Tensor,
        score: torch.Tensor,
        keep_n: int,
        important_ratio: float,
    ) -> torch.Tensor:
        """Original VisPruner selection rule.

        1) Keep T_imp tokens with the highest visual saliency score.
        2) For the remaining budget, start from score-sorted residual tokens and
           iteratively remove duplicate/similar tokens by pairwise matching.

        This mirrors llava/model/llava_arch.py in the original VisPruner code,
        but uses the current layer's vision hidden states as image_features and
        the selected vision-score mode as image_attentions.
        """
        v = int(vision_idx.numel())
        if keep_n >= v:
            return torch.arange(v, device=vision_idx.device)
        keep_n = max(1, min(int(keep_n), v))
        important_n = min(keep_n, max(0, int(round(float(keep_n) * float(important_ratio)))))
        diverse_n = keep_n - important_n

        order = torch.argsort(score.float(), descending=True)
        important = order[:important_n]
        residual = order[important_n:]

        if diverse_n <= 0:
            return important
        if residual.numel() <= diverse_n:
            return torch.cat([important, residual], dim=0)[:keep_n]

        feats = key_states_b[:, vision_idx, :].float().mean(dim=0)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        residual_indices = residual.clone()

        while diverse_n > 0:
            R = int(residual_indices.numel())
            r = min(8, R - diverse_n)
            if r <= 0:
                break
            # Original VisPruner pairs even/odd residual streams and drops the
            # most duplicate even-side tokens according to max similarity.
            even = residual_indices[::2]
            odd = residual_indices[1::2]
            pair_n = min(int(even.numel()), int(odd.numel()))
            if pair_n <= 0:
                break
            even = even[:pair_n]
            odd = odd[:pair_n]
            scores_pair = feats[even] @ feats[odd].transpose(0, 1)
            duplicate_score = scores_pair.max(dim=-1).values
            r = min(r, int(duplicate_score.numel()))
            keep_even_local = duplicate_score.argsort(descending=True)[r:]
            kept_even = even[keep_even_local]
            tail = residual_indices[2 * pair_n :]
            residual_indices = torch.cat([kept_even, odd, tail], dim=0)
            if residual_indices.numel() <= diverse_n:
                break

        selected = torch.cat([important, residual_indices[:diverse_n]], dim=0)
        return selected[:keep_n]

    def _select_vision_keep(
        self,
        key_states_b: torch.Tensor,
        vision_idx: torch.Tensor,
        score: torch.Tensor,
        keep_n: int,
        important_ratio: float,
    ) -> torch.Tensor:
        v = int(vision_idx.numel())
        if keep_n >= v:
            return torch.arange(v, device=vision_idx.device)
        imp_n = min(keep_n, max(0, int(round(keep_n * important_ratio))))
        selected = []
        if imp_n > 0:
            selected.extend(torch.topk(score, k=imp_n, largest=True).indices.tolist())
        div_n = keep_n - len(selected)
        if div_n <= 0:
            return torch.tensor(selected, device=vision_idx.device, dtype=torch.long)

        feats = key_states_b[:, vision_idx, :].float().mean(dim=0)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        remaining = torch.ones(v, device=vision_idx.device, dtype=torch.bool)
        if selected:
            remaining[torch.tensor(selected, device=vision_idx.device)] = False
            sel = feats[torch.tensor(selected, device=vision_idx.device)]
            min_dist = (1.0 - feats @ sel.T).amin(dim=1)
        else:
            first = int(torch.argmax(score).item())
            selected.append(first)
            remaining[first] = False
            min_dist = 1.0 - feats @ feats[first : first + 1].T.squeeze(-1)
            div_n -= 1

        for _ in range(div_n):
            cand = torch.where(remaining, min_dist, torch.full_like(min_dist, -1.0))
            nxt = int(torch.argmax(cand).item())
            selected.append(nxt)
            remaining[nxt] = False
            dist = 1.0 - feats @ feats[nxt : nxt + 1].T.squeeze(-1)
            min_dist = torch.minimum(min_dist, dist)
        return torch.tensor(selected, device=vision_idx.device, dtype=torch.long)

    def _query_to_key_mass_scores(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        query_idx: torch.Tensor,
        target_key_idx: torch.Tensor,
    ) -> torch.Tensor:
        device = query_states.device
        query_idx = query_idx.to(device=device, dtype=torch.long)
        target_key_idx = target_key_idx.to(device=device, dtype=torch.long)
        if query_idx.numel() == 0 or target_key_idx.numel() == 0:
            return torch.zeros((query_idx.numel(),), device=device)
        key_len = int(key_states.shape[2])
        all_key_idx = torch.arange(key_len, device=device, dtype=torch.long)
        k = key_states[0].float()
        n_heads = int(query_states.shape[1])
        if k.shape[0] != n_heads:
            repeat = max(1, n_heads // int(k.shape[0]))
            k = k.repeat_interleave(repeat, dim=0)
        k_all = k[:, all_key_idx, :]
        scale = float(query_states.shape[-1]) ** 0.5
        out = torch.zeros((query_idx.numel(),), device=device, dtype=torch.float32)
        try:
            chunk_size = int(os.environ.get("VISPRUNER_SCORE_QUERY_CHUNK", "128"))
        except Exception:
            chunk_size = 128
        chunk_size = max(1, min(chunk_size, int(query_idx.numel())))
        mask_4d = None
        active_key_2d = None
        if attention_mask is not None:
            if attention_mask.ndim == 4:
                mask_4d = attention_mask[0, 0]
            elif attention_mask.ndim == 2:
                active_key_2d = attention_mask[0, :key_len].bool()
        target_valid = target_key_idx[target_key_idx < key_len]
        if target_valid.numel() == 0:
            return out
        for start in range(0, int(query_idx.numel()), chunk_size):
            qi = query_idx[start : start + chunk_size]
            q = query_states[0, :, qi, :].float()
            logits = torch.einsum("hqd,hkd->hqk", q, k_all) / scale
            visible = qi[:, None] >= all_key_idx[None, :]
            if mask_4d is not None:
                mask_slice = mask_4d[qi][:, :key_len]
                finite = torch.isfinite(mask_slice) & (mask_slice > torch.finfo(mask_slice.dtype).min / 2)
                visible = visible & finite
                logits = logits + mask_slice.float().unsqueeze(0)
            elif active_key_2d is not None:
                visible = visible & active_key_2d[None, :]
            logits = logits.masked_fill(~visible.unsqueeze(0), torch.finfo(logits.dtype).min)
            probs = torch.softmax(logits, dim=-1).masked_fill(~visible.unsqueeze(0), 0.0)
            mass = probs[:, :, target_valid].sum(dim=-1).mean(dim=0)
            out[start : start + qi.numel()] = mass
        return out

    def _linearized_causal_utilization_scores(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        query_idx: torch.Tensor,
        key_idx: torch.Tensor,
        batch_idx: int = 0,
    ) -> torch.Tensor:
        """First-order O(Td) approximation of causal received attention.

        Exact text utilization divides each text key's received attention by
        its causal uniform-attention baseline.  Linearizing each query's
        softmax around uniform attention gives

            A_ji ~= (1 / n_j) * (1 + z_ji - mean_m z_jm),

        where the mean is over every active causally visible key.  Prefix key
        means and suffix query sums evaluate the resulting utilization without
        materializing a text-by-sequence logit or attention matrix.  Q/K are
        the already projected, post-RoPE tensors reused by the observer.
        """
        device = query_states.device
        query_idx = query_idx.to(device=device, dtype=torch.long)
        key_idx = key_idx.to(device=device, dtype=torch.long)
        if query_idx.numel() == 0 or key_idx.numel() == 0:
            return torch.zeros((key_idx.numel(),), device=device)
        if query_idx.numel() != key_idx.numel() or not torch.equal(query_idx, key_idx):
            raise RuntimeError(
                "linearized_causal_utilization requires identical ordered text query/key indices"
            )

        key_len = min(int(key_states.shape[2]), int(self.prompt_len))
        if bool(((key_idx < 0) | (key_idx >= key_len)).any().item()):
            raise RuntimeError(
                f"linearized text index outside prompt: key_len={key_len} "
                f"min={int(key_idx.min().item())} max={int(key_idx.max().item())}"
            )

        q = query_states[0, :, query_idx, :].float()
        k_all = key_states[0, :, :key_len, :].float()
        if int(k_all.shape[0]) != int(q.shape[0]):
            if int(q.shape[0]) % int(k_all.shape[0]) != 0:
                raise RuntimeError(
                    f"query/key head mismatch: q={int(q.shape[0])} kv={int(k_all.shape[0])}"
                )
            k_all = k_all.repeat_interleave(
                int(q.shape[0]) // int(k_all.shape[0]), dim=0
            )
        k_text = k_all[:, key_idx, :]

        if (
            self._active is not None
            and 0 <= int(batch_idx) < int(self._active.shape[0])
            and int(self._active.shape[1]) >= key_len
        ):
            active = self._active[int(batch_idx), :key_len].to(device=device).bool()
        else:
            active = torch.ones((key_len,), device=device, dtype=torch.bool)
        # Text indices originate from active prompt positions.  Treat a mismatch
        # as a state error rather than silently changing the causal baseline.
        if not bool(active[key_idx].all().item()):
            raise RuntimeError("linearized utilization received an inactive text index")

        active_f = active.to(dtype=k_all.dtype)
        visible_count_all = torch.cumsum(active_f, dim=0).clamp_min(1.0)
        visible_count = visible_count_all[key_idx]
        inverse_visible = visible_count.reciprocal()

        prefix_key_sum = torch.cumsum(
            k_all * active_f.view(1, -1, 1), dim=1
        )
        visible_key_mean = prefix_key_sum[:, key_idx, :] / visible_count.view(1, -1, 1)
        scale = float(q.shape[-1]) ** 0.5
        mean_logit = (q * visible_key_mean).sum(dim=-1) / scale

        weighted_q = q * inverse_visible.view(1, -1, 1)
        suffix_q = torch.flip(
            torch.cumsum(torch.flip(weighted_q, dims=(1,)), dim=1), dims=(1,)
        )
        weighted_mean_logit = mean_logit * inverse_visible.view(1, -1)
        suffix_mean_logit = torch.flip(
            torch.cumsum(torch.flip(weighted_mean_logit, dims=(1,)), dim=1),
            dims=(1,),
        )
        baseline = torch.flip(
            torch.cumsum(torch.flip(inverse_visible, dims=(0,)), dim=0), dims=(0,)
        ).clamp_min(1e-8)

        key_suffix_logit = (k_text * suffix_q).sum(dim=-1) / scale
        score_by_head = 1.0 + (key_suffix_logit - suffix_mean_logit) / baseline.view(1, -1)
        score = torch.nan_to_num(
            score_by_head.mean(dim=0), nan=1.0, posinf=1e4, neginf=-1e4
        )

        token_count = int(key_idx.numel())
        heads = int(q.shape[0])
        head_dim = int(q.shape[-1])
        self.record("text_score_linearized_utilization_calls", 1)
        self.record("text_score_linearized_utilization_tokens", token_count)
        self.record(
            "text_score_linearized_utilization_feature_ops",
            token_count * heads * head_dim,
        )
        self.record(
            "text_score_linearized_utilization_negative",
            int((score < 0.0).sum().item()),
        )
        return score

    def _linear_causal_query_scores(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        query_idx: torch.Tensor,
        key_idx: torch.Tensor,
    ) -> torch.Tensor:
        """O(Td) proxy for text-key received attention.

        For each text key, average the projected queries of causally later text
        tokens, then measure key alignment with that suffix-mean query.  The
        exponential score is normalized to mean one so the existing relative
        threshold semantics remain valid.  This deliberately avoids constructing
        a T-by-sequence logit matrix or running an extra attention softmax.
        """
        device = query_states.device
        query_idx = query_idx.to(device=device, dtype=torch.long)
        key_idx = key_idx.to(device=device, dtype=torch.long)
        key_len = int(key_states.shape[2])
        key_idx = key_idx[(key_idx >= 0) & (key_idx < key_len)]
        if query_idx.numel() == 0 or key_idx.numel() == 0:
            return torch.zeros((key_idx.numel(),), device=device)
        if query_idx.numel() != key_idx.numel() or not torch.equal(query_idx, key_idx):
            raise RuntimeError("linear_causal_query requires identical ordered text query/key indices")

        q = query_states[0, :, query_idx, :].float()
        k = key_states[0, :, key_idx, :].float()
        if int(k.shape[0]) != int(q.shape[0]):
            if int(q.shape[0]) % int(k.shape[0]) != 0:
                raise RuntimeError(f"query/key head mismatch: q={int(q.shape[0])} kv={int(k.shape[0])}")
            k = k.repeat_interleave(int(q.shape[0]) // int(k.shape[0]), dim=0)

        suffix_sum = torch.flip(torch.cumsum(torch.flip(q, dims=(1,)), dim=1), dims=(1,))
        suffix_count = torch.arange(
            int(q.shape[1]), 0, -1, device=device, dtype=q.dtype
        ).view(1, -1, 1)
        suffix_mean = suffix_sum / suffix_count
        logits = (suffix_mean * k).sum(dim=-1).mean(dim=0) / (float(q.shape[-1]) ** 0.5)
        logits = (logits - logits.mean()).clamp(min=-12.0, max=12.0)
        score = torch.exp(logits)
        score = score / score.mean().clamp_min(1e-8)

        token_count = int(query_idx.numel())
        self.record("text_score_linear_causal_calls", 1)
        self.record("text_score_linear_causal_tokens", token_count)
        self.record("text_score_linear_causal_feature_ops", token_count * int(q.shape[-1]) * int(q.shape[0]))
        return score

    def _deterministic_random_text_scores(
        self,
        key_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Layer-varying deterministic random ranking control."""
        device = key_idx.device
        call_idx = int(self.stats.get("text_score_random_control_calls", 0.0))
        seed = int(os.environ.get("VISPRUNER_TEXT_RANDOM_SEED", "0") or 0)
        x = key_idx.to(device=device, dtype=torch.int64)
        x = (
            x * 1103515245
            + int(self._analysis_prompt_idx) * 12345
            + call_idx * 2654435761
            + seed * 1013904223
        ) & 0x7FFFFFFF
        score = (x.float() + 1.0) / 2147483648.0
        self.record("text_score_random_control_calls", 1)
        self.record("text_score_random_control_tokens", int(key_idx.numel()))
        self.record(f"text_score_random_seed_{seed}_calls", 1)
        return score

    def _text_update_functional_signatures(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        query_idx: torch.Tensor,
        key_idx: torch.Tensor,
        output_projection,
    ) -> torch.Tensor:
        """Sum each visual key's exact attention-value write over text queries.

        For token i, the pre-projection vector is the concatenation across
        heads of ``sum_q A[h,q,i] * V[h,i]``. Applying the native attention
        output projection gives one model-space functional signature per
        visual token without an extra model forward.
        """
        if value_states is None or value_states.ndim != 4:
            raise RuntimeError("functional coreset requires native value states")
        if output_projection is None:
            raise RuntimeError("functional coreset requires the native attention output projection")
        device = query_states.device
        query_idx = query_idx.to(device=device, dtype=torch.long)
        key_idx = key_idx.to(device=device, dtype=torch.long)
        if query_idx.numel() == 0 or key_idx.numel() == 0:
            return torch.zeros((int(key_idx.numel()), 1), device=device, dtype=torch.float32)
        q = query_states[0].float()
        k = key_states[0].float()
        v = value_states[0].float()
        n_heads = int(q.shape[0])
        if int(k.shape[0]) != n_heads:
            if n_heads % int(k.shape[0]) != 0 or int(v.shape[0]) != int(k.shape[0]):
                raise RuntimeError(
                    f"functional Q/K/V head mismatch q={n_heads} k={int(k.shape[0])} v={int(v.shape[0])}"
                )
            repeat = n_heads // int(k.shape[0])
            k = k.repeat_interleave(repeat, dim=0)
            v = v.repeat_interleave(repeat, dim=0)
        key_len = min(int(k.shape[1]), int(v.shape[1]))
        query_idx = query_idx[(query_idx >= 0) & (query_idx < int(q.shape[1]))]
        key_idx = key_idx[(key_idx >= 0) & (key_idx < key_len)]
        mass = torch.zeros((n_heads, int(key_idx.numel())), device=device, dtype=torch.float32)
        all_keys = torch.arange(key_len, device=device)
        chunk_size = max(1, int(os.environ.get("VISPRUNER_FUNCTIONAL_QUERY_CHUNK", "32") or 32))
        scale = 1.0 / math.sqrt(float(q.shape[-1]))
        for start in range(0, int(query_idx.numel()), chunk_size):
            qpos = query_idx[start : start + chunk_size]
            qchunk = q.index_select(1, qpos)
            logits = torch.einsum("hqd,hkd->hqk", qchunk, k[:, :key_len]) * scale
            visible = all_keys.view(1, -1) <= qpos.view(-1, 1)
            if torch.is_tensor(attention_mask):
                if attention_mask.ndim == 4:
                    rows = attention_mask[0, 0].index_select(0, qpos)[:, :key_len].float()
                    finite = torch.isfinite(rows) & (rows > torch.finfo(rows.dtype).min / 2)
                    visible &= finite
                    logits = logits + rows.unsqueeze(0)
                elif attention_mask.ndim == 2:
                    visible &= attention_mask[0, :key_len].to(device=device).bool().unsqueeze(0)
            logits = logits.masked_fill(~visible.unsqueeze(0), torch.finfo(logits.dtype).min)
            probs = torch.softmax(logits, dim=-1).masked_fill(~visible.unsqueeze(0), 0.0)
            mass += probs.index_select(2, key_idx).sum(dim=1)
        mass = mass / float(max(1, int(query_idx.numel())))
        selected_v = v[:, :key_len].index_select(1, key_idx)
        pre_projection = (mass.unsqueeze(-1) * selected_v).permute(1, 0, 2).reshape(
            int(key_idx.numel()), -1
        )
        try:
            projection_dtype = next(output_projection.parameters()).dtype
        except (StopIteration, AttributeError):
            projection_dtype = value_states.dtype
        projected = output_projection(pre_projection.to(dtype=projection_dtype)).float()
        self.record("vision_functional_signature_events", 1)
        self.record("vision_functional_signature_tokens", int(key_idx.numel()))
        self.record("vision_functional_signature_queries", int(query_idx.numel()))
        self.record("vision_functional_signature_dims", int(projected.shape[-1]))
        self.record("vision_functional_signature_norm_sum", float(projected.norm(dim=-1).sum().item()))
        return projected.detach()

    def _text_update_role_resolved_functional_signatures(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        query_idx: torch.Tensor,
        key_idx: torch.Tensor,
        output_projection,
    ) -> torch.Tensor:
        """Preserve user-question and remaining-text visual writes separately.

        The previous functional signature averages every prompt-text query into
        one write vector per visual token.  This opt-in diagnostic keeps two
        exact native attention-value writes instead: user-content queries
        (role 2) and all remaining text queries.  Each group is averaged by its
        own query count, passed through the same native output projection, and
        concatenated.  No task label, GT, extra model forward, or learned head
        is used.
        """
        if value_states is None or value_states.ndim != 4:
            raise RuntimeError("role-resolved functional coreset requires native value states")
        if output_projection is None:
            raise RuntimeError("role-resolved functional coreset requires native output projection")
        if self._role_ids is None or int(self._role_ids.numel()) != self.prompt_len:
            raise RuntimeError("role-resolved functional coreset requires aligned prompt roles")
        device = query_states.device
        query_idx = query_idx.to(device=device, dtype=torch.long)
        key_idx = key_idx.to(device=device, dtype=torch.long)
        if query_idx.numel() == 0 or key_idx.numel() == 0:
            raise RuntimeError("role-resolved functional coreset received an empty query/key set")
        q = query_states[0].float()
        k = key_states[0].float()
        v = value_states[0].float()
        n_heads = int(q.shape[0])
        if int(k.shape[0]) != n_heads:
            if n_heads % int(k.shape[0]) != 0 or int(v.shape[0]) != int(k.shape[0]):
                raise RuntimeError(
                    f"role-resolved functional Q/K/V head mismatch "
                    f"q={n_heads} k={int(k.shape[0])} v={int(v.shape[0])}"
                )
            repeat = n_heads // int(k.shape[0])
            k = k.repeat_interleave(repeat, dim=0)
            v = v.repeat_interleave(repeat, dim=0)
        key_len = min(int(k.shape[1]), int(v.shape[1]))
        query_idx = query_idx[(query_idx >= 0) & (query_idx < int(q.shape[1]))]
        key_idx = key_idx[(key_idx >= 0) & (key_idx < key_len)]
        roles = self._role_ids.to(device=device).index_select(0, query_idx)
        user_queries = query_idx[roles.eq(2)]
        other_queries = query_idx[~roles.eq(2)]
        if user_queries.numel() == 0 or other_queries.numel() == 0:
            raise RuntimeError(
                f"role-resolved functional query split is empty: "
                f"user={int(user_queries.numel())} other={int(other_queries.numel())}"
            )

        all_keys = torch.arange(key_len, device=device)
        chunk_size = max(1, int(os.environ.get("VISPRUNER_FUNCTIONAL_QUERY_CHUNK", "32") or 32))
        scale = 1.0 / math.sqrt(float(q.shape[-1]))
        selected_v = v[:, :key_len].index_select(1, key_idx)
        try:
            projection_dtype = next(output_projection.parameters()).dtype
        except (StopIteration, AttributeError):
            projection_dtype = value_states.dtype
        projected_groups = []
        for group_queries in (user_queries, other_queries):
            mass = torch.zeros(
                (n_heads, int(key_idx.numel())), device=device, dtype=torch.float32
            )
            for start in range(0, int(group_queries.numel()), chunk_size):
                qpos = group_queries[start : start + chunk_size]
                qchunk = q.index_select(1, qpos)
                logits = torch.einsum("hqd,hkd->hqk", qchunk, k[:, :key_len]) * scale
                visible = all_keys.view(1, -1) <= qpos.view(-1, 1)
                if torch.is_tensor(attention_mask):
                    if attention_mask.ndim == 4:
                        rows = attention_mask[0, 0].index_select(0, qpos)[:, :key_len].float()
                        finite = torch.isfinite(rows) & (rows > torch.finfo(rows.dtype).min / 2)
                        visible &= finite
                        logits = logits + rows.unsqueeze(0)
                    elif attention_mask.ndim == 2:
                        visible &= attention_mask[0, :key_len].to(device=device).bool().unsqueeze(0)
                logits = logits.masked_fill(~visible.unsqueeze(0), torch.finfo(logits.dtype).min)
                probs = torch.softmax(logits, dim=-1).masked_fill(~visible.unsqueeze(0), 0.0)
                mass += probs.index_select(2, key_idx).sum(dim=1)
            mass = mass / float(int(group_queries.numel()))
            pre_projection = (mass.unsqueeze(-1) * selected_v).permute(1, 0, 2).reshape(
                int(key_idx.numel()), -1
            )
            projected_groups.append(
                output_projection(pre_projection.to(dtype=projection_dtype)).float()
            )
        signature = torch.cat(projected_groups, dim=-1)
        self.record("vision_functional_signature_events", 1)
        self.record("vision_functional_signature_tokens", int(key_idx.numel()))
        self.record("vision_functional_signature_queries", int(query_idx.numel()))
        self.record("vision_role_functional_signature_events", 1)
        self.record("vision_role_functional_user_queries", int(user_queries.numel()))
        self.record("vision_role_functional_other_queries", int(other_queries.numel()))
        self.record("vision_role_functional_signature_dims", int(signature.shape[-1]))
        self.record("vision_functional_signature_dims", int(signature.shape[-1]))
        self.record("vision_functional_signature_norm_sum", float(signature.norm(dim=-1).sum().item()))
        return signature.detach()

    def _text_update_user_functional_signatures(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        query_idx: torch.Tensor,
        key_idx: torch.Tensor,
        output_projection,
    ) -> torch.Tensor:
        """Visual-write signatures from user-content queries only.

        This removes system, assistant, and generated-token feedback while
        keeping the same exact native attention-value write and coreset rule.
        Role 2 is the aligned user-content span; no task identity or output
        token is consulted.
        """
        if self._role_ids is None or int(self._role_ids.numel()) != self.prompt_len:
            raise RuntimeError("user functional coreset requires aligned prompt roles")
        device = query_states.device
        query_idx = query_idx.to(device=device, dtype=torch.long)
        valid = query_idx[(query_idx >= 0) & (query_idx < self.prompt_len)]
        if valid.numel() == 0:
            raise RuntimeError("user functional coreset received no aligned text queries")
        roles = self._role_ids.to(device=device).index_select(0, valid)
        user_queries = valid[roles.eq(2)]
        if user_queries.numel() == 0:
            raise RuntimeError("user functional coreset found no user-content queries")
        cache = getattr(self, "_received_user_mass_cache", None)
        use_cache = bool(getattr(self, "_functional_mass_reuse_enabled", False)) and isinstance(cache, dict)
        if use_cache:
            if value_states is None or value_states.ndim != 4 or output_projection is None:
                raise RuntimeError("cached functional signature requires native values and output projection")
            if int(cache.get("key_count", -1)) != int(key_idx.numel()) or int(cache.get("user_queries", -1)) != int(user_queries.numel()):
                raise RuntimeError("cached functional mass alignment mismatch")
            mass = cache["mass"].to(device=device, dtype=torch.float32)
            v = value_states[0].float()
            if int(v.shape[0]) != int(mass.shape[0]):
                if int(mass.shape[0]) % int(v.shape[0]) != 0:
                    raise RuntimeError("cached functional mass/value head mismatch")
                v = v.repeat_interleave(int(mass.shape[0]) // int(v.shape[0]), dim=0)
            key_len = min(int(cache["key_len"]), int(v.shape[1]))
            local = key_idx.to(device=device, dtype=torch.long)
            if int(local.numel()) != int(mass.shape[1]):
                raise RuntimeError("cached functional mass/key count mismatch")
            selected_v = v[:, :key_len].index_select(1, local)
            pre_projection = (mass.unsqueeze(-1) * selected_v).permute(1, 0, 2).reshape(int(local.numel()), -1)
            try:
                projection_dtype = next(output_projection.parameters()).dtype
            except (StopIteration, AttributeError):
                projection_dtype = value_states.dtype
            if bool(getattr(self, "_functional_signature_pre_projection", False)):
                signature = pre_projection.float().detach()
                self.record("vision_functional_signature_pre_projection_events", 1)
            else:
                signature = output_projection(pre_projection.to(dtype=projection_dtype)).float().detach()
            self._received_user_mass_cache = None
            self.record("vision_functional_mass_reuse_consumed", 1)
            self.record("vision_functional_signature_events", 1)
            self.record("vision_functional_signature_tokens", int(local.numel()))
            self.record("vision_functional_signature_queries", int(user_queries.numel()))
            self.record("vision_functional_signature_dims", int(signature.shape[-1]))
            if not bool(getattr(self, "_skip_functional_runtime_diagnostics", False)):
                self.record("vision_functional_signature_norm_sum", float(signature.norm(dim=-1).sum().item()))
            else:
                self.record("vision_functional_signature_norm_diagnostics_skipped", 1)
        else:
            signature = self._text_update_functional_signatures(
                query_states, key_states, value_states, attention_mask,
                user_queries, key_idx, output_projection,
            )
        self.record("vision_user_functional_signature_events", 1)
        self.record("vision_user_functional_queries", int(user_queries.numel()))
        self.record("vision_user_functional_signature_dims", int(signature.shape[-1]))
        return signature

    def _functional_continuous_alpha(
        self,
        signatures: torch.Tensor,
        encoder_score: torch.Tensor,
        pmi_score: torch.Tensor,
    ) -> tuple[float, torch.Tensor, float]:
        """Fit a real-valued encoder/PMI blend to the observed visual write.

        Each visual token's functional target is its positive signed alignment
        with the aggregate visual-to-text attention write at the selection
        layer.  The scalar PMI coefficient is the clipped least-squares fit of
        ``encoder + beta * PMI`` to that target after removing score means.
        ``alpha = 1 - beta`` is therefore produced directly without a learned
        head, a candidate-alpha grid, GT, or an extra model forward.
        """
        z = signatures.detach().float()
        if z.ndim != 2 or int(z.shape[0]) == 0:
            raise RuntimeError(f"functional alpha requires [tokens, dims] signatures, got {tuple(z.shape)}")
        if int(encoder_score.numel()) != int(z.shape[0]) or int(pmi_score.numel()) != int(z.shape[0]):
            raise RuntimeError("functional alpha score/signature alignment mismatch")
        total = z.sum(dim=0)
        aligned = z @ total
        target = aligned.clamp_min(0.0)
        if not bool((target > 0).any().item()):
            target = z.norm(dim=-1)
        target = target / target.mean().clamp_min(1e-8)
        enc = self._normalize_score(encoder_score).detach().float()
        pmi = self._normalize_score(pmi_score).detach().float()
        enc_centered = enc - enc.mean()
        pmi_centered = pmi - pmi.mean()
        target_centered = target - target.mean()
        ee = enc_centered.square().sum()
        pp = pmi_centered.square().sum()
        ep = (enc_centered * pmi_centered).sum()
        et = (enc_centered * target_centered).sum()
        pt = (pmi_centered * target_centered).sum()
        determinant = ee * pp - ep.square()
        if not torch.isfinite(determinant) or float(determinant.abs().item()) <= 1e-12:
            beta = torch.zeros((), device=z.device, dtype=torch.float32)
        else:
            encoder_coef = (et * pp - pt * ep) / determinant
            pmi_coef = (pt * ee - et * ep) / determinant
            if not torch.isfinite(encoder_coef) or float(encoder_coef.item()) <= 1e-8:
                beta = torch.ones((), device=z.device, dtype=torch.float32) if float(pmi_coef.item()) > 0.0 else torch.zeros((), device=z.device, dtype=torch.float32)
            else:
                beta = pmi_coef / encoder_coef
                beta = torch.nan_to_num(beta, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        candidate = enc_centered + beta * pmi_centered
        candidate_scale = (candidate * target_centered).sum() / candidate.square().sum().clamp_min(1e-8)
        base_scale = (enc_centered * target_centered).sum() / enc_centered.square().sum().clamp_min(1e-8)
        base_error = (target_centered - base_scale * enc_centered).square().mean()
        fit_error = (target_centered - candidate_scale * candidate).square().mean()
        fit_gain = float(((base_error - fit_error) / base_error.clamp_min(1e-8)).item())
        alpha = float((1.0 - beta).item())
        return alpha, target, fit_gain

    @staticmethod
    def _functional_coverage_score(
        signatures: torch.Tensor,
        selected: torch.Tensor,
    ) -> float:
        """Magnitude-weighted directional coverage of all functional writes."""
        features = torch.nn.functional.normalize(signatures.float(), dim=-1)
        selected = selected.to(device=features.device, dtype=torch.long)
        similarity = features @ features.index_select(0, selected).T
        covered = similarity.max(dim=1).values.clamp(min=0.0, max=1.0)
        weight = signatures.float().norm(dim=-1)
        if not bool((weight > 0).any().item()):
            weight = torch.ones_like(weight)
        return float((covered * weight).sum().div(weight.sum().clamp_min(1e-8)).item())

    def _select_functional_coreset(
        self,
        signatures: torch.Tensor,
        relevance: torch.Tensor,
        keep_n: int,
    ) -> torch.Tensor:
        """Greedy relevance-weighted coverage in visual-to-text update space."""
        count = int(signatures.shape[0])
        keep_n = max(1, min(int(keep_n), count))
        if keep_n >= count:
            return torch.arange(count, device=signatures.device, dtype=torch.long)
        features = torch.nn.functional.normalize(signatures.float(), dim=-1)
        rel = relevance.detach().float().clamp_min(0.0)
        rel = rel / rel.mean().clamp_min(1e-8)
        first = int(torch.argmax(rel).item())
        selected = [first]
        remaining = torch.ones(count, device=features.device, dtype=torch.bool)
        remaining[first] = False
        similarity = features @ features[first]
        min_distance = (1.0 - similarity).clamp_min(0.0)
        for _ in range(keep_n - 1):
            utility = rel * min_distance
            utility = torch.where(remaining, utility, torch.full_like(utility, -1.0))
            if float(utility.max().item()) <= 0.0:
                utility = torch.where(remaining, rel, torch.full_like(rel, -1.0))
            nxt = int(torch.argmax(utility).item())
            selected.append(nxt)
            remaining[nxt] = False
            distance = (1.0 - features @ features[nxt]).clamp_min(0.0)
            min_distance = torch.minimum(min_distance, distance)
        out = torch.tensor(selected, device=signatures.device, dtype=torch.long)
        if int(torch.unique(out).numel()) != keep_n:
            raise RuntimeError("functional coreset selection produced duplicate indices")
        self.record("vision_functional_coreset_select_events", 1)
        self.record("vision_functional_coreset_selected", keep_n)
        return out

    def _select_functional_guarded_coreset(
        self,
        signatures: torch.Tensor,
        relevance: torch.Tensor,
        keep_n: int,
    ) -> torch.Tensor:
        """Accept functional replacements only near the relevance boundary.

        Let q be the number of functional proposals outside the current top-K.
        A proposal is accepted only when it also lies in the current top-(K+q).
        Thus the guard has no tuned score threshold or task/budget branch and an
        empty accepted set reproduces the native current top-K exactly.
        """
        proposal = self._select_functional_coreset(signatures, relevance, keep_n)
        n = int(relevance.numel())
        k = max(1, min(int(keep_n), n))
        current = torch.topk(relevance.float(), k=k, largest=True).indices
        current_mask = torch.zeros(n, device=relevance.device, dtype=torch.bool)
        current_mask[current] = True
        outsiders = proposal[~current_mask.index_select(0, proposal)]
        proposed_n = int(outsiders.numel())
        if proposed_n > 0:
            guard_n = min(n, k + proposed_n)
            guard = torch.topk(relevance.float(), k=guard_n, largest=True).indices
            guard_mask = torch.zeros(n, device=relevance.device, dtype=torch.bool)
            guard_mask[guard] = True
            accepted = outsiders[guard_mask.index_select(0, outsiders)]
        else:
            guard_n = k
            accepted = outsiders
        accepted_n = int(accepted.numel())
        used = torch.zeros(n, device=relevance.device, dtype=torch.bool)
        if accepted_n:
            used[accepted] = True
        fill = current[~used.index_select(0, current)][: k - accepted_n]
        out = torch.cat((accepted, fill), dim=0)
        if int(out.numel()) != k or int(torch.unique(out).numel()) != k:
            raise RuntimeError("functional guard produced an invalid selection")
        self.record("vision_functional_guard_events", 1)
        self.record("vision_functional_guard_proposed", proposed_n)
        self.record("vision_functional_guard_accepted", accepted_n)
        self.record("vision_functional_guard_rejected", proposed_n - accepted_n)
        self.record("vision_functional_guard_window", guard_n)
        return out

    def _last_query_value_contribution_scores(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        query_pos: int,
        key_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Pre-O-projection norm of each key's contribution to one query."""
        if value_states is None or value_states.ndim != 4:
            raise RuntimeError("last-query contribution scoring requires native value states")
        device = query_states.device
        key_idx = key_idx.to(device=device, dtype=torch.long)
        q = query_states[0, :, int(query_pos), :].float()
        k = key_states[0].float()
        v = value_states[0].float()
        n_heads = int(q.shape[0])
        if int(k.shape[0]) != n_heads:
            if n_heads % int(k.shape[0]) != 0 or int(v.shape[0]) != int(k.shape[0]):
                raise RuntimeError(
                    f"contribution Q/K/V head mismatch q={n_heads} k={int(k.shape[0])} v={int(v.shape[0])}"
                )
            repeat = n_heads // int(k.shape[0])
            k = k.repeat_interleave(repeat, dim=0)
            v = v.repeat_interleave(repeat, dim=0)
        key_len = min(int(k.shape[1]), int(v.shape[1]))
        key_idx = key_idx[(key_idx >= 0) & (key_idx < key_len)]
        logits = torch.einsum("hd,hkd->hk", q, k[:, :key_len]) / math.sqrt(float(q.shape[-1]))
        visible = torch.arange(key_len, device=device) <= int(query_pos)
        if torch.is_tensor(attention_mask):
            if attention_mask.ndim == 4:
                row = attention_mask[0, 0, int(query_pos), :key_len].float()
                visible &= torch.isfinite(row) & (row > torch.finfo(row.dtype).min / 2)
                logits = logits + row.unsqueeze(0)
            elif attention_mask.ndim == 2:
                visible &= attention_mask[0, :key_len].to(device=device).bool()
        logits = logits.masked_fill(~visible.unsqueeze(0), torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1).masked_fill(~visible.unsqueeze(0), 0.0)
        selected_w = weights.index_select(1, key_idx)
        selected_v = v[:, :key_len].index_select(1, key_idx)
        per_token = (selected_w.unsqueeze(-1) * selected_v).permute(1, 0, 2).reshape(int(key_idx.numel()), -1)
        self.record("vision_last_query_contribution_scores", int(key_idx.numel()))
        return per_token.norm(dim=-1)

    def _received_scores(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        query_idx: torch.Tensor,
        key_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Received attention for target keys under the full causal softmax.

        Queries and returned targets may both be text tokens, but the softmax
        denominator contains every causally visible key (surviving vision and
        text). This matches the legacy LLaVA masking definition and avoids
        comparing a text-only softmax against a full-sequence uniform baseline.
        """
        # Strip the explicit high-score (inverse) direction here. The caller
        # applies the rank reversal after this exact base score is computed.
        mode, _ = self._text_score_source()
        linear_modes = {"linear_causal_query", "linear_received", "mean_query_alignment"}
        random_modes = {"random_control", "deterministic_random"}
        native_proxy_modes = {
            "causal_qkv_contribution", "causal_qkv_contribution_inverse",
            "last_query_qk", "last_query_qk_inverse",
            "residual_update", "residual_update_inverse",
            "residual_alignment", "residual_alignment_inverse",
            "attention_output_norm", "attention_output_norm_inverse",
            "mlp_output_norm", "mlp_output_norm_inverse",
            "local_redundancy", "local_redundancy_inverse",
            "linear_causal_query_inverse",
            "linearized_causal_utilization",
            "linearized_causal_utilization_inverse",
            "linearized_utilization",
            "first_order_utilization",
        }
        same_ordered_tokens = bool(
            query_idx.numel() == key_idx.numel()
            and torch.equal(query_idx.to(device=key_idx.device), key_idx)
        )
        if mode in linear_modes and same_ordered_tokens:
            return self._linear_causal_query_scores(query_states, key_states, query_idx, key_idx)
        if mode in random_modes and same_ordered_tokens:
            return self._deterministic_random_text_scores(key_idx)
        if mode in linear_modes | random_modes | native_proxy_modes:
            # Vision PMI also uses this generic received-score helper with text
            # queries and vision keys. Preserve its validated exact correction;
            # cheap proxies and controls replace text-self importance only.
            mode = "exposure_baseline_ratio"

        device = query_states.device
        query_idx = query_idx.to(device=device, dtype=torch.long)
        key_idx = key_idx.to(device=device, dtype=torch.long)
        key_len = int(key_states.shape[2])
        key_idx = key_idx[(key_idx >= 0) & (key_idx < key_len)]
        if query_idx.numel() == 0 or key_idx.numel() == 0:
            return torch.zeros((key_idx.numel(),), device=device)

        q_total = int(query_idx.numel())
        k_total = int(key_idx.numel())
        try:
            chunk_size = int(os.environ.get("VISPRUNER_SCORE_QUERY_CHUNK", "128"))
        except Exception:
            chunk_size = 128
        if chunk_size <= 0:
            chunk_size = q_total
        chunk_size = max(1, min(chunk_size, q_total))

        k = key_states[0].float()
        n_heads = int(query_states.shape[1])
        if k.shape[0] != n_heads:
            if n_heads % int(k.shape[0]) != 0:
                raise RuntimeError(f"query/key head mismatch: q={n_heads} kv={int(k.shape[0])}")
            k = k.repeat_interleave(n_heads // int(k.shape[0]), dim=0)
        all_key_idx = torch.arange(key_len, device=device, dtype=torch.long)
        k_all = k[:, :key_len, :]
        scale = float(query_states.shape[-1]) ** 0.5

        received = torch.zeros((k_total,), device=device, dtype=torch.float32)
        need_correction = mode != "none"
        exposure = torch.zeros((k_total,), device=device, dtype=torch.float32) if need_correction else None
        baseline = torch.zeros((k_total,), device=device, dtype=torch.float32) if mode not in {"none", "exposure"} else None

        if q_total > chunk_size:
            self.record("score_chunked_calls", 1)
            self.record("score_chunks", (q_total + chunk_size - 1) // chunk_size)

        active_key_2d = None
        mask_4d = None
        if attention_mask is not None:
            if attention_mask.ndim == 4:
                mask_4d = attention_mask[0, 0]
            elif attention_mask.ndim == 2:
                active_key_2d = attention_mask[0, :key_len].bool()

        # Exact inference reuse for the user-functional selector.  The native
        # PMI received score and the functional signature previously rebuilt
        # the same full causal QK/softmax matrix in two consecutive calls.
        # Accumulate the per-head user-query mass while that matrix is already
        # live; the signature path consumes it immediately afterwards.
        reuse_user_mass = bool(getattr(self, "_functional_mass_reuse_enabled", False))
        user_role_mask = None
        user_mass = None
        user_query_count = 0
        if reuse_user_mass:
            self._received_user_mass_cache = None
            if self._role_ids is None or int(self._role_ids.numel()) != self.prompt_len:
                raise RuntimeError("functional mass reuse requires aligned prompt roles")
            roles = self._role_ids.to(device=device).index_select(0, query_idx)
            user_role_mask = roles.eq(2)
            user_query_count = int(user_role_mask.sum().item())
            if user_query_count <= 0:
                raise RuntimeError("functional mass reuse found no user-content queries")
            user_mass = torch.zeros((n_heads, k_total), device=device, dtype=torch.float32)

        visible_pair_total = 0
        for chunk_start in range(0, q_total, chunk_size):
            qi = query_idx[chunk_start : chunk_start + chunk_size]
            q = query_states[0, :, qi, :].float()
            logits = torch.einsum("hqd,hkd->hqk", q, k_all) / scale
            visible = qi[:, None] >= all_key_idx[None, :]

            if mask_4d is not None:
                mask_slice = mask_4d[qi][:, :key_len]
                finite = torch.isfinite(mask_slice) & (mask_slice > torch.finfo(mask_slice.dtype).min / 2)
                visible = visible & finite
                logits = logits + mask_slice.float().unsqueeze(0)
            elif active_key_2d is not None:
                visible = visible & active_key_2d[None, :]

            logits = logits.masked_fill(~visible.unsqueeze(0), torch.finfo(logits.dtype).min)
            probs = torch.softmax(logits, dim=-1).masked_fill(~visible.unsqueeze(0), 0.0)
            target_probs = probs.index_select(2, key_idx)
            received += target_probs.mean(dim=0).sum(dim=0)
            if user_mass is not None and user_role_mask is not None:
                local_user = user_role_mask[chunk_start : chunk_start + int(qi.numel())]
                user_mass += (target_probs * local_user.view(1, -1, 1)).sum(dim=1)
            if not bool(getattr(self, "_skip_functional_runtime_diagnostics", False)):
                visible_pair_total += int(visible.sum().item())

            if need_correction:
                visible_target = visible[:, key_idx].float()
                exposure += visible_target.sum(dim=0)
                if baseline is not None:
                    visible_all = visible.float().sum(dim=1).clamp_min(1.0)
                    baseline += (visible_target / visible_all[:, None]).sum(dim=0)

        if user_mass is not None:
            self._received_user_mass_cache = {
                "mass": (user_mass / float(user_query_count)).detach(),
                "key_count": k_total,
                "key_len": key_len,
                "user_queries": user_query_count,
            }
            self.record("vision_functional_mass_reuse_produced", 1)
        self.record("text_score_full_key_calls", 1)
        self.record("text_score_full_key_query_tokens", q_total)
        self.record("text_score_full_key_target_tokens", k_total)
        if not bool(getattr(self, "_skip_functional_runtime_diagnostics", False)):
            self.record("text_score_full_key_visible_pairs", visible_pair_total)
        else:
            self.record("text_score_full_key_visible_pair_diagnostics_skipped", 1)
        if mode == "none":
            return received
        exposure = exposure.clamp_min(1.0)
        received_exp = received / exposure
        if mode == "exposure":
            return received_exp

        baseline_exp = baseline / exposure
        eps = 1e-8
        if mode in {"baseline_ratio", "exposure_baseline_ratio"}:
            return received / (baseline + eps)
        if mode == "baseline_diff":
            return received - baseline
        if mode == "exposure_baseline_diff":
            return received_exp - baseline_exp
        return received

    def _visible_counts_for_queries(
        self,
        query_idx: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        key_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        counts = (query_idx + 1).to(device=device, dtype=torch.float32).clamp_max(float(key_len))
        if attention_mask is None or attention_mask.ndim != 4:
            return counts
        m = attention_mask[0, 0]
        out = []
        for q in query_idx.tolist():
            row = m[q, :key_len]
            out.append((torch.isfinite(row) & (row > torch.finfo(row.dtype).min / 2)).float().sum())
        return torch.stack(out).to(device=device).float()
