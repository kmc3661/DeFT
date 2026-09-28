"""Analysis-only L9 multi-view coverage selector.

The current user-functional coreset covers decoder L9 visual-to-user writes.
This patch adds two independent residual views without changing L0, M0, K, or
pruning depth:

* global-feature: global-summary saliency weighted diversity of native L9 keys;
* spatial: global-summary saliency weighted dispersion in the original visual grid.

Channel utilities are normalized at every greedy step.  Consequently there is
no fixed slot quota: a view receives another token only while it still contains
large uncovered residual information.
"""
from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np
import torch

from .interface_ablation import InterfaceComponentAblationState
from .interface_topology import classify_visual_interface_topology

_BASE_SELECTOR = InterfaceComponentAblationState._select_functional_coreset
_BASE_USER_SIGNATURES = InterfaceComponentAblationState._text_update_user_functional_signatures
_VALID = {"current", "global_feature", "spatial", "joint", "adaptive"}


def _mode(state) -> str:
    value = str(getattr(state, "_l9_multiview_mode", os.environ.get(
        "VISPRUNER_L9_MULTIVIEW_MODE", "current"
    )) or "current").strip().lower()
    if value not in _VALID:
        raise ValueError(f"unknown L9 multi-view mode: {value}")
    return value


def _resolve_adaptive_mode(family: str, revision_to_final: float) -> str:
    family = str(family or "").strip().lower()
    if family in {"single_merger", "single_projector"}:
        return "global_feature"
    if family == "multi_injection":
        return "global_feature" if float(revision_to_final) > 1.0 + 1e-12 else "current"
    raise RuntimeError(f"unsupported adaptive visual-interface topology: {family!r}")


def _adaptive_revision_to_final(self, candidate_count: int, keep_n: int) -> float:
    candidate_count = max(1, int(candidate_count))
    keep_n = max(1, min(int(keep_n), candidate_count))
    num_layers = max(1, int(getattr(self.config, "num_layers", 1) or 1))
    final_layer = max(0, int(self._effective_vision_layer()))
    depth_fraction = max(0.0, min(1.0, float(final_layer) / float(num_layers)))
    fixed = int(getattr(self.config, "vision_keep_tokens", 0) or 0)
    if fixed > 0:
        source_n = candidate_count
        initial = tuple(int(x) for x in getattr(self, "_initial_vision_counts", ()))
        if initial:
            source_n = max(1, initial[0])
        retention = float(min(keep_n, source_n)) / float(source_n)
    else:
        retention = max(1e-12, min(1.0, float(getattr(self.config, "vision_keep_ratio", 1.0))))
    ratio = depth_fraction * (1.0 - retention) / retention
    self.record("l9_multiview_adaptive_revision_to_final_x1e6", int(round(ratio * 1e6)))
    return ratio


def configure_adaptive_topology(state, model) -> dict:
    topology = classify_visual_interface_topology(model)
    state._l9_multiview_topology_family = topology.family
    state._l9_multiview_topology_paths = topology.path_names
    state.record("l9_multiview_topology_config_events", 1)
    state.record("l9_multiview_topology_path_count", topology.path_count)
    state.record(f"l9_multiview_topology_{topology.family}_events", 1)
    return topology.as_dict()


def _capture_user_signatures(
    self,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    query_idx: torch.Tensor,
    key_idx: torch.Tensor,
    output_projection,
) -> torch.Tensor:
    requested_mode = _mode(self)
    family = None
    candidate_count = int(key_idx.numel())
    keep_n = int(self.final_vision_budget(candidate_count, 0))
    resolved_mode = requested_mode
    if requested_mode == "adaptive":
        family = getattr(self, "_l9_multiview_topology_family", None)
        if family is None:
            raise RuntimeError("adaptive L9 selector requires configured visual-interface topology")
        revision_to_final = _adaptive_revision_to_final(self, candidate_count, keep_n)
        resolved_mode = _resolve_adaptive_mode(family, revision_to_final)
        if (
            resolved_mode == "current"
            and bool(getattr(self, "_l9_exact_fast_path", False))
        ):
            # The adaptive-current endpoint is exact relevance top-K and never
            # consumes a functional signature.  Return a shape-correct empty
            # carrier so the shared selection path remains unchanged while
            # avoiding a redundant QK/softmax/value-projection pass.
            self._l9_multiview_key_features = None
            self.record("l9_multiview_adaptive_key_capture_skipped", 1)
            self.record("l9_exact_fast_unused_signature_skipped", 1)
            return torch.empty(
                (candidate_count, 0), device=query_states.device, dtype=torch.float32
            )

    fast_mode = str(getattr(self, "_l9_coverage_fast_mode", "functional_greedy") or "functional_greedy").strip().lower()
    if fast_mode not in {"functional_greedy", "key_greedy", "global_blend_topk", "balanced_rank_union", "single_anchor", "dual_anchor", "truncated_sqrt", "truncated_quarter", "truncated_discard"}:
        raise RuntimeError(f"unsupported L9 coverage fast mode: {fast_mode}")
    key_only = (
        bool(getattr(self, "_l9_key_only_coverage_fast_path", False))
        or fast_mode in {"key_greedy", "global_blend_topk", "balanced_rank_union", "single_anchor", "dual_anchor"}
    )
    signatures = None
    if not key_only:
        signatures = _BASE_USER_SIGNATURES(
            self, query_states, key_states, value_states, attention_mask,
            query_idx, key_idx, output_projection,
        )
    if requested_mode == "adaptive" and resolved_mode == "current":
        self._l9_multiview_key_features = None
        self.record("l9_multiview_adaptive_key_capture_skipped", 1)
        if signatures is None:
            raise RuntimeError("key-only coverage cannot be routed to relevance top-K")
        return signatures
    if fast_mode in {"global_blend_topk", "balanced_rank_union"}:
        self._l9_multiview_key_features = None
        self.record("l9_global_blend_signature_skipped_events", 1)
        return torch.empty(
            (candidate_count, 0), device=query_states.device, dtype=torch.float32
        )
    keys = key_states[0].detach().float()
    key_len = int(keys.shape[1])
    local = key_idx.to(device=keys.device, dtype=torch.long)
    if local.numel() == 0 or bool(((local < 0) | (local >= key_len)).any().item()):
        raise RuntimeError("multi-view key capture received invalid visual indices")
    features = keys.index_select(1, local).permute(1, 0, 2).reshape(int(local.numel()), -1)
    if not bool(torch.isfinite(features).all().item()):
        raise RuntimeError("multi-view L9 key features are non-finite")
    self._l9_multiview_key_features = features.detach()
    self.record("l9_multiview_key_capture_events", 1)
    self.record("l9_multiview_key_capture_tokens", int(features.shape[0]))
    self.record("l9_multiview_key_capture_dims", int(features.shape[1]))
    if key_only:
        signatures = features.detach()
        self.record("l9_key_only_coverage_signature_events", 1)
    if signatures is None:
        raise RuntimeError("multi-view coverage did not construct a signature")
    return signatures


def _infer_grid(token_count: int, aspect: float) -> tuple[int, int, float]:
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    if not math.isfinite(aspect) or aspect <= 0.0:
        aspect = 1.0
    best = None
    for height in range(1, int(math.sqrt(token_count)) + 1):
        if token_count % height:
            continue
        width = token_count // height
        for h, w in ((height, width), (width, height)):
            error = abs(math.log(max(1e-12, float(w) / float(h))) - math.log(aspect))
            candidate = (error, abs(h - w), h, w)
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    return int(best[2]), int(best[3]), float(best[0])


def _spatial_features(self, count: int, device: torch.device) -> torch.Tensor:
    original = getattr(self, "_feature_reserve_original_local_indices", None)
    initial = tuple(int(x) for x in getattr(self, "_initial_vision_counts", ()))
    if original is None or len(initial) != 1 or int(original.numel()) != count:
        raise RuntimeError("multi-view spatial coverage requires aligned original visual indices")
    native_n = int(initial[0])
    aspect = float(getattr(self, "_l9_multiview_image_aspect", 1.0) or 1.0)
    height, width, error = _infer_grid(native_n, aspect)
    local = original.to(device=device, dtype=torch.long)
    if bool(((local < 0) | (local >= native_n)).any().item()):
        raise RuntimeError("multi-view original visual index is out of range")
    row = torch.div(local, width, rounding_mode="floor").float()
    col = torch.remainder(local, width).float()
    row = row / float(max(1, height - 1))
    col = col / float(max(1, width - 1))
    coords = torch.stack((row, col), dim=-1)
    self.record("l9_multiview_spatial_events", 1)
    self.record("l9_multiview_grid_height", height)
    self.record("l9_multiview_grid_width", width)
    self.record("l9_multiview_grid_log_aspect_error_x1e6", int(round(error * 1e6)))
    return coords


def _positive_mean_normalize(values: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
    out = values.clamp_min(0.0)
    positive = out[available & (out > 0)]
    if positive.numel() == 0:
        return out
    return out / positive.mean().clamp_min(1e-8)


def _stable_relevance_topk(relevance: torch.Tensor, keep_n: int) -> torch.Tensor:
    count = int(relevance.numel())
    keep_n = max(1, min(int(keep_n), count))
    return torch.argsort(relevance.detach().float(), descending=True, stable=True)[:keep_n]



def _cpu_precomputed_greedy(
    rel: torch.Tensor,
    glob: torch.Tensor,
    user_similarity: torch.Tensor,
    key_similarity: torch.Tensor,
    keep_n: int,
    mode: str,
    coords: Optional[torch.Tensor],
    native: bool = False,
    packed_transfer: bool = False,
) -> list[int]:
    """Run the exact sequential rule on CPU after one batched GPU Gram pass.

    Candidate counts are only a few hundred.  Moving two compact Gram matrices
    once is substantially cheaper than synchronising one or two tiny CUDA
    reductions for every selected token.
    """
    if packed_transfer:
        count = int(rel.numel())
        flat = torch.cat((
            rel.detach().float().flatten(),
            glob.detach().float().flatten(),
            user_similarity.detach().float().flatten(),
            key_similarity.detach().float().flatten(),
        ), dim=0).cpu().numpy()
        offset = 0
        rel_np = flat[offset : offset + count]; offset += count
        glob_np = flat[offset : offset + count]; offset += count
        user_sim = flat[offset : offset + count * count].reshape(count, count); offset += count * count
        key_sim = flat[offset : offset + count * count].reshape(count, count)
    else:
        packed = torch.stack((rel, glob), dim=0).detach().float().cpu().numpy()
        user_sim = user_similarity.detach().float().cpu().numpy()
        key_sim = key_similarity.detach().float().cpu().numpy()
        rel_np = packed[0]
        glob_np = packed[1]
    coord_np = coords.detach().float().cpu().numpy() if coords is not None else None
    if native:
        if mode != "global_feature" or coord_np is not None:
            raise RuntimeError("native CPU greedy currently supports global_feature only")
        from .native.coverage_native import global_feature_greedy
        return global_feature_greedy(rel_np, glob_np, user_sim, key_sim, keep_n)
    count = int(rel_np.shape[0])
    first = int(np.argmax(rel_np))
    selected = [first]
    available = np.ones(count, dtype=np.bool_)
    available[first] = False
    user_residual = np.clip(1.0 - user_sim[:, first], 0.0, 2.0)
    key_residual = np.clip(1.0 - key_sim[:, first], 0.0, 2.0)
    if coord_np is not None:
        spatial_residual = np.clip(
            np.linalg.norm(coord_np - coord_np[first], axis=-1) / math.sqrt(2.0), 0.0, 1.0
        )
    else:
        spatial_residual = None

    def normalize(values: np.ndarray) -> np.ndarray:
        out = np.maximum(values, np.float32(0.0))
        denom = max(float(out[available].mean()), 1e-8) if bool(available.any()) else 1.0
        return out / np.float32(denom)

    for _ in range(int(keep_n) - 1):
        channels = [normalize(rel_np * user_residual)]
        if mode in {"global_feature", "joint"}:
            channels.append(normalize(glob_np * key_residual))
        if mode in {"spatial", "joint"}:
            assert spatial_residual is not None
            channels.append(normalize(glob_np * spatial_residual))
        utility = np.mean(np.stack(channels, axis=0), axis=0, dtype=np.float32)
        utility = np.where(available, utility, np.float32(-1.0))
        if float(utility.max()) <= 0.0:
            utility = np.where(available, rel_np + glob_np, np.float32(-1.0))
        nxt = int(np.argmax(utility))
        selected.append(nxt)
        available[nxt] = False
        user_residual = np.minimum(user_residual, np.clip(1.0 - user_sim[:, nxt], 0.0, 2.0))
        key_residual = np.minimum(key_residual, np.clip(1.0 - key_sim[:, nxt], 0.0, 2.0))
        if spatial_residual is not None:
            distance = np.clip(
                np.linalg.norm(coord_np - coord_np[nxt], axis=-1) / math.sqrt(2.0), 0.0, 1.0
            )
            spatial_residual = np.minimum(spatial_residual, distance)
    return selected

def _multi_view_selector(self, signatures: torch.Tensor, relevance: torch.Tensor, keep_n: int) -> torch.Tensor:
    requested_mode = _mode(self)
    count = int(signatures.shape[0])
    keep_n = max(1, min(int(keep_n), count))
    mode = requested_mode
    if requested_mode == "adaptive":
        family = getattr(self, "_l9_multiview_topology_family", None)
        if family is None:
            raise RuntimeError("adaptive L9 selector requires configured visual-interface topology")
        revision_to_final = _adaptive_revision_to_final(self, count, keep_n)
        mode = _resolve_adaptive_mode(family, revision_to_final)
        self.record("l9_multiview_adaptive_route_events", 1)
        self.record(f"l9_multiview_adaptive_{mode}_events", 1)
    if mode == "current":
        self.record("l9_multiview_current_events", 1)
        if requested_mode == "adaptive":
            self.record("l9_multiview_adaptive_current_topk_events", 1)
            return _stable_relevance_topk(relevance, keep_n)
        return _BASE_SELECTOR(self, signatures, relevance, keep_n)

    if keep_n >= count:
        return torch.arange(count, device=signatures.device, dtype=torch.long)
    if int(relevance.numel()) != count:
        raise RuntimeError("multi-view relevance/signature mismatch")

    fast_mode = str(getattr(self, "_l9_coverage_fast_mode", "functional_greedy") or "functional_greedy").strip().lower()
    if fast_mode in {"global_blend_topk", "balanced_rank_union"}:
        detail = getattr(self, "_cls_detail_candidate_score", None)
        if detail is None or int(detail.numel()) != count:
            raise RuntimeError("one-shot L9 selection requires aligned global-summary scores")
        rel = torch.nan_to_num(relevance.detach().float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        glob = torch.nan_to_num(detail.to(device=rel.device).float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        if fast_mode == "global_blend_topk":
            utility = rel / rel.mean().clamp_min(1e-8) + glob / glob.mean().clamp_min(1e-8)
            selected = torch.argsort(utility, descending=True, stable=True)[:keep_n]
            self.record("l9_global_blend_topk_events", 1)
        else:
            # Parameter-free reciprocal rank union: alternate the stable
            # question-relevance and global-image rankings, skipping duplicate
            # indices, until exactly K unique tokens are selected.
            rel_order = torch.argsort(rel, descending=True, stable=True).detach().cpu().tolist()
            glob_order = torch.argsort(glob, descending=True, stable=True).detach().cpu().tolist()
            chosen = []
            seen = set()
            for rank in range(count):
                for idx in (rel_order[rank], glob_order[rank]):
                    idx = int(idx)
                    if idx not in seen:
                        seen.add(idx)
                        chosen.append(idx)
                        if len(chosen) == keep_n:
                            break
                if len(chosen) == keep_n:
                    break
            if len(chosen) != keep_n:
                raise RuntimeError("balanced rank union failed to produce exact K")
            selected = torch.tensor(chosen, device=rel.device, dtype=torch.long)
            self.record("l9_balanced_rank_union_events", 1)
        self.record("l9_multiview_select_events", 1)
        self.record(f"l9_multiview_mode_{mode}_events", 1)
        self.record("l9_multiview_selected", keep_n)
        return selected

    user = torch.nn.functional.normalize(signatures.detach().float(), dim=-1)
    keys = getattr(self, "_l9_multiview_key_features", None)
    if keys is None or int(keys.shape[0]) != count:
        raise RuntimeError("multi-view selector is missing aligned L9 key features")
    keys = torch.nn.functional.normalize(keys.to(device=user.device).float(), dim=-1)
    detail = getattr(self, "_cls_detail_candidate_score", None)
    if detail is None or int(detail.numel()) != count:
        raise RuntimeError("multi-view selector requires aligned global-summary scores")

    rel = torch.nan_to_num(relevance.detach().float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    glob = torch.nan_to_num(detail.to(device=user.device).float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    rel = rel / rel.mean().clamp_min(1e-8)
    glob = glob / glob.mean().clamp_min(1e-8)
    coords = _spatial_features(self, count, user.device) if mode in {"spatial", "joint"} else None

    if fast_mode in {"single_anchor", "dual_anchor"}:
        # A fixed role-based simplification of iterative coverage.  The first
        # anchor is the strongest question-relevant token.  The dual variant
        # adds exactly one non-redundant global-image anchor; all remaining
        # slots are filled by one stable residual ranking.  No model, task, or
        # compression identity is consulted.
        first = torch.argmax(rel)
        selected_parts = [first.reshape(1)]
        available = torch.ones(count, device=rel.device, dtype=torch.bool)
        available[first] = False
        residual = (1.0 - keys @ keys.index_select(0, first.reshape(1)).squeeze(0)).clamp(0.0, 2.0)
        if fast_mode == "dual_anchor" and keep_n > 1:
            anchor_utility = _positive_mean_normalize(glob * residual, available)
            fallback = _positive_mean_normalize((rel + glob) * residual, available)
            anchor_utility = torch.where(
                anchor_utility.max() > 0.0, anchor_utility, fallback
            ).masked_fill(~available, torch.finfo(anchor_utility.dtype).min)
            second = torch.argmax(anchor_utility)
            selected_parts.append(second.reshape(1))
            available[second] = False
            second_residual = (
                1.0 - keys @ keys.index_select(0, second.reshape(1)).squeeze(0)
            ).clamp(0.0, 2.0)
            residual = torch.minimum(residual, second_residual)
        remaining = keep_n - len(selected_parts)
        if remaining > 0:
            channel_rel = _positive_mean_normalize(rel * residual, available)
            channel_glob = _positive_mean_normalize(glob * residual, available)
            utility = 0.5 * (channel_rel + channel_glob)
            utility = utility.masked_fill(~available, torch.finfo(utility.dtype).min)
            selected_parts.append(
                torch.argsort(utility, descending=True, stable=True)[:remaining]
            )
        selected = torch.cat(selected_parts, dim=0)
        if int(selected.numel()) != keep_n or int(torch.unique(selected).numel()) != keep_n:
            raise RuntimeError(f"{fast_mode} coverage produced an invalid fixed-K set")
        self.record(f"l9_{fast_mode}_coverage_events", 1)
        self.record("l9_multiview_select_events", 1)
        self.record(f"l9_multiview_mode_{mode}_events", 1)
        self.record("l9_multiview_selected", keep_n)
        return selected

    device_greedy_fast = bool(getattr(self, "_l9_device_greedy_fast_path", False))
    precompute_similarity = bool(getattr(self, "_l9_precompute_similarity_matrix", False))
    reduction_fast = bool(getattr(self, "_l9_single_reduction_fast_path", False))
    key_only = (
        bool(getattr(self, "_l9_key_only_coverage_fast_path", False))
        or fast_mode == "key_greedy"
    )
    key_similarity = keys @ keys.T if precompute_similarity else None
    user_similarity = (
        key_similarity if key_only and precompute_similarity
        else user @ user.T if precompute_similarity
        else None
    )
    if precompute_similarity:
        self.record("l9_similarity_matrix_precompute_events", 1)
    cuda_greedy_fast = bool(getattr(self, "_l9_native_cuda_greedy_fast_path", False))
    if cuda_greedy_fast:
        if mode != "global_feature" or coords is not None:
            raise RuntimeError("native CUDA greedy currently supports global_feature only")
        if user_similarity is None or key_similarity is None:
            raise RuntimeError("native CUDA greedy requires precomputed similarity matrices")
        from .native.coverage_cuda import global_feature_greedy_cuda
        selected_cuda = global_feature_greedy_cuda(
            rel, glob, user_similarity, key_similarity, keep_n
        )
        self.record("l9_native_cuda_greedy_events", 1)
        self.record("l9_multiview_select_events", 1)
        self.record(f"l9_multiview_mode_{mode}_events", 1)
        self.record("l9_multiview_selected", keep_n)
        return selected_cuda
    if fast_mode in {"truncated_sqrt", "truncated_quarter", "truncated_discard"}:
        if mode != "global_feature" or coords is not None:
            raise RuntimeError("truncated coverage currently supports global_feature only")
        if user_similarity is None or key_similarity is None:
            raise RuntimeError("truncated coverage requires precomputed similarity matrices")
        if fast_mode == "truncated_sqrt":
            steps = int(math.ceil(math.sqrt(float(keep_n))))
        elif fast_mode == "truncated_quarter":
            steps = int(math.ceil(float(keep_n) / 4.0))
        else:
            # Use the observed L9 discard fraction as the sequential-coverage
            # fraction. This varies continuously with M0/K and has no named
            # model, task, or pruning-budget branch.
            steps = int(math.ceil(float(keep_n) * float(count - keep_n) / float(count)))
        from .native.coverage_native import global_feature_truncated_greedy
        selected_cpu = global_feature_truncated_greedy(
            rel.detach().float().cpu().numpy(), glob.detach().float().cpu().numpy(),
            user_similarity.detach().float().cpu().numpy(), key_similarity.detach().float().cpu().numpy(),
            keep_n, steps,
        )
        if len(selected_cpu) != keep_n or len(set(selected_cpu)) != keep_n:
            raise RuntimeError("truncated coverage produced an invalid fixed-K set")
        self.record("l9_native_cpu_truncated_greedy_events", 1)
        self.record("l9_native_cpu_truncated_greedy_steps", steps)
        self.record(f"l9_native_cpu_{fast_mode}_events", 1)
        self.record("l9_multiview_select_events", 1)
        self.record(f"l9_multiview_mode_{mode}_events", 1)
        self.record("l9_multiview_selected", keep_n)
        return torch.tensor(selected_cpu, device=user.device, dtype=torch.long)

    cpu_greedy_fast = bool(getattr(self, "_l9_cpu_greedy_fast_path", False))
    if cpu_greedy_fast:
        if user_similarity is None or key_similarity is None:
            raise RuntimeError("CPU greedy fast path requires precomputed similarity matrices")
        selected_cpu = _cpu_precomputed_greedy(
            rel, glob, user_similarity, key_similarity, keep_n, mode, coords,
            native=bool(getattr(self, "_l9_native_cpu_greedy_fast_path", False)),
            packed_transfer=bool(getattr(self, "_l9_packed_cpu_transfer_fast_path", False)),
        )
        if len(selected_cpu) != keep_n or len(set(selected_cpu)) != keep_n:
            raise RuntimeError("CPU greedy fast path produced an invalid fixed-K set")
        self.record("l9_cpu_precomputed_greedy_events", 1)
        if bool(getattr(self, "_l9_native_cpu_greedy_fast_path", False)):
            self.record("l9_native_cpu_greedy_events", 1)
        self.record("l9_multiview_select_events", 1)
        self.record(f"l9_multiview_mode_{mode}_events", 1)
        self.record("l9_multiview_selected", keep_n)
        return torch.tensor(selected_cpu, device=user.device, dtype=torch.long)
    if device_greedy_fast:
        first = torch.argmax(rel)
        selected = [first]
    else:
        first = int(torch.argmax(rel).item())
        selected = [first]
    available = torch.ones(count, device=user.device, dtype=torch.bool)
    available[first] = False
    user_column = user_similarity[:, first] if user_similarity is not None else user @ user[first]
    key_column = key_similarity[:, first] if key_similarity is not None else keys @ keys[first]
    user_residual = (1.0 - user_column).clamp(0.0, 2.0)
    key_residual = (1.0 - key_column).clamp(0.0, 2.0)
    if coords is not None:
        spatial_residual = torch.linalg.vector_norm(coords - coords[first], dim=-1).div(math.sqrt(2.0)).clamp(0.0, 1.0)
    else:
        spatial_residual = None

    for _ in range(keep_n - 1):
        channels = [_positive_mean_normalize(rel * user_residual, available)]
        if mode in {"global_feature", "joint"}:
            channels.append(_positive_mean_normalize(glob * key_residual, available))
        if mode in {"spatial", "joint"}:
            assert spatial_residual is not None
            channels.append(_positive_mean_normalize(glob * spatial_residual, available))
        utility = torch.stack(channels, dim=0).mean(dim=0)
        utility = torch.where(available, utility, torch.full_like(utility, -1.0))
        if device_greedy_fast:
            # Keep the reference fallback rule but make the decision entirely
            # on device.  This removes two host synchronizations per selected
            # token without changing the utility or tie-breaking definition.
            primary = torch.argmax(utility)
            fallback = torch.where(
                available, rel + glob, torch.full_like(rel, -1.0)
            )
            fallback_idx = torch.argmax(fallback)
            nxt = torch.where(utility[primary] > 0.0, primary, fallback_idx)
        else:
            if reduction_fast:
                max_value, max_index = torch.max(utility, dim=0)
                if float(max_value.item()) <= 0.0:
                    fallback = rel + glob
                    utility = torch.where(available, fallback, torch.full_like(fallback, -1.0))
                    nxt = int(torch.argmax(utility).item())
                else:
                    nxt = int(max_index.item())
            else:
                if float(utility.max().item()) <= 0.0:
                    fallback = rel + glob
                    utility = torch.where(available, fallback, torch.full_like(fallback, -1.0))
                nxt = int(torch.argmax(utility).item())
        selected.append(nxt)
        available[nxt] = False
        user_column = user_similarity[:, nxt] if user_similarity is not None else user @ user[nxt]
        key_column = key_similarity[:, nxt] if key_similarity is not None else keys @ keys[nxt]
        user_residual = torch.minimum(user_residual, (1.0 - user_column).clamp(0.0, 2.0))
        key_residual = torch.minimum(key_residual, (1.0 - key_column).clamp(0.0, 2.0))
        if spatial_residual is not None:
            distance = torch.linalg.vector_norm(coords - coords[nxt], dim=-1).div(math.sqrt(2.0)).clamp(0.0, 1.0)
            spatial_residual = torch.minimum(spatial_residual, distance)

    result = torch.stack(selected).to(device=user.device, dtype=torch.long) if device_greedy_fast else torch.tensor(selected, device=user.device, dtype=torch.long)
    if device_greedy_fast:
        self.record("l9_exact_fast_device_greedy_events", 1)
    if int(result.numel()) != keep_n or int(torch.unique(result).numel()) != keep_n:
        raise RuntimeError("multi-view selector produced an invalid fixed-K set")
    self.record("l9_multiview_select_events", 1)
    self.record(f"l9_multiview_mode_{mode}_events", 1)
    self.record("l9_multiview_selected", keep_n)
    return result


def install() -> None:
    if getattr(InterfaceComponentAblationState, "_l9_multiview_coverage_patch_installed", False):
        return
    InterfaceComponentAblationState._text_update_user_functional_signatures = _capture_user_signatures
    InterfaceComponentAblationState._select_functional_coreset = _multi_view_selector
    InterfaceComponentAblationState._l9_multiview_coverage_patch_installed = True
