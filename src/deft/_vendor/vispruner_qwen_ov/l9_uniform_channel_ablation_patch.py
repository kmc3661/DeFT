"""Opt-in evidence-channel ablation for the uniform-start L9 selector.

This module is analysis-only. It preserves the single L9-key residual geometry
and changes only which scalar evidence channels enter the same greedy rule.
"""
from __future__ import annotations

import numpy as np

from .native import coverage_native
from .native.coverage_uniform_native import global_feature_uniform_start_greedy

_MODE = "joint"
_ORIGINAL = coverage_native.global_feature_greedy


def set_mode(mode: str) -> None:
    global _MODE
    value = str(mode).strip().lower()
    if value not in {"joint", "image_only", "question_only", "joint_no_residual_update"}:
        raise ValueError(f"unsupported L9 evidence ablation mode: {mode}")
    _MODE = value


def current_mode() -> str:
    return _MODE


def _adapter(relevance, global_importance, user_similarity, key_similarity, keep_n: int) -> list[int]:
    if user_similarity.shape != key_similarity.shape or not np.array_equal(user_similarity, key_similarity):
        raise RuntimeError("channel ablation requires one shared L9-key geometry")
    rel = np.asarray(relevance, dtype=np.float32)
    glob = np.asarray(global_importance, dtype=np.float32)
    if _MODE == "image_only":
        rel = np.zeros_like(rel)
    elif _MODE == "question_only":
        glob = np.zeros_like(glob)
    if _MODE == "joint_no_residual_update":
        # Preserve the same sequential normalization and exclusion logic while
        # setting d_i(S)=1 for every still-available token. Only the
        # selected-set redundancy update is removed.
        similarity = np.zeros_like(key_similarity, dtype=np.float32)
    else:
        similarity = key_similarity
    return global_feature_uniform_start_greedy(rel, glob, similarity, keep_n)


def install() -> None:
    coverage_native.global_feature_greedy = _adapter


def uninstall() -> None:
    coverage_native.global_feature_greedy = _ORIGINAL
