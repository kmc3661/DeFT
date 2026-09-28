"""Opt-in patch making the single-L9-space greedy rule uniform from step zero.

This module does not alter the active benchmark source.  A future paired
runner explicitly installs it in its own process.  The existing key-greedy
path then calls the same one-space native selector, except that the empty-set
residual is defined as one and no relevance-only first-token exception exists.
"""
from __future__ import annotations

from .native import coverage_native
from .native.coverage_uniform_native import global_feature_uniform_start_greedy

_ORIGINAL = coverage_native.global_feature_greedy


def _uniform_start_adapter(
    relevance,
    global_importance,
    user_similarity,
    key_similarity,
    keep_n: int,
) -> list[int]:
    if user_similarity.shape != key_similarity.shape:
        raise ValueError("uniform-start selector requires aligned similarity matrices")
    # The single-L9-space route supplies the same matrix for both arguments.
    # Fail closed if a two-space caller attempts to use this opt-in patch.
    if user_similarity is not key_similarity and not (
        user_similarity.shape == key_similarity.shape
        and (user_similarity == key_similarity).all()
    ):
        raise RuntimeError("uniform-start patch is valid only for one shared L9 geometry")
    return global_feature_uniform_start_greedy(
        relevance,
        global_importance,
        key_similarity,
        keep_n,
    )


def install() -> None:
    coverage_native.global_feature_greedy = _uniform_start_adapter


def uninstall() -> None:
    coverage_native.global_feature_greedy = _ORIGINAL
