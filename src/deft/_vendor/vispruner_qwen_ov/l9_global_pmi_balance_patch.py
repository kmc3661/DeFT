"""Runtime-only common L9 global-prior PMI and balanced endpoint ablation.

Global visual prior is defined structurally: CLS-query saliency when the visual
encoder owns a CLS token, otherwise the native all-query received saliency.
The additive encoder term stays unchanged; only the PMI reference prior changes.
"""
from __future__ import annotations

import os
import torch

from .interface_ablation import InterfaceComponentAblationState
from .progressive_pruning import ProgressivePruningState

_ORIGINAL_PMI = ProgressivePruningState._pmi_like_score
_ORIGINAL_CORESET = ProgressivePruningState._select_functional_coreset


def _global_prior_pmi(self, text_score, encoder_score):
    enabled = bool(getattr(self, "_l9_global_prior_pmi_enabled", False))
    if not enabled:
        return _ORIGINAL_PMI(self, text_score, encoder_score)
    global_prior = getattr(self, "_cls_detail_candidate_score", None)
    if global_prior is None or int(global_prior.numel()) != int(encoder_score.numel()):
        raise RuntimeError("global-prior PMI requires an aligned native global-summary prior")
    global_prior = global_prior.to(device=encoder_score.device, dtype=torch.float32).flatten()
    result = _ORIGINAL_PMI(self, text_score, global_prior)
    self.record("l9_global_prior_pmi_events", 1)
    self.record("l9_global_prior_pmi_tokens", int(result.numel()))
    return result


def _harmonic(a: float, b: float) -> float:
    return 0.0 if a <= 0.0 or b <= 0.0 else 2.0 * a * b / (a + b)


def _balanced_coreset(self, signatures, relevance, keep_n):
    mode = str(getattr(self, "_l9_balance_mode", os.environ.get(
        "VISPRUNER_L9_BALANCE_MODE", "functional"
    )) or "functional").strip().lower()
    if mode not in {"topk", "functional", "balanced"}:
        raise ValueError(f"unknown L9 balance mode: {mode}")
    n = int(relevance.numel()); k = max(1, min(int(keep_n), n))
    top = torch.argsort(relevance.detach().float(), descending=True, stable=True)[:k]
    if mode == "topk":
        self.record("l9_balance_topk_events", 1)
        return top
    functional = _ORIGINAL_CORESET(self, signatures, relevance, k)
    if mode == "functional":
        self.record("l9_balance_functional_events", 1)
        return functional

    rel = relevance.detach().float().clamp_min(0.0)
    rel_top = float(rel.index_select(0, top).sum().item())
    rel_functional = float(rel.index_select(0, functional).sum().item())
    rel_best = max(rel_top, rel_functional, 1e-12)
    cov_top = float(self._functional_coverage_score(signatures, top))
    cov_functional = float(self._functional_coverage_score(signatures, functional))
    cov_best = max(cov_top, cov_functional, 1e-12)
    top_objective = _harmonic(rel_top / rel_best, cov_top / cov_best)
    functional_objective = _harmonic(rel_functional / rel_best, cov_functional / cov_best)
    choose_functional = functional_objective > top_objective + 1e-12
    selected = functional if choose_functional else top
    overlap = int(torch.isin(top, functional).sum().item())
    self.record("l9_balance_decision_events", 1)
    self.record("l9_balance_functional_decisions", int(choose_functional))
    self.record("l9_balance_topk_decisions", int(not choose_functional))
    self.record("l9_balance_top_functional_overlap", overlap)
    self.record("l9_balance_selected", k)
    self.record("l9_balance_rel_top_x1e6", int(round(1e6 * rel_top / rel_best)))
    self.record("l9_balance_rel_functional_x1e6", int(round(1e6 * rel_functional / rel_best)))
    self.record("l9_balance_cov_top_x1e6", int(round(1e6 * cov_top / cov_best)))
    self.record("l9_balance_cov_functional_x1e6", int(round(1e6 * cov_functional / cov_best)))
    return selected


def install() -> None:
    if getattr(InterfaceComponentAblationState, "_l9_global_pmi_balance_patch_installed", False):
        return
    ProgressivePruningState._pmi_like_score = _global_prior_pmi
    InterfaceComponentAblationState._pmi_like_score = _global_prior_pmi
    InterfaceComponentAblationState._select_functional_coreset = _balanced_coreset
    InterfaceComponentAblationState._l9_global_pmi_balance_patch_installed = True
