"""Task/model-name-free deployment of the image-only M0 influence prior.

The paper-level definition is path-complete visual influence. A single shared,
token-aligned post-encoder interface is measured by its bias-centered actual
contribution. If the architecture injects visual features through multiple
post-encoder paths, no one projector score is complete, so the common
pre-interface encoder evidence is used.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .interface_topology import classify_visual_interface_topology


@dataclass(frozen=True)
class TotalInfluenceDeployment:
    topology_family: str
    path_count: int
    stage_variant: str
    estimator: str | None
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "topology_family": self.topology_family,
            "path_count": self.path_count,
            "stage_variant": self.stage_variant,
            "estimator": self.estimator,
            "reason": self.reason,
        }


def configure_total_influence_stage(state: Any, model: Any) -> TotalInfluenceDeployment:
    """Configure one architecture-only M0 policy on an instantiated model.

    This function never inspects model/checkpoint names, tasks, datasets,
    prompts, outputs, pruning percentages, or benchmark scores.
    """
    topology = classify_visual_interface_topology(model)
    if topology.saliency_policy == "encoder":
        stage_variant = "encoder_feature_diverse"
        estimator = None
        reason = "multiple visual injection paths require shared path-complete encoder evidence"
    elif topology.saliency_policy == "interface_contribution" and topology.path_count == 1:
        stage_variant = "interface_feature_diverse"
        estimator = "contribution"
        reason = "one token-aligned interface permits exact bias-centered contribution"
    else:
        raise RuntimeError(f"unsupported total-influence topology: {topology.as_dict()}")

    if not hasattr(state, "interface_stage_variant"):
        raise RuntimeError("total-influence policy requires interface-aware pruning state")
    state.interface_stage_variant = stage_variant
    state.config.vision_stage_score_mode = "feature_diverse"
    if estimator is not None:
        os.environ["VISPRUNER_INTERFACE_ESTIMATOR"] = estimator
        install = getattr(state, "install_interface_projector_hooks", None)
        if not callable(install):
            raise RuntimeError("single-interface total influence requires projector hooks")
        install(model)
    record = getattr(state, "record", None)
    if callable(record):
        record("total_influence_policy_events", 1)
        record(f"total_influence_topology_{topology.family}", 1)
        record(f"total_influence_stage_{stage_variant}", 1)
    return TotalInfluenceDeployment(
        topology_family=topology.family,
        path_count=topology.path_count,
        stage_variant=stage_variant,
        estimator=estimator,
        reason=reason,
    )
