"""Model-name-free runtime configuration for the frozen common pruning candidate."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from .l9_multiview_coverage_patch import configure_adaptive_topology
from .total_influence_policy import TotalInfluenceDeployment, configure_total_influence_stage

@dataclass(frozen=True)
class CommonCandidateDeployment:
    topology: dict[str, Any]
    influence: TotalInfluenceDeployment
    l9_mode: str
    balance_mode: str
    preserve_stage_positions: bool
    preserve_final_positions: bool
    merge: bool
    def as_dict(self)->dict[str,Any]:
        return {
            "topology":self.topology,
            "influence":self.influence.as_dict(),
            "l9_mode":self.l9_mode,
            "balance_mode":self.balance_mode,
            "preserve_stage_positions":self.preserve_stage_positions,
            "preserve_final_positions":self.preserve_final_positions,
            "merge":self.merge,
        }

def configure_common_candidate_runtime(wrapper:Any)->CommonCandidateDeployment:
    """Apply the frozen candidate using topology only, never benchmark metadata."""
    state=getattr(wrapper,"_progressive",None)
    model=getattr(wrapper,"model",None)
    if state is None or model is None:
        raise RuntimeError("common candidate requires a progressive wrapper and loaded model")
    topology=configure_adaptive_topology(state,model)
    influence=configure_total_influence_stage(state,model)
    # One fixed L9 rule across every supported topology and compression:
    # question relevance weighted global-feature functional coverage.
    state._l9_multiview_mode="global_feature"
    state._l9_balance_mode="functional"
    state._preserve_stage_original_position_ids=True
    state._preserve_final_original_position_ids=True
    state._l9_position_continuity_gate_enabled=False
    state._l9_position_topology_gate_enabled=False
    # Exact inference fast path: preserve the selected set while removing
    # unused functional-signature work and analysis-only coverage diagnostics.
    # The GPU-resident greedy variant is deliberately disabled: controlled
    # latency tests show the small iterative selector is faster with its
    # reference host-synchronised implementation on both target models.
    state._l9_exact_fast_path=True
    state._l9_device_greedy_fast_path=False
    state._skip_functional_runtime_diagnostics=True
    # PMI and the functional signature consume the same user-query QK/softmax
    # mass. Reuse it exactly instead of recomputing it, in every topology.
    # A 120-input/model paired audit reproduced every selected set and output.
    state._functional_mass_reuse_enabled=True
    state._l9_precompute_similarity_matrix=True
    state._l9_single_reduction_fast_path=True
    state._l9_cpu_greedy_fast_path=True
    state._l9_native_cpu_greedy_fast_path=True
    state._l9_native_cuda_greedy_fast_path=False
    state._l9_packed_cpu_transfer_fast_path=False
    state._l9_coverage_fast_mode="functional_greedy"
    # L0 exact fast path: only the selected set is consumed, so avoid full
    # stable sorts and synchronization-heavy analysis checks in deployment.
    # Keep the stable selector: interface scores can tie at the boundary.
    # Only remove analysis-only synchronizations, which is exactly equivalent.
    state._feature_diverse_fast_topk=False
    state._skip_selection_runtime_diagnostics=True
    state._runtime_exact_fast_path=True
    state.config.vision_merge=False
    return CommonCandidateDeployment(
        topology=topology,
        influence=influence,
        l9_mode="global_feature",
        balance_mode="functional",
        preserve_stage_positions=True,
        preserve_final_positions=True,
        merge=False,
    )
