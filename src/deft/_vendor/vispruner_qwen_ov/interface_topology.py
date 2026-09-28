"""Architecture-only visual-interface topology classification.

This module intentionally does not inspect checkpoint names, task labels, data, or
outputs. It answers whether one cheap interface-contribution score can cover all
post-encoder visual injection paths.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VisualInterfaceTopology:
    family: str
    path_names: tuple[str, ...]
    saliency_policy: str
    reason: str

    @property
    def path_count(self) -> int:
        return len(self.path_names)

    @property
    def is_single_token_aligned_path(self) -> bool:
        return self.path_count == 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "path_names": list(self.path_names),
            "path_count": self.path_count,
            "is_single_token_aligned_path": self.is_single_token_aligned_path,
            "saliency_policy": self.saliency_policy,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class VisualSaliencyDeployment:
    stage_variant: str
    final_variant: str
    interface_estimator: str | None
    requested_component_condition: str | None
    effective_component_condition: str
    reason: str

    @property
    def needs_interface_contribution(self) -> bool:
        return self.interface_estimator == "contribution"

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage_variant": self.stage_variant,
            "final_variant": self.final_variant,
            "interface_estimator": self.interface_estimator,
            "requested_component_condition": self.requested_component_condition,
            "effective_component_condition": self.effective_component_condition,
            "needs_interface_contribution": self.needs_interface_contribution,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ClsDetailRouterDeployment:
    control_condition: str
    stage_variant: str
    control_final_variant: str
    pure_final_variant: str
    routed_final_variant: str
    interface_estimator: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "control_condition": self.control_condition,
            "stage_variant": self.stage_variant,
            "control_final_variant": self.control_final_variant,
            "pure_final_variant": self.pure_final_variant,
            "routed_final_variant": self.routed_final_variant,
            "interface_estimator": self.interface_estimator,
            "reason": self.reason,
        }


_COMPONENT_VARIANTS = {
    "e0_e9": ("encoder_feature_diverse", "encoder_pmi"),
    "i0_e9": ("interface_feature_diverse", "encoder_pmi"),
    "e0_i9": ("encoder_feature_diverse", "interface_pmi"),
    "i0_i9": ("interface_feature_diverse", "interface_pmi"),
}


def resolve_visual_saliency_deployment(
    topology: VisualInterfaceTopology,
    selected_component_condition: str | None,
) -> VisualSaliencyDeployment:
    """Resolve one architecture-only deployment policy after paired selection.

    Multi-injection models fail safe to the shared encoder coordinate regardless
    of an OV-selected component. A path-complete single interface may use only
    the preregistered component condition selected by the paired gate.
    """

    requested = (
        None if selected_component_condition is None
        else str(selected_component_condition).strip().lower()
    )
    if topology.saliency_policy == "encoder":
        return VisualSaliencyDeployment(
            stage_variant="encoder_feature_diverse",
            final_variant="encoder_pmi",
            interface_estimator=None,
            requested_component_condition=requested,
            effective_component_condition="e0_e9",
            reason=(
                "Multiple post-encoder injection paths make one interface score "
                "incomplete; deploy the byte-equivalent Current encoder endpoint."
            ),
        )
    if topology.saliency_policy != "interface_contribution" or topology.path_count != 1:
        raise RuntimeError("inconsistent or unsupported visual saliency topology")
    if requested not in _COMPONENT_VARIANTS:
        raise ValueError(
            "single-interface deployment requires one selected component in "
            f"{sorted(_COMPONENT_VARIANTS)}"
        )
    stage, final = _COMPONENT_VARIANTS[requested]
    needs_interface = requested != "e0_e9"
    return VisualSaliencyDeployment(
        stage_variant=stage,
        final_variant=final,
        interface_estimator="contribution" if needs_interface else None,
        requested_component_condition=requested,
        effective_component_condition=requested,
        reason=(
            "One shared token-aligned interface is path-complete; deploy the "
            "ratio-specific component endpoint selected by the paired gate."
        ),
    )


def resolve_cls_detail_router_deployment(
    topology: VisualInterfaceTopology,
    selected_component_condition: str | None,
) -> ClsDetailRouterDeployment:
    """Bind the frozen CLS router to the exact selected single-path control.

    The router needs interface statistics as input even when its false route
    preserves an encoder-PMI control. Multi-injection models have no equivalent
    independent CLS structural route and therefore fail closed here.
    """

    if topology.saliency_policy != "interface_contribution" or topology.path_count != 1:
        raise RuntimeError(
            "CLS-detail routing requires one token-aligned post-encoder visual path"
        )
    control = resolve_visual_saliency_deployment(
        topology, selected_component_condition
    )
    endpoint_variants = {
        "encoder_pmi": ("encoder_pmi_cls_quota", "encoder_pmi_cls_routed"),
        "interface_pmi": ("interface_pmi_cls_quota", "interface_pmi_cls_routed"),
    }.get(control.final_variant)
    if endpoint_variants is None:
        raise RuntimeError(
            f"unsupported CLS-router control prior {control.final_variant!r}"
        )
    pure_final, routed_final = endpoint_variants
    return ClsDetailRouterDeployment(
        control_condition=control.effective_component_condition,
        stage_variant=control.stage_variant,
        control_final_variant=control.final_variant,
        pure_final_variant=pure_final,
        routed_final_variant=routed_final,
        interface_estimator="contribution",
        reason=(
            "The false route is the ratio-selected component endpoint; the true "
            "route changes only the final endpoint to the frozen pure-CLS quota."
        ),
    )


def _nonempty_children(module: Any) -> tuple[Any, ...]:
    if module is None:
        return ()
    try:
        return tuple(module)
    except TypeError:
        return ()


def classify_visual_interface_topology(model: Any) -> VisualInterfaceTopology:
    """Classify a loaded VLM using only its post-encoder module topology.

    A single shared projector or merger permits the bias-centered actual
    contribution norm(f(x_i)-f(0)) to cover the full token-aligned interface.
    When extra post-encoder injection mergers exist, one interface score is not
    path-complete, so the conservative policy keeps encoder saliency.
    """

    visual = getattr(model, "visual", None)
    main_merger = getattr(visual, "merger", None)
    if main_merger is not None:
        deepstack = _nonempty_children(
            getattr(visual, "deepstack_merger_list", None)
        )
        paths = ("main_merger",) + tuple(
            f"deepstack_merger_{index}" for index in range(len(deepstack))
        )
        if len(paths) == 1:
            return VisualInterfaceTopology(
                family="single_merger",
                path_names=paths,
                saliency_policy="interface_contribution",
                reason=(
                    "All post-encoder visual tokens traverse one shared merger, "
                    "so its bias-centered contribution is path-complete."
                ),
            )
        return VisualInterfaceTopology(
            family="multi_injection",
            path_names=paths,
            saliency_policy="encoder",
            reason=(
                "The model has a main merger plus additional post-encoder "
                "injection mergers; one merger-local score is not path-complete."
            ),
        )

    inner = getattr(model, "model", None)
    projector = getattr(inner, "multi_modal_projector", None)
    if projector is not None:
        return VisualInterfaceTopology(
            family="single_projector",
            path_names=("multi_modal_projector",),
            saliency_policy="interface_contribution",
            reason=(
                "The loaded decoder exposes one shared token-aligned multimodal "
                "projector and no additional post-encoder injection merger."
            ),
        )

    raise RuntimeError(
        "Unsupported visual interface: expected visual.merger or "
        "model.multi_modal_projector"
    )
