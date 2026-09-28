"""Final DeFT policy on the frozen sequence-compaction backend.

Internal historical mode names are initialization compatibility identifiers.
The final selector below explicitly excludes both encoder priors and PMI.
"""
import os
import sys
from pathlib import Path
from types import MethodType
from .config import DeFTConfig

_prepared = False


def prepare_backend():
    global _prepared
    if _prepared:
        return
    # Fail closed: do not let an experimental shell silently change the method.
    conflicting = [k for k in os.environ if k.startswith("VISPRUNER_")]
    if conflicting:
        raise RuntimeError("Unset inherited VISPRUNER_* experiment variables: " + ", ".join(conflicting))
    vendor = Path(__file__).parent/"_vendor"
    sys.path.insert(0, str(vendor))
    sys.path.insert(0, str(vendor/"scripts"))
    os.environ.update(
        VISPRUNER_INTERFACE_COMPONENT_ABLATION="1",
        VISPRUNER_INTERFACE_FINAL="encoder_pmi_functional_user_cls_stage",
        VISPRUNER_INTERFACE_ESTIMATOR="contribution", VISPRUNER_INTERFACE_BRANCHES="main",
        VISPRUNER_ENCODER_SALIENCY_AGGREGATION="last1",
        VISPRUNER_L9_MULTIVIEW_RUNTIME="1", VISPRUNER_L9_MULTIVIEW_MODE="global_feature",
        VISPRUNER_L9_BALANCE_MODE="functional",
        VISPRUNER_CAPTION_PROMPT="Output only a concise caption of at most 12 words for the image.",
        VISPRUNER_CAPTION_MAX_NEW_TOKENS="32",
    )
    from vispruner_qwen_ov import l9_uniform_start_patch, l9_uniform_channel_ablation_patch
    from vispruner_qwen_ov import l9_global_pmi_balance_patch, l9_multiview_coverage_patch
    l9_uniform_start_patch.install()
    l9_uniform_channel_ablation_patch.set_mode("joint_no_residual_update")
    l9_uniform_channel_ablation_patch.install()
    l9_global_pmi_balance_patch.install()
    l9_multiview_coverage_patch.install()
    _prepared = True


def install_policy(model, family: str, config: DeFTConfig, audit=True):
    import torch
    from vispruner_qwen_ov.common_candidate_runtime import configure_common_candidate_runtime
    state = model._progressive
    if getattr(state, "_deft_installed", False):
        raise RuntimeError("DeFT is already installed on this model")
    configure_common_candidate_runtime(model)
    state._functional_mass_reuse_enabled = False
    state._l9_coverage_fast_mode = "key_greedy"
    state.config.vision_layer = config.boundary(state.config.num_layers)
    state.config.vision_stage_surplus_fraction = config.alpha
    state.config.vision_stage_budget_mode = "surplus_fraction"
    state.config.vision_merge = False
    state.config.vision_merge_placement = "none"

    def initial(self, score, features, keep_n, anchor_n):
        values = self._cls_detail_stage_score if family == "ov15" else score
        if values is None or values.numel() != features.shape[0]:
            raise RuntimeError("Visual scores do not align with native visual tokens")
        selected = torch.argsort(values.detach().float().flatten(), descending=True, stable=True)[:int(keep_n)]
        self.record("deft_initial_events", 1)
        self.record("deft_initial_selected", int(keep_n))
        return selected.to(features.device)
    state._select_vision_keep_feature_diverse = MethodType(initial, state)
    original = state._select_vision_drop
    aligned = state._aligned_encoder_score_for_vision_idx
    diagnostic = state._record_vision_score_analysis

    def verify(self, **kw):
        assert kw['mode'] == 'text_topk'
        assert kw['score_encoder'] is None and kw['score_pmi'] is None
        scores = kw['score_text'].float()
        assert torch.isfinite(scores).all() and torch.equal(kw['score_final'], kw['score_text'])
        expected = torch.zeros_like(kw['keep_mask'])
        expected[torch.topk(scores, kw['keep_n']).indices] = True
        assert torch.equal(expected, kw['keep_mask'])
        self.record("deft_text_only_verified", 1)
        return diagnostic(**kw)

    def select(self, *args, **kw):
        assert kw.get('stage_budget_mode') is None
        old = self.config.vision_score_mode
        self.config.vision_score_mode = "text_topk"
        self._aligned_encoder_score_for_vision_idx = lambda *a, **k: None
        if audit:
            self._record_vision_score_analysis = MethodType(verify, self)
        try:
            return original(*args, **kw)
        finally:
            self.config.vision_score_mode = old
            self._aligned_encoder_score_for_vision_idx = aligned
            self._record_vision_score_analysis = diagnostic
    state._select_vision_drop = MethodType(select, state)
    state._deft_installed = True
    model._deft_config = config
    model.use_cache = False
    return model


def load_model(model_path, family="qwen3", config=None, backend=None, device="cuda:0", audit=True):
    if family not in ("qwen3", "ov15"):
        raise ValueError("family must be qwen3 (4B/8B) or ov15")
    prepare_backend()
    from vispruner_qwen_ov.models import VisPrunerQwen3VL, VisPrunerLlavaOV15
    config = config or DeFTConfig()
    cls = VisPrunerQwen3VL if family == "qwen3" else VisPrunerLlavaOV15
    backend = backend or ("flash_attention_2" if family == "qwen3" else "sdpa")
    model = cls(pretrained=str(model_path), device=device, device_map=device,
                attn_implementation=backend, min_pixels=200704, max_pixels=1605632,
                **config.model_kwargs())
    model.model.eval()
    return install_policy(model, family, config, audit=audit)
