"""Public configuration; token counts use the paper's rounding conventions."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DeFTConfig:
    prune_ratio: float = 0.8
    alpha: float = 0.2
    selection_depth: float = 0.5

    def __post_init__(self):
        if not 0 <= self.prune_ratio < 1:
            raise ValueError("prune_ratio must be in [0, 1)")
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be in [0, 1]; the recommended sweep is [0.1, 0.3]")
        if not 0 < self.selection_depth < 1:
            raise ValueError("selection_depth must be between 0 and 1")

    def counts(self, visual_tokens: int) -> tuple[int, int]:
        if visual_tokens < 1:
            raise ValueError("visual_tokens must be positive")
        # Percent subtraction matches the evaluated p70/p80/p90 commands.
        keep = max(1, round(visual_tokens * ((100 - 100*self.prune_ratio)/100)))
        candidates = min(visual_tokens, keep + math.ceil(self.alpha*(visual_tokens-keep)))
        return candidates, keep

    def boundary(self, decoder_layers: int) -> int:
        blocks = int(decoder_layers*self.selection_depth)
        if not 1 <= blocks < decoder_layers:
            raise ValueError("selection depth leaves an empty decoder segment")
        return blocks-1

    def model_kwargs(self):
        return dict(
            use_cache=False,
            progressive_vision_pruning=True, progressive_text_masking=False,
            progressive_vision_layer=17,
            progressive_vision_keep_ratio=(100-100*self.prune_ratio)/100,
            progressive_vision_keep_tokens=0,
            progressive_vision_score_mode="encoder_pmi_add_topk",
            progressive_vision_score_lambda=1, progressive_vision_anchor_alpha=1,
            progressive_vision_stage_keep_tokens=0, progressive_vision_stage_keep_ratio=1,
            progressive_vision_stage_budget_mode="surplus_fraction",
            progressive_vision_stage_surplus_fraction=self.alpha,
            progressive_vision_stage_layer=0,
            progressive_vision_stage_score_mode="feature_diverse",
            progressive_vision_stage_physical_drop=True,
            progressive_vision_encoder_layer=-1,
            progressive_vision_merge=False, progressive_vision_merge_placement="none",
            progressive_vision_physical_drop=True,
            progressive_text_correction="exposure_baseline_ratio",
        )
