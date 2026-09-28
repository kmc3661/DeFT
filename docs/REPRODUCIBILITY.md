# Reproduction scope

## Main and alpha evaluations

Use the three checkpoint families, all eight tasks, p70/p80/p90, alpha=.2,
midpoint selection, greedy decoding and the documented backend. Test alpha
.1/.15/.2/.25/.3 at p80 without changing any other setting. The final OV policy
is CLS-only initial selection. Do not mix results from older interface-based
OV ranking or old image/PMI midpoint scoring with this release.

All telemetry belongs to one output file. A process exits on failed assertions;
the public evaluator does not silently resume or overwrite partial answers.
After sharding, check unique IDs and full expected counts before scoring.

## Depth ablation

`--selection-depth .25`, `.5` and `.75` select after 9,18,27 blocks for a
36-block backbone. For the matched token-processing budget use

```
alpha_L = 0.2 * 18 / L
```

Thus L9 uses .4, L18 uses .2, L22 uses .163636..., and L27 uses .133333... .
Use `--selection-depth 0.6111111111111112` for L22. Integer rounding can leave
small per-input differences. Matching visual-token × decoder-block workload is
not exactly matching nonlinear attention FLOPs or measured latency. Fixed-alpha
depth sweeps are a different experiment and should be labeled accordingly.

## Timing

Paper measurements use one RTX A6000, SDPA, batch one, cache off, 100 fixed
InfoVQA inputs, warm-up passes and three timed repetitions. Decoder prefill
includes selection overhead. End-to-end starts at GPU-ready inputs and stops
at the first output token; CPU preprocessing, input transfer and later decoding
are excluded. End-to-end passes must run without profiler hooks. Never overlap
timing with another GPU job. The package's timing tool implements this boundary
and reports both quantities, but new-host timing is not a copy of paper results.

## Diagnostic and baseline scope

The question-depth diagnostic uses 300 images with two naturally occurring
questions per image on each of TextVQA, ChartQA, InfoVQA and AI2D. Each image is
partitioned into eight native-grid regions. Individually masking each region's
visual keys across all decoder blocks gives a reference-answer NLL difference.
Prompt-only per-layer attention is compared with this region-importance target
using Spearman correlation. Matching-minus-swapped gain keeps the image fixed.
Bootstrap resamples image clusters within task, then equally averages tasks.

The exploratory diagnostic runners, historical score-fusion ablations, baseline
ports and figure-editing scripts are not part of this cleaned implementation
package. Releasing those requires a separate dataset/attribution audit. The
included code reproduces the final DeFT policy and its basic budget sweeps,
not every historical analysis automatically.
