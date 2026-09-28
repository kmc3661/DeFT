# Release validation

## Completed

- CPU tests: 15 passed (budgets, rounding, depth, input manifest, archive scan,
  normalized ZIP metadata and deterministic archive creation).
- All included Python sources compile. Inference and evaluation CLI help runs.
- Private backend imports successfully without the original research tree on
  Python's import path.
- A wheel builds successfully without installing into or modifying the running
  experiment environments. The unpacked wheel's backend imports from an isolated
  temporary directory, without the source checkout on Python's path.
- TextVQA preparation reproduces all 5,000 original question-file records exactly.
- GPU replay against the original final-method runner: two AI2D and two TextCaps inputs on each
  of Qwen3-VL-4B, Qwen3-VL-8B and LLaVA-OV-1.5-8B. Generated answers, cumulative
  candidate/final counts, and final original-token IDs match exactly on all
  generation passes (12 inputs total). See `gpu_smoke_validation.json` and
  `gpu_caption_validation.json`.
- Source packaging excludes model weights, datasets, caches, build artifacts,
  logs, symlinks and machine-specific source paths. ZIP integrity is checked and
  per-file hashes are included. Upstream attribution remains intact.

## Validation limits

- Smoke parity is not a new full-benchmark evaluation of the packaged code.
- Clean-environment dependency installation has not been validated from scratch.
  CUDA extension ABI compatibility remains environment-specific.
- The portable latency utility uses fixed warm-up passes; paper numbers used
  an adaptive warm-up/stability controller. Do not claim the new utility has
  already reproduced the latency table. It deliberately refuses concurrent GPU
  jobs, and was not timed while the alpha=.3 performance sweep was running.
- Original diagnostic/baseline reproduction scripts are outside this cleaned
  package's scope; see `REPRODUCIBILITY.md`.
- The source package does not contain model weights, benchmark data or every
  exploratory analysis. Check the original licenses before redistributing
  third-party checkpoints or datasets.
