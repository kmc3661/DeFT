# Environment

Paper environments shared Transformers 4.57.3, NumPy 1.26.4, qwen-vl-utils 0.0.14,
and the lmms-eval 0.5.0 API. The Qwen3 wrapper missing from that API version is
included in `_vendor` with its license. The exact OV wrapper is also included.

| Use | PyTorch | Attention |
|---|---|---|
| Qwen full performance runs | 2.6.0+cu126 | flash-attn 2.8.3 |
| OV performance and reference SDPA environment | 2.9.0 | SDPA |

Use separate virtual environments when reproducing these environments. Install
the appropriate PyTorch build first, then `pip install -e '.[eval,test]'`.
For the Qwen environment, install `flash-attn==2.8.3 --no-build-isolation` with
a matching compiler/CUDA toolkit. Do not install an extension wheel built for a
different PyTorch ABI. Python 3.11 is the declared packaging target; the original
Qwen environment used Python 3.10. Fresh-environment installation is a separate
validation requirement, not implied by source import tests on the research host.

Caption metrics require Java (PTB tokenization/METEOR). Runtime model loading
can download files unless Hugging Face offline variables are set. Set `HF_HOME`
and `HF_DATASETS_CACHE` to your own locations. No original machine path is needed.

For a portable initial smoke test use SDPA. Use the paper backend for reported
benchmark comparisons. Neither different kernels nor newer Transformers builds
are guaranteed bitwise-equivalent.
