# Environment

Use Python 3.11 and a CUDA-enabled PyTorch installation. From the repository
root, install DeFT and its evaluation dependencies:

```bash
pip install -e '.[eval]'
```

Qwen evaluation uses FlashAttention-2 by default. Install a `flash-attn` build
compatible with your PyTorch and CUDA versions, or pass `--backend sdpa` for a
quick functional check. LLaVA-OneVision uses SDPA by default. Its checkpoint
provides custom model code, so review the checkpoint before running it.

Caption scoring requires Java. Model checkpoints and benchmark data are
downloaded separately; see the [README](../README.md) and [data setup](DATA.md).
