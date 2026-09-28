# Method

DeFT performs two-stage, training-free visual-token pruning:

1. **Before the decoder:** Rank visual tokens by vision-encoder attention and
   retain an expanded candidate set. Qwen uses received visual attention;
   LLaVA-OneVision uses CLS attention.
2. **At an intermediate decoder layer:** Rank the candidates by text-to-visual
   attention and keep the final visual-token budget. Text tokens are unchanged.

For `N` visual tokens, pruning ratio `p`, and candidate-reserve fraction `alpha`:

```text
Final tokens:       K = max(1, round(N × (1 − p)))
Initial candidates: M = min(N, K + ceil(alpha × (N − K)))
```

The default is `p=0.8`, `alpha=0.2`, with final selection halfway through the
decoder. `--prune` and `--alpha` expose these settings in both command-line
interfaces. DeFT physically removes discarded visual tokens; it does not merge
them or use a learned scoring module.
