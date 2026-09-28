# DeFT: Deferred Text-Guided Visual Token Pruning

Code for *When Text Matters: Design Principles for Visual Token Pruning in Vision-Language Models*.

DeFT prunes visual tokens in two stages. Before the language model, visual attention selects a compact set of candidates while preserving a small reserve. At an intermediate decoder layer, text-to-visual attention selects the final tokens. The method requires no training or token merging. See [method details](docs/METHOD.md).

## Installation

Python 3.11 and a CUDA-capable GPU are required. See [environment details](docs/ENVIRONMENT.md).

```bash
pip install -e '.[eval]'
```

For Qwen benchmark evaluation, also install a FlashAttention-2 build compatible
with your PyTorch and CUDA versions.

## Models and data

Download a checkpoint from [Qwen3-VL-4B](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct), [Qwen3-VL-8B](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), or [LLaVA-OneVision-1.5-8B](https://huggingface.co/lmms-lab/LLaVA-OneVision-1.5-8B-Instruct), and pass its local directory as `--model-path`. The eight benchmark download links and preparation steps are in [data setup](docs/DATA.md). We do not redistribute weights or benchmark images.

## Inference

Run one image and prompt; the answer is printed to the terminal. This quick-start
example uses SDPA so it does not require FlashAttention-2.

```bash
deft-infer \
  --model-path /path/to/Qwen3-VL-8B-Instruct \
  --family qwen3 \
  --backend sdpa \
  --image /path/to/image.jpg \
  --prompt 'What is shown in the image?' \
  --prune 0.8 --alpha 0.2
```

`--prune` is the fraction of visual tokens removed. `--alpha` controls the fraction of otherwise-pruned tokens kept as candidates for the second stage.

## Evaluation

Run a benchmark split and save its predictions for scoring. Qwen evaluation
defaults to FlashAttention-2; LLaVA-OneVision defaults to SDPA. ChartQA and
InfoVQA are fetched automatically; other tasks may require local data
preparation.

```bash
deft-eval \
  --task chartqa \
  --family qwen3 \
  --model-path /path/to/Qwen3-VL-8B-Instruct \
  --prune 0.8 --alpha 0.2 \
  --output outputs/chartqa.jsonl
```

## Citation

Citation details will be added with the arXiv release.

## License

Released under the [Apache 2.0 license](LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.md) for included upstream code.
