# When Text Matters: Design Principles for Visual Token Pruning in Vision-Language Models

Official implementation of **DeFT** (Deferred Text-Guided Visual Token Pruning).

DeFT prunes visual tokens in two stages. Before the language model, visual attention selects a compact set of candidates while preserving a small reserve. At an intermediate decoder layer, text-to-visual attention selects the final tokens. The method requires no training or token merging.

## Installation

Python 3.11 and a CUDA-capable GPU are required. We recommend a separate environment for each model family; see [environment details](docs/ENVIRONMENT.md).

```bash
pip install -e '.[eval]'
```

Download the model weights separately. The supported checkpoints are Qwen3-VL-4B/8B-Instruct and LLaVA-OneVision-1.5-8B-Instruct.

## Inference

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

```bash
deft-eval \
  --task chartqa \
  --family qwen3 \
  --backend sdpa \
  --model-path /path/to/Qwen3-VL-8B-Instruct \
  --prune 0.8 --alpha 0.2 \
  --output outputs/chartqa.jsonl
```

See [data and metrics](docs/DATA.md) for all eight benchmarks, [reproduction settings](docs/REPRODUCIBILITY.md) for the paper experiments, and [method details](docs/METHOD.md) for the implementation. Model weights and benchmark data are not included.

## Citation

Citation details will be added with the arXiv release.

## License

Released under the [Apache 2.0 license](LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.md) for included upstream code.
