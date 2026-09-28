# Benchmark data

The evaluation data is available from the original sources below. Download it
separately and follow each dataset's terms of use.

| Task | Download | Split |
|---|---|---|
| TextVQA | [TextVQA](https://textvqa.org/dataset/) | Validation |
| MMMU | [MMMU](https://huggingface.co/datasets/lmms-lab/MMMU) | Validation |
| AI2D | [AI2D](https://huggingface.co/datasets/lmms-lab/ai2d) | Test |
| MMStar | [MMStar](https://huggingface.co/datasets/Lin-Chen/MMStar) | `mmstar.parquet` |
| ChartQA | [ChartQA](https://huggingface.co/datasets/lmms-lab/ChartQA) | Test |
| InfoVQA | [DocVQA / InfographicVQA](https://huggingface.co/datasets/lmms-lab/DocVQA) | InfographicVQA validation |
| TextCaps | [TextCaps](https://huggingface.co/datasets/lmms-lab/TextCaps) | Validation |
| NoCaps | [NoCaps](https://huggingface.co/datasets/lmms-lab/NoCaps) | Validation |

`deft-eval` downloads ChartQA and InfoVQA automatically. For TextCaps and
NoCaps, run `python tools/download_caption_data.py` before evaluation.

For datasets requiring local files:

- **TextVQA:** Download the official validation annotations and training-image
  archive (which contains the validation images). Run
  `python tools/prepare_textvqa.py --annotations /path/to/TextVQA_0.5.1_val.json --output-dir data/textvqa`.
  Pass `--question-file data/textvqa/questions.jsonl` and the extracted image
  directory as `--image-folder`.
- **AI2D and MMMU:** Download the linked test/validation Parquet files. Run
  `python tools/prepare_qa_data.py --task ai2d --source /path/to/test-*.parquet --output-root data`
  or the same command with `--task mmmu` and its validation Parquet file. Pass
  `data/<task>/questions.jsonl` as `--question-file` and `data/<task>` as
  `--image-folder`.
- **MMStar:** Download `mmstar.parquet` and pass its path as `--question-file`.

Use `deft-eval --help` for all task names and options. Evaluation writes model
predictions; scoring utilities are in `tools/`.
