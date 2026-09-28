# Data and metrics

## Benchmark sources

Datasets and images must be acquired separately under their original terms.
No model output is used to select evaluation examples. Main scores use full sets.

| Task | Source used | Split |
|---|---|---|
| TextVQA | [Official TextVQA v0.5.1](https://textvqa.org/dataset/) | Validation annotations and images |
| MMMU | [lmms-lab/MMMU](https://huggingface.co/datasets/lmms-lab/MMMU) | Validation |
| AI2D | [lmms-lab/ai2d](https://huggingface.co/datasets/lmms-lab/ai2d) | Test |
| MMStar | [Lin-Chen/MMStar](https://huggingface.co/datasets/Lin-Chen/MMStar) | `mmstar.parquet` |
| ChartQA | [lmms-lab/ChartQA](https://huggingface.co/datasets/lmms-lab/ChartQA) | Test |
| InfoVQA | [lmms-lab/DocVQA](https://huggingface.co/datasets/lmms-lab/DocVQA) | InfographicVQA validation |
| TextCaps | [lmms-lab/TextCaps](https://huggingface.co/datasets/lmms-lab/TextCaps) | Validation |
| NoCaps | [lmms-lab/NoCaps](https://huggingface.co/datasets/lmms-lab/NoCaps) | Validation |

`deft-eval` downloads the ChartQA and InfoVQA splits automatically. For
the two captioning tasks, run `python tools/download_caption_data.py`; it
downloads only the validation parquet files into the Hugging Face cache.
The other tasks need local files:

```bash
hf download lmms-lab/ai2d data/test-00000-of-00002.parquet data/test-00001-of-00002.parquet \
  --repo-type dataset --local-dir data/raw/ai2d
python tools/prepare_qa_data.py --task ai2d --output-root data \
  --source data/raw/ai2d/data/test-00000-of-00002.parquet data/raw/ai2d/data/test-00001-of-00002.parquet

hf download lmms-lab/MMMU data/validation-00000-of-00001.parquet \
  --repo-type dataset --local-dir data/raw/mmmu
python tools/prepare_qa_data.py --task mmmu --output-root data \
  --source data/raw/mmmu/data/validation-00000-of-00001.parquet

hf download Lin-Chen/MMStar mmstar.parquet --repo-type dataset \
  --local-dir data/mmstar
```

Pass `--question-file data/ai2d/questions.jsonl --image-folder data/ai2d` for
AI2D, the corresponding `data/mmmu` paths for MMMU, or
`--question-file data/mmstar/mmstar.parquet` for MMStar.

| Task | Inputs | Metric | Maximum generated tokens |
|---|---:|---|---:|
| TextVQA validation | 5000 | VQA accuracy | 16 |
| MMMU validation | 900 | Accuracy | 32 |
| AI2D | 3088 | Accuracy | 4 |
| MMStar | 1500 | Accuracy | 4 |
| ChartQA test | 2500 | Relaxed accuracy | 16 |
| InfoVQA validation | 2801 | ANLS | 32 |
| TextCaps validation | 3166 | CIDEr | 32 |
| NoCaps validation | 4500 | CIDEr | 32 |

For TextVQA supply the canonical JSONL (`question_id`, `text`, `image`) and image
directory. Keep the official `TextVQA_0.5.1_val.json` annotations beside the question
file (or its `data/` directory) for reference lookup. Download the official
training-image archive (which contains the validation images) and extract it to
`data/textvqa/train_images`. Do not prepend OCR tokens. Prepare the no-OCR
questions with:

```bash
python tools/prepare_textvqa.py \
  --annotations /path/to/TextVQA_0.5.1_val.json --output-dir data/textvqa
```

Then pass `--question-file data/textvqa/questions.jsonl --image-folder
data/textvqa/train_images` to `deft-eval`.

For AI2D/MMMU supply canonical JSONL with `question_id`, `text`, `images` (relative
paths), `answer`, and optionally `options`/`question_type`. The loader preserves
MMMU multi-image prompts. MMStar takes its benchmark parquet as `--question-file`.

Prepared AI2D/MMMU/TextVQA files are **not redistributed here**. When preparing
them from the official benchmarks, preserve question wording, image order,
options, splits and answer format. A newly formatted dataset is not automatically
an exact paper replay. See the retained loader for the precise accepted schema.

`tools/prepare_qa_data.py` accepts either the downloaded Parquet files shown
above or cached Arrow files, in their original shard order. It preserves the
AI2D/MMMU prompt formatting. Verify IDs, counts and prompts when preparing
data from a source other than the linked benchmarks.

Caption prompt: `Output only a concise caption of at most 12 words for the image.`
Generation is greedy, batch one, cache off. Reference captions are stored only
as output metadata for evaluation and are not passed to the pruning selector.

Scorers are preserved under `tools/`. Caption CIDEr from the evaluator is on its
native scale; multiply by 100 for the paper's displayed CIDEr values. Recovery
is `100 * method_score / dense_score`, computed separately for each task and
then averaged equally. Do not average different tasks' raw metrics.

## Fixed cost inputs

`configs/infovqa_latency_100.jsonl` contains only IDs, hashes and image-size
metadata. Starting from a deterministic image-disjoint pool of 500 inputs
(one question per image), examples were sorted by clipped image-area proxy,
aspect ratio and fixed hash. One hash-selected example from each of 100 rank
strata was retained. The area proxy is clipped to [200704,1605632], not an exact
native-token count. Sampling seed string: `20260905-paper-efficiency`.
All configurations use this same manifest. No correctness/performance filtering
was used. The manifest's external image URLs are identifiers, not bundled images.
