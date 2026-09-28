# Data and metrics

Datasets and images must be acquired separately under their original terms.
No model output is used to select evaluation examples. Main scores use full sets.

| Task | Inputs | Metric | Maximum generated tokens |
|---|---:|---|---:|
| TextVQA validation | 5000 | VQA accuracy | 16 |
| MMMU development | 900 | Accuracy | 32 |
| AI2D | 3088 | Accuracy | 4 |
| MMStar | 1500 | Accuracy | 4 |
| ChartQA test | 2500 | Relaxed accuracy | 16 |
| InfoVQA validation | 2801 | ANLS | 32 |
| TextCaps validation | 3166 | CIDEr | 32 |
| NoCaps validation | 4500 | CIDEr | 32 |

ChartQA and InfoVQA are loaded through the Hugging Face datasets API. Caption
inputs use pinned parquet snapshots; run `python tools/download_caption_data.py`
before caption evaluation. Revisions are recorded in the retained loaders.

For TextVQA supply the canonical JSONL (`question_id`, `text`, `image`) and image
directory. Keep the official `TextVQA_0.5.1_val.json` annotations beside the question
file (or its `data/` directory) for reference lookup. Do not prepend OCR tokens.
Use `python tools/prepare_textvqa.py --annotations /path/to/TextVQA_0.5.1_val.json
--output-dir data/textvqa` to create the no-OCR questions and copy annotations.
For AI2D/MMMU supply canonical JSONL with `question_id`, `text`, `images` (relative
paths), `answer`, and optionally `options`/`question_type`. The loader preserves
MMMU multi-image prompts. MMStar takes its benchmark parquet as `--question-file`.

Prepared AI2D/MMMU/TextVQA files are **not redistributed here**. When preparing
them from the official benchmarks, preserve question wording, image order,
options, splits and answer format. A newly formatted dataset is not automatically
an exact paper replay. See the retained loader for the precise accepted schema.

`tools/prepare_qa_data.py` preserves the original AI2D/MMMU formatting while
accepting explicit cached Arrow paths instead of machine-specific directories:

```bash
python tools/prepare_qa_data.py --task ai2d --output-root data \
  --arrow /path/to/ai2d-test-00000-of-00002.arrow /path/to/ai2d-test-00001-of-00002.arrow
python tools/prepare_qa_data.py --task mmmu --output-root data \
  --arrow /path/to/mmmu-validation.arrow
```

The original cached source fingerprints were AI2D
`c83a9b9692933aff8349157c88a413df9d02c4e5` and MMMU
`364f2e2eb107b36e07ff4c5a15f5947a759cef47`. These are cache fingerprints,
not necessarily downloadable Git revisions. Verify IDs, counts and prompts.

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
