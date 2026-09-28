#!/usr/bin/env python
"""Caption-benchmark adapter for the audited Qwen3/OneVision local runner.

The adapter intentionally leaves ``run_qwen_ov_local_bench.py`` unchanged so
the fingerprints of completed QA experiments remain valid.  It adds the
official NoCaps validation and TextCaps val parquet splits while reusing the
same model construction, image preprocessing, decoding, and telemetry path.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from datasets import load_dataset as hf_load_dataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_qwen_ov_local_bench as base_runner


CAPTION_SPECS = {
    "nocaps": {
        "repo": "lmms-lab/NoCaps",
        "revision": "a26b3fe1e0021164ec430c57c48085d58f2fe922",
        "glob": "validation-*.parquet",
        "rows": 4500,
    },
    "textcaps": {
        "repo": "lmms-lab/TextCaps",
        "revision": "e9e223338832318a6161d04648871c18a024ce85",
        "glob": "val-*.parquet",
        "rows": 3166,
    },
}

CAPTION_PROMPT = os.environ.get(
    "VISPRUNER_CAPTION_PROMPT",
    "Provide a one-sentence caption for the provided image.",
)
_ORIGINAL_LOAD_DATASET = base_runner.load_dataset
_ORIGINAL_GENERATE_ANSWERS = base_runner.generate_answers


def _snapshot_data_dir(spec: dict) -> Path:
    hub_cache = Path(
        os.environ.get(
            "HF_HUB_CACHE",
            str(Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"),
        )
    )
    repo_dir = "datasets--" + spec["repo"].replace("/", "--")
    return hub_cache / repo_dir / "snapshots" / spec["revision"] / "data"


def load_caption_dataset(name: str) -> list[dict]:
    spec = CAPTION_SPECS[name]
    data_dir = _snapshot_data_dir(spec)
    files = sorted(data_dir.glob(spec["glob"]))
    if not files:
        raise FileNotFoundError(
            f"missing pinned {name} parquet files: {data_dir / spec['glob']}"
        )
    dataset = hf_load_dataset(
        "parquet",
        data_files=[str(path) for path in files],
        split="train",
        cache_dir=os.environ.get("HF_DATASETS_CACHE"),
    )
    if len(dataset) != int(spec["rows"]):
        raise RuntimeError(f"{name} row mismatch: {len(dataset)} != {spec['rows']}")
    metadata = dataset.remove_columns(["image"])
    rows = []
    for idx in range(len(metadata)):
        row = metadata[idx]
        if name == "nocaps":
            image_id = int(row["image_id"])
            references = [str(value) for value in row["annotations_captions"]]
        else:
            image_id = str(row["image_id"])
            references = [str(value) for value in row["caption_str"]]
        rows.append(
            {
                "question_id": f"{name}-{image_id}",
                "prompt": CAPTION_PROMPT,
                "_hf_dataset": dataset,
                "_hf_index": idx,
                "extra": {
                    "answer": references,
                    "image_id": image_id,
                    "dataset_revision": spec["revision"],
                },
            }
        )
    identities = [(row["question_id"], row["prompt"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise RuntimeError(f"duplicate {name} image identities")
    return rows


def load_dataset(args):
    if args.dataset in CAPTION_SPECS:
        return load_caption_dataset(args.dataset)
    return _ORIGINAL_LOAD_DATASET(args)


def generate_answers(args):
    requested = os.environ.get("VISPRUNER_CAPTION_DATASET", "").strip().lower()
    if requested:
        if requested not in CAPTION_SPECS:
            raise ValueError(f"unsupported caption dataset: {requested}")
        args.dataset = requested
    return _ORIGINAL_GENERATE_ANSWERS(args)


def main() -> None:
    try:
        index = sys.argv.index("--dataset") + 1
        requested = sys.argv[index].strip().lower()
    except (ValueError, IndexError) as exc:
        raise SystemExit("--dataset {nocaps,textcaps} is required") from exc
    if requested not in CAPTION_SPECS:
        raise SystemExit(f"unsupported caption dataset: {requested}")
    os.environ["VISPRUNER_CAPTION_DATASET"] = requested
    # The base parser predates these tasks.  Use an accepted placeholder and
    # restore the requested dataset before loading/generation.
    sys.argv[index] = "chartqa_official"
    base_runner.load_dataset = load_dataset
    base_runner.generate_answers = generate_answers
    base_runner.main()


if __name__ == "__main__":
    main()
