#!/usr/bin/env python3
import argparse
import ast
import json
import re
from pathlib import Path

from datasets import Dataset


MMMU_SOURCE = None
AI2D_SOURCES = []


def load_source(path):
    if path.suffix == ".arrow":
        return Dataset.from_file(str(path))
    if path.suffix == ".parquet":
        return Dataset.from_parquet(str(path))
    raise ValueError(f"Expected an Arrow or Parquet file: {path}")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def prepare_mmmu(output_root):
    output_dir = output_root / "mmmu"
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_source(MMMU_SOURCE)
    rows = []
    for doc in dataset:
        options = ast.literal_eval(doc["options"])
        choices = "\n".join(f"{chr(65 + i)}. {option}" for i, option in enumerate(options))
        if doc["question_type"] == "multiple-choice":
            text = (
                f'{doc["question"]}\n{choices}\n\n'
                "Answer with the option's letter from the given choices directly."
            )
        else:
            text = f'{doc["question"]}\n\nAnswer the question using a single word or phrase.'

        image_paths_by_key = {}
        for token in sorted(set(re.findall(r"<image \d+>", text))):
            image_key = token.strip("<>").replace(" ", "_")
            image_path = image_dir / f'{doc["id"]}_{image_key}.png'
            if not image_path.exists():
                doc[image_key].convert("RGB").save(image_path)
            image_paths_by_key[token] = str(image_path.relative_to(output_dir))
        # Each image placeholder needs one matching visual feature.
        image_paths = [image_paths_by_key[token] for token in re.findall(r"<image \d+>", text)]
        text = re.sub(r"<image \d+>", "<image>", text)
        rows.append(
            {
                "question_id": doc["id"],
                "text": text,
                "images": image_paths,
                "answer": doc["answer"],
                "question_type": doc["question_type"],
                "options": options,
                "subfield": doc["subfield"],
            }
        )
    write_jsonl(output_dir / "questions.jsonl", rows)
    print(f"MMMU validation: {len(rows)} samples -> {output_dir}")


def prepare_ai2d(output_root):
    output_dir = output_root / "ai2d"
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    datasets = [load_source(path) for path in AI2D_SOURCES]
    rows = []
    index = 0
    for dataset in datasets:
        for doc in dataset:
            choices = "\n".join(f"{chr(65 + i)}. {option}" for i, option in enumerate(doc["options"]))
            text = (
                f"<image>\n{doc['question']}\n{choices}\n"
                "Answer with the option's letter from the given choices directly."
            )
            image_path = image_dir / f"{index:05d}.png"
            if not image_path.exists():
                doc["image"].convert("RGB").save(image_path)
            rows.append(
                {
                    "question_id": f"ai2d_{index}",
                    "text": text,
                    "images": [str(image_path.relative_to(output_dir))],
                    "answer": chr(65 + int(doc["answer"])),
                    "options": doc["options"],
                }
            )
            index += 1
    write_jsonl(output_dir / "questions.jsonl", rows)
    print(f"AI2D test: {len(rows)} samples -> {output_dir}")


def main():
    global MMMU_SOURCE, AI2D_SOURCES
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument("--task", choices=["ai2d","mmmu"], required=True)
    parser.add_argument("--source", "--arrow", type=Path, nargs="+", required=True,
                        help="Official Arrow or Parquet files in original shard order")
    args = parser.parse_args()
    if (args.output_root/args.task).exists():
        parser.error("Task output directory already exists; refusing to overwrite")
    if not all(p.is_file() and p.suffix in {".arrow", ".parquet"} for p in args.source):
        parser.error("Missing or unsupported input file (expected Arrow or Parquet)")
    if args.task=="mmmu":
        if len(args.source)!=1:parser.error("MMMU requires one validation file")
        MMMU_SOURCE=args.source[0]
        prepare_mmmu(args.output_root)
    else:
        AI2D_SOURCES=args.source
        prepare_ai2d(args.output_root)


if __name__ == "__main__":
    main()
