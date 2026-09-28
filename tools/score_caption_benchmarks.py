#!/usr/bin/env python
"""Official COCO-caption metrics for NoCaps validation and TextCaps val."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer


EXPECTED_ROWS = {"nocaps": 4500, "textcaps": 3166}


def read_rows(path: Path, dataset: str, expected: int | None) -> list[dict]:
    rows = [json.loads(line) for line in path.open(errors="replace") if line.strip()]
    identities = [(str(row.get("question_id")), str(row.get("prompt", ""))) for row in rows]
    if len(identities) != len(set(identities)):
        raise RuntimeError(f"duplicate answer identities in {path}")
    required = EXPECTED_ROWS[dataset] if expected is None else expected
    if len(rows) != required:
        raise RuntimeError(f"{dataset} row mismatch: {len(rows)} != {required}")
    return rows


def score_rows(rows: list[dict]) -> dict:
    gts = {}
    res = {}
    for idx, row in enumerate(rows):
        refs = row.get("metadata", {}).get("answer")
        if not isinstance(refs, list) or not refs:
            raise RuntimeError(f"missing caption references at row {idx}")
        gts[idx] = [{"caption": str(ref)} for ref in refs]
        res[idx] = [{"caption": str(row.get("text", ""))}]
    tokenizer = PTBTokenizer()
    tokenized_gts = tokenizer.tokenize(gts)
    tokenized_res = tokenizer.tokenize(res)
    bleu, _ = Bleu(4).compute_score(tokenized_gts, tokenized_res)
    meteor_scorer = Meteor()
    try:
        meteor, _ = meteor_scorer.compute_score(tokenized_gts, tokenized_res)
    finally:
        close = getattr(meteor_scorer, "close", None)
        if callable(close):
            close()
    rouge, _ = Rouge().compute_score(tokenized_gts, tokenized_res)
    cider, per_image_cider = Cider().compute_score(tokenized_gts, tokenized_res)
    return {
        "metric": "CIDEr",
        "score": float(cider),
        "CIDEr": float(cider),
        "BLEU-1": float(bleu[0]),
        "BLEU-2": float(bleu[1]),
        "BLEU-3": float(bleu[2]),
        "BLEU-4": float(bleu[3]),
        "METEOR": float(meteor),
        "ROUGE-L": float(rouge),
        "count": len(rows),
        "per_image_cider_mean": float(sum(per_image_cider) / len(per_image_cider)),
        "protocol": "lmms-eval NoCaps/TextCaps prompt and PTB-tokenized pycocoevalcap metrics",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(EXPECTED_ROWS), required=True)
    parser.add_argument("--answers-file", type=Path, required=True)
    parser.add_argument("--score-file", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=None)
    args = parser.parse_args()
    rows = read_rows(args.answers_file, args.dataset, args.expected)
    result = score_rows(rows)
    result.update({"dataset": args.dataset, "answers_file": str(args.answers_file)})
    args.score_file.parent.mkdir(parents=True, exist_ok=True)
    args.score_file.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
