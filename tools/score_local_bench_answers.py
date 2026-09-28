#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import json
import re
import string
from collections import defaultdict
from pathlib import Path


def normalize_answer(text: str) -> str:
    text = str(text or "").lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def parse_yes_no(text: str):
    s = normalize_answer(text)
    if re.search(r"\byes\b", s):
        return "yes"
    if re.search(r"\bno\b", s):
        return "no"
    return None


def load_textvqa_evaluator():
    import importlib.util

    evaluator_path = Path(__file__).resolve().parents[2] / "llava/eval/m4c_evaluator.py"
    spec = importlib.util.spec_from_file_location("vispruner_textvqa_eval", evaluator_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load TextVQA evaluator from {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EvalAIAnswerProcessor()


_TEXTVQA_ANSWER_PROCESSOR = None


def textvqa_score(pred: str, answers) -> float:
    if not isinstance(answers, list) or len(answers) != 10:
        return 0.0
    global _TEXTVQA_ANSWER_PROCESSOR
    if _TEXTVQA_ANSWER_PROCESSOR is None:
        _TEXTVQA_ANSWER_PROCESSOR = load_textvqa_evaluator()
    processor = _TEXTVQA_ANSWER_PROCESSOR
    pred_n = processor(pred)
    matches = sum(processor(answer) == pred_n for answer in answers)
    same_answer_score = min(1.0, max(0, matches - 1) / 3.0)
    other_answer_score = min(1.0, matches / 3.0)
    return (
        matches * same_answer_score + (10 - matches) * other_answer_score
    ) / 10.0


def textvqa_question_from_prompt(prompt: str) -> str:
    lines = str(prompt).splitlines()
    if not lines:
        return ""
    if lines[0].startswith("Reference OCR token:") and len(lines) > 1:
        question = lines[1]
    else:
        question = lines[0]
    return " ".join(question.lower().strip().split())


def load_textvqa_ground_truth(question_file: Path | None) -> dict[tuple[str, str], list[str]]:
    if question_file is None:
        annotation_file = Path("playground/data/eval/textvqa/TextVQA_0.5.1_val.json")
    else:
        annotation_file = question_file.parent / "TextVQA_0.5.1_val.json"
        if not annotation_file.is_file():
            annotation_file = Path("playground/data/eval/textvqa/TextVQA_0.5.1_val.json")
    data = json.loads(annotation_file.read_text(errors="replace")).get("data", [])
    ground_truth = {
        (
            str(row.get("image_id")),
            " ".join(str(row.get("question", "")).lower().strip().split()),
        ): row.get("answers", [])
        for row in data
    }
    if len(ground_truth) != len(data):
        raise RuntimeError(
            f"TextVQA composite keys are not unique: rows={len(data)} keys={len(ground_truth)}"
        )
    return ground_truth


def chartqa_relaxed_correctness(prediction: str, target: str, max_relative_change: float = 0.05) -> bool:
    def to_float(text: str):
        try:
            value = str(text).strip()
            return float(value[:-1]) / 100.0 if value.endswith("%") else float(value)
        except ValueError:
            return None

    pred = str(prediction).strip()
    gold = str(target).strip()
    pred_float = to_float(pred)
    gold_float = to_float(gold)
    if pred_float is not None and gold_float not in (None, 0.0):
        return abs(pred_float - gold_float) / abs(gold_float) <= max_relative_change
    return pred.lower() == gold.lower()


def parse_choice(text: str, valid_letters: str = "ABCD"):
    if text is None:
        return None
    letters = "".join(x for x in str(valid_letters).upper() if x in string.ascii_uppercase)
    if not letters:
        return None
    char_class = re.escape(letters)
    s = str(text).strip()
    # Common direct formats: A, A., (A), Answer: A, The answer is A.
    m = re.search(
        rf"(?i)(?:^|\b)(?:answer\s*(?:is|:)?\s*)?\(?\s*([{char_class}])\s*\)?(?:\.|\b)",
        s,
    )
    if m:
        return m.group(1).upper()
    m = re.search(rf"\b([{char_class}])\b", s.upper())
    if m:
        return m.group(1).upper()
    return None


def open_answer_aliases(answer) -> list[str]:
    if isinstance(answer, list):
        return [str(x) for x in answer]
    value = str(answer)
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except (SyntaxError, ValueError):
            pass
    return [value]


# Deterministic port of MMMU eval/eval_utils.py open-response normalization.
def mmmu_extract_numbers(value: str) -> list[str]:
    comma = re.findall(r"-?\b\d{1,3}(?:,\d{3})+\b", value)
    scientific = re.findall(r"-?\d+(?:\.\d+)?[eE][+-]?\d+", value)
    simple = re.findall(r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])", value)
    return comma + scientific + simple


def mmmu_normalize(value) -> list[str | float]:
    value = str(value).strip()
    try:
        return [round(float(value.replace(",", "")), 2)]
    except ValueError:
        value = value.lower()
        return [" " + value, value + " "] if len(value) == 1 else [value]


def parse_mmmu_open_response(response: str) -> list[str | float]:
    response = str(response).strip().strip(".").lower()
    sub_responses = response.splitlines() or [response]
    indicators = ("could be ", "so ", "is ", "thus ", "therefore ", "final ", "answer ", "result ", "=")
    keys = []
    for part in sub_responses:
        candidates = [part.split(marker)[-1].strip() for marker in indicators if marker in part]
        candidates = [x for x in candidates if x and x not in {":", ",", ".", "!", "?", ";", "'"}]
        keys.append(min(candidates, key=len) if candidates else part.strip())
    parsed = list(keys)
    for key in keys:
        parsed.extend(mmmu_extract_numbers(key))
    normalized = []
    for value in parsed:
        normalized.extend(mmmu_normalize(value))
    return list(dict.fromkeys(normalized))


def eval_mmmu_open(answer, parsed: list[str | float]) -> bool:
    normalized_answers = []
    for alias in open_answer_aliases(answer):
        normalized_answers.extend(mmmu_normalize(alias))
    for pred in parsed:
        if isinstance(pred, str):
            if any(isinstance(ans, str) and ans in pred for ans in normalized_answers):
                return True
        elif pred in normalized_answers:
            return True
    return False


def pope_local_question_id(value) -> str:
    return str(int(value) % 10_000_000)


def load_pope_ground_truth(question_file: Path) -> dict[tuple[str, str], str]:
    root = question_file.parent
    ground_truth = {}
    for category in ("adversarial", "popular", "random"):
        path = root / f"coco_pope_{category}.json"
        if not path.is_file():
            path = root / "coco" / f"coco_pope_{category}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing POPE annotation: {path}")
        for line in path.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            ground_truth[(category, pope_local_question_id(row["question_id"]))] = str(row["label"]).lower()
    return ground_truth


def parse_pope_prediction(text: str) -> str:
    # Match llava/eval/eval_pope.py: first sentence, then any no/not token means no.
    first_sentence = str(text or "").split(".", 1)[0].replace(",", "")
    words = first_sentence.split(" ")
    return "no" if any(word in {"No", "no", "not"} for word in words) else "yes"


def pope_category_metrics(pairs: list[tuple[str, str]]) -> dict[str, float | int]:
    tp = sum(pred == gt == "yes" for pred, gt in pairs)
    tn = sum(pred == gt == "no" for pred, gt in pairs)
    fp = sum(pred == "yes" and gt == "no" for pred, gt in pairs)
    fn = sum(pred == "no" and gt == "yes" for pred, gt in pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "total": len(pairs), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": (tp + tn) / len(pairs) if pairs else 0.0,
        "precision": precision, "recall": recall, "f1": f1,
        "yes_ratio": sum(pred == "yes" for pred, _ in pairs) / len(pairs) if pairs else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers-file", type=Path, required=True)
    ap.add_argument("--score-file", type=Path, required=True)
    ap.add_argument("--dataset", default="")
    ap.add_argument("--question-file", type=Path, default=None)
    ap.add_argument("--allow-incomplete-pope-categories", action="store_true")
    args = ap.parse_args()

    dataset = str(args.dataset).lower()
    pope_gt = {}
    textvqa_gt = {}
    if dataset == "pope":
        question_file = args.question_file or Path("playground/data/eval/pope/llava_pope_test.jsonl")
        pope_gt = load_pope_ground_truth(question_file)
    elif dataset in {"textvqa", "textvqa_val", "textvqa_dev"}:
        textvqa_gt = load_textvqa_ground_truth(args.question_file)

    total = 0
    correct = 0.0
    missing_gt = 0
    unparsed = 0
    pope_pairs = defaultdict(list)
    for line in args.answers_file.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        metadata = r.get("metadata") or {}
        gt = metadata.get("answer")
        category = str(metadata.get("category") or "")
        if dataset == "pope" and gt is None:
            gt = pope_gt.get((category, pope_local_question_id(r.get("question_id"))))
        elif dataset in {"textvqa", "textvqa_val", "textvqa_dev"}:
            gt = textvqa_gt.get(
                (
                    str(r.get("question_id")),
                    textvqa_question_from_prompt(r.get("prompt", "")),
                )
            )
        if gt is None:
            missing_gt += 1
            continue
        text = r.get("text", "")
        if dataset == "mmmu_dev":
            options = metadata.get("options") or []
            question_type = str(metadata.get("question_type") or "").lower()
            if options and question_type != "open":
                valid_letters = string.ascii_uppercase[: len(options)]
                gt_norm = str(gt).strip().upper()
                pred = parse_choice(text, valid_letters)
                if pred is None:
                    unparsed += 1
                ok = pred == gt_norm
            else:
                pred = parse_mmmu_open_response(text)
                ok = bool(pred) and eval_mmmu_open(gt, pred)
                if not pred:
                    unparsed += 1
            score = float(ok)
        elif dataset in {"textvqa", "textvqa_val", "textvqa_dev"} and isinstance(gt, list):
            score = textvqa_score(text, gt)
            pred = normalize_answer(text)
            ok = score > 0
        elif dataset == "pope":
            gt_norm = str(gt).strip().lower()
            pred = parse_pope_prediction(text)
            ok = pred == gt_norm
            score = float(ok)
            pope_pairs[category].append((pred, gt_norm))
        elif str(gt).strip().lower() in {"yes", "no"}:
            gt_norm = str(gt).strip().lower()
            pred = parse_yes_no(text)
            if pred is None:
                unparsed += 1
            ok = pred == gt_norm
            score = float(ok)
        elif dataset == "gqa":
            gt_norm = normalize_answer(gt)
            pred = normalize_answer(text)
            ok = pred == gt_norm
            score = float(ok)
        elif dataset == "chartqa":
            pred = str(text).strip()
            ok = chartqa_relaxed_correctness(pred, str(gt))
            score = float(ok)
        else:
            gt_norm = str(gt).strip().upper()
            pred = parse_choice(text)
            if pred is None:
                unparsed += 1
            ok = pred == gt_norm
            score = float(ok)
        total += 1
        correct += float(score)

    if total <= 0:
        raise RuntimeError(f"no scorable rows in {args.answers_file}")
    if dataset == "pope" and missing_gt:
        raise RuntimeError(f"POPE ground truth missing for {missing_gt} rows")
    if dataset in {"textvqa", "textvqa_val", "textvqa_dev"} and missing_gt:
        raise RuntimeError(f"TextVQA ground truth missing for {missing_gt} rows")

    accuracy = correct / total
    out = {
        "dataset": args.dataset,
        "metric": "acc",
        "acc": accuracy,
        "correct": correct,
        "total": total,
        "missing_gt": missing_gt,
        "unparsed": unparsed,
    }
    if dataset == "chartqa":
        out.update({"metric": "chartqa_relaxed_accuracy", "chartqa_relaxed_accuracy": accuracy})
    if dataset == "pope":
        category_metrics = {
            category: pope_category_metrics(pairs)
            for category, pairs in sorted(pope_pairs.items()) if pairs
        }
        expected_categories = {"adversarial", "popular", "random"}
        if not expected_categories.issubset(category_metrics) and not args.allow_incomplete_pope_categories:
            raise RuntimeError(
                f"POPE categories incomplete: expected={sorted(expected_categories)} "
                f"actual={sorted(category_metrics)}"
            )
        averaged_categories = expected_categories if expected_categories.issubset(category_metrics) else set(category_metrics)
        average_f1 = sum(category_metrics[x]["f1"] for x in averaged_categories) / len(averaged_categories)
        out.update({
            "metric": "pope_f1",
            "acc": average_f1,
            "pope_f1": average_f1,
            "accuracy": accuracy,
            "category_metrics": category_metrics,
        })

    args.score_file.parent.mkdir(parents=True, exist_ok=True)
    args.score_file.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
