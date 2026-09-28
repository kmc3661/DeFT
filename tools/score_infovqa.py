#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def normalize(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def levenshtein(left: str, right: str) -> int:
    if len(left) > len(right):
        left, right = right, left
    distances = list(range(len(left) + 1))
    for row, char_right in enumerate(right, start=1):
        updated = [row]
        for col, char_left in enumerate(left, start=1):
            if char_left == char_right:
                updated.append(distances[col - 1])
            else:
                updated.append(1 + min(distances[col - 1], distances[col], updated[-1]))
        distances = updated
    return distances[-1]


def anls(prediction: str, answers: list[str], threshold: float = 0.5) -> float:
    pred = normalize(prediction)
    similarities = []
    for answer in answers:
        gold = normalize(answer)
        width = max(len(gold), len(pred))
        similarity = 0.0 if width == 0 else 1.0 - levenshtein(gold, pred) / width
        similarities.append(similarity)
    best = max(similarities) if similarities else 0.0
    return best if best >= threshold else 0.0


def mean(values):
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--answers-file', type=Path, required=True)
    ap.add_argument('--score-file', type=Path, required=True)
    ap.add_argument('--expected', type=int, default=2801)
    args = ap.parse_args()
    rows=[json.loads(line) for line in args.answers_file.open(errors='replace') if line.strip()]
    identities=[(str(row.get('question_id')),str(row.get('prompt',''))) for row in rows]
    if len(rows)!=args.expected or len(identities)!=len(set(identities)):
        raise RuntimeError(f'answer identity audit failed: rows={len(rows)} expected={args.expected} unique={len(set(identities))}')
    scores=[]
    by_answer_type=defaultdict(list)
    by_operation=defaultdict(list)
    per_sample=[]
    for row in rows:
        metadata=row.get('metadata') or {}
        answers=metadata.get('answer')
        if not isinstance(answers,list) or not answers:
            raise RuntimeError(f'missing answer aliases: {row.get("question_id")}')
        score=anls(str(row.get('text','')), [str(x) for x in answers])
        scores.append(score)
        for value in metadata.get('answer_type') or ['unspecified']:
            by_answer_type[str(value)].append(score)
        for value in metadata.get('operation_reasoning') or ['none']:
            by_operation[str(value)].append(score)
        per_sample.append({'question_id':str(row['question_id']),'score':score,'prediction':str(row.get('text','')),'answers':[str(x) for x in answers]})
    result={
        'dataset':'infovqa','metric':'ANLS','score':mean(scores),'score_percent':100*mean(scores),'count':len(rows),
        'threshold':0.5,
        'protocol':'Official InfographicVQA/lmms-eval ANLS: whitespace-normalized lowercase Levenshtein similarity, max over aliases, values below 0.5 set to zero.',
        'by_answer_type':{key:{'score':mean(vals),'count':len(vals)} for key,vals in sorted(by_answer_type.items())},
        'by_operation_reasoning':{key:{'score':mean(vals),'count':len(vals)} for key,vals in sorted(by_operation.items())},
        'per_sample':per_sample,
        'answers_file':str(args.answers_file),
    }
    args.score_file.parent.mkdir(parents=True,exist_ok=True)
    args.score_file.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='per_sample'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
