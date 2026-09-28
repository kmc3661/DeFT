#!/usr/bin/env python
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import sys
import time
import uuid
from io import BytesIO
from pathlib import Path

_extra_site = os.environ.get("VISPRUNER_EXTRA_SITE_PACKAGES", "")
for _p in [x for x in _extra_site.split(os.pathsep) if x]:
    if _p not in sys.path:
        sys.path.append(_p)

import pandas as pd
import torch
from datasets import load_dataset as hf_load_dataset
from PIL import Image
from tqdm import tqdm
from qwen_vl_utils import process_vision_info

if os.environ.get("VISPRUNER_L9_MULTIVIEW_RUNTIME", "0").strip().lower() in {"1", "true", "yes", "on"}:
    from vispruner_qwen_ov.l9_global_pmi_balance_patch import install as _install_l9_balance
    from vispruner_qwen_ov.l9_multiview_coverage_patch import install as _install_l9_multiview
    _install_l9_balance()
    _install_l9_multiview()

if os.environ.get("VISPRUNER_L9_POSITION_CONTINUITY_RUNTIME", "0").strip().lower() in {"1", "true", "yes", "on"}:
    from vispruner_qwen_ov.l9_position_continuity_patch import install as _install_l9_position_continuity
    _install_l9_position_continuity()

if os.environ.get("VISPRUNER_L9_POSITION_TOPOLOGY_RUNTIME", "0").strip().lower() in {"1", "true", "yes", "on"}:
    from vispruner_qwen_ov.l9_position_topology_patch import install as _install_l9_position_topology
    _install_l9_position_topology()

from vispruner_qwen_ov.models import VisPrunerLlavaOV15, VisPrunerQwen25VL, VisPrunerQwen3VL


def str2bool(x):
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"1", "true", "yes", "y", "on"}


def split_chunk(rows, num_chunks: int, chunk_idx: int):
    if num_chunks <= 1:
        return rows
    chunk_size = math.ceil(len(rows) / num_chunks)
    return rows[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]


def answer_identity(row) -> tuple[str, str]:
    return str(row.get("question_id")), str(row.get("prompt", ""))


def answers_complete(path: Path, examples) -> bool:
    if not path.is_file():
        return False
    try:
        rows = [json.loads(line) for line in path.open(errors="replace") if line.strip()]
    except (OSError, json.JSONDecodeError):
        return False
    expected = [
        (str(x["question_id"]), str(x.get("prompt", "")))
        for x in examples
    ]
    actual = [answer_identity(x) for x in rows]
    return actual == expected and len(set(actual)) == len(actual)


def textvqa_question_from_prompt(prompt: str) -> str:
    lines = str(prompt).splitlines()
    if not lines:
        return ""
    if lines[0].startswith("Reference OCR token:") and len(lines) > 1:
        question = lines[1]
    else:
        question = lines[0]
    return " ".join(question.lower().strip().split())


def image_to_data_url(image: Image.Image) -> str:
    buf = BytesIO()
    image.convert("RGB").save(buf, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def build_messages(question: str, images: list[Image.Image], system_prompt: str):
    image_items = [
        {"type": "image", "image": image_to_data_url(image)} for image in images
    ]
    if "<image>" in question:
        parts = question.split("<image>")
        content = []
        for idx, part in enumerate(parts):
            if part.strip():
                content.append({"type": "text", "text": part})
            if idx < len(image_items):
                content.append(image_items[idx])
        content.extend(image_items[len(parts) - 1 :])
    else:
        content = image_items + [{"type": "text", "text": question}]
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def _token_ids(tokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


def _find_subsequence(haystack: list[int], needle: list[int], start: int = 0) -> tuple[int, int] | None:
    if not needle:
        return None
    stop = len(haystack) - len(needle) + 1
    for pos in range(max(0, start), max(0, stop)):
        if haystack[pos : pos + len(needle)] == needle:
            return pos, pos + len(needle)
    return None


def _mark_content_tokens(
    ids: list[int], tokenizer, text: str, role_ids: list[int], role: int, start: int,
) -> tuple[int, int]:
    """Mark content tokens without assuming a model-specific chat template."""
    variants = [text, text.strip(), "\n" + text.strip(), " " + text.strip()]
    seen = set()
    for variant in variants:
        token_ids = _token_ids(tokenizer, variant)
        key = tuple(token_ids)
        if not token_ids or key in seen:
            continue
        seen.add(key)
        match = _find_subsequence(ids, token_ids, start=start)
        if match is None:
            match = _find_subsequence(ids, token_ids, start=0)
        if match is not None:
            lo, hi = match
            role_ids[lo:hi] = [role] * (hi - lo)
            return hi, hi - lo
    return start, 0


def attach_prompt_token_metadata(model, inputs, system_prompt: str, user_prompt: str) -> None:
    """Attach analysis-only system/user labels to the pruning state."""
    prog = getattr(model, "_progressive", None)
    if prog is None or not hasattr(prog, "set_prompt_token_metadata"):
        return
    ids = [int(x) for x in inputs.input_ids[0].detach().cpu().tolist()]
    roles = [0] * len(ids)
    cursor, system_n = _mark_content_tokens(
        ids, model.tokenizer, system_prompt, roles, 1, 0
    )
    user_n = 0
    parts = user_prompt.split("<image>")
    for part in parts:
        if not part.strip():
            continue
        cursor, marked = _mark_content_tokens(
            ids, model.tokenizer, part, roles, 2, cursor
        )
        user_n += marked
    raw = [str(x) for x in model.tokenizer.convert_ids_to_tokens(ids)]
    decoded = [
        model.tokenizer.decode([x], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        for x in ids
    ]
    prog.set_prompt_token_metadata(torch.tensor(roles), raw, decoded)
    if hasattr(prog, "set_alpha_router_prompt"):
        prog.set_alpha_router_prompt(user_prompt)
    if hasattr(prog, "set_continuous_alpha_prompt"):
        prog.set_continuous_alpha_prompt(user_prompt)
    if hasattr(prog, "set_selector_router_prompt"):
        prog.set_selector_router_prompt(user_prompt)
    prog.record("prompt_metadata_calls", 1)
    prog.record("prompt_metadata_system_tokens", system_n)
    prog.record("prompt_metadata_user_tokens", user_n)
    prog.record("prompt_metadata_unlabeled_tokens", sum(int(x == 0) for x in roles))
    if system_n == 0:
        prog.record("prompt_metadata_system_match_failures", 1)
    if user_n == 0:
        prog.record("prompt_metadata_user_match_failures", 1)


def instantiate_model(args):
    cls = {"qwen25": VisPrunerQwen25VL, "qwen3": VisPrunerQwen3VL, "ov15": VisPrunerLlavaOV15}[args.model]
    return cls(
        pretrained=str(args.model_path),
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        attn_implementation=args.attn_implementation,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        use_cache=True,
        progressive_vision_pruning=args.progressive_vision_pruning,
        progressive_text_masking=args.progressive_text_masking,
        progressive_layer_k=args.progressive_layer_k,
        progressive_vision_layer=args.progressive_vision_layer,
        progressive_vision_keep_tokens=args.progressive_vision_keep_tokens,
        progressive_vision_keep_ratio=args.progressive_vision_keep_ratio,
        progressive_vision_text_aware=args.progressive_vision_text_aware,
        progressive_vision_score_mode=args.progressive_vision_score_mode,
        progressive_vision_score_lambda=args.progressive_vision_score_lambda,
        progressive_vision_anchor_alpha=args.progressive_vision_anchor_alpha,
        progressive_vision_stage_keep_tokens=args.progressive_vision_stage_keep_tokens,
        progressive_vision_stage_keep_ratio=args.progressive_vision_stage_keep_ratio,
        progressive_vision_stage_budget_mode=args.progressive_vision_stage_budget_mode,
        progressive_vision_stage_layer=args.progressive_vision_stage_layer,
        progressive_vision_stage_score_mode=args.progressive_vision_stage_score_mode,
        progressive_vision_stage_score_lambda=args.progressive_vision_stage_score_lambda,
        progressive_vision_stage_physical_drop=args.progressive_vision_stage_physical_drop,
        progressive_vision_auto_rule=args.progressive_vision_auto_rule,
        progressive_vision_auto_min_history=args.progressive_vision_auto_min_history,
        progressive_vision_encoder_layer=args.progressive_vision_encoder_layer,
        progressive_vision_merge=args.progressive_vision_merge,
        progressive_vision_merge_placement=args.progressive_vision_merge_placement,
        progressive_vision_physical_drop=args.progressive_vision_physical_drop,
        progressive_vision_merge_mode=args.progressive_vision_merge_mode,
        progressive_vision_merge_temperature=args.progressive_vision_merge_temperature,
        progressive_important_ratio=args.progressive_important_ratio,
        progressive_text_correction=args.progressive_text_correction,
        progressive_text_mask_mode=args.progressive_text_mask_mode,
        progressive_text_threshold=args.progressive_text_threshold,
        progressive_text_mask_ratio=args.progressive_text_mask_ratio,
        progressive_text_mask_ratio_end=args.progressive_text_mask_ratio_end,
        progressive_text_layer_start=args.progressive_text_layer_start,
        progressive_text_layer_end=args.progressive_text_layer_end,
        progressive_text_piecewise=args.progressive_text_piecewise,
        progressive_text_adaptive_tau=args.progressive_text_adaptive_tau,
        progressive_text_adaptive_alpha=args.progressive_text_adaptive_alpha,
        progressive_text_adaptive_floor=args.progressive_text_adaptive_floor,
        progressive_text_adaptive_min_ratio=args.progressive_text_adaptive_min_ratio,
        progressive_text_adaptive_max_ratio=args.progressive_text_adaptive_max_ratio,
        progressive_text_adaptive_floor_mode=args.progressive_text_adaptive_floor_mode,
        progressive_text_adaptive_margin=args.progressive_text_adaptive_margin,
        progressive_text_adaptive_ramp_layers=args.progressive_text_adaptive_ramp_layers,
        progressive_text_adaptive_gate_tau=args.progressive_text_adaptive_gate_tau,
        progressive_text_adaptive_gate_min_frac=args.progressive_text_adaptive_gate_min_frac,
        progressive_text_soft_gamma=args.progressive_text_soft_gamma,
        progressive_text_grounding_lambda=args.progressive_text_grounding_lambda,
        progressive_text_ema_beta=args.progressive_text_ema_beta,
        progressive_analysis_path=args.progressive_analysis_path,
        progressive_debug=args.progressive_debug,
    )


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.open(errors="replace") if line.strip()]


def is_none(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "nan", "none"}:
        return True
    return False


def load_image_from_base64(value: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(value))).convert("RGB")


OFFICIAL_DATASETS = {
    "chartqa_official": {
        "repo": "lmms-lab/ChartQA",
        "revision": "9e63b7df1592a1c2158e735cc1725454aef0d6d9",
        "config": None,
        "split": "test",
    },
    "infovqa": {
        "repo": "lmms-lab/DocVQA",
        "revision": "539088ef8a8ada01ac8e2e6d4e372586748a265e",
        "config": "InfographicVQA",
        "split": "validation",
    },
    "ocrbench_v2": {
        "repo": "ling99/OCRBench_v2",
        "revision": "c7e7cdf23bdb6774661e9b0caf0d9935a42feb8b",
        "config": None,
        "split": "test",
    },
    "hrbench4k": {
        "repo": "DreamMr/HR-Bench",
        "revision": "83b9013d6293b85dc507e87199ca52517536939c",
        "config": "hrbench_version_split",
        "split": "hrbench_4k",
    },
    "hrbench8k": {
        "repo": "DreamMr/HR-Bench",
        "revision": "83b9013d6293b85dc507e87199ca52517536939c",
        "config": "hrbench_version_split",
        "split": "hrbench_8k",
    },
}


def load_official_hf_dataset(name: str):
    spec = OFFICIAL_DATASETS[name]
    kwargs = {
        "split": spec["split"],
        "revision": spec["revision"],
        "cache_dir": os.environ.get("HF_DATASETS_CACHE"),
    }
    if spec["config"] is None:
        return hf_load_dataset(spec["repo"], **kwargs)
    return hf_load_dataset(spec["repo"], spec["config"], **kwargs)


def open_image_payload(value) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, str):
        return load_image_from_base64(value)
    if isinstance(value, dict):
        image_bytes = value.get("bytes")
        image_path = value.get("path")
        if image_bytes is not None:
            return Image.open(BytesIO(image_bytes)).convert("RGB")
        if image_path:
            return Image.open(image_path).convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(BytesIO(bytes(value))).convert("RGB")
    raise TypeError(f"unsupported image payload: {type(value)!r}")


def hrbench_prompt(row: dict) -> str:
    options_prompt = "".join(f"{key}. {row[key]}\n" for key in "ABCD")
    return f"{str(row['question']).strip()}\n{options_prompt}Answer the option letter directly."


def pope_local_question_id(value) -> str:
    """Map LLaVA aggregate IDs (10M/20M offsets) to per-split POPE IDs."""
    return str(int(value) % 10_000_000)


def mmbench_prompt(row) -> str:
    question = str(row["question"])
    hint = row.get("hint", None)
    if not is_none(hint):
        question = str(hint) + "\n" + question
    for option in ["A", "B", "C", "D"]:
        val = row.get(option, None)
        if is_none(val):
            break
        question += f"\n{option}. {val}"
    question += "\nAnswer with the option's letter from the given choices directly."
    return question


def load_dataset(args):
    ds = args.dataset
    if ds in {"gqa", "textvqa", "textvqa_val", "pope"}:
        rows = load_jsonl(args.question_file)
        gqa_gt = {}
        textvqa_gt = {}
        pope_gt = {}
        if ds == "gqa":
            gt_path = args.question_file.parent / "data" / "questions" / "testdev_balanced_questions.json"
            if not gt_path.is_file():
                gt_path = Path("playground/data/eval/gqa/data/questions/testdev_balanced_questions.json")
            if gt_path.exists():
                gqa_gt = json.loads(gt_path.read_text(errors="replace"))
        elif ds in {"textvqa", "textvqa_val"}:
            gt_path = args.question_file.parent / "TextVQA_0.5.1_val.json"
            if gt_path.exists():
                data = json.loads(gt_path.read_text(errors="replace")).get("data", [])
                textvqa_gt = {
                    (
                        str(x.get("image_id")),
                        " ".join(str(x.get("question", "")).lower().strip().split()),
                    ): x.get("answers", [])
                    for x in data
                }
        elif ds == "pope":
            for category in ("adversarial", "popular", "random"):
                gt_path = args.question_file.parent / f"coco_pope_{category}.json"
                if not gt_path.exists():
                    gt_path = args.question_file.parent / "coco" / f"coco_pope_{category}.json"
                if gt_path.exists():
                    for line in gt_path.read_text(errors="replace").splitlines():
                        if not line.strip():
                            continue
                        item = json.loads(line)
                        pope_gt[(category, pope_local_question_id(item.get("question_id")))] = item.get("label")
        out = []
        for row in rows:
            answer = None
            if ds == "gqa":
                item = gqa_gt.get(str(row["question_id"]), {})
                answer = item.get("answer")
            elif ds in {"textvqa", "textvqa_val"}:
                answer = textvqa_gt.get(
                    (
                        str(row["question_id"]),
                        textvqa_question_from_prompt(row.get("text", "")),
                    )
                )
            elif ds == "pope":
                answer = pope_gt.get(
                    (row.get("category", ""), pope_local_question_id(row["question_id"]))
                )
            out.append({
                "question_id": row["question_id"],
                "prompt": row["text"],
                "image_path": args.image_folder / row["image"],
                "extra": {"category": row.get("category", ""), "answer": answer},
            })
        return out
    if ds in {"ai2d", "ai2d_dev", "mmmu_dev"}:
        rows = load_jsonl(args.question_file)
        out = []
        for row in rows:
            image_rels = row.get("images") or []
            if not image_rels:
                raise ValueError(f"missing images for {ds} question {row.get('question_id')}")
            prompt = row["text"].strip()
            if ds != "mmmu_dev":
                prompt = prompt.replace("<image>\n", "").replace("<image>", "").strip()
            out.append({
                "question_id": row["question_id"],
                "prompt": prompt,
                "image_paths": [args.image_folder / image_rel for image_rel in image_rels],
                "extra": {
                    "answer": row.get("answer"),
                    "options": row.get("options"),
                    "question_type": row.get("question_type"),
                },
            })
        return out
    if ds in OFFICIAL_DATASETS:
        hf_dataset = load_official_hf_dataset(ds)
        drop_columns = ["image"] + (["ocr"] if ds == "infovqa" else [])
        metadata_dataset = hf_dataset.remove_columns(drop_columns)
        out = []
        for idx in range(len(metadata_dataset)):
            row = metadata_dataset[idx]
            if ds == "chartqa_official":
                question_id = f"chartqa-{idx}"
                prompt = str(row["question"]).strip() + "\nAnswer the question with a single word."
                extra = {"answer": str(row["answer"]), "type": str(row["type"])}
            elif ds == "infovqa":
                question_id = f"infovqa-{row['questionId']}"
                prompt = str(row["question"]).strip() + "\nAnswer the question using a single word or phrase."
                extra = {
                    "answer": [str(value) for value in row["answers"]],
                    "answer_type": row.get("answer_type", []),
                    "operation_reasoning": row.get("operation/reasoning", []),
                    "image_url": str(row.get("image_url", "")),
                }
            elif ds == "ocrbench_v2":
                question_id = f"ocrbench_v2-{row['id']}"
                prompt = str(row["question"]).strip()
                extra = {
                    key: value for key, value in row.items()
                    if value is not None and not (isinstance(value, str) and value == "None")
                }
            else:
                question_id = f"{ds}-{row['index']}"
                prompt = hrbench_prompt(row)
                extra = dict(row)
            out.append({
                "question_id": question_id,
                "prompt": prompt,
                "_hf_dataset": hf_dataset,
                "_hf_index": idx,
                "extra": extra,
            })
        return out
    if ds == "chartqa":
        df = pd.read_parquet(args.question_file)
        out = []
        for idx, row in df.iterrows():
            image = row["image"]
            if isinstance(image, dict):
                image_bytes = image.get("bytes")
                image_path = image.get("path")
                if image_bytes is None and image_path:
                    image_bytes = Path(image_path).read_bytes()
            elif isinstance(image, (bytes, bytearray)):
                image_bytes = bytes(image)
            else:
                raise TypeError(f"unsupported ChartQA image payload: {type(image)!r}")
            if not image_bytes:
                raise ValueError(f"missing ChartQA image bytes at row {idx}")
            out.append({
                "question_id": f"chartqa-{idx}",
                "prompt": str(row["question"]).strip() + "\nAnswer the question with a single word.",
                "image_bytes": image_bytes,
                "extra": {"answer": str(row["answer"]), "type": str(row["type"])},
            })
        return out
    if ds == "mmstar":
        df = pd.read_parquet(args.question_file)
        out = []
        for row_idx, row in df.iterrows():
            image = row["image"]
            if isinstance(image, dict):
                image_bytes = image.get("bytes")
                image_path = image.get("path")
                if image_bytes is None and image_path:
                    image_bytes = Path(image_path).read_bytes()
            elif isinstance(image, (bytes, bytearray)):
                image_bytes = bytes(image)
            else:
                raise TypeError(f"unsupported MMStar image payload: {type(image)!r}")
            if not image_bytes:
                raise ValueError(f"missing MMStar image bytes at row {row_idx}")
            sample_id = int(row["index"])
            out.append({
                "question_id": f"mmstar-{sample_id}",
                "prompt": str(row["question"]).strip() + "\nAnswer with the option's letter from the given choices directly.",
                "image_bytes": image_bytes,
                "extra": {
                    "answer": str(row["answer"]).strip().upper(),
                    "category": str(row["category"]),
                    "l2_category": str(row["l2_category"]),
                },
            })
        return out
    if ds in {"mmbench", "mmbench_en_dev"}:
        df = pd.read_table(args.question_file)
        out = []
        for _, row in df.iterrows():
            out.append({
                "question_id": int(row["index"]),
                "prompt": mmbench_prompt(row),
                "image_b64": row["image"],
                "extra": {
                    "options": [str(row[o]) for o in ["A", "B", "C", "D"] if o in row and not is_none(row[o])],
                    "answer": None if is_none(row.get("answer")) else str(row.get("answer")),
                    "category": None if is_none(row.get("category")) else str(row.get("category")),
                },
            })
        return out
    raise ValueError(f"unsupported dataset: {ds}")


def write_compact_telemetry(model, args, runtime: dict | None = None) -> None:
    path = str(getattr(args, "progressive_telemetry_path", "") or "").strip()
    if not path:
        return
    prog = getattr(model, "_progressive", None)
    stats = dict(getattr(prog, "stats", {}) or {})
    payload = {
        "schema_version": 1,
        "model": args.model,
        "dataset": args.dataset,
        "answers_file": str(args.answers_file),
        "text_correction": args.progressive_text_correction,
        "text_mask_mode": args.progressive_text_mask_mode,
        "stats": stats,
        "runtime": dict(runtime or {}),
    }
    records = float(stats.get("text_actual_mask_ratio_records", 0.0))
    recovery_records = float(stats.get("text_mask_recovery_records", 0.0))
    jaccard_records = float(stats.get("text_mask_jaccard_records", 0.0))
    payload["summary"] = {
        "actual_mask_ratio_mean": (
            float(stats.get("text_actual_mask_ratio_sum", 0.0)) / records if records else None
        ),
        "recovery_fraction_mean": (
            float(stats.get("text_mask_recovery_sum", 0.0)) / recovery_records if recovery_records else None
        ),
        "mask_jaccard_mean": (
            float(stats.get("text_mask_jaccard_sum", 0.0)) / jaccard_records if jaccard_records else None
        ),
        "masked_attention_edges": float(stats.get("text_masked_attention_edges", 0.0)),
        "exact_scoring_visible_pairs": float(stats.get("text_score_full_key_visible_pairs", 0.0)),
        "linear_score_feature_ops": float(stats.get("text_score_linear_causal_feature_ops", 0.0)),
        "observe_reused_qk": float(stats.get("observe_reused_qk", 0.0)),
        "generation_cache_sessions": float(stats.get("text_mask_generation_cache_sessions", 0.0)),
        "generation_cache_store_layers": float(stats.get("text_mask_generation_cache_store_layers", 0.0)),
        "generation_cache_reuse_forwards": float(stats.get("text_mask_generation_cache_reuse_forwards", 0.0)),
        "generation_cache_restore_layers": float(stats.get("text_mask_generation_cache_restore_layers", 0.0)),
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def generate_answers(args):
    examples = load_dataset(args)
    if args.sample_seed >= 0:
        import random
        rng = random.Random(int(args.sample_seed))
        rng.shuffle(examples)
    examples = split_chunk(examples, args.num_chunks, args.chunk_idx)
    if args.sample_offset > 0:
        examples = examples[args.sample_offset :]
    if args.limit and args.limit > 0:
        examples = examples[: args.limit]

    args.answers_file.parent.mkdir(parents=True, exist_ok=True)
    if args.reuse_answers and answers_complete(args.answers_file, examples):
        print(f"[reuse-answers] {args.answers_file} rows={len(examples)} validated_ids=true")
        return

    model = instantiate_model(args)
    model.model.eval()
    if os.environ.get("VISPRUNER_L9_MULTIVIEW_MODE", "").strip().lower() == "adaptive":
        from vispruner_qwen_ov.l9_multiview_coverage_patch import configure_adaptive_topology
        prog = getattr(model, "_progressive", None)
        if prog is None:
            raise RuntimeError("adaptive L9 selector requires progressive pruning state")
        configure_adaptive_topology(prog, model.model)
    if os.environ.get("VISPRUNER_L9_POSITION_CONTINUITY_RUNTIME", "0").strip().lower() in {"1", "true", "yes", "on"}:
        prog = getattr(model, "_progressive", None)
        if prog is None:
            raise RuntimeError("position-continuity gate requires progressive pruning state")
        prog._preserve_final_original_position_ids = False
        prog._l9_position_continuity_gate_enabled = True
    if os.environ.get("VISPRUNER_L9_POSITION_TOPOLOGY_RUNTIME", "0").strip().lower() in {"1", "true", "yes", "on"}:
        prog = getattr(model, "_progressive", None)
        if prog is None:
            raise RuntimeError("position-topology gate requires progressive pruning state")
        prog._preserve_final_original_position_ids = False
        prog._l9_position_topology_gate_enabled = True
    if os.environ.get("VISPRUNER_PRESERVE_COMPACTION_POSITIONS_RUNTIME", "0").strip().lower() in {"1", "true", "yes", "on"}:
        if os.environ.get("VISPRUNER_L9_POSITION_CONTINUITY_RUNTIME", "0").strip().lower() in {"1", "true", "yes", "on"} or os.environ.get("VISPRUNER_L9_POSITION_TOPOLOGY_RUNTIME", "0").strip().lower() in {"1", "true", "yes", "on"}:
            raise RuntimeError("all-boundary position preservation is mutually exclusive with legacy position gates")
        prog = getattr(model, "_progressive", None)
        if prog is None:
            raise RuntimeError("all-boundary position preservation requires progressive pruning state")
        prog._preserve_stage_original_position_ids = True
        prog._preserve_final_original_position_ids = True
        prog._l9_position_continuity_gate_enabled = False
        prog._l9_position_topology_gate_enabled = False
    if args.alpha_router_artifact is not None:
        from scripts.analysis.runtime_risk_alpha_router import install_runtime_router
        prog = getattr(model, "_progressive", None)
        if prog is None:
            raise RuntimeError("alpha router requires progressive pruning state")
        install_runtime_router(prog, args.alpha_router_artifact)
    if bool(getattr(args, "progressive_vision_deferred_rescore", False)):
        prog = getattr(model, "_progressive", None)
        if prog is None:
            raise RuntimeError("deferred vision re-scoring requires progressive pruning state")
        prog.config.vision_deferred_drop_layer = int(args.progressive_vision_deferred_drop_layer)
        prog.config.vision_deferred_reserve_fraction = 0.0
        prog.config.vision_deferred_rescore = True
        prog.config.vision_deferred_rescore_min_swaps = int(args.progressive_vision_deferred_rescore_min_swaps)
    runtime = {"samples": 0, "generation_seconds": 0.0}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with args.answers_file.open("w") as out:
        for ex in tqdm(examples, desc=f"{args.dataset} local generation"):
            if "image_paths" in ex:
                images = [Image.open(path).convert("RGB") for path in ex["image_paths"]]
            elif "image_path" in ex:
                images = [Image.open(ex["image_path"]).convert("RGB")]
            elif "image_bytes" in ex:
                images = [Image.open(BytesIO(ex["image_bytes"])).convert("RGB")]
            elif "_hf_dataset" in ex:
                image_payload = ex["_hf_dataset"][ex["_hf_index"]]["image"]
                images = [open_image_payload(image_payload)]
            else:
                images = [load_image_from_base64(ex["image_b64"])]
            messages = [build_messages(ex["prompt"], images, model.system_prompt)]
            texts = [model.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = model.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
            if model.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(model.device)
            attach_prompt_token_metadata(model, inputs, model.system_prompt, ex["prompt"])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            generation_start = time.perf_counter()
            with torch.inference_mode():
                cont = model.model.generate(
                    **inputs,
                    eos_token_id=model.tokenizer.eos_token_id,
                    pad_token_id=model.tokenizer.pad_token_id or model.tokenizer.eos_token_id,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    num_beams=1,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=model.use_cache,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            runtime["generation_seconds"] += time.perf_counter() - generation_start
            runtime["samples"] += 1
            generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            runtime["generated_tokens"] = runtime.get("generated_tokens", 0) + sum(int(x.numel()) for x in generated_ids)
            text = model.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
            record = {
                "question_id": ex["question_id"],
                "prompt": ex["prompt"],
                "text": text,
                "answer_id": uuid.uuid4().hex,
                "model_id": args.model,
                "metadata": ex.get("extra", {}),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
    if runtime["samples"]:
        runtime["seconds_per_sample"] = runtime["generation_seconds"] / runtime["samples"]
    if runtime.get("generated_tokens", 0):
        runtime["seconds_per_generated_token"] = runtime["generation_seconds"] / runtime["generated_tokens"]
    if torch.cuda.is_available():
        runtime["peak_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        runtime["peak_memory_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
    write_compact_telemetry(model, args, runtime)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["gqa", "textvqa", "textvqa_val", "textvqa_dev", "pope", "ai2d", "ai2d_dev", "mmmu_dev", "mmbench", "mmbench_en_dev", "chartqa", "chartqa_official", "infovqa", "ocrbench_v2", "hrbench4k", "hrbench8k", "mmstar"], required=True)
    ap.add_argument("--model", choices=["qwen25", "qwen3", "ov15"], required=True)
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--question-file", type=Path, required=True)
    ap.add_argument("--image-folder", type=Path, default=Path("."))
    ap.add_argument("--answers-file", type=Path, required=True)
    ap.add_argument("--reuse-answers", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample-offset", type=int, default=0)
    ap.add_argument("--sample-seed", type=int, default=-1)
    ap.add_argument("--num-chunks", type=int, default=1)
    ap.add_argument("--chunk-idx", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--attn-implementation", default="sdpa")
    ap.add_argument("--min-pixels", type=int, default=200704)
    ap.add_argument("--max-pixels", type=int, default=1605632)
    ap.add_argument("--progressive-vision-pruning", type=str2bool, default=False)
    ap.add_argument("--progressive-text-masking", type=str2bool, default=False)
    ap.add_argument("--progressive-layer-k", type=int, default=4)
    ap.add_argument("--progressive-vision-layer", type=int, default=0)
    ap.add_argument("--progressive-vision-keep-tokens", type=int, default=0)
    ap.add_argument("--progressive-vision-keep-ratio", type=float, default=1.0 / 9.0)
    ap.add_argument("--progressive-vision-text-aware", type=str2bool, default=False)
    ap.add_argument("--progressive-vision-score-mode", default="text")
    ap.add_argument("--progressive-vision-score-lambda", type=float, default=1.0)
    ap.add_argument("--progressive-vision-anchor-alpha", type=float, default=0.0)
    ap.add_argument("--alpha-router-artifact", type=Path, default=None)
    ap.add_argument("--progressive-vision-stage-keep-tokens", type=int, default=0)
    ap.add_argument("--progressive-vision-stage-keep-ratio", type=float, default=0.5)
    ap.add_argument("--progressive-vision-stage-budget-mode", default="ratio", choices=["ratio", "depth_coupled"])
    ap.add_argument("--progressive-vision-stage-layer", type=int, default=0)
    ap.add_argument("--progressive-vision-stage-score-mode", default="vispruner")
    ap.add_argument("--progressive-vision-stage-score-lambda", type=float, default=0.0)
    ap.add_argument("--progressive-vision-stage-physical-drop", type=str2bool, default=False)
    ap.add_argument("--progressive-vision-deferred-rescore", type=str2bool, default=False)
    ap.add_argument("--progressive-vision-deferred-drop-layer", type=int, default=17)
    ap.add_argument("--progressive-vision-deferred-rescore-min-swaps", type=int, default=6)
    ap.add_argument("--progressive-vision-auto-rule", default="corr_pos")
    ap.add_argument("--progressive-vision-auto-min-history", type=int, default=2)
    ap.add_argument("--progressive-vision-encoder-layer", type=int, default=-1)
    ap.add_argument("--progressive-vision-merge", type=str2bool, default=False)
    ap.add_argument(
        "--progressive-vision-merge-placement",
        default="legacy",
        choices=["legacy", "none", "early", "final", "both"],
    )
    ap.add_argument("--progressive-vision-physical-drop", type=str2bool, default=False)
    ap.add_argument("--progressive-vision-merge-mode", default="weighted", choices=["weighted", "mean", "hard", "nearest_mean", "nearest_residual", "nearest_saliency_anchor", "segment_saliency_anchor", "segment_saliency_mean"])
    ap.add_argument("--progressive-vision-merge-temperature", type=float, default=0.07)
    ap.add_argument("--progressive-important-ratio", type=float, default=0.5)
    ap.add_argument("--progressive-text-correction", default="exposure_baseline_ratio")
    ap.add_argument("--progressive-text-mask-mode", default="threshold")
    ap.add_argument("--progressive-text-threshold", type=float, default=0.0)
    ap.add_argument("--progressive-text-mask-ratio", type=float, default=0.0)
    ap.add_argument("--progressive-text-mask-ratio-end", type=float, default=-1.0)
    ap.add_argument("--progressive-text-layer-start", type=float, default=0.0)
    ap.add_argument("--progressive-text-layer-end", type=float, default=1.0)
    ap.add_argument("--progressive-text-piecewise", default="")
    ap.add_argument("--progressive-text-adaptive-tau", type=float, default=1.0)
    ap.add_argument("--progressive-text-adaptive-alpha", type=float, default=0.45)
    ap.add_argument("--progressive-text-adaptive-floor", type=float, default=0.15)
    ap.add_argument("--progressive-text-adaptive-min-ratio", type=float, default=0.0)
    ap.add_argument("--progressive-text-adaptive-max-ratio", type=float, default=0.2)
    ap.add_argument("--progressive-text-adaptive-floor-mode", default="warmup_median")
    ap.add_argument("--progressive-text-adaptive-margin", type=float, default=0.0)
    ap.add_argument("--progressive-text-adaptive-ramp-layers", type=int, default=0)
    ap.add_argument("--progressive-text-adaptive-gate-tau", type=float, default=0.0)
    ap.add_argument("--progressive-text-adaptive-gate-min-frac", type=float, default=0.0)
    ap.add_argument("--progressive-text-soft-gamma", type=float, default=2.0)
    ap.add_argument("--progressive-text-grounding-lambda", type=float, default=0.0)
    ap.add_argument("--progressive-text-ema-beta", type=float, default=0.8)
    ap.add_argument("--progressive-analysis-path", default="")
    ap.add_argument("--progressive-telemetry-path", default="")
    ap.add_argument("--progressive-debug", type=str2bool, default=False)
    args = ap.parse_args()
    if args.dataset == "textvqa_dev":
        args.dataset = "textvqa"
        if not args.limit:
            args.limit = 256
    elif args.dataset == "textvqa_val":
        args.dataset = "textvqa"
    if args.dataset == "mmbench_en_dev":
        args.dataset = "mmbench"
    generate_answers(args)


if __name__ == "__main__":
    main()
