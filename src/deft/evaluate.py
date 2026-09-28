"""Canonical eight-task evaluation with a small public argument surface."""
import argparse
import sys
from pathlib import Path
from .config import DeFTConfig

TASKS = {
    "textvqa": ("textvqa", 16), "mmmu": ("mmmu_dev", 32),
    "ai2d": ("ai2d", 4), "mmstar": ("mmstar", 4),
    "chartqa": ("chartqa_official", 16), "infovqa": ("infovqa", 32),
    "textcaps": ("textcaps", 32), "nocaps": ("nocaps", 32),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--family", choices=["qwen3", "ov15"], default="qwen3")
    p.add_argument("--question-file", default=".")
    p.add_argument("--image-folder", default=".")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--alpha", type=float, default=.2)
    p.add_argument("--prune", type=float, default=.8)
    p.add_argument("--selection-depth", type=float, default=.5)
    p.add_argument("--backend", choices=["sdpa", "flash_attention_2"])
    p.add_argument("--limit", type=int, default=0, help="0 uses the full benchmark")
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--trace", type=Path, help="Optional token-ID audit, not for timing")
    args = p.parse_args()
    if not 0 <= args.shard_index < args.shards:
        p.error("shard-index must be in [0, shards)")
    if args.output.exists():
        p.error("Output already exists; choose a new path (no silent overwrite/resume)")
    config = DeFTConfig(args.prune, args.alpha, args.selection_depth)
    from .runtime import prepare_backend, load_model
    prepare_backend()
    from scripts import run_qwen_ov_local_bench as base
    sys.modules['run_qwen_ov_local_bench'] = base
    from scripts import run_qwen_ov_caption_bench as caption
    traces=[]
    def construct(_):
        model=load_model(args.model_path,args.family,config,args.backend)
        if args.trace:
            from .trace import capture_trace
            traces.extend([capture_trace(model)])
        return model
    base.instantiate_model = construct
    task, maxnew = TASKS[args.task]
    cli = dict(dataset=task,model=args.family,model_path=args.model_path,
               question_file=args.question_file,image_folder=args.image_folder,
               answers_file=str(args.output),limit=args.limit,num_chunks=args.shards,
               chunk_idx=args.shard_index,max_new_tokens=maxnew,
               progressive_telemetry_path=str(args.output.with_suffix('.telemetry.json')))
    sys.argv = ['deft-eval'] + [v for k,value in cli.items() for v in ('--'+k.replace('_','-'),str(value))]
    (caption.main if args.task in ('textcaps','nocaps') else base.main)()
    if args.trace:
        import json
        args.trace.parent.mkdir(parents=True,exist_ok=True)
        with args.trace.open('x') as out:json.dump(traces[0],out)


if __name__ == "__main__":
    main()
