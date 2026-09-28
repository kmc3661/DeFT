"""Isolated SDPA first-token timing on the frozen 100-input InfoVQA manifest.

This portable runner is not a substitute for cross-host timing validation.
Clean end-to-end passes and CUDA-event decoder passes are kept separate.
"""
import argparse,hashlib,json,os,random,statistics,subprocess,time
from pathlib import Path
from types import SimpleNamespace


def isolated():
    ids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    other={int(x) for x in ids.splitlines() if x.strip()}-{os.getpid()}
    if other:raise RuntimeError('Timing requires all GPUs idle except this process')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path',required=True)
    p.add_argument('--family',choices=['qwen3','ov15'],default='qwen3')
    p.add_argument('--alpha',type=float,default=.2)
    p.add_argument('--prune',type=float,default=.8)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=Path(__file__).resolve().parents[1]/'configs/infovqa_latency_100.jsonl')
    p.add_argument('--warmup-passes',type=int,default=2)
    a=p.parse_args()
    if a.output.exists():p.error('Output exists')
    if a.warmup_passes<2:p.error('Use at least two warm-up passes')
    isolated()
    import torch,numpy as np
    from qwen_vl_utils import process_vision_info
    from deft import DeFTConfig
    from deft.runtime import load_model
    model=load_model(a.model_path,a.family,DeFTConfig(a.prune,a.alpha),backend='sdpa',audit=False)
    from scripts.run_qwen_ov_local_bench import load_dataset,open_image_payload,build_messages,attach_prompt_token_metadata
    allrows=load_dataset(SimpleNamespace(dataset='infovqa'))
    manifest=[json.loads(s) for s in a.manifest.read_text().splitlines()]
    assert len(manifest)==100 and len({r['image_group'] for r in manifest})==100
    examples=[]
    for record in manifest:
        row=allrows[record['row_index']]
        assert str(row['question_id'])==record['question_id']
        assert hashlib.sha256(row['prompt'].encode()).hexdigest()==record['prompt_sha256']
        image=open_image_payload(row['_hf_dataset'][row['_hf_index']]['image'])
        examples.append((row,image))
    def run(events=None):
        isolated();random.seed(1234);np.random.seed(1234);torch.manual_seed(1234);torch.cuda.manual_seed_all(1234)
        records=[]
        for row,image in examples:
            messages=[build_messages(row['prompt'],[image],model.system_prompt)]
            text=[model.processor.apply_chat_template(m,tokenize=False,add_generation_prompt=True) for m in messages]
            images,videos=process_vision_info(messages)
            inputs=model.processor(text=text,images=images,videos=videos,padding=True,return_tensors='pt').to('cuda:0')
            attach_prompt_token_metadata(model,inputs,model.system_prompt,row['prompt'])
            torch.cuda.synchronize();start=time.perf_counter()
            with torch.inference_mode():
                output=model.model.generate(**inputs,do_sample=False,temperature=None,top_p=None,num_beams=1,
                    max_new_tokens=1,use_cache=False,eos_token_id=model.tokenizer.eos_token_id,
                    pad_token_id=model.tokenizer.pad_token_id or model.tokenizer.eos_token_id)
            torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
            records.append(dict(id=row['question_id'],e2e_ms=elapsed,
                prefill_ms=events[0].elapsed_time(events[1]) if events else None,
                output_token=int(output[0,-1])))
        isolated()
        return records
    for _ in range(a.warmup_passes):run()
    clean=[run() for _ in range(3)]
    language=model.model.model.language_model
    events=[torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)]
    handles=[language.register_forward_pre_hook(lambda *_:events[0].record()),
             language.register_forward_hook(lambda *_:events[1].record())]
    try:
        run(events)
        profiled=[run(events) for _ in range(3)]
    finally:
        for h in handles:h.remove()
        for _,image in examples:image.close()
    outputs=[[r['output_token'] for r in run] for run in clean+profiled]
    assert all(x==outputs[0] for x in outputs),'Output replay mismatch'
    e2e=[statistics.mean(r['e2e_ms'] for r in run) for run in clean]
    prefill=[statistics.mean(r['prefill_ms'] for r in run) for run in profiled]
    report=dict(model_family=a.family,alpha=a.alpha,prune=a.prune,samples=100,repeats=3,
        backend='sdpa',cache=False,warmup_passes=a.warmup_passes,
        manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
        prefill_ms=statistics.mean(prefill),e2e_ms=statistics.mean(e2e),
        prefill_repeat_sd=statistics.stdev(prefill),e2e_repeat_sd=statistics.stdev(e2e),
        timing_warning=any((max(v)-min(v))/statistics.median(v)>.1 for v in (prefill,e2e)),
        clean_passes=clean,profiled_passes=profiled,
        protocol='fixed warmup portable runner; not the historical adaptive-warmup controller')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if not k.endswith('_passes')},indent=2))


if __name__=='__main__':main()
