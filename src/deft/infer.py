"""Single-image greedy inference using the same prompt construction as evaluation."""
import argparse
import json
from .config import DeFTConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--family", choices=["qwen3", "ov15"], default="qwen3")
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--alpha", type=float, default=.2)
    parser.add_argument("--prune", type=float, default=.8, help="Fraction pruned, not retained")
    parser.add_argument("--backend", choices=["sdpa", "flash_attention_2"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    from .runtime import load_model
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    model = load_model(args.model_path, args.family, DeFTConfig(args.prune, args.alpha), args.backend, args.device)
    from scripts.run_qwen_ov_local_bench import build_messages, attach_prompt_token_metadata
    with Image.open(args.image) as raw:
        images = [raw.convert("RGB")]
    messages = [build_messages(args.prompt, images, model.system_prompt)]
    texts = [model.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
    image_inputs, videos = process_vision_info(messages)
    inputs = model.processor(text=texts, images=image_inputs, videos=videos, padding=True, return_tensors="pt").to(args.device)
    attach_prompt_token_metadata(model, inputs, model.system_prompt, args.prompt)
    with torch.inference_mode():
        output = model.model.generate(**inputs, do_sample=False, temperature=None, top_p=None, num_beams=1,
            eos_token_id=model.tokenizer.eos_token_id,
            pad_token_id=model.tokenizer.pad_token_id or model.tokenizer.eos_token_id,
            max_new_tokens=args.max_new_tokens, use_cache=False)
    answer = model.tokenizer.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    print(json.dumps({"answer":answer,"alpha":args.alpha,"prune_ratio":args.prune},ensure_ascii=False))


if __name__ == "__main__":
    main()
