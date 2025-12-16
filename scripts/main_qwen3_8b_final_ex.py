# scripts/main_qwen3_8b_final_ex.py
import argparse
import os
import re
from tqdm import tqdm
from llama_cpp import Llama

from main_llama3_8b_final_ex import (
    get_prompt_config,
    build_messages,
    decode_to_json,
    load_jsonl,
    save_jsonl,
)

def strip_qwen_think(text: str) -> str:
    """
    Qwen3 can emit <think>...</think>. Remove it so JSON extraction is cleaner.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def maybe_prefix_no_think(system_text: str, use_no_think: bool) -> str:
    """
    If enabled, prefix /no_think in the system prompt (common recommendation for Qwen3).
    """
    if use_no_think:
        return "/no_think\n" + system_text
    return system_text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_path",
        default="models/Qwen3-8B-Q4_K_M.gguf",
        help="Path to GGUF model file (Qwen3-8B-Q4_K_M.gguf)",
    )
    ap.add_argument("--input", default="data/data_llm_io.jsonl")
    ap.add_argument("--out", default="output/preds_qwen3_8b_promptvar.jsonl")
    ap.add_argument("--ctx", type=int, default=4096, help="Context window")
    ap.add_argument(
        "--n_gpu_layers",
        type=int,
        default=0,
        help="Set >0 (or -1) to offload layers to GPU if built with CUDA/Metal",
    )
    ap.add_argument("--n_threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)


    ap.add_argument(
        "--no_think",
        action="store_true",
        default=True,
        help="Prefix /no_think in the system prompt (default: True).",
    )
    ap.add_argument(
        "--allow_think",
        action="store_true",
        default=False,
        help="Disable /no_think prefix (lets Qwen output <think> blocks).",
    )

    ap.add_argument(
        "--prompt_variant",
        default="qa_en",
        choices=["qa_en", "qa_bn", "plain_en", "en_bn_QA", "en_bn_Description", "zero_shot"],
        help="Same prompt variants as your Llama script",
    )

    ap.add_argument(
        "--chat_format",
        default="chatml",
        help=(
            "Chat format/template for create_chat_completion(). "
            "Common values: chatml (often works for Qwen), qwen, qwen2, etc. "
            "If outputs look wrong, try another."
        ),
    )

    args = ap.parse_args()

    print(f"Loading GGUF: {args.model_path}")
    print(f"Using prompt variant: {args.prompt_variant}")
    print(f"Using chat_format: {args.chat_format}")

    cfg = get_prompt_config(args.prompt_variant)

    llm = Llama(
        model_path=args.model_path,
        n_ctx=args.ctx,
        n_gpu_layers=args.n_gpu_layers,
        n_threads=args.n_threads,
        seed=args.seed,
        chat_format=args.chat_format,  
        verbose=False,
    )

    records = list(load_jsonl(args.input))
    out_rows = []

    use_no_think = args.no_think and (not args.allow_think)

    for r in tqdm(records, desc="Generating (Qwen3 chat_completion)"):
        msgs = build_messages(llm, r["Input"], args.ctx, args.max_tokens, cfg)

        msgs[0]["content"] = maybe_prefix_no_think(msgs[0]["content"], use_no_think)

        out = llm.create_chat_completion(
            messages=msgs,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )

        content = out["choices"][0]["message"]["content"]
        content = strip_qwen_think(content)

        pred = decode_to_json(content)
        out_rows.append({"ID": r["ID"], "Input": r["Input"], "Pred": pred})

    save_jsonl(args.out, out_rows)
    print(f"Wrote {len(out_rows)} predictions -> {args.out}")

if __name__ == "__main__":
    main()
