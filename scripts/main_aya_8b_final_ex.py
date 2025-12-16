# scripts/main_aya_8b_final_ex.py
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

def strip_think_blocks(text: str) -> str:
    """
    Some models may emit <think>...</think> style blocks.
    Safe to remove for cleaner JSON extraction (no harm if not present).
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def main():
    ap = argparse.ArgumentParser()

    # ---- model / io ----
    ap.add_argument(
        "--model_path",
        default="models/aya-expanse-8b-q4_k_m.gguf",
        help="Path to GGUF model file (Aya Expanse 8B)",
    )
    ap.add_argument("--input", default="data/data_llm_io_top50.jsonl")
    ap.add_argument("--out", default="output/preds_aya_8b_promptvar.jsonl")

    # ---- generation ----
    ap.add_argument("--ctx", type=int, default=4096, help="Context window")
    ap.add_argument(
        "--n_gpu_layers",
        type=int,
        default=-1,
        help="Set >0 (or -1) to offload layers to GPU if built with CUDA/Metal",
    )
    ap.add_argument("--n_threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)

    # ---- prompt variant (same as Llama/Qwen scripts) ----
    ap.add_argument(
        "--prompt_variant",
        default="zero_shot",
        choices=["qa_en", "qa_bn", "plain_en", "en_bn_QA", "en_bn_Description", "zero_shot"],
        help="Same prompt variants as your Llama/Qwen scripts",
    )

    # ---- chat template ----
    ap.add_argument(
        "--chat_format",
        default="chatml",
        help="Chat format/template for create_chat_completion() (chatml is a good default for Aya GGUFs).",
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

    for r in tqdm(records, desc="Generating (Aya chat_completion)"):
        msgs = build_messages(llm, r["Input"], args.ctx, args.max_tokens, cfg)

        out = llm.create_chat_completion(
            messages=msgs,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )

        content = out["choices"][0]["message"]["content"]
        content = strip_think_blocks(content)

        pred = decode_to_json(content)
        out_rows.append({"ID": r["ID"], "Input": r["Input"], "Pred": pred})

    save_jsonl(args.out, out_rows)
    print(f"Wrote {len(out_rows)} predictions -> {args.out}")

if __name__ == "__main__":
    main()
