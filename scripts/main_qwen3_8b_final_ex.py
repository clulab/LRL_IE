# scripts/main_qwen3_8b_final_ex.py
import argparse
import os
import re
from tqdm import tqdm
from llama_cpp import Llama

# Reuse *your* existing prompt variants + truncation + JSON parsing
# (This is why this file stays short and matches your repo behavior.)
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

def chatml_prompt(system_text: str, user_text: str, no_think: bool = True) -> str:
    """
    Minimal ChatML wrapper that works well for Qwen-style instruct models.

    We add an assistant "generation prompt" at the end.
    """
    if no_think:
        # Qwen model card suggests /no_think to disable reasoning blocks.
        system_text = "/no_think\n" + system_text

    # ChatML format:
    # <|im_start|>system\n...\n<|im_end|>
    # <|im_start|>user\n...\n<|im_end|>
    # <|im_start|>assistant\n
    return (
        "<|im_start|>system\n" + system_text.strip() + "\n<|im_end|>\n"
        "<|im_start|>user\n" + user_text.strip() + "\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

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
        help="Prefix /no_think in the system prompt (recommended for clean JSON)",
    )
    ap.add_argument(
        "--prompt_variant",
        default="qa_en",
        choices=["qa_en", "qa_bn", "plain_en", "en_bn_QA", "en_bn_Description"],
        help="Same prompt variants as your Llama script",
    )

    args = ap.parse_args()

    print(f"Loading GGUF: {args.model_path}")
    print(f"Using prompt variant: {args.prompt_variant}")

    cfg = get_prompt_config(args.prompt_variant)

    # NOTE:
    # - We do NOT set chat_format="llama-3" (that is Llama-specific).
    # - We will generate using create_completion() with ChatML prompt.
    llm = Llama(
        model_path=args.model_path,
        n_ctx=args.ctx,
        n_gpu_layers=args.n_gpu_layers,
        n_threads=args.n_threads,
        seed=args.seed,
        verbose=False,
    )

    records = list(load_jsonl(args.input))
    out_rows = []

    for r in tqdm(records, desc="Generating (Qwen3)"):
        # Reuse your existing message builder (token budgeting + truncation)
        msgs = build_messages(llm, r["Input"], args.ctx, args.max_tokens, cfg)
        system_text = msgs[0]["content"]
        user_text = msgs[1]["content"]

        prompt = chatml_prompt(system_text, user_text, no_think=args.no_think)

        # For ChatML, stopping at <|im_end|> prevents the model from continuing extra turns.
        out = llm.create_completion(
            prompt=prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            stop=["<|im_end|>"],
        )

        # llama-cpp-python returns: {"choices":[{"text": "..."}], ...}
        text = out["choices"][0].get("text", "")
        text = strip_qwen_think(text)

        pred = decode_to_json(text)
        out_rows.append({"ID": r["ID"], "Input": r["Input"], "Pred": pred})

    save_jsonl(args.out, out_rows)
    print(f"Wrote {len(out_rows)} predictions -> {args.out}")

if __name__ == "__main__":
    main()
