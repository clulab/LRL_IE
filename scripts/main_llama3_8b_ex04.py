import argparse
import json
import yaml
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm
from llama_cpp import Llama


# ---------- JSON helpers ----------
def extract_first_json(s: str) -> Optional[str]:
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(s[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i+1]
    return None

def normalize_json_obj(obj: dict, schema_keys: List[str]) -> dict:
    fixed = {}
    for k in schema_keys:
        vals = obj.get(k, [])
        if not isinstance(vals, list):
            vals = []
        norm = []
        seen = set()
        for v in vals:
            if isinstance(v, str):
                vv = " ".join(v.split()).strip()
                if vv and vv not in seen:
                    seen.add(vv)
                    norm.append(vv)
        fixed[k] = norm
    return fixed

def decode_to_json(text: str, schema_keys: List[str]) -> dict:
    js = extract_first_json(text)
    if not js:
        return {k: [] for k in schema_keys}
    try:
        obj = json.loads(js)
    except Exception:
        js2 = (js.replace("None", "null")
               .replace("True", "true")
               .replace("False", "false"))
        try:
            obj = json.loads(js2)
        except Exception:
            return {k: [] for k in schema_keys}
    return normalize_json_obj(obj, schema_keys)

def load_jsonl(path: str):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s:
            yield json.loads(s)

def save_jsonl(path: str, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---------- Token budgeting helpers ----------
def count_tokens(llm: Llama, text: str) -> int:
    return len(llm.tokenize(text.encode("utf-8"), add_bos=False))

def fit_system_with_examples(llm: Llama, ctx: int, max_gen_tokens: int, system_prompt_base: str, fewshots: List[str]) -> str:
    SAFETY = 64
    budget = ctx - max_gen_tokens - SAFETY
    tokens = count_tokens(llm, system_prompt_base)
    pieces: List[str] = []
    for ex in fewshots:
        ex_tokens = count_tokens(llm, ex)
        if tokens + ex_tokens > budget:
            break
        pieces.append(ex)
        tokens += ex_tokens
    return system_prompt_base + "".join(pieces)

def maybe_truncate_input(llm: Llama, input_text: str, ctx: int, sys_str: str, max_gen_tokens: int, user_template: str) -> Tuple[str, int, int]:
    SAFETY = 64
    budget = ctx - max_gen_tokens - SAFETY
    user_block = user_template.format(input_text=input_text)
    total = count_tokens(llm, sys_str) + count_tokens(llm, user_block)
    if total <= budget:
        return input_text, total, budget

    toks_input = llm.tokenize(input_text.encode("utf-8"), add_bos=False)
    keep = max(64, budget - count_tokens(llm, sys_str) - count_tokens(llm, user_template.format(input_text="")))
    if keep < 64:
        keep = 64
    truncated_ids = toks_input[-keep:]
    truncated_text = llm.detokenize(truncated_ids).decode("utf-8", errors="ignore")
    user_block2 = user_template.format(input_text=truncated_text)
    total2 = count_tokens(llm, sys_str) + count_tokens(llm, user_block2)
    return truncated_text, total2, budget

def build_messages(llm: Llama, input_text: str, ctx: int, max_gen_tokens: int, prompt_data: dict, fewshots: List[str]) -> List[Dict[str, str]]:
    system_str = fit_system_with_examples(llm, ctx, max_gen_tokens, prompt_data['prompt_base'], fewshots)
    input_text_fitted, _, _ = maybe_truncate_input(llm, input_text, ctx, system_str, max_gen_tokens, prompt_data['user_template'])
    return [
        {"role": "system", "content": system_str},
        {"role": "user", "content": prompt_data['user_template'].format(input_text=input_text_fitted.strip())},
    ]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                    help="Path to GGUF model file")
    ap.add_argument("--input", default="data/input/in.jsonl")
    ap.add_argument("--out", default="data/out/preds_llama31_8b_ex04.jsonl")
    ap.add_argument("--prompt", default="data/prompts/prompt.jsonl")
    ap.add_argument("--examples", default="data/examples/examples.jsonl")
    ap.add_argument("--ctx", type=int, default=4096, help="Context window")
    ap.add_argument("--n_gpu_layers", type=int, default=0,
                    help="Set >0 (or -1) to offload layers to GPU if built with CUDA/Metal/OpenCL")
    ap.add_argument("--n_threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.1, help="Slight recall nudge; 0.0 for max determinism")
    ap.add_argument("--top_p", type=float, default=0.9, help="Slight recall nudge; 1.0 for max determinism")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"Loading GGUF: {args.model_path}")
    llm = Llama(
        model_path=args.model_path,
        n_ctx=args.ctx,
        n_gpu_layers=args.n_gpu_layers,
        n_threads=args.n_threads,
        seed=args.seed,
        chat_format="llama-3",   # LLaMA-3 / 3.1 instruct
        verbose=False,
    )

    with open(args.prompt) as yml_file:
        prompt_data = yaml.safe_load(yml_file)

    with open(args.examples) as json_file:
        fewshots = [json.loads(ex) for ex in json_file]

    records = list(load_jsonl(args.input))
    out_rows = []
    for r in tqdm(records, desc="Generating"):
        messages = build_messages(llm, r["Input"], args.ctx, args.max_tokens, prompt_data, fewshots)
        out = llm.create_chat_completion(
            messages=messages,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
        content = out["choices"][0]["message"]["content"]
        pred = decode_to_json(content, prompt_data['schema'])
        out_rows.append({"ID": r["ID"], "Input": r["Input"], "Pred": pred})

    save_jsonl(args.out, out_rows)
    print(f"Wrote {len(out_rows)} predictions -> {args.out}")

if __name__ == "__main__":
    main()
