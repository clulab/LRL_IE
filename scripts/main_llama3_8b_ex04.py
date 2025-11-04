import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm
from llama_cpp import Llama

SCHEMA_KEYS = ["Age","Symptom","Medicine","Health_Condition","Specialist","Medical_Procedure"]

# ======================
# QA-style SYSTEM PROMPT (LLaMA)
# ======================
SYSTEM_PROMPT_BASE = (
    "You are a question-answering assistant that performs medical Named Entity Recognition (NER).\n"
    "For each question, identify ONLY the entities that are explicitly present in the provided text and answer STRICTLY in JSON.\n"
    "Your answer MUST be a single JSON object with EXACTLY these six keys (even if empty), in this order:\n"
    "{\"Age\":[],\"Symptom\":[],\"Medicine\":[],\"Health_Condition\":[],\"Specialist\":[],\"Medical_Procedure\":[]}\n\n"
    "Rules:\n"
    "- Copy spans verbatim from the text (Bangla/English as they appear). No paraphrasing or hallucination.\n"
    "- Duration vs Age: if a number modifies time words (\"গত/পিছনের/ধরে\" + দিন/সপ্তাহ/মাস, or \"last/for/since\" + days/weeks/months), DO NOT label it as Age.\n"
    "- Negation: if a symptom is negated within ~5 tokens (\"না/নাই/নেই/করিনি/হয়নি/হয়ে নাই/হয় নি\"), DO NOT extract it.\n"
    "- Lab/test terms alone (e.g., Triglyceride, কোলেস্টেরল, HbA1c) are NOT symptoms unless the text explicitly states a complaint.\n"
    "- Prefer concise head+modifier spans; exclude extra function words/punctuation. Do not output standalone single letters (e.g., X/RT/S).\n"
    "- Only label Age when it is an age expression (e.g., \"৪০ বছর\", \"years old\", \"Age 27\"), not bare numbers.\n"
    "- Lists contain strings only; no duplicates; no commentary before or after the JSON.\n\n"
    "Label hints (recall boosters without adding false positives):\n"
    "- Symptom: Extract single-word symptoms (e.g., \"জ্বর\", \"কাশি\", \"বমি\") when a complaint verb appears nearby (\"আছে/হচ্ছে/লাগছে/অনুভব/ভুগছি/সমস্যা\").\n"
    "- Symptom: Also extract collocates exactly (\"মাথা ব্যথা\", \"গলা ব্যথা\", \"ঘন কফ\", \"বুকে ব্যথা\").\n"
    "- Health_Condition: Keep disease/diagnosis nouns (\"ডায়াবেটিস\", \"উচ্চ রক্তচাপ\", \"অ্যাজমা\", \"থাইরয়েড\", \"মাইগ্রেন\", \"গ্যাস্ট্রাইটিস\").\n"
    "- Specialist: Titles/specialities (\"ডাক্তার\", \"চিকিৎসক\", \"বিশেষজ্ঞ\", \"ইএনটি বিশেষজ্ঞ\", \"গ্যাস্ট্রোএন্টারোলজিস্ট\", \"নিউরোলজিস্ট\").\n"
    "- Medical_Procedure: Extract tests/imaging when performed/ordered (\"করা হয়েছে/করাতে বলেছেন\", \"done/ordered\").\n\n"
    "You will be asked the question:\n"
    "\"Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) are present in this text?\"\n\n"
    "Examples:\n"
)

# ======================
# Few-shots — QA style
# ======================
FEWSHOTS: List[str] = [
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) are present in this text?\n"
        "Text: \"গত সপ্তাহ থেকে জ্বর আছে, রাতে কাশি বাড়ে, মাঝে মাঝে বমি হয়.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[\"জ্বর\",\"কাশি\",\"বমি\"], \"Medicine\":[], "
        "\"Health_Condition\":[], \"Specialist\":[], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) are present in this text?\n"
        "Text: \"বুকে ব্যথা হচ্ছে এবং শ্বাসকষ্ট আছে; খুব অস্বস্তি লাগছে.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[\"বুকে ব্যথা\",\"শ্বাসকষ্ট\",\"অস্বস্তি\"], \"Medicine\":[], "
        "\"Health_Condition\":[], \"Specialist\":[], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) are present in this text?\n"
        "Text: \"রোগী অ্যাজমা ও গ্যাস্ট্রাইটিসের রোগী; গতকাল ধরা পড়েছে মাইগ্রেন.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[], \"Medicine\":[], "
        "\"Health_Condition\":[\"অ্যাজমা\",\"গ্যাস্ট্রাইটিস\",\"মাইগ্রেন\"], \"Specialist\":[], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) are present in this text?\n"
        "Text: \"থাইরয়েড সমস্যা আছে; history of ডায়াবেটিস উল্লেখ আছে.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[], \"Medicine\":[], "
        "\"Health_Condition\":[\"থাইরয়েড সমস্যা\",\"ডায়াবেটিস\"], \"Specialist\":[], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) are present in this text?\n"
        "Text: \"গত চার মাস ধরে মাথা ব্যথা; কাশি না.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[\"মাথা ব্যথা\"], \"Medicine\":[], "
        "\"Health_Condition\":[], \"Specialist\":[], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) are present in this text?\n"
        "Text: \"Your serum Triglyceride is slightly raised; HbA1c 6.5%.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[], \"Medicine\":[], "
        "\"Health_Condition\":[], \"Specialist\":[], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) are present in this text?\n"
        "Text: \"গত তিন দিন ধরে কাশি ও জ্বর আছে। একজন মেডিসিন বিশেষজ্ঞ আমাকে সেফিক্সিম দিয়েছেন। এক্স-রে করা হয়নি.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[\"কাশি\",\"জ্বর\"], \"Medicine\":[\"সেফিক্সিম\"], "
        "\"Health_Condition\":[], \"Specialist\":[\"মেডিসিন বিশেষজ্ঞ\"], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) are present in this text?\n"
        "Text: \"রোগীর বয়স ৫৫ বছর। তিনি ডায়াবেটিস ও উচ্চ রক্তচাপের রোগী এবং মেটফরমিন ও লোসারটান খাচ্ছেন.\"\n"
        "Answer: {\"Age\":[\"৫৫ বছর\"], \"Symptom\":[], \"Medicine\":[\"মেটফরমিন\",\"লোসারটান\"], "
        "\"Health_Condition\":[\"ডায়াবেটিস\",\"উচ্চ রক্তচাপ\"], \"Specialist\":[], \"Medical_Procedure\":[]}\n\n"
    ),
    (
        "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) are present in this text?\n"
        "Text: \"ইএনটি বিশেষজ্ঞ টিম্পানোমেট্রি করতে বলেছেন.\"\n"
        "Answer: {\"Age\":[], \"Symptom\":[], \"Medicine\":[], "
        "\"Health_Condition\":[], \"Specialist\":[\"ইএনটি বিশেষজ্ঞ\"], \"Medical_Procedure\":[\"টিম্পানোমেট্রি\"]}\n\n"
    ),
]

# ======================
# QA-style user template
# ======================
USER_TEMPLATE = (
    "Question: Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) are present in this text?\n"
    "Text:\n<<<\n{input_text}\n>>>\n"
    "Answer (a single JSON object with those six keys):\n"
)

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

def normalize_json_obj(obj: dict) -> dict:
    fixed = {}
    for k in SCHEMA_KEYS:
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

def decode_to_json(text: str) -> dict:
    js = extract_first_json(text)
    if not js:
        return {k: [] for k in SCHEMA_KEYS}
    try:
        obj = json.loads(js)
    except Exception:
        js2 = (js.replace("None", "null")
               .replace("True", "true")
               .replace("False", "false"))
        try:
            obj = json.loads(js2)
        except Exception:
            return {k: [] for k in SCHEMA_KEYS}
    return normalize_json_obj(obj)

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

def fit_system_with_examples(llm: Llama, ctx: int, max_gen_tokens: int) -> str:
    SAFETY = 64
    budget = ctx - max_gen_tokens - SAFETY
    base = SYSTEM_PROMPT_BASE
    tokens = count_tokens(llm, base)
    pieces: List[str] = []
    for ex in FEWSHOTS:
        ex_tokens = count_tokens(llm, ex)
        if tokens + ex_tokens > budget:
            break
        pieces.append(ex)
        tokens += ex_tokens
    return base + "".join(pieces)

def maybe_truncate_input(llm: Llama, input_text: str, ctx: int, sys_str: str, max_gen_tokens: int) -> Tuple[str, int, int]:
    SAFETY = 64
    budget = ctx - max_gen_tokens - SAFETY
    user_block = USER_TEMPLATE.format(input_text=input_text)
    total = count_tokens(llm, sys_str) + count_tokens(llm, user_block)
    if total <= budget:
        return input_text, total, budget

    toks_input = llm.tokenize(input_text.encode("utf-8"), add_bos=False)
    keep = max(64, budget - count_tokens(llm, sys_str) - count_tokens(llm, USER_TEMPLATE.format(input_text="")))
    if keep < 64:
        keep = 64
    truncated_ids = toks_input[-keep:]
    truncated_text = llm.detokenize(truncated_ids).decode("utf-8", errors="ignore")
    user_block2 = USER_TEMPLATE.format(input_text=truncated_text)
    total2 = count_tokens(llm, sys_str) + count_tokens(llm, user_block2)
    return truncated_text, total2, budget

def build_messages(llm: Llama, input_text: str, ctx: int, max_gen_tokens: int) -> List[Dict[str, str]]:
    system_str = fit_system_with_examples(llm, ctx, max_gen_tokens)
    input_text_fitted, _, _ = maybe_truncate_input(llm, input_text, ctx, system_str, max_gen_tokens)
    return [
        {"role": "system", "content": system_str},
        {"role": "user", "content": USER_TEMPLATE.format(input_text=input_text_fitted.strip())},
    ]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                    help="Path to GGUF model file")
    ap.add_argument("--input", default="data/data_llm_io.jsonl")
    ap.add_argument("--out", default="data/preds_llama31_8b_ex04.jsonl")
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

    records = list(load_jsonl(args.input))
    out_rows = []
    for r in tqdm(records, desc="Generating"):
        messages = build_messages(llm, r["Input"], args.ctx, args.max_tokens)
        out = llm.create_chat_completion(
            messages=messages,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
        content = out["choices"][0]["message"]["content"]
        pred = decode_to_json(content)
        out_rows.append({"ID": r["ID"], "Input": r["Input"], "Pred": pred})

    save_jsonl(args.out, out_rows)
    print(f"Wrote {len(out_rows)} predictions -> {args.out}")

if __name__ == "__main__":
    main()
