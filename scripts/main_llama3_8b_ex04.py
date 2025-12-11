import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm
from llama_cpp import Llama

SCHEMA_KEYS = ["Sign_or_Symptom", "Disease_or_Syndrome", "Pathologic_Function", "Finding", "Other_Clinical_Disorder", "Patient", "H-Professional", "Other_Role"]

# ======================
# QA-style SYSTEM PROMPT (LLaMA)
# ======================
SYSTEM_PROMPT_BASE = (
    "You are a question-answering assistant that performs medical Named Entity Recognition (NER).\n"
    "For each question, identify ONLY the entities that are explicitly present in the provided text and answer STRICTLY in JSON.\n"
    "Your answer MUST be a single JSON object with EXACTLY these six keys (even if empty), in this order:\n"
    "{\"Sign_or_Symptom\":[],\"Disease_or_Syndrome\":[],\"Pathologic_Function\":[],\"Finding\":[],\"Other_Clinical_Disorder\":[],\"Patient\":[],\"H-Professional\":[],\"Other_Role\":[]}\n\n"
    "Rules:\n"
    "- Copy spans verbatim from the text (Basque/English as they appear). No paraphrasing or hallucination.\n"
    "- Annotate the maximum extent of each entity. This means that all modifiers of a noun phrase should be included.\n"
    "- Each occurrence of a disorder in the text should be considered independently.\n"
    "- All the disorders present in the text are to be annotated, even if they do not pertain to the patient or are presented in hypothetical or generic contexts.\n"
    "- Lists contain strings only; no commentary before or after the JSON.\n\n"
    "Label hints (recall boosters without adding false positives):\n"
    "- Sign_or_Symptom is an observable manifestation of a disease or condition based on clinical judgment, or a manifestation of a disease or condition which is experienced by the patient and reported as a subjective observation.\n"
    "- Disease_or_Syndrome is a condition which alters or interferes with a normal process, state, or activity of an organism. It is usually characterized by the abnormal functioning of one or more of the host's systems, parts, or organs.\n"
    "- Pathologic_Function is a disordered process, activity, or state of the organism as a whole, of a body system or systems, or of multiple organs or tissues.\n"
    "- Finding is that which is discovered by direct observation or measurement of an organism attribute or condition, including the clinical history of the patient.\n"
    "- Other_Clinical_Disorder represents other disorders such as neoplastic process, injury or poisoning, mental or behavioral dysfunction, anatomical abnormality, acquired abnormality, and congenital abnormality.\n"
    "- A Patient is the person to whom a clinical narrative refers and that is being treated by a health professional.\n"
    "- A H-Professional is a professional that takes care of the patients and interact with them. These are normally doctors, nurses, emergency department paramedicals, etc.\n"
    "- Other_Role represents the actors that are neither the patient or any health professional present in text. These can be animals, family and acquaintances of the patient, lawyers, generic mentions, etc.\n"
    "You will be asked the question:\n"
    "\"Which entities (Sign_or_Symptom, Disease_or_Syndrome, Pathologic_Function, Finding, Other_Clinical_Disorder, Patient, H-Professional, Other_Role) are present in this text?\"\n\n"
    "Examples:\n"
)

# ======================
# Few-shots — QA style
# ======================
FEWSHOTS: List[str] = [
    (
        "Question: Which entities (Sign_or_Symptom, Disease_or_Syndrome, Pathologic_Function, Finding, Other_Clinical_Disorder, Patient, H-Professional, Other_Role) are present in this text?\n"
        "Text: \"Bitan ingresatu behar izan dute, oka dela eta pairatu duen deshidratazioagatik.\"\n"
        "Answer: {\"Sign_or_Symptom\": [\"oka\"], \"Disease_or_Syndrome\": [\"deshidratazioagatik\"], \"Pathologic_Function\": [], \"Finding\": [], \"Other_Clinical_Disorder\": [], \"Patient\": [], \"H-Professional\": [], \"Other_Role\": []}\n\n"
    ),
    (
        "Question: Which entities (Sign_or_Symptom, Disease_or_Syndrome, Pathologic_Function, Finding, Other_Clinical_Disorder, Patient, H-Professional, Other_Role) are present in this text?\n"
        "Text: \"PPRB (Giltzurrunaren alde bietako ukabil-perkusioa): minik ez.\"\n"
        "Answer: {\"Sign_or_Symptom\": [\"minik\"], \"Disease_or_Syndrome\": [], \"Pathologic_Function\": [], \"Finding\": [], \"Other_Clinical_Disorder\": [], \"Patient\": [], \"H-Professional\": [], \"Other_Role\": []}\n\n"
    ),
    (
        "Question: Which entities (Sign_or_Symptom, Disease_or_Syndrome, Pathologic_Function, Finding, Other_Clinical_Disorder, Patient, H-Professional, Other_Role) are present in this text?\n"
        "Text: \"Aita oso zorrotza eta jeloskorra, ama bakegilea eta depresioarekin.\"\n"
        "Answer: {\"Sign_or_Symptom\": [], \"Disease_or_Syndrome\": [], \"Pathologic_Function\": [], \"Finding\": [], \"Other_Clinical_Disorder\": [\"depresioarekin\"], \"Patient\": [], \"H-Professional\": [], \"Other_Role\": [\"Aita\", \"ama\"]}\n\n"
    ),
    (
        "Question: Which entities (Sign_or_Symptom, Disease_or_Syndrome, Pathologic_Function, Finding, Other_Clinical_Disorder, Patient, H-Professional, Other_Role) are present in this text?\n"
        "Text: \"Kirurgia orokorrean egindako analitikan ez zen infekzio daturik objektibatu, baina baibilirrubinaren eta Ca 19.9-ren igoera, minbizia susmoa handiagotuz.\"\n"
        "Answer: {\"Sign_or_Symptom\": [], \"Disease_or_Syndrome\": [\"infekzio\"], \"Pathologic_Function\": [], \"Finding\": [], \"Other_Clinical_Disorder\": [\"minbizia\"], \"Patient\": [], \"H-Professional\": [], \"Other_Role\": []}\n\n"
    ),
    (
        "Question: Which entities (Sign_or_Symptom, Disease_or_Syndrome, Pathologic_Function, Finding, Other_Clinical_Disorder, Patient, H-Professional, Other_Role) are present in this text?\n"
        "Text: \"Gaixoa sindrome hemofagozitikoa zuen 71 urteko emakumezkoa zen.\"\n"
        "Answer: {\"Sign_or_Symptom\": [], \"Disease_or_Syndrome\": [\"sindrome hemofagozitikoa\"], \"Pathologic_Function\": [], \"Finding\": [], \"Other_Clinical_Disorder\": [], \"Patient\": [\"Gaixoa\", \"71 urteko emakumezkoa\"], \"H-Professional\": [], \"Other_Role\": []}\n\n"
    ),
    (
        "Question: Which entities (Sign_or_Symptom, Disease_or_Syndrome, Pathologic_Function, Finding, Other_Clinical_Disorder, Patient, H-Professional, Other_Role) are present in this text?\n"
        "Text: \"Pazienteak duela bi aste edemak nabaritu zituen zangoetan.\"\n"
        "Answer: {\"Sign_or_Symptom\": [], \"Disease_or_Syndrome\": [], \"Pathologic_Function\": [\"edemak\"], \"Finding\": [], \"Other_Clinical_Disorder\": [], \"Patient\": [\"Pazienteak\"], \"H-Professional\": [], \"Other_Role\": []}\n\n"
    ),
    (
        "Question: Which entities (Sign_or_Symptom, Disease_or_Syndrome, Pathologic_Function, Finding, Other_Clinical_Disorder, Patient, H-Professional, Other_Role) are present in this text?\n"
        "Text: \"Martxoaren 3an, txertatutako gunean flebitisa duela ohartzen da erizaina.\"\n"
        "Answer: {\"Sign_or_Symptom\": [], \"Disease_or_Syndrome\": [\"flebitisa\"], \"Pathologic_Function\": [], \"Finding\": [], \"Other_Clinical_Disorder\": [], \"Patient\": [], \"H-Professional\": [\"erizaina\"], \"Other_Role\": []}\n\n"
    ),
    (
        "Question: Which entities (Sign_or_Symptom, Disease_or_Syndrome, Pathologic_Function, Finding, Other_Clinical_Disorder, Patient, H-Professional, Other_Role) are present in this text?\n"
        "Text: \"Bihotzerrea sentitzen du maiz, edozein janari hartuta, eta goragalea ere bai.\"\n"
        "Answer: {\"Sign_or_Symptom\": [\"Bihotzerrea\", \"goragalea\"], \"Disease_or_Syndrome\": [], \"Pathologic_Function\": [], \"Finding\": [], \"Other_Clinical_Disorder\": [], \"Patient\": [], \"H-Professional\": [], \"Other_Role\": []}\n\n"
    ),
    (
        "Question: Which entities (Sign_or_Symptom, Disease_or_Syndrome, Pathologic_Function, Finding, Other_Clinical_Disorder, Patient, H-Professional, Other_Role) are present in this text?\n"
        "Text: \"Kirurgiak 4 ordu irauten ditu, azidosia, hipotermia eta koagulopatia agertuz.\"\n"
        "Answer: {\"Sign_or_Symptom\": [], \"Disease_or_Syndrome\": [\"koagulopatia\"], \"Pathologic_Function\": [\"azidosia\"], \"Finding\": [\"hipotermia\"], \"Other_Clinical_Disorder\": [], \"Patient\": [], \"H-Professional\": [], \"Other_Role\": []}\n\n"
    ),
    (
        "Question: Which entities (Sign_or_Symptom, Disease_or_Syndrome, Pathologic_Function, Finding, Other_Clinical_Disorder, Patient, H-Professional, Other_Role) are present in this text?\n"
        "Text: \"Ea krisi honi buruz hitz egiteko lagun edo familiarik duen, pentsamendu suizidak etortzen bazaizkio adi egon daitezen.\"\n"
        "Answer: {\"Sign_or_Symptom\": [], \"Disease_or_Syndrome\": [], \"Pathologic_Function\": [], \"Finding\": [\"krisi\", \"pentsamendu suizidak\"], \"Other_Clinical_Disorder\": [], \"Patient\": [], \"H-Professional\": [], \"Other_Role\": []}\n\n"
    )
]

# ======================
# QA-style user template
# ======================
USER_TEMPLATE = (
    "Question: Which entities (Sign_or_Symptom, Disease_or_Syndrome, Pathologic_Function, Finding, Other_Clinical_Disorder, Patient, H-Professional, Other_Role) are present in this text?\n"
    "Text:\n<<<\n{input_text}\n>>>\n"
    "Answer (a single JSON object with those eight keys):\n"
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
    ap.add_argument("--input", default="data/in.jsonl")
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
