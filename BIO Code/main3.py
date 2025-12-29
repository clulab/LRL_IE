import os
import json
import argparse
from typing import List, Dict, Any
from collections import Counter
from tqdm import tqdm
from llama_cpp import Llama

# -----------------------------
# Tag schema (must match eval)
# -----------------------------
VALID_TAGS = {
    "B-Age","I-Age",
    "B-Dosage","I-Dosage",
    "B-Health_Condition","I-Health_Condition",
    "B-Medicine","I-Medicine",
    "B-Medical_Procedure","I-Medical_Procedure",
    "B-Specialist","I-Specialist",
    "B-Symptom","I-Symptom",
    "O"
}
BASE_TYPES = {
    "Age","Dosage","Health_Condition","Medicine",
    "Medical_Procedure","Specialist","Symptom","O"
}

# -----------------------------
# Tiny few-shot prefix (fallback)
# -----------------------------
FEW_SHOT_PREFIX = """You are a Entity recognizer that performs medical Named Entity Recognition (NER).
For each input, identify ONLY the entities that are explicitly present in the provided text and output STRICTLY in BIO format.

Output requirements:
- Output exactly one tag per token, space-separated.
- The number of output tags MUST exactly match the number of input tokens.
- Use ONLY these tags:
B-Age I-Age B-Dosage I-Dosage B-Health_Condition I-Health_Condition B-Medicine I-Medicine B-Medical_Procedure I-Medical_Procedure B-Specialist I-Specialist B-Symptom I-Symptom O

Rules:
- Copy spans verbatim from the text (Bangla/English as they appear). No paraphrasing or hallucination.
- Duration vs Age: if a number modifies time words ("গত/পিছনের/ধরে" + দিন/সপ্তাহ/মাস, or "last/for/since" + days/weeks/months), DO NOT label it as Age.
- Negation: if a symptom is negated within ~5 tokens ("না/নাই/নেই/করিনি/হয়নি/হয়ে নাই/হয় নি"), DO NOT tag it as Symptom.
- Lab/test terms alone (e.g., Triglyceride, কোলেস্টেরল, HbA1c) are NOT symptoms unless the text explicitly states a complaint.
- Prefer concise head+modifier spans; exclude extra function words/punctuation. Do not tag standalone single letters (e.g., X/RT/S).
- Only label Age when it is an age expression (e.g., "৪০ বছর", "years old", "Age 27"), not bare numbers.
- No commentary before or after the tags line.

Label hints (recall boosters without adding false positives):
- Symptom: Tag single-word symptoms (e.g., "জ্বর", "কাশি", "বমি") when a complaint verb appears nearby ("আছে/হচ্ছে/লাগছে/অনুভব/ভুগছি/সমস্যা").
- Symptom: Also tag collocates exactly ("মাথা ব্যথা", "গলা ব্যথা", "ঘন কফ", "বুকে ব্যথা").
- Health_Condition: Tag disease/diagnosis nouns ("ডায়াবেটিস", "উচ্চ রক্তচাপ", "অ্যাজমা", "থাইরয়েড", "মাইগ্রেন", "গ্যাস্ট্রাইটিস").
- Specialist: Tag titles/specialities ("ডাক্তার", "চিকিৎসক", "বিশেষজ্ঞ", "ইএনটি বিশেষজ্ঞ", "গ্যাস্ট্রোএন্টারোলজিস্ট", "নিউরোলজিস্ট").
- Medical_Procedure: Tag tests/imaging when performed/ordered ("করা হয়েছে/করাতে বলেছেন", "done/ordered").

You will be asked:
Which entities (Age, Symptom, Medicine, Health_Condition, Specialist, Medical_Procedure) are present in this text?

Examples (BIO):

Example:
Sentence: আমার পিত্ত থলিতে পাথর আছে ।
Tags: O O O O B-Health_Condition I-Health_Condition O

Example:
Sentence: আমার বয়স 22 বছর .
Tags: O O B-Age I-Age O

Example:
Sentence: তিনি ECG এবং ECHO করাতে বলেন ।
Tags: O B-Medical_Procedure O B-Medical_Procedure O O

Example:
Sentence: তিনি PROPRANOLOL HCL 10 mg খেতে বলেন ।
Tags: O B-Medicine I-Medicine I-Medicine I-Medicine O O

Example:
Sentence: আমার জ্বর নেই ।
Tags: O B-Symptom O

Example:
Sentence: শ্বাস নিতে কষ্ট হচ্ছে ।
Tags: B-Symptom I-Symptom I-Symptom O

Example:
Sentence: বুকে ব্যথা হচ্ছে ।
Tags: B-Symptom I-Symptom O

Example:
Sentence: ডাক্তার এর পরামর্শ চাচ্ছি ।
Tags: B-Specialist O O O

Example:
Sentence: চেস্ট এক্সরে রিপোর্ট নরমাল ।
Tags: B-Medical_Procedure I-Medical_Procedure O O
""".strip()

# -----------------------------
# IO helpers
# -----------------------------
def load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# -----------------------------
# Tag cleaning / normalization
# -----------------------------
def clean_tag(tag: str) -> str:
    t = str(tag).strip()
    if not t:
        return "O"
    t = (
        t.replace(" ", "")
         .replace("–", "-")
         .replace("—", "-")
         .replace("\u200c", "")
         .replace("\u200b", "")
    )
    if t.upper().startswith("B-"):
        t = "B-" + t[2:]
    elif t.upper().startswith("I-"):
        t = "I-" + t[2:]
    if t in VALID_TAGS or t == "O":
        return t
    return t  # may be base type; handled later

def bioify_from_base_types(raw_tags: List[str], n_tokens: int) -> List[str]:
    out: List[str] = []
    prev = None
    for i in range(n_tokens):
        raw = raw_tags[i] if i < len(raw_tags) else "O"
        cleaned = clean_tag(raw)

        if cleaned in VALID_TAGS or cleaned == "O":
            out.append(cleaned if cleaned in VALID_TAGS else "O")
            prev = cleaned.split("-", 1)[1] if cleaned != "O" else None
            continue

        base = cleaned
        if base in BASE_TYPES and base != "O":
            out.append(("I-" if prev == base else "B-") + base)
            prev = base
        else:
            out.append("O"); prev = None
    return out

# -----------------------------
# Model init (plain completion)
# -----------------------------
def make_model(model_path: str, n_ctx: int, n_threads: int, n_gpu_layers: int, n_batch: int) -> Llama:
    if not os.path.exists(model_path):
        raise ValueError(f"Model path does not exist: {model_path}")
    return Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=(n_threads if n_threads > 0 else os.cpu_count()),
        n_batch=n_batch,
        n_gpu_layers=n_gpu_layers,     # -1 = full Metal offload on Mac; reduce if OOM
        # IMPORTANT: use raw completions (no chat_format) to match SFT
    )

# -----------------------------
# Prompt builders
# -----------------------------
def build_prompt(sentence_tokens: List[str], mode: str,
                 prefix: str, suffix: str, add_tags_token: bool) -> str:
    sent = " ".join(sentence_tokens)

    # Default: the common SFT pattern
    if mode == "st":
        # NOTE the trailing space after "Tags: "
        return f"Sentence: {sent}\nTags: "

    if mode == "fewshot":
        return f"{FEW_SHOT_PREFIX}\n\nSentence: {sent}\nTags: "

    # mode == "raw"
    built = ""
    if prefix:
        built += prefix.format(sent=sent)
    if suffix:
        built += suffix.format(sent=sent)
    if not prefix and not suffix:
        built = sent + "\n"
    if add_tags_token:
        # trailing space helps decoding of first tag token
        built += "Tags: "
    return built

# -----------------------------
# One forward pass → tags
# -----------------------------
def decode_tags_from_text(text: str, n_tokens: int) -> List[str]:
    if text.lower().startswith("tags:"):
        text = text[5:].strip()
    raw_tags = text.replace(",", " ").split()
    cleaned = [clean_tag(t) for t in raw_tags]
    invalid = sum(1 for t in cleaned if t not in VALID_TAGS)

    # Heuristic: if many invalid, try BIO-ify from base types
    if invalid > max(2, int(0.2 * n_tokens)):
        tags = bioify_from_base_types(raw_tags, n_tokens)
    else:
        tags = cleaned

    # enforce exact length
    if len(tags) > n_tokens:
        tags = tags[:n_tokens]
    elif len(tags) < n_tokens:
        tags += ["O"] * (n_tokens - len(tags))

    return [t if t in VALID_TAGS else "O" for t in tags]

def generate_tags(llm: Llama, tokens: List[str], prompt: str) -> List[str]:
    # room for all tags; cap to avoid runaway
    max_new = min(max(len(tokens) + 32, 32), 1024)
    out = llm(
        prompt=prompt,
        max_tokens=max_new,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        stop=["\n"]  # expect a single line of tags
    )
    text = out["choices"][0]["text"].strip()
    return decode_tags_from_text(text, len(tokens))

# -----------------------------
# CLI / Main
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate Bangla medical NER.")
    ap.add_argument("--data_path", type=str, default="data/test_v1_fixed.json",
                    help="Test JSON with TOKEN (list[str]) and NER_TAG (list[str]).")
    ap.add_argument("--model_path", type=str, default="models/Qwen3-8B-Q4_K_M.gguf.gguf",
                    help="GGUF path. Use F16 for best accuracy; Q4_K_M is smaller/faster.")
    ap.add_argument("--out_path", type=str, default="data/qwen0p6_preds.json",
                    help="Where to save predictions JSON.")

    ap.add_argument("--mode", choices=["st", "fewshot", "raw"], default="st",
                    help="Input template: st=Sentence/Tags, fewshot=tiny preamble+st, raw=custom.")
    ap.add_argument("--prefix", type=str, default="Sentence: {sent}\n",
                    help="RAW mode/custom: prefix text; include {sent} to inject the sentence.")
    ap.add_argument("--suffix", type=str, default="",
                    help="RAW mode/custom: suffix text; include {sent} if you need it there.")
    ap.add_argument("--add_tags_token", action="store_true",
                    help="Append literal 'Tags: ' after prefix+suffix (with trailing space).")

    ap.add_argument("--n_ctx", type=int, default=2048)
    ap.add_argument("--n_threads", type=int, default=0)
    ap.add_argument("--n_gpu_layers", type=int, default=-1)
    ap.add_argument("--n_batch", type=int, default=512)
    ap.add_argument("--debug_first", type=int, default=0,
                    help="If >0, print the built prompt and model output for the first N samples.")
    return ap.parse_args()

def main():
    args = parse_args()
    data = load_dataset(args.data_path)
    llm = make_model(
        model_path=args.model_path,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_gpu_layers=args.n_gpu_layers,
        n_batch=args.n_batch,
    )

    preds = []
    pred_counter = Counter()

    for i, item in enumerate(tqdm(data, desc=f"Evaluating ({args.mode} → fallback if all-O)")):
        tokens: List[str] = item["TOKEN"]
        gold:   List[str] = item["NER_TAG"]

        # 1) primary prompt
        prompt = build_prompt(tokens, mode=args.mode,
                              prefix=args.prefix, suffix=args.suffix,
                              add_tags_token=(args.mode != "st"))

        try:
            tags = generate_tags(llm, tokens, prompt)
        except Exception as e:
            print(f"[WARN] sample {i} primary failed: {e}")
            tags = ["O"] * len(tokens)

        # 2) if model produced almost all O → retry with few-shot stabilizer
        if sum(1 for t in tags if t == "O") >= int(0.95 * len(tags)):
            try:
                prompt2 = build_prompt(tokens, mode="fewshot",
                                       prefix="", suffix="", add_tags_token=False)
                tags2 = generate_tags(llm, tokens, prompt2)
                # accept retry if better (more non-O)
                if sum(1 for t in tags2 if t != "O") > sum(1 for t in tags if t != "O"):
                    tags = tags2
                    if args.debug_first and i < args.debug_first:
                        print("\n[Retry with fewshot accepted]")
            except Exception as e:
                print(f"[WARN] sample {i} fewshot retry failed: {e}")

        if args.debug_first and i < args.debug_first:
            print("\n--- DEBUG SAMPLE", i, "---")
            print(prompt)
            print("PRED:", " ".join(tags))

        preds.append({
            "true_labels": gold,
            "predicted_labels_flat": tags
        })
        pred_counter.update(tags)

    save_json(preds, args.out_path)
    print(f"✅ Saved predictions to: {args.out_path}")

    # quick distribution peek (helps catch "all O")
    total = sum(pred_counter.values())
    print("Prediction distribution (top 8):")
    for tag, cnt in pred_counter.most_common(8):
        print(f"  {tag:>22s}: {cnt:7d}  ({cnt/total*100:.2f}%)")

if __name__ == "__main__":
    main()
