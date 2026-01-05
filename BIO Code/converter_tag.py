import json
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any

# =====================
# CONFIG
# =====================
IN_PATH   = "data/train.json"   # input: list[{"ID","TOKEN","NER_TAG"}] (or keys 'tokens','labels'). 
            #test_v1_fixed.json test data
OUT_JSONL = "data/dat_llm_io.jsonl"    # LLM-friendly (one record per line)
OUT_JSON  = "data/dat_llm_io.json"     # pretty JSON list for inspection

#dat_llm_io.jsonl. train data
#data_llm_io.jsonl. test data

# If True, also include character offsets for each span (under "Entities")
ADD_OFFSETS = False

# Fixed schema to allow (extend if needed)
ALLOWED_TYPES = {
    "Age",
    "Symptom",
    "Medicine",
    "Health_Condition",
    "Specialist",
    "Medical_Procedure",
}

# Canonicalize tag variants to a key in ALLOWED_TYPES
TAG_CANON = {
    "Health Condition": "Health_Condition",
    "Health-Condition": "Health_Condition",
    "Condition": "Health_Condition",
    "Procedure": "Medical_Procedure",
    "Medical-Procedure": "Medical_Procedure",
    "Speciality": "Specialist",
    "Specialist": "Specialist",
    "Age": "Age",
    "Symptom": "Symptom",
    "Medicine": "Medicine",
    "Drug": "Medicine",
}

# Punctuation that attaches to the previous token (Bangla + English)
_PUNCT_RIGHT = {".", ",", ":", ";", "!", "?", "।"}

# Characters to trim from span edges after detokenization
_STRIP_CHARS = " \t\r\n.,;:!?()[]{}'\"“”‘’|/\\，。；：！？（）【】"

# =====================
# Helpers
# =====================

def detok(tokens: List[str]) -> str:
    """Detokenize with Bangla + English punctuation rules."""
    out = ""
    for i, tok in enumerate(tokens):
        if i == 0:
            out += tok
        else:
            if tok in _PUNCT_RIGHT:
                out += tok
            else:
                out += " " + tok
    return out

def build_char_offsets(tokens: List[str]) -> List[Tuple[int, int]]:
    """
    Return (start_char, end_char_inclusive) per token according to detok() rule above.
    Useful for character-level spans if ADD_OFFSETS=True.
    """
    spans = []
    text = ""
    for i, tok in enumerate(tokens):
        if i == 0:
            start = 0
            text = tok
        else:
            if tok in _PUNCT_RIGHT:
                start = len(text)
                text += tok
            else:
                start = len(text) + 1
                text += " " + tok
        end = len(text) - 1
        spans.append((start, end))
    return spans

def canon_type_from_bio(tag: str) -> str | None:
    """Normalize BIO tag like 'B-Health Condition' → 'Health_Condition' (only if allowed)."""
    if not tag or tag == "O" or "-" not in tag:
        return None
    _, raw = tag.split("-", 1)
    raw = raw.strip()
    mapped = TAG_CANON.get(raw, raw.replace(" ", "_"))
    return mapped if mapped in ALLOWED_TYPES else None

def extract_spans_bio_strict(tokens: List[str], tags: List[str]) -> List[Tuple[str,int,int]]:
    """
    BIO-strict extraction:
      - Start only on B-TYPE
      - Continue while next tags are I-TYPE
      - Stop on first non I-TYPE (O, B-*, I-other, etc.)
      - Never include tokens tagged O
    Returns list of (type, start_idx, end_idx).
    """
    spans: List[Tuple[str,int,int]] = []
    i = 0
    n = len(tags)

    while i < n:
        tag = tags[i]
        if tag.startswith("B-"):
            typ = canon_type_from_bio(tag)
            if typ is None:
                i += 1
                continue
            start = i
            j = i + 1
            # consume contiguous I-typ
            while j < n and tags[j].startswith("I-"):
                if canon_type_from_bio(tags[j]) != typ:
                    break
                j += 1
            end = j - 1
            spans.append((typ, start, end))
            i = j
        else:
            i += 1

    return spans

def span_text(tokens: List[str], s: int, e: int) -> str:
    """Detokenize a span and lightly clean it."""
    text = detok(tokens[s:e+1]).strip(_STRIP_CHARS)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def collect_output(tokens: List[str], spans: List[Tuple[str,int,int]]) -> Dict[str, List[str]]:
    """
    Build Output dict: type → list of span strings (BIO-strict).
    Remove empty types.
    """
    out: Dict[str, List[str]] = {}
    for typ, s, e in spans:
        t = span_text(tokens, s, e)
        if t:
            out.setdefault(typ, []).append(t)
    return out

def validate_bio_strict(spans: List[Tuple[str,int,int]], tags: List[str]) -> List[str]:
    """
    Validate each span: must be B-typ followed by >=0 I-typ, no O’s inside, no I of other type.
    Returns list of error strings (empty if OK).
    """
    errs = []
    for typ, s, e in spans:
        if not tags[s].startswith("B-"):
            errs.append(f"Span {s}-{e} type={typ}: start not B- (got {tags[s]!r})")
        if canon_type_from_bio(tags[s]) != typ:
            errs.append(f"Span {s}-{e} type={typ}: B- type mismatch (got {canon_type_from_bio(tags[s])})")
        for k in range(s+1, e+1):
            if not tags[k].startswith("I-"):
                errs.append(f"Span {s}-{e} type={typ}: token {k} not I- (got {tags[k]!r})")
            elif canon_type_from_bio(tags[k]) != typ:
                errs.append(f"Span {s}-{e} type={typ}: I- type mismatch at {k} (got {canon_type_from_bio(tags[k])})")
    return errs

# =====================
# Main
# =====================

def main():
    data = json.loads(Path(IN_PATH).read_text(encoding="utf-8"))

    out_records: List[Dict[str, Any]] = []
    total = 0
    skipped = 0
    warned = 0

    for ex in data:
        tokens = ex.get("TOKEN") or ex.get("tokens")
        tags   = ex.get("NER_TAG") or ex.get("labels") or ex.get("tags")
        ex_id  = ex.get("ID") or ex.get("id")

        if not tokens or not tags:
            skipped += 1
            continue

        if len(tokens) != len(tags):
            print(f"[warn] ID={ex_id} token/tag length mismatch: {len(tokens)} vs {len(tags)} — skipped")
            warned += 1
            skipped += 1
            continue

        total += 1

        # 1) Input sentence via detok
        sent = detok(tokens)

        # 2) BIO-strict spans
        spans = extract_spans_bio_strict(tokens, tags)

        # 3) Validate strictness
        errs = validate_bio_strict(spans, tags)
        if errs:
            print(f"[warn] ID={ex_id} BIO-strict validation found {len(errs)} issue(s).")
            for e in errs[:5]:
                print("   -", e)
            if len(errs) > 5:
                print("   - ...")

        # 4) Output dictionary
        output = collect_output(tokens, spans)

        rec: Dict[str, Any] = {"ID": ex_id, "Input": sent, "Output": output}

        # 5) (Optional) add character offsets for each span (nice for eval)
        if ADD_OFFSETS and spans:
            tok_char = build_char_offsets(tokens)
            ents = []
            for typ, s, e in spans:
                start_char = tok_char[s][0]
                end_char   = tok_char[e][1]
                ents.append({
                    "type": typ,
                    "text": span_text(tokens, s, e),
                    "start": start_char,
                    "end": end_char
                })
            rec["Entities"] = ents  # extra, not required by your training format

        out_records.append(rec)

    # 6) Write JSONL
    with Path(OUT_JSONL).open("w", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 7) Write pretty JSON
    Path(OUT_JSON).write_text(json.dumps(out_records, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"processed: {total} | written: {len(out_records)} | skipped: {skipped} | warnings: {warned}")
    print(f"→ {OUT_JSONL}")
    print(f"→ {OUT_JSON}")

if __name__ == "__main__":
    main()
