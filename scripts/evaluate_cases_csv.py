#!/usr/bin/env python3
import argparse
import csv
import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

# -----------------------------
# Schema (keep consistent)
# -----------------------------
SCHEMA: List[str] = [
    "Age",
    "Symptom",
    "Medicine",
    "Health_Condition",
    "Specialist",
    "Medical_Procedure",
]

# -----------------------------
# Outcome / error category names (UPDATED NAMES ONLY)
# -----------------------------
CAT_EXACT_MATCH = "Exact-Match"          # correct (unchanged)
CAT_CORRECT_EMPTY = "Correct-Empty"      # correct (unchanged)

# error categories only (renamed):
CAT_EMPTY_GOLD_FP = "Hallucinated Entities"        # gold empty, pred non-empty
CAT_ALL_MISSED = "All-Missed"                      # gold non-empty, pred empty
CAT_RECALL_ONLY = "Missed Entities"                # FN>0, FP=0
CAT_PRECISION_ONLY = "Extra Entities"              # FP>0, FN=0
CAT_TYPE_CONFUSION = "Type Confusion"              # same surface span, wrong label
CAT_OVER_MIXED = "Extra-Dominant Errors"           # FP>FN
CAT_UNDER_MIXED = "Missed-Dominant Errors"         # FN>FP
CAT_BAL_MIXED = "Boundary mismatch"                # FP==FN (or ties / balanced mixed)

NON_ERROR_CATEGORIES = {CAT_EXACT_MATCH, CAT_CORRECT_EMPTY}

# Order to print in summary
ERROR_ORDER = [
    CAT_EMPTY_GOLD_FP,
    CAT_ALL_MISSED,
    CAT_RECALL_ONLY,
    CAT_PRECISION_ONLY,
    CAT_TYPE_CONFUSION,
    CAT_OVER_MIXED,
    CAT_UNDER_MIXED,
    CAT_BAL_MIXED,
]


# -----------------------------
# IO + helpers
# -----------------------------
def load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_span(span: Any) -> str:
    """Strict-ish normalization (close to your evaluate_details.py behavior)."""
    if span is None:
        return ""
    if not isinstance(span, str):
        span = str(span)
    span = span.strip()
    span = " ".join(span.split())  # collapse whitespace
    return span


def to_type_sets(obj: dict) -> Dict[str, Set[str]]:
    """
    obj example: {"Symptom": ["জ্বর", "কাশি"], "Medicine": ["Paracetamol"]}
    returns dict for all keys in SCHEMA: tag -> set(normalized_spans)
    """
    out: Dict[str, Set[str]] = {}
    obj = obj or {}
    for t in SCHEMA:
        vals = obj.get(t, []) or []
        spans: Set[str] = set()
        if isinstance(vals, list):
            for v in vals:
                vv = normalize_span(v)
                if vv:
                    spans.add(vv)
        out[t] = spans
    return out


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f


def flat_repr(type_sets: Dict[str, Set[str]]) -> str:
    """Readable compact string: Type: a|b ; Type2: c"""
    parts = []
    for t in SCHEMA:
        spans = sorted(type_sets.get(t, set()))
        if spans:
            parts.append(f"{t}: " + " | ".join(spans))
    return " ; ".join(parts)


def json_repr(obj: dict) -> str:
    return json.dumps(obj or {}, ensure_ascii=False, sort_keys=True)


def ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def best_near_miss(
    target_span: str,
    candidates: List[Tuple[str, str]],  # (span, type)
    min_ratio: float = 0.85,
) -> str:
    """
    Find best fuzzy match among candidates.
    Returns formatted string or "".
    """
    best = ("", "", 0.0)  # span, type, score
    for cand_span, cand_type in candidates:
        if cand_span == target_span:
            continue
        s = ratio(target_span, cand_span)
        if s > best[2]:
            best = (cand_span, cand_type, s)
    if best[2] >= min_ratio:
        return f"{target_span} ~ {best[0]} ({best[1]}) r={best[2]:.2f}"
    return ""


def span_to_types(type_sets: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    m: Dict[str, Set[str]] = {}
    for t in SCHEMA:
        for sp in type_sets.get(t, set()):
            m.setdefault(sp, set()).add(t)
    return m


# -----------------------------
# Category logic
# -----------------------------
def categorize(
    gold_total: int,
    pred_total: int,
    fp_total: int,
    fn_total: int,
    has_confusion: bool,
) -> str:
    """
    Returns one category string (including correct outcomes).
    Note: we *exclude* correct outcomes from the ERROR-ONLY CSV.
    """
    if gold_total == 0 and pred_total == 0:
        return CAT_CORRECT_EMPTY
    if gold_total > 0 and pred_total == 0:
        return CAT_ALL_MISSED
    if gold_total == 0 and pred_total > 0:
        return CAT_EMPTY_GOLD_FP
    if fp_total == 0 and fn_total == 0:
        return CAT_EXACT_MATCH

    # If any same surface span appears in both but with different labels → Type Confusion
    if has_confusion:
        return CAT_TYPE_CONFUSION

    # Pure FN / pure FP
    if fn_total > 0 and fp_total == 0:
        return CAT_RECALL_ONLY
    if fp_total > 0 and fn_total == 0:
        return CAT_PRECISION_ONLY

    # Mixed
    if fp_total > fn_total:
        return CAT_OVER_MIXED
    if fn_total > fp_total:
        return CAT_UNDER_MIXED
    return CAT_BAL_MIXED


def main():
    ap = argparse.ArgumentParser(
        description="ERROR-ONLY per-example CSV + strict span metrics for Bangla medical NER (readable categories + %)."
    )
    ap.add_argument("--gold", required=True, help="Gold JSONL with fields: ID, Input, Output")
    ap.add_argument("--pred", required=True, help="Pred JSONL with fields: ID, Input, Pred")
    ap.add_argument("--out_csv", required=True, help="Main metrics CSV (per-type + micro/macro)")
    ap.add_argument(
        "--out_errors_csv",
        default="",
        help="ERROR-ONLY one-row-per-example CSV. Default: <out_csv_stem>_errors.csv",
    )
    ap.add_argument("--near_miss_ratio", type=float, default=0.85, help="Similarity threshold for near-miss reporting.")
    args = ap.parse_args()

    gold_rows = load_jsonl(args.gold)
    pred_rows = load_jsonl(args.pred)

    gold_by_id: Dict[str, dict] = {str(r["ID"]): r for r in gold_rows}
    pred_by_id: Dict[str, dict] = {str(r["ID"]): r for r in pred_rows}

    common_ids = sorted(set(gold_by_id) & set(pred_by_id), key=lambda x: int(x) if x.isdigit() else x)

    # Aggregate per-type counts across ALL evaluated examples
    per_type = {t: {"TP": 0, "FP": 0, "FN": 0, "SUPPORT": 0} for t in SCHEMA}
    micro_tp = micro_fp = micro_fn = 0

    # Error-only output path
    out_csv_path = Path(args.out_csv)
    errors_path = Path(args.out_errors_csv) if args.out_errors_csv else out_csv_path.with_name(out_csv_path.stem + "_errors.csv")
    errors_path.parent.mkdir(parents=True, exist_ok=True)

    # Error-only category counts
    err_counts: Dict[str, int] = {k: 0 for k in ERROR_ORDER}

    # Case-level fields (error-only CSV)
    fieldnames = [
        "ID",
        "Input",
        "Gold_JSON",
        "Pred_JSON",
        "Gold_NER_Flat",
        "Pred_NER_Flat",
        "Gold_Total",
        "Pred_Total",
        "TP_Total",
        "FP_Total",
        "FN_Total",
        "Micro_P",
        "Micro_R",
        "Micro_F1",
        "Missing_spans_all(FN)",
        "Spurious_spans_all(FP)",
        "Confusions(span_label)",
        "Near_miss_FN",
        "Near_miss_FP",
        "Outcome_category",
    ]

    total_eval = 0
    total_errors_written = 0
    total_correct_exact = 0
    total_correct_empty = 0

    with errors_path.open("w", newline="", encoding="utf-8") as f_err:
        w = csv.DictWriter(f_err, fieldnames=fieldnames)
        w.writeheader()

        for _id in common_ids:
            total_eval += 1

            g_obj = (gold_by_id[_id].get("Output") or {})
            p_obj = (pred_by_id[_id].get("Pred") or {})

            g_sets = to_type_sets(g_obj)
            p_sets = to_type_sets(p_obj)

            # Totals
            gold_total = sum(len(g_sets[t]) for t in SCHEMA)
            pred_total = sum(len(p_sets[t]) for t in SCHEMA)

            # Candidates for near-miss
            pred_candidates: List[Tuple[str, str]] = []
            gold_candidates: List[Tuple[str, str]] = []
            for t in SCHEMA:
                for sp in p_sets[t]:
                    pred_candidates.append((sp, t))
                for sp in g_sets[t]:
                    gold_candidates.append((sp, t))

            # Confusions
            g_span_types = span_to_types(g_sets)
            p_span_types = span_to_types(p_sets)
            confusions: List[str] = []
            for sp in sorted(set(g_span_types) & set(p_span_types)):
                for gt in sorted(g_span_types[sp]):
                    for pt in sorted(p_span_types[sp]):
                        if gt != pt:
                            confusions.append(f"{sp}: {gt}->{pt}")

            missing_all: List[str] = []
            spurious_all: List[str] = []

            # Per-type stats
            tp_total = fp_total = fn_total = 0
            for t in SCHEMA:
                G = g_sets[t]
                P = p_sets[t]
                tp = len(G & P)
                fp = len(P - G)
                fn = len(G - P)

                per_type[t]["TP"] += tp
                per_type[t]["FP"] += fp
                per_type[t]["FN"] += fn
                per_type[t]["SUPPORT"] += len(G)

                tp_total += tp
                fp_total += fp
                fn_total += fn

                for sp in sorted(G - P):
                    missing_all.append(f"{t}:{sp}")
                for sp in sorted(P - G):
                    spurious_all.append(f"{t}:{sp}")

            # Micro (per-example)
            micro_tp += tp_total
            micro_fp += fp_total
            micro_fn += fn_total
            mp, mr, mf = prf(tp_total, fp_total, fn_total)

            cat = categorize(
                gold_total=gold_total,
                pred_total=pred_total,
                fp_total=fp_total,
                fn_total=fn_total,
                has_confusion=(len(confusions) > 0),
            )

            # Track correct outcomes but DO NOT write them to the error CSV
            if cat == CAT_EXACT_MATCH:
                total_correct_exact += 1
                continue
            if cat == CAT_CORRECT_EMPTY:
                total_correct_empty += 1
                continue

            # Near-miss diagnostics
            near_fn: List[str] = []
            for t in SCHEMA:
                for gold_span in sorted(g_sets[t] - p_sets[t]):
                    hit = best_near_miss(gold_span, pred_candidates, min_ratio=args.near_miss_ratio)
                    if hit:
                        near_fn.append(hit)

            near_fp: List[str] = []
            for t in SCHEMA:
                for pred_span in sorted(p_sets[t] - g_sets[t]):
                    hit = best_near_miss(pred_span, gold_candidates, min_ratio=args.near_miss_ratio)
                    if hit:
                        near_fp.append(hit)

            # Count this error category
            err_counts[cat] = err_counts.get(cat, 0) + 1
            total_errors_written += 1

            w.writerow(
                {
                    "ID": _id,
                    "Input": gold_by_id[_id].get("Input", ""),
                    "Gold_JSON": json_repr(g_obj),
                    "Pred_JSON": json_repr(p_obj),
                    "Gold_NER_Flat": flat_repr(g_sets),
                    "Pred_NER_Flat": flat_repr(p_sets),
                    "Gold_Total": gold_total,
                    "Pred_Total": pred_total,
                    "TP_Total": tp_total,
                    "FP_Total": fp_total,
                    "FN_Total": fn_total,
                    "Micro_P": f"{mp:.6f}",
                    "Micro_R": f"{mr:.6f}",
                    "Micro_F1": f"{mf:.6f}",
                    "Missing_spans_all(FN)": " ; ".join(missing_all),
                    "Spurious_spans_all(FP)": " ; ".join(spurious_all),
                    "Confusions(span_label)": " ; ".join(confusions),
                    "Near_miss_FN": " ; ".join(near_fn),
                    "Near_miss_FP": " ; ".join(near_fp),
                    "Outcome_category": cat,
                }
            )

    # Write main metrics CSV (per-type + micro/macro) across ALL evaluated examples
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with out_csv_path.open("w", newline="", encoding="utf-8") as f_out:
        w = csv.writer(f_out)
        w.writerow(["Type", "Precision", "Recall", "F1", "Support", "Correctly Caught (%)"])
        macro_p = macro_r = macro_f = 0.0

        for t in SCHEMA:
            TP = per_type[t]["TP"]
            FP = per_type[t]["FP"]
            FN = per_type[t]["FN"]
            S = per_type[t]["SUPPORT"]
            P, R, F1 = prf(TP, FP, FN)
            macro_p += P
            macro_r += R
            macro_f += F1
            caught = (TP / S * 100.0) if S else 0.0
            w.writerow([t, f"{P:.6f}", f"{R:.6f}", f"{F1:.6f}", S, f"{caught:.2f}"])

        n = len(SCHEMA)
        macro_p /= n
        macro_r /= n
        macro_f /= n
        micro_p, micro_r, micro_f = prf(micro_tp, micro_fp, micro_fn)

        w.writerow([])
        w.writerow(["AVERAGE(macro)", f"{macro_p:.6f}", f"{macro_r:.6f}", f"{macro_f:.6f}", "", ""])
        w.writerow(["MICRO", f"{micro_p:.6f}", f"{micro_r:.6f}", f"{micro_f:.6f}", "", ""])

    # -----------------------------
    # Print summary + percentages
    # -----------------------------
    print(f"Saved main metrics CSV     -> {out_csv_path}")
    print(f"Saved ERROR-ONLY cases CSV -> {errors_path}")
    print(f"Common IDs evaluated       -> {total_eval}")
    print(f"Exact-Match (correct)      -> {total_correct_exact}")
    print(f"Correct-Empty (correct)    -> {total_correct_empty}")
    print(f"Total error cases          -> {total_errors_written}")

    # error share of all evaluated items
    err_rate = (total_errors_written / total_eval * 100.0) if total_eval else 0.0
    print(f"Error rate                 -> {err_rate:.2f}%")

    print("\nError type breakdown (count and % of *errors*):")
    if total_errors_written == 0:
        print("  (No errors)")
    else:
        for k in ERROR_ORDER:
            c = err_counts.get(k, 0)
            pct = (c / total_errors_written * 100.0)
            print(f"  {k}: {c}  ({pct:.2f}%)")


if __name__ == "__main__":
    main()
