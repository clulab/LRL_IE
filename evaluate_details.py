import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Fixed NER schema used everywhere
SCHEMA: List[str] = [
    "Age",
    "Symptom",
    "Medicine",
    "Health_Condition",
    "Specialist",
    "Medical_Procedure",
]


def load_jsonl(path: str) -> List[dict]:
    """Load a .jsonl file into a list of dicts."""
    rows: List[dict] = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def index_by_id(rows: List[dict]) -> Dict[int, dict]:
    """Index rows by their 'ID' field."""
    out: Dict[int, dict] = {}
    for r in rows:
        _id = r.get("ID")
        if _id is None:
            raise ValueError("Every row must have an 'ID' field, missing in: %r" % (r,))
        if _id in out:
            raise ValueError("Duplicate ID %r in file" % (_id,))
        out[_id] = r
    return out


def normalize_span(span: str) -> str:
    """
    Normalize an entity span for strict comparison.

    We keep case and characters as-is (strict), but:
    - strip leading/trailing whitespace
    - collapse internal whitespace to a single space
    """
    if not isinstance(span, str):
        span = str(span)
    span = span.strip()
    # collapse multiple spaces / newlines etc. to a single space
    span = " ".join(span.split())
    return span


def to_type_sets(obj: dict) -> Dict[str, Set[str]]:
    """
    Convert a nested dict like:
        {"Symptom": ["জ্বর", "কাশি"], "Medicine": ["Paracetamol"]}
    into:
        {"Symptom": {"জ্বর", "কাশি"}, "Medicine": {"Paracetamol"}, ...}
    for all keys in SCHEMA.
    """
    out: Dict[str, Set[str]] = {}
    for t in SCHEMA:
        vals = obj.get(t, []) or []
        norm_vals = [normalize_span(v) for v in vals if v is not None and str(v).strip() != ""]
        out[t] = set(norm_vals)
    return out


def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    if tp + fp > 0:
        p = tp / float(tp + fp)
    else:
        p = 0.0
    if tp + fn > 0:
        r = tp / float(tp + fn)
    else:
        r = 0.0
    if p + r > 0:
        f1 = 2.0 * p * r / (p + r)
    else:
        f1 = 0.0
    return p, r, f1


def evaluate(
    gold_rows: List[dict],
    pred_rows: List[dict],
):
    """
    Core evaluation logic.

    gold_rows: each row must contain:
        - "ID"
        - "Output": {tag -> [spans]}
    pred_rows: each row must contain:
        - "ID"
        - "Pred": {tag -> [spans]}
    """
    gold_by_id = index_by_id(gold_rows)
    pred_by_id = index_by_id(pred_rows)

    gold_ids = set(gold_by_id.keys())
    pred_ids = set(pred_by_id.keys())
    common_ids = sorted(gold_ids & pred_ids)

    if not common_ids:
        raise ValueError("No overlapping IDs between gold and pred!")

    if gold_ids - pred_ids:
        missing = sorted(gold_ids - pred_ids)
        print(f"WARNING: {len(missing)} gold IDs have no prediction (first few: {missing[:10]})")
    if pred_ids - gold_ids:
        extra = sorted(pred_ids - gold_ids)
        print(f"WARNING: {len(extra)} prediction IDs have no gold (first few: {extra[:10]})")

    # Per-type counters
    per_type_counts: Dict[str, Dict[str, int]] = {
        t: {"TP": 0, "FP": 0, "FN": 0, "SUPPORT": 0} for t in SCHEMA
    }

    # Detailed per-example error rows
    per_example_rows: List[dict] = []

    # Cross-type confusion statistics: (gold_type, pred_type) -> count
    confusion_counts: Dict[Tuple[str, str], int] = {}

    for _id in common_ids:
        gold_obj = gold_by_id[_id].get("Output", {}) or {}
        pred_obj = pred_by_id[_id].get("Pred", {}) or {}

        g_sets = to_type_sets(gold_obj)
        p_sets = to_type_sets(pred_obj)

        # Build span -> type sets for confusion analysis
        gold_span_to_types: Dict[str, Set[str]] = {}
        pred_span_to_types: Dict[str, Set[str]] = {}

        for t in SCHEMA:
            for span in g_sets[t]:
                gold_span_to_types.setdefault(span, set()).add(t)
            for span in p_sets[t]:
                pred_span_to_types.setdefault(span, set()).add(t)

        # Per-type TP/FP/FN and per-example breakdown
        for t in SCHEMA:
            G = g_sets[t]
            P = p_sets[t]

            tp_spans = G & P
            fp_spans = P - G
            fn_spans = G - P

            tp = len(tp_spans)
            fp = len(fp_spans)
            fn = len(fn_spans)
            support = len(G)

            per_type_counts[t]["TP"] += tp
            per_type_counts[t]["FP"] += fp
            per_type_counts[t]["FN"] += fn
            per_type_counts[t]["SUPPORT"] += support

            per_example_rows.append(
                {
                    "ID": _id,
                    "Type": t,
                    "Gold_spans": "; ".join(sorted(G)) if G else "",
                    "Pred_spans": "; ".join(sorted(P)) if P else "",
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "Missing_spans(FN)": "; ".join(sorted(fn_spans)) if fn_spans else "",
                    "Spurious_spans(FP)": "; ".join(sorted(fp_spans)) if fp_spans else "",
                }
            )

        # Cross-type confusion: same surface form but different label
        all_spans = set(gold_span_to_types.keys()) | set(pred_span_to_types.keys())
        for span in all_spans:
            g_types = gold_span_to_types.get(span, set())
            p_types = pred_span_to_types.get(span, set())
            for gt in g_types:
                for pt in p_types:
                    if gt != pt:
                        confusion_counts[(gt, pt)] = confusion_counts.get((gt, pt), 0) + 1

    # Aggregate metrics
    rows_for_csv: List[dict] = []

    total_tp = total_fp = total_fn = 0.0
    macro_p = macro_r = macro_f = 0.0

    for t in SCHEMA:
        stats = per_type_counts[t]
        tp = stats["TP"]
        fp = stats["FP"]
        fn = stats["FN"]
        support = stats["SUPPORT"]

        p, r, f1 = precision_recall_f1(tp, fp, fn)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        macro_p += p
        macro_r += r
        macro_f += f1

        correctly_caught = (tp / support * 100.0) if support > 0 else 0.0

        rows_for_csv.append(
            {
                "Type": t,
                "Precision": p,
                "Recall": r,
                "F1": f1,
                "Support": support,
                "Correctly Caught (%)": correctly_caught,
            }
        )

    # Macro = unweighted average over types
    macro_p /= float(len(SCHEMA))
    macro_r /= float(len(SCHEMA))
    macro_f /= float(len(SCHEMA))

    # Micro = pool all TP/FP/FN
    micro_p, micro_r, micro_f = precision_recall_f1(int(total_tp), int(total_fp), int(total_fn))

    metrics_summary = {
        "rows": rows_for_csv,
        "macro": (macro_p, macro_r, macro_f),
        "micro": (micro_p, micro_r, micro_f),
        "per_example": per_example_rows,
        "confusions": confusion_counts,
    }
    return metrics_summary


def save_main_metrics_csv(out_csv: Path, metrics_summary) -> None:
    """Write the original-style metrics CSV (kept backward compatible)."""
    rows = metrics_summary["rows"]
    macro_p, macro_r, macro_f = metrics_summary["macro"]
    micro_p, micro_r, micro_f = metrics_summary["micro"]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Type", "Precision", "Recall", "F1", "Support", "Correctly Caught (%)"])
        for r in rows:
            w.writerow(
                [
                    r["Type"],
                    f"{r['Precision']:.6f}",
                    f"{r['Recall']:.6f}",
                    f"{r['F1']:.6f}",
                    r["Support"],
                    f"{r['Correctly Caught (%)']:.2f}",
                ]
            )
        w.writerow([])
        w.writerow(["Macro", f"{macro_p:.6f}", f"{macro_r:.6f}", f"{macro_f:.6f}", "", ""])
        w.writerow(["Micro", f"{micro_p:.6f}", f"{micro_r:.6f}", f"{micro_f:.6f}", "", ""])


def save_per_example_csv(out_csv: Path, per_example_rows: List[dict]) -> None:
    """One row per (ID, Type) with gold/pred spans and FP/FN details."""
    per_example_path = out_csv.with_name(out_csv.stem + "_per_example.csv")
    per_example_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "ID",
        "Type",
        "Gold_spans",
        "Pred_spans",
        "TP",
        "FP",
        "FN",
        "Missing_spans(FN)",
        "Spurious_spans(FP)",
    ]

    with per_example_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in per_example_rows:
            w.writerow(row)


def save_confusions_csv(out_csv: Path, confusion_counts: Dict[Tuple[str, str], int]) -> None:
    """Summarize cross-type confusions: when the same span got a different label."""
    conf_path = out_csv.with_name(out_csv.stem + "_confusions.csv")
    conf_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["Gold_Type", "Pred_Type", "Span_Confusion_Count"]

    with conf_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(fieldnames)
        # Sort for readability
        for (gt, pt), cnt in sorted(confusion_counts.items(), key=lambda x: (-x[1], x[0][0], x[0][1])):
            w.writerow([gt, pt, cnt])


def main():
    ap = argparse.ArgumentParser(description="Strict span-level evaluation for Bangla medical NER.")
    ap.add_argument("--gold", required=True, help="Gold JSONL file with fields: ID, Input, Output")
    ap.add_argument("--pred", required=True, help="Prediction JSONL file with fields: ID, Input, Pred")
    ap.add_argument("--out_csv", required=True, help="Path to the main metrics CSV")

    args = ap.parse_args()

    gold_rows = load_jsonl(args.gold)
    pred_rows = load_jsonl(args.pred)

    print(f"Loaded {len(gold_rows)} gold rows from {args.gold}")
    print(f"Loaded {len(pred_rows)} pred rows from {args.pred}")

    metrics_summary = evaluate(gold_rows, pred_rows)

    out_csv_path = Path(args.out_csv)
    save_main_metrics_csv(out_csv_path, metrics_summary)
    save_per_example_csv(out_csv_path, metrics_summary["per_example"])
    save_confusions_csv(out_csv_path, metrics_summary["confusions"])

    print(f"\nSaved main metrics CSV   -> {out_csv_path}")
    print(f"Saved per-example CSV    -> {out_csv_path.with_name(out_csv_path.stem + '_per_example.csv')}")
    print(f"Saved confusion CSV      -> {out_csv_path.with_name(out_csv_path.stem + '_confusions.csv')}")


if __name__ == "__main__":
    main()
