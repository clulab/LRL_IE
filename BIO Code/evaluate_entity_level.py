#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from typing import List, Dict, Any, Tuple
from seqeval.metrics import precision_score, recall_score, f1_score
from seqeval.metrics.sequence_labeling import get_entities
from seqeval.scheme import IOB2


def load_predictions(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def align_lengths(true_seq: List[str], pred_seq: List[str]) -> Tuple[List[str], List[str]]:
    """Pad/truncate prediction to match the true sequence length (pads with 'O')."""
    if len(pred_seq) < len(true_seq):
        pred_seq = pred_seq + ["O"] * (len(true_seq) - len(pred_seq))
    elif len(pred_seq) > len(true_seq):
        pred_seq = pred_seq[:len(true_seq)]
    return true_seq, pred_seq


def token_accuracy(y_true: List[List[str]], y_pred: List[List[str]]) -> float:
    correct = total = 0
    for t_seq, p_seq in zip(y_true, y_pred):
        for t, p in zip(t_seq, p_seq):
            total += 1
            if t == p:
                correct += 1
    return correct / total if total > 0 else 0.0


def collect_entity_sets(y_true: List[List[str]], y_pred: List[List[str]]) -> Dict[str, Dict[str, set]]:
    per_type: Dict[str, Dict[str, set]] = {}
    for i, (t_seq, p_seq) in enumerate(zip(y_true, y_pred)):
        t_ents = get_entities(t_seq)
        p_ents = get_entities(p_seq)
        for t, s, e in t_ents:
            per_type.setdefault(t, {"true": set(), "pred": set()})
            per_type[t]["true"].add((i, s, e))
        for t, s, e in p_ents:
            per_type.setdefault(t, {"true": set(), "pred": set()})
            per_type[t]["pred"].add((i, s, e))
    return per_type


def prf_from_sets(true_set: set, pred_set: set) -> Tuple[float, float, float, int]:
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    support = len(true_set)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1, support


def format_row(name: str, prec: float, rec: float, f1: float, support: int) -> str:
    # Recall × 100 = percentage of entities correctly caught
    recall_pct = rec * 100
    return f"{name:<20}  {prec:7.4f}  {rec:7.4f}  {f1:7.4f}  {support:7d}    {recall_pct:6.2f}%"


def main(pred_path: str, out_path: str):
    data = load_predictions(pred_path)

    # Align sentence-level sequences
    y_true: List[List[str]] = []
    y_pred: List[List[str]] = []
    for item in data:
        t, p = align_lengths(item.get("true_labels", []), item.get("predicted_labels_flat", []))
        y_true.append(t)
        y_pred.append(p)

    # Overall micro scores
    micro_p = precision_score(y_true, y_pred, mode="strict", scheme=IOB2)
    micro_r = recall_score(y_true, y_pred, mode="strict", scheme=IOB2)
    micro_f = f1_score(y_true, y_pred, mode="strict", scheme=IOB2)

    # Per-type metrics
    per_type = collect_entity_sets(y_true, y_pred)

    header = (
        "\n=== Entity-level (seqeval, strict IOB2) — Per-type Metrics ===\n"
        "Type                 Precision  Recall    F1       Support   Correctly Caught (%)"
    )
    rows = []
    for t in sorted(per_type.keys()):
        stats = per_type[t]
        prec, rec, f1, sup = prf_from_sets(stats["true"], stats["pred"])
        rows.append(format_row(t, prec, rec, f1, sup))

    # Token-level accuracy for context
    acc = token_accuracy(y_true, y_pred)

    # Final report
    text_lines = [
        header,
        *rows,
        "",
        f"overall (micro)     P={micro_p:.4f}  R={micro_r:.4f}  F1={micro_f:.4f}",
        f"(token-level accuracy, context only) accuracy={acc:.4f}",
    ]
    report = "\n".join(text_lines)

    print(report)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entity-level NER evaluation with recall% shown per type.")
    parser.add_argument(
        "--pred", type=str, default="data/aya8b.json",
        #"--pred", type=str, default="data/q14B_test_v1_fixed_v2.json",
        #q8B_test_v1_fixed
        help="Path to predictions JSON (fields: true_labels, predicted_labels_flat)"
    )
    parser.add_argument(
        "--out", type=str, default="metrics_entity_aya8b.txt",
        help="Where to save the report"
    )
    args = parser.parse_args()
    main(args.pred, args.out)
