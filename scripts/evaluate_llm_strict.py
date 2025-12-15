# entity_llm.py
import argparse
import csv
import json
import yaml
from pathlib import Path
from typing import Dict, Set, List


def load_jsonl(path: str) -> List[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows

def norm_span(s: str) -> str:
    return " ".join(s.split()).strip()

def to_type_sets(obj: dict, schema: List[str]) -> Dict[str, Set[str]]:
    out = {k:set() for k in schema}
    for k in schema:
        vals = obj.get(k, [])
        if isinstance(vals, list):
            for v in vals:
                if isinstance(v, str):
                    vv = norm_span(v)
                    if vv:
                        out[k].add(vv)
    return out

def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2*p*r/(p+r) if (p+r) else 0.0
    return p, r, f

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/input/data_llm_io.jsonl")
    ap.add_argument("--pred", default="data/out/preds_llamacpp.jsonl")
    ap.add_argument("--out_csv", default="results/entity_metrics.csv")
    ap.add_argument("--prompt", default="data/prompts/prompt.jsonl")
    args = ap.parse_args()

    gold_rows = load_jsonl(args.gold)
    pred_rows = load_jsonl(args.pred)

    with open(args.prompt) as yml_file:
        prompt_data = yaml.safe_load(yml_file)
        schema = prompt_data['schema']

    gold_by_id = {r["ID"]: r for r in gold_rows}
    pred_by_id = {r["ID"]: r for r in pred_rows}
    common = sorted(set(gold_by_id) & set(pred_by_id))

    per_type = {k: {"TP":0,"FP":0,"FN":0,"SUPPORT":0} for k in schema}

    for _id in common:
        g = to_type_sets(gold_by_id[_id].get("Output", {}), schema)
        p = to_type_sets(pred_by_id[_id].get("Pred", {}), schema)
        for k in schema:
            G = g[k]; P = p[k]
            tp = len(G & P)
            fp = len(P - G)
            fn = len(G - P)
            per_type[k]["TP"] += tp
            per_type[k]["FP"] += fp
            per_type[k]["FN"] += fn
            per_type[k]["SUPPORT"] += len(G)

    # collect rows and micro/macro
    rows = []
    micro_tp = micro_fp = micro_fn = 0
    for k in schema:
        TP = per_type[k]["TP"]; FP = per_type[k]["FP"]; FN = per_type[k]["FN"]; S = per_type[k]["SUPPORT"]
        P,R,F = prf(TP,FP,FN)
        micro_tp += TP; micro_fp += FP; micro_fn += FN
        caught = (TP / S * 100.0) if S else 0.0
        rows.append({"Type":k,"Precision":P,"Recall":R,"F1":F,"Support":S,"Correctly Caught (%)":caught})

    macro_p = sum(r["Precision"] for r in rows)/len(rows) if rows else 0.0
    macro_r = sum(r["Recall"] for r in rows)/len(rows) if rows else 0.0
    macro_f = sum(r["F1"] for r in rows)/len(rows) if rows else 0.0

    micro_p, micro_r, micro_f = prf(micro_tp, micro_fp, micro_fn)

    # pretty print
    def fmt(x): return f"{x:.4f}"
    print("\n=== Entity-level (set exact-match) — Per-type Metrics ===")
    print(f"{'Type':20s} {'Precision':>10s} {'Recall':>8s} {'F1':>8s} {'Support':>9s} {'Correctly Caught (%)':>22s}")
    for r in rows:
        print(f"{r['Type']:20s} {fmt(r['Precision']):>10s} {fmt(r['Recall']):>8s} {fmt(r['F1']):>8s} {r['Support']:>9d} {fmt(r['Correctly Caught (%)']):>22s}")

    print("\n=== Macro-averaged ===")
    print(f"Precision={fmt(macro_p)}  Recall={fmt(macro_r)}  F1={fmt(macro_f)}")

    print("\n=== Micro-averaged ===")
    print(f"Precision={fmt(micro_p)}  Recall={fmt(micro_r)}  F1={fmt(micro_f)}")

    # save csv
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out_csv).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Type","Precision","Recall","F1","Support","Correctly Caught (%)"])
        for r in rows:
            w.writerow([r["Type"], f"{r['Precision']:.6f}", f"{r['Recall']:.6f}", f"{r['F1']:.6f}", r["Support"], f"{r['Correctly Caught (%)']:.2f}"])
        w.writerow([])
        w.writerow(["Macro", f"{macro_p:.6f}", f"{macro_r:.6f}", f"{macro_f:.6f}", "", ""])
        w.writerow(["Micro", f"{micro_p:.6f}", f"{micro_r:.6f}", f"{micro_f:.6f}", "", ""])
    print(f"\nSaved CSV -> {args.out_csv}")

if __name__ == "__main__":
    main()
