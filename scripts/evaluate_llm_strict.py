import argparse
import csv
import sys
from pathlib import Path

# Ensure repo root on path when run as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from lrl_ie.eval import evaluate
from lrl_ie.config import load_experiment_config
from lrl_ie.io import load_jsonl, load_yaml
from lrl_ie.paths import resolve_pred_out_path, resolve_templated_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, help="Experiment name defined in experiments.yaml")
    ap.add_argument("--model", required=True, help="Model block name in configs/config.yaml (e.g., llama, qwen)")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--experiments", default="configs/experiments.yaml")
    args = ap.parse_args()

    cfg = load_experiment_config(args.config, args.experiments, args.experiment, args.model)
    model_name = cfg.get("model", args.model)

    gold_path = cfg["input"]
    pred_path = resolve_pred_out_path(str(cfg["out"]), args.experiment, model_name)
    prompt_path = cfg["prompt"]
    results_path = resolve_templated_path(str(cfg["results_out"]), args.experiment, model_name)

    gold_rows = load_jsonl(gold_path)
    pred_rows = load_jsonl(pred_path)
    gold_ids = {r.get("ID") for r in gold_rows if "ID" in r}
    pred_ids = {r.get("ID") for r in pred_rows if "ID" in r}
    common_ids = gold_ids & pred_ids

    prompt_data = load_yaml(prompt_path)
    schema = prompt_data["schema"]

    rows, macro, micro = evaluate(gold_rows, pred_rows, schema)
    macro_p, macro_r, macro_f = macro
    micro_p, micro_r, micro_f = micro

    print(f"Experiment: {args.experiment}")
    print(f"Model: {model_name}")
    print(f"Gold: {gold_path}")
    print(f"Pred: {pred_path}")
    print(f"# of Samples: {len(common_ids)}")

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
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Experiment", args.experiment, "", "", "", ""])
        w.writerow(["Model", model_name, "", "", "", ""])
        w.writerow(["Gold", gold_path, "", "", "", ""])
        w.writerow(["Pred", pred_path, "", "", "", ""])
        w.writerow(["# of Samples", len(common_ids), "", "", "", ""])
        w.writerow([])
        w.writerow(["Type","Precision","Recall","F1","Support","Correctly Caught (%)"])
        for r in rows:
            w.writerow([r["Type"], f"{r['Precision']:.6f}", f"{r['Recall']:.6f}", f"{r['F1']:.6f}", r["Support"], f"{r['Correctly Caught (%)']:.2f}"])
        w.writerow([])
        w.writerow(["Macro", f"{macro_p:.6f}", f"{macro_r:.6f}", f"{macro_f:.6f}", "", ""])
        w.writerow(["Micro", f"{micro_p:.6f}", f"{micro_r:.6f}", f"{micro_f:.6f}", "", ""])
    print(f"\nSaved CSV -> {results_path}")

if __name__ == "__main__":
    main()
