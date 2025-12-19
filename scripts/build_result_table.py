#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def build_table(experiment: str, results_dir: Path) -> Path:
    paths = sorted(results_dir.glob(f"metrics_{experiment}_*.csv"))
    if not paths:
        raise SystemExit(f"No metrics files found for experiment: {experiment}")

    models = []
    model_metrics = []
    labels = None

    for path in paths:
        with path.open(newline="") as f:
            rows = list(csv.reader(f))

        model_row = next((r for r in rows if r and r[0] == "Model"), None)
        if not model_row:
            continue
        model = model_row[1]

        for i, r in enumerate(rows):
            if r and r[0] == "Type":
                header = r
                start = i + 1
                break
        else:
            continue

        metric_cols = ["Precision", "Recall", "F1"]
        idx = [header.index(m) for m in metric_cols]

        table = {}
        label_list = []
        for r in rows[start:]:
            if not r or not r[0]:
                continue
            while len(r) < len(header):
                r.append("")
            label = r[0]
            label_list.append(label)
            table[label] = {m: r[i] for m, i in zip(metric_cols, idx)}

        if labels is None:
            labels = label_list

        models.append(model)
        model_metrics.append(table)

    if not models:
        raise SystemExit("No valid metrics files with a Model row found.")

    out_path = results_dir / f"metrics_{experiment}_model_table.csv"
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "Score"] + labels)
        for model, table in zip(models, model_metrics):
            for score in ["Precision", "Recall", "F1"]:
                row = [model, score]
                for label in labels:
                    row.append(table.get(label, {}).get(score, ""))
                w.writerow(row)

    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a per-experiment metrics table across models."
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment name, e.g., en_zshot_en",
    )
    parser.add_argument(
        "--results_dir",
        default="results",
        help="Directory with metrics_*.csv files (default: results)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = build_table(args.experiment, Path(args.results_dir))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
