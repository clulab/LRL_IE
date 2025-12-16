from typing import Dict, List, Set, Tuple


def norm_span(s: str) -> str:
    return " ".join(str(s).split()).strip()


def to_type_sets(obj: dict, schema: List[str]) -> Dict[str, Set[str]]:
    out = {k: set() for k in schema}
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
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def evaluate(gold_rows: List[dict], pred_rows: List[dict], schema: List[str]):
    gold_by_id = {r["ID"]: r for r in gold_rows}
    pred_by_id = {r["ID"]: r for r in pred_rows}
    common = sorted(set(gold_by_id) & set(pred_by_id))

    per_type = {k: {"TP": 0, "FP": 0, "FN": 0, "SUPPORT": 0} for k in schema}

    for _id in common:
        g = to_type_sets(gold_by_id[_id].get("Output", {}), schema)
        p = to_type_sets(pred_by_id[_id].get("Pred", {}), schema)
        for k in schema:
            G = g[k]
            P = p[k]
            tp = len(G & P)
            fp = len(P - G)
            fn = len(G - P)
            per_type[k]["TP"] += tp
            per_type[k]["FP"] += fp
            per_type[k]["FN"] += fn
            per_type[k]["SUPPORT"] += len(G)

    rows = []
    micro_tp = micro_fp = micro_fn = 0
    for k in schema:
        TP = per_type[k]["TP"]
        FP = per_type[k]["FP"]
        FN = per_type[k]["FN"]
        S = per_type[k]["SUPPORT"]
        P, R, F = prf(TP, FP, FN)
        micro_tp += TP
        micro_fp += FP
        micro_fn += FN
        caught = (TP / S * 100.0) if S else 0.0
        rows.append({"Type": k, "Precision": P, "Recall": R, "F1": F, "Support": S, "Correctly Caught (%)": caught})

    macro_p = sum(r["Precision"] for r in rows) / len(rows) if rows else 0.0
    macro_r = sum(r["Recall"] for r in rows) / len(rows) if rows else 0.0
    macro_f = sum(r["F1"] for r in rows) / len(rows) if rows else 0.0

    micro_p, micro_r, micro_f = prf(micro_tp, micro_fp, micro_fn)

    return rows, (macro_p, macro_r, macro_f), (micro_p, micro_r, micro_f)

