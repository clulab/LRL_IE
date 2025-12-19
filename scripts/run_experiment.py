import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm
from llama_cpp import Llama

# Ensure repo root is on sys.path when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from lrl_ie.config import load_experiment_config
from lrl_ie.io import load_jsonl, save_jsonl, load_yaml
from lrl_ie.json_utils import decode_to_json
from lrl_ie.prompting import build_messages
from lrl_ie.qwen import chatml_prompt, strip_think
from lrl_ie.paths import resolve_pred_out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, help="Experiment name defined in experiments.yaml")
    ap.add_argument("--model", default="llama", help="Model block to use from configs/config.yaml (e.g., llama, qwen)")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--experiments", default="configs/experiments.yaml")
    ap.add_argument("--model_path", default=None, help="Override model_path from config (absolute or relative)")
    ap.add_argument("--debug_samples", type=int, default=0, help="If >0, run on a random subset of this many examples")
    args = ap.parse_args()

    cfg = load_experiment_config(args.config, args.experiments, args.experiment, args.model)
    model_family = cfg.get("model", "llama")

    if args.model_path:
        cfg["model_path"] = str(Path(args.model_path).expanduser())

    cfg["out"] = str(resolve_pred_out_path(str(cfg["out"]), args.experiment, model_family))
    raw_out_template = str(cfg.get("raw_out", "")).strip()
    if not raw_out_template:
        raw_out_template = str(cfg["out"]).replace("preds_", "raw_")
    cfg["raw_out"] = str(resolve_pred_out_path(raw_out_template, args.experiment, model_family))

    print(f"Running experiment: {args.experiment}")
    print(f"Loading GGUF: {cfg['model_path']}")
    llm_kwargs = dict(
        model_path=cfg["model_path"],
        n_ctx=cfg["ctx"],
        n_gpu_layers=cfg["n_gpu_layers"],
        n_threads=cfg["n_threads"],
        seed=cfg["seed"],
        verbose=False,
    )
    if model_family == "llama":
        llm_kwargs["chat_format"] = "llama-3"  # LLaMA-3 / 3.1 instruct
    if model_family == "aya":
        llm_kwargs["chat_format"] = "chatml"
    llm = Llama(**llm_kwargs)

    prompt_data = load_yaml(cfg["prompt"])

    fewshots = []
    if "examples" in cfg:
        with open(cfg["examples"], encoding="utf-8") as json_file:
            if cfg["examples"].endswith('.jsonl'):
                fewshots = [json.loads(ex) for ex in json_file]
            else:
                examples = json.load(json_file)
                for ex in examples:
                    shot = "".join([ex[field] for field in cfg["example_fields"]])
                    fewshots.append(shot)

    records = list(load_jsonl(cfg["input"]))
    if args.debug_samples and args.debug_samples > 0:
        import random
        random.seed(cfg.get("seed", 0))
        records = random.sample(records, k=min(len(records), args.debug_samples))
        print(f"DEBUG: running on {len(records)} sampled examples")
    out_rows = []
    raw_rows = []
    for r in tqdm(records, desc="Generating"):
        messages = build_messages(llm, r["Input"], cfg["ctx"], cfg["max_tokens"], prompt_data, fewshots)
        if model_family == "qwen":
            system_text = messages[0]["content"]
            user_text = messages[1]["content"]
            no_think = cfg.get("no_think", True)
            prompt = chatml_prompt(system_text, user_text, no_think=no_think)
            out = llm.create_completion(
                prompt=prompt,
                temperature=cfg["temperature"],
                top_p=cfg["top_p"],
                max_tokens=cfg["max_tokens"],
                stop=cfg.get("stop", ["<|im_end|>"]),
            )
            text_raw = out["choices"][0].get("text", "")
        else:
            out = llm.create_chat_completion(
                messages=messages,
                temperature=cfg["temperature"],
                top_p=cfg["top_p"],
                max_tokens=cfg["max_tokens"],
            )
            text_raw = out["choices"][0]["message"]["content"]

        text = strip_think(text_raw)
        pred = decode_to_json(text, prompt_data['schema'])
        out_rows.append({"ID": r["ID"], "Input": r["Input"], "Pred": pred})
        raw_rows.append({"ID": r["ID"], "Input": r["Input"], "Raw": text_raw})

    save_jsonl(cfg["out"], out_rows)
    print(f"Wrote {len(out_rows)} predictions -> {cfg['out']}")
    save_jsonl(cfg["raw_out"], raw_rows)
    print(f"Wrote {len(raw_rows)} raw outputs -> {cfg['raw_out']}")

if __name__ == "__main__":
    main()
