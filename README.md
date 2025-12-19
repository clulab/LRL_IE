# lrl_ie — bangla/basque ner with llama.cpp

## layout
- `configs/config.yaml`: model defaults, separated blocks for `llama`, `qwen`, and `aya`.
- `configs/experiments.yaml`: per-experiment overrides (input/prompt/examples/out/raw_out/results_out).
- `data/`: inputs, prompts (yaml), few-shot examples (json or jsonl).
- `scripts/run_experiment.py`: run inference (supports llama, qwen, or aya).
- `scripts/evaluate_llm_strict.py`: strict span-level evaluation (micro/macro, per-type).
- `lrl_ie/`: shared helpers (io, config merge, prompting, json parsing, eval, qwen chatml).

## setup
```bash
python -m venv .venv
source .venv/bin/activate  # windows: .venv\scripts\activate
pip install -r requirements.txt  # llama-cpp-python with your backend (cpu/metal/cuda)
```
download gguf models somewhere and update the model paths in `configs/config.yaml`, or override at runtime with `--model_path /absolute/path/to/model.gguf`.

## run inference (config-driven)
pick an experiment from `configs/experiments.yaml` and a model block from `configs/config.yaml`.

### experiment types
- `bn_qa_en`: Bangla medical ner; qa-style wrapper in English (`Question/Text/Answer`); uses `data/prompts/bn_qa_en.yaml` + `data/examples/bn_en_10.jsonl`.
- `bn_qa_bn`: same task/data, but qa-style wrapper in Bangla; uses `data/prompts/bn_qa_bn.yaml` + `data/examples/bn_bn_10.jsonl`.
- `bn_plain_en`: same task/data, but instruction-style (no explicit question); uses `data/prompts/bn_plain_en.yaml` + `data/examples/bn_en_10.jsonl`.
- `eu_zshot_en`: Basque clinical ner; plain-style wrapper in English (`Text/Answer`); uses `data/prompts/eu_zshot_en.yaml`.
- `eu_zshot_eu`: same task/data; plain-style wrapper in Basque (`Text/Answer`); uses `data/prompts/eu_zshot_eu.yaml`.
- `eu_plain_en`: same task/data; plain-style wrapper in English (`Text/Answer`); uses `data/prompts/eu_plain_en.yaml` + `data/examples/eu_en_10.jsonl`.
- `eu_plain_eu`: same task/data; plain-style wrapper in Basque (`Text/Answer`); uses `data/prompts/eu_plain_eu.yaml` + `data/examples/eu_eu_10.jsonl`.
- `eu_tr_en`: same task/data; plain-style wrapper in English (`Translation/Text/Answer`); uses `data/prompts/eu_plain_en.yaml` + `data/examples/eu_en_10.jsonl`.
- `eu_tr_eu`: same task/data; plain-style wrapper in Basque (`Translation/Text/Answer`); uses `data/prompts/eu_plain_eu.yaml` + `data/examples/eu_eu_10.jsonl`.
- `eu_qa_en`: same task/data; qa-style wrapper in English (`Question/Text/Answer`); uses `data/prompts/eu_qa_en.yaml` + `data/examples/eu_en_10.jsonl`.
- `eu_qa_eu`: same task/data, but qa-style wrapper in Basque (`Question/Text/Answer`); uses `data/prompts/eu_qa_eu.yaml` + `data/examples/eu_eu_10.jsonl`.
- `es_zshot_en`: Spanish clinical ner; plain-style wrapper in English (`Text/Answer`); uses `data/prompts/es_zshot_en.yaml`.
- `es_zshot_es`: same task/data; plain-style wrapper in Spanish (`Text/Answer`); uses `data/prompts/es_zshot_es.yaml`.
- `es_plain_en`: same task/data; plain-style wrapper in English (`Text/Answer`); uses `data/prompts/es_plain_en.yaml` + `data/examples/es_en_10.jsonl`.
- `es_plain_es`: same task/data; plain-style wrapper in Spanish (`Text/Answer`); uses `data/prompts/es_plain_es.yaml` + `data/examples/es_es_10.jsonl`.
- `es_tr_en`: same task/data; plain-style wrapper in English (`Translation/Text/Answer`); uses `data/prompts/es_plain_en.yaml` + `data/examples/es_en_10.jsonl`.
- `es_tr_es`: same task/data; plain-style wrapper in Spanish (`Translation/Text/Answer`); uses `data/prompts/es_plain_es.yaml` + `data/examples/es_es_10.jsonl`.
- `es_qa_en`: Spanish clinical ner; qa-style wrapper in English (`Question/Text/Answer`); uses `data/prompts/es_qa_en.yaml` + `data/examples/es_en_10.jsonl`.
- `es_qa_es`: same task/data; but qa-style wrapper in Spanish (`Question/Text/Answer`); uses `data/prompts/es_qa_es.yaml` + `data/examples/es_es_10.jsonl`.
- `en_zshot_en`: English clinical ner; plain-style wrapper in English (`Text/Answer`); uses `data/prompts/en_zshot_en.yaml`.
- `en_plain_en`: same task/data; plain-style wrapper in English (`Text/Answer`); uses `data/prompts/en_plain_en.yaml` + `data/examples/en_en_10.jsonl`.
- `en_tr_en`: same task/data; plain-style wrapper in English (`Translation/Text/Answer`); uses `data/prompts/en_plain_en.yaml` + `data/examples/en_en_10.jsonl`.
- `en_qa_en`: same task/data; but qa-style wrapper in English (`Question/Text/Answer`); uses `data/prompts/en_qa_en.yaml` + `data/examples/en_en_10.jsonl`.

### datasets and labels

**Bangla (bn) medical ner dataset** (`data/input/bn.jsonl`):
- `Age`: patient age mentions
- `Symptom`: medical symptoms
- `Medicine`: medication names
- `Health_Condition`: medical conditions/diseases
- `Specialist`: medical specialist types
- `Medical_Procedure`: medical procedures/tests

**Basque (eu) clinical ner dataset** (`data/input/eu.jsonl`):
- `Disorder`: pathologic process with a set of signs and symptoms
- `Patient`: patient mentions
- `H-Professional`: healthcare professionals

### model types
- `llama`: uses `create_chat_completion` with `chat_format="llama-3"` (llama-3/3.1 instruct ggufs).
- `qwen`: uses `create_completion` with a manual ChatML prompt; supports `/no_think`, stop tokens, and (when the model emits a reasoning block) stripping everything up to `</think>` (see `configs/config.yaml`).
- `aya`: uses `create_chat_completion` with `chat_format="chatml"` (ChatML-style instruct ggufs).

```bash
# llama (default)
python scripts/run_experiment.py --experiment bn_qa_bn --model llama

# qwen (chatml, /no_think, </think>-based stripping)
python scripts/run_experiment.py --experiment bn_qa_en --model qwen

# aya (chatml via chat_format="chatml")
python scripts/run_experiment.py --experiment bn_qa_en --model aya

# debug on a random subset (e.g., 100 examples)
python scripts/run_experiment.py --experiment bn_plain_en --model llama --debug_samples 100

python scripts/run_experiment.py --experiment bn_qa_bn --model qwen --debug_samples 100

python scripts/run_experiment.py --experiment bn_qa_en --model aya --debug_samples 100
```
the script merges model defaults with the experiment block, loads prompt/examples/input, and writes predictions to the `out` path defined in the experiment (templated like `data/out/preds_{experiment}_{model}.jsonl`).

## evaluate
```bash
# llama run
python scripts/evaluate_llm_strict.py --experiment bn_plain_en --model llama

# qwen run
python scripts/evaluate_llm_strict.py --experiment bn_qa_bn --model qwen

# aya run
python scripts/evaluate_llm_strict.py --experiment bn_qa_en --model aya
```
outputs per-type precision/recall/f1/support and macro/micro averages to console, and writes a csv under the experiment’s `results_out` directory.
csv path defaults to `results/metrics_{experiment}_{model}.csv` (and the parent directory is created automatically).
if you used `--debug_samples`, metrics reflect only that sampled subset (id intersection of gold/pred).
