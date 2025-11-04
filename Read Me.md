bangla_qwen_llamacpp/
│
├── llama.cpp/                ← llama.cpp repo
├── models/                   ← Qwen GGUF model goes here
│   └── qwen-1.5b-chat.gguf
|   └── qwen3_bner_lora_f16.gguf
├── data/
│   └── dev.json              ← Downloaded HF dataset
├── prompts/
│   └── ner_prompt.txt        ← Prompt format
├── main.py                   ← Runs prediction
├── utils.py                  ← Helper functions
├── requirements.txt          ← Python dependencies
└── evaluate.py               ← (Optional) Evaluation script

============================================================
Step 01: requirements.txt (inside)
datasets
numpy
scikit-learn
                               CMAKE_ARGS="-DGGML_METAL=ON" pip install --force-reinstall --no-cache-dir "llama-cpp-python[metal]"
                               pip install llama-cpp-python
                               pip install -r requirements.txt

============================================================
Step 02: Download Qwen GGUF Model
Use a GGUF-compatible model like:
🔗 https://huggingface.co/TheBloke/Qwen1.5-1.8B-Chat-GGUF

Download qwen1_5-1.8b-chat.Q4_K_M.gguf
Put it in: models/

============================================================
Step 03: Download the Dataset from Hugging Face
dev.json

============================================================
Step 04: Re-clone & Build with CMake
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir build
cd build
cmake -DLLAMA_METAL=on ..
cmake --build . --config Release

============================================================
Step 05: Run this code into terminal
cd /Users/ayeshakhatun/Downloads/Projects/bangla_qwen_llamacpp

                                                        # 1) (optional) new venv
                                                        python -m venv .venv
                                                        source .venv/bin/activate   
# Windows: .venv\Scripts\activate

# 2) install
pip install -r requirements.txt
# (If you need GPU offload, reinstall llama-cpp-python with your backend enabled.)

# 3) generate predictions with llama.cpp


=====================. Experinment 04: Question English + Bangla (llama3-8B) ========

python scripts/main_llama3_8b_ex04.py \
  --model_path models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  --input data/data_llm_io.jsonl \
  --out data/preds_llama31_8b_ex04.jsonl \
  --n_gpu_layers -1 \
  --ctx 4096 \
  --temperature 0.0 \
  --max_tokens 256


python evaluate_llm_strict.py \
  --gold data/data_llm_io.jsonl \
  --pred data/preds_llama31_8b_ex04.jsonl \
  --out_csv results/metrics_llama31_8b_ex04.csv


