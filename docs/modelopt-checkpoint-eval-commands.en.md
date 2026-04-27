# Bash Command Examples for Evaluating ModelOpt Quantized Checkpoints

This cookbook shows different ways to evaluate ModelOpt-quantized checkpoints
with `lm-eval`. Commands are written assuming they are run from the
`lm-eval-example` repository root.

## Version Validity

These commands were written against the local environment used to create this
example repository:

```bash
# lm-evaluation-harness
# package version: 0.4.12.dev0
# git describe: v0.4.11-24-g620262e0
# commit: 620262e0
# note date: 2026-04-27
```

Quantization arguments may change across vLLM, SGLang, and ModelOpt versions.
Record versions in CI:

```bash
python -m pip freeze | grep -E 'lm_eval|vllm|sglang|nvidia-modelopt|torch|transformers'
```

If a command fails, reduce to `--limit 10` first, then verify the installed
backend's supported `quantization=` names.

## 0. Common Variables

```bash
export LMEVAL_REPO=/localhome/swqa/workspace/aaa/lm-evaluation-harness
export EXAMPLE_REPO=/localhome/swqa/workspace/aaa/lm-eval-example
export MODEL_ROOT=/path/to/modelopt/checkpoints

export TASKS=arc_challenge,hellaswag,piqa,winogrande
export LIMIT=100
export NUM_FEWSHOT=0
```

Install the HF backend:

```bash
cd "${LMEVAL_REPO}"
pip install -e ".[hf]"
```

Install the vLLM backend:

```bash
cd "${LMEVAL_REPO}"
pip install -e ".[vllm]"
```

For SGLang, install SGLang using its official instructions, then install the
harness:

```bash
cd "${LMEVAL_REPO}"
pip install -e .
```

## 1. Validate Tasks

```bash
cd "${EXAMPLE_REPO}"
TASKS="${TASKS}" ./scripts/validate_tasks.sh
```

List available tasks:

```bash
lm-eval ls tasks
lm-eval ls groups
lm-eval ls subtasks
```

## 2. HF Backend: First Compatibility Check

Use this as the reference/correctness path.

```bash
cd "${EXAMPLE_REPO}"

MODEL_DIR="${MODEL_ROOT}/fp8" \
TASKS="${TASKS}" \
LIMIT="${LIMIT}" \
NUM_FEWSHOT="${NUM_FEWSHOT}" \
BATCH_SIZE=8 \
OUTPUT_DIR=artifacts/eval/hf-fp8 \
./scripts/run_hf_modelopt.sh
```

NVFP4 checkpoint:

```bash
MODEL_DIR="${MODEL_ROOT}/nvfp4" \
TASKS="${TASKS}" \
LIMIT="${LIMIT}" \
BATCH_SIZE=8 \
OUTPUT_DIR=artifacts/eval/hf-nvfp4 \
./scripts/run_hf_modelopt.sh
```

Note: HF `DTYPE=auto/bfloat16/float16` is not the ModelOpt `fp8/nvfp4`
quantization format. Quantization should come from the checkpoint config or a
custom loader.

## 3. vLLM Backend: ModelOpt FP8

`lm-eval --model vllm` uses the vLLM offline engine. It does not start a server.

```bash
MODEL_DIR="${MODEL_ROOT}/fp8" \
VLLM_QUANTIZATION=modelopt \
TASKS="${TASKS}" \
LIMIT="${LIMIT}" \
BATCH_SIZE=auto \
TP_SIZE=1 \
GPU_MEMORY_UTILIZATION=0.85 \
MAX_MODEL_LEN=4096 \
OUTPUT_DIR=artifacts/eval/vllm-fp8 \
./scripts/run_vllm_modelopt.sh
```

If vLLM can auto-detect the checkpoint from `hf_quant_config.json`, leave the
quantization argument empty:

```bash
MODEL_DIR="${MODEL_ROOT}/fp8" \
VLLM_QUANTIZATION= \
OUTPUT_DIR=artifacts/eval/vllm-fp8-auto \
./scripts/run_vllm_modelopt.sh
```

## 4. vLLM Backend: ModelOpt NVFP4

```bash
MODEL_DIR="${MODEL_ROOT}/nvfp4" \
VLLM_QUANTIZATION=modelopt_fp4 \
TASKS="${TASKS}" \
LIMIT="${LIMIT}" \
BATCH_SIZE=auto \
TP_SIZE=1 \
GPU_MEMORY_UTILIZATION=0.85 \
MAX_MODEL_LEN=4096 \
OUTPUT_DIR=artifacts/eval/vllm-nvfp4 \
./scripts/run_vllm_modelopt.sh
```

Blackwell/NVFP4 paths are often hardware- and backend-version-sensitive. Start
with a small smoke run:

```bash
MODEL_DIR="${MODEL_ROOT}/nvfp4" \
VLLM_QUANTIZATION=modelopt_fp4 \
LIMIT=10 \
OUTPUT_DIR=artifacts/eval/vllm-nvfp4-smoke \
./scripts/run_vllm_modelopt.sh
```

## 5. vLLM Backend: Tensor Parallel

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL_DIR="${MODEL_ROOT}/fp8" \
VLLM_QUANTIZATION=modelopt \
TP_SIZE=4 \
GPU_MEMORY_UTILIZATION=0.85 \
MAX_MODEL_LEN=4096 \
BATCH_SIZE=auto \
OUTPUT_DIR=artifacts/eval/vllm-fp8-tp4 \
./scripts/run_vllm_modelopt.sh
```

For early CI, avoid `data_parallel_size>1` because it uses Ray actors and adds
more dependencies.

## 6. SGLang Backend: ModelOpt FP8

`lm-eval --model sglang` uses the SGLang offline engine. It does not start a
server.

```bash
MODEL_DIR="${MODEL_ROOT}/fp8" \
SGLANG_QUANTIZATION=modelopt_fp8 \
TASKS="${TASKS}" \
LIMIT="${LIMIT}" \
BATCH_SIZE=auto \
TP_SIZE=1 \
MEM_FRACTION_STATIC=0.70 \
MAX_MODEL_LEN=4096 \
OUTPUT_DIR=artifacts/eval/sglang-fp8 \
./scripts/run_sglang_modelopt.sh
```

If the installed SGLang version does not accept `modelopt_fp8`, try:

```bash
MODEL_DIR="${MODEL_ROOT}/fp8" \
SGLANG_QUANTIZATION=modelopt \
OUTPUT_DIR=artifacts/eval/sglang-fp8-modelopt \
./scripts/run_sglang_modelopt.sh
```

Or leave it empty and let the backend parse the checkpoint config:

```bash
MODEL_DIR="${MODEL_ROOT}/fp8" \
SGLANG_QUANTIZATION= \
OUTPUT_DIR=artifacts/eval/sglang-fp8-auto \
./scripts/run_sglang_modelopt.sh
```

## 7. SGLang Backend: ModelOpt NVFP4/FP4

```bash
MODEL_DIR="${MODEL_ROOT}/nvfp4" \
SGLANG_QUANTIZATION=modelopt_fp4 \
TASKS="${TASKS}" \
LIMIT="${LIMIT}" \
BATCH_SIZE=auto \
TP_SIZE=1 \
MEM_FRACTION_STATIC=0.70 \
MAX_MODEL_LEN=4096 \
OUTPUT_DIR=artifacts/eval/sglang-nvfp4 \
./scripts/run_sglang_modelopt.sh
```

If OOM happens, reduce:

```bash
MEM_FRACTION_STATIC=0.60
MAX_MODEL_LEN=2048
BATCH_SIZE=8
```

## 8. Fixed Samples for Stable CI

```bash
SAMPLES_JSON="$(cat config/sample_indices.json)" \
MODEL_DIR="${MODEL_ROOT}/fp8" \
TASKS="${TASKS}" \
LIMIT=none \
OUTPUT_DIR=artifacts/eval/hf-fp8-fixed \
./scripts/run_hf_modelopt.sh
```

`--samples` and `--limit` are mutually exclusive, so set `LIMIT=none`.

## 9. Run Multiple Quant Formats and Backends

Expected layout:

```text
/path/to/modelopt/checkpoints/
  bf16/
  fp8/
  nvfp4/
```

Run:

```bash
MODEL_ROOT="${MODEL_ROOT}" \
BACKENDS="hf vllm sglang" \
QUANT_FORMATS="bf16 fp8 nvfp4" \
TASKS="${TASKS}" \
LIMIT=100 \
NUM_FEWSHOT=0 \
OUTPUT_ROOT=artifacts/eval/matrix \
./scripts/run_modelopt_matrix.sh
```

vLLM only:

```bash
MODEL_ROOT="${MODEL_ROOT}" \
BACKENDS="vllm" \
QUANT_FORMATS="fp8 nvfp4" \
LIMIT=100 \
./scripts/run_modelopt_matrix.sh
```

## 10. Baseline vs Quantized Flow

Run the BF16 baseline:

```bash
MODEL_DIR="${MODEL_ROOT}/bf16" \
TASKS="${TASKS}" \
LIMIT=200 \
OUTPUT_DIR=artifacts/eval/baseline-bf16 \
./scripts/run_hf_modelopt.sh
```

Run the quantized checkpoint:

```bash
MODEL_DIR="${MODEL_ROOT}/fp8" \
TASKS="${TASKS}" \
LIMIT=200 \
OUTPUT_DIR=artifacts/eval/quant-fp8 \
./scripts/run_vllm_modelopt.sh
```

Find result files:

```bash
find artifacts/eval -name 'results_*.json' -print
```

Extract metrics:

```bash
jq '.results.arc_challenge["acc_norm,none"], .results.hellaswag["acc_norm,none"]' \
  "$(find artifacts/eval/quant-fp8 -name 'results_*.json' | sort | tail -1)"
```

## 11. CI Threshold Gate

```bash
RESULT_JSON="$(find artifacts/eval/quant-fp8 -name 'results_*.json' | sort | tail -1)"

python - <<'PY' "${RESULT_JSON}"
import json
import sys

path = sys.argv[1]
data = json.load(open(path))

thresholds = {
    ("arc_challenge", "acc_norm,none"): 0.40,
    ("hellaswag", "acc_norm,none"): 0.55,
    ("piqa", "acc_norm,none"): 0.70,
    ("winogrande", "acc,none"): 0.55,
}

failed = []
for (task, metric), threshold in thresholds.items():
    value = data["results"].get(task, {}).get(metric)
    if value is None or value < threshold:
        failed.append((task, metric, value, threshold))

if failed:
    for task, metric, value, threshold in failed:
        print(f"FAIL {task} {metric}: got={value} threshold={threshold}")
    sys.exit(1)

print("PASS")
PY
```

## 12. Python API for Custom ModelOpt Restore

If the checkpoint cannot be loaded directly by HF/vLLM/SGLang, replace the
loader inside the Python API example:

```bash
python python/simple_eval_modelopt_hf.py \
  --model-dir "${MODEL_ROOT}/custom-restore-checkpoint" \
  --tasks "${TASKS}" \
  --limit 100 \
  --num-fewshot 0 \
  --batch-size 8 \
  --trust-remote-code \
  --output-json artifacts/eval/python-api/results.json
```

The final objects must provide:

```text
model(input_ids).logits
model.generate(...)
tokenizer.encode/decode or a compatible HF tokenizer API
```

## 13. Generative Task Research

Generative tasks are slower and more prompt/chat-template-sensitive. Start with
small samples:

```bash
MODEL_DIR="${MODEL_ROOT}/fp8" \
TASKS=gsm8k_cot,ifeval \
LIMIT=20 \
BATCH_SIZE=auto \
EXTRA_MODEL_ARGS="max_gen_toks=512" \
OUTPUT_DIR=artifacts/eval/vllm-generative-smoke \
./scripts/run_vllm_modelopt.sh
```

For chat/instruct models, add chat template support:

```bash
lm-eval run \
  --model vllm \
  --model_args pretrained="${MODEL_ROOT}/fp8",dtype=auto,trust_remote_code=True,max_model_len=4096 \
  --tasks gsm8k_cot \
  --apply_chat_template \
  --batch_size auto \
  --limit 20 \
  --output_path artifacts/eval/chat-template-smoke
```

## 14. Debugging

Minimal debug run:

```bash
MODEL_DIR="${MODEL_ROOT}/fp8" \
TASKS=arc_challenge \
LIMIT=5 \
BATCH_SIZE=1 \
OUTPUT_DIR=artifacts/eval/debug \
./scripts/run_hf_modelopt.sh
```

OOM controls:

```bash
MAX_MODEL_LEN=2048
BATCH_SIZE=4
GPU_MEMORY_UTILIZATION=0.75
MEM_FRACTION_STATIC=0.60
```

Inspect prompt formatting:

```bash
lm-eval run \
  --model hf \
  --model_args pretrained="${MODEL_ROOT}/fp8",dtype=auto,trust_remote_code=True \
  --tasks arc_challenge \
  --limit 1 \
  --write_out
```

