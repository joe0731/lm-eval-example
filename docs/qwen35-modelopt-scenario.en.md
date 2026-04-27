# Qwen3.5 ModelOpt Multi-Format Checkpoint Scenario

This scenario assumes local Qwen3.5 quantized artifacts:

```text
/data/models/qwen3.5-modelopt/
  bf16/
  fp16/
  fp8/
  int8_awq/
  nvfp4/
```

The goal is to run consistent `lm-eval` quality regression across these
artifacts and compare HF, vLLM, and SGLang paths.

## Version Validity

The examples use these backend conventions:

- vLLM ModelOpt FP8: `quantization=modelopt`
- vLLM ModelOpt NVFP4: `quantization=modelopt_fp4`
- vLLM AWQ: `quantization=awq`
- SGLang ModelOpt FP8: `quantization=modelopt_fp8`
- SGLang ModelOpt FP4/NVFP4: `quantization=modelopt_fp4`
- SGLang AWQ: `quantization=awq`

If your installed backend rejects these values, inspect `hf_quant_config.json`
with `scripts/inspect_modelopt_checkpoint.sh` and adjust to the installed
backend documentation.

## 1. Common Variables

```bash
cd /localhome/swqa/workspace/aaa/lm-eval-example

export MODEL_ROOT=/data/models/qwen3.5-modelopt
export TASKS=arc_challenge,hellaswag,piqa,winogrande
export LIMIT=100
export NUM_FEWSHOT=0
```

## 2. Inspect Checkpoint Quantization Metadata

```bash
for format in bf16 fp16 fp8 int8_awq nvfp4; do
  MODEL_DIR="${MODEL_ROOT}/${format}" ./scripts/inspect_modelopt_checkpoint.sh
done
```

The script prints suggested:

```bash
VLLM_QUANTIZATION=...
SGLANG_QUANTIZATION=...
```

## 3. HF Backend Smoke Runs

```bash
MODEL_DIR="${MODEL_ROOT}/bf16" DTYPE=bfloat16 OUTPUT_DIR=artifacts/eval/qwen35/hf-bf16 ./scripts/run_hf_modelopt.sh
MODEL_DIR="${MODEL_ROOT}/fp16" DTYPE=float16  OUTPUT_DIR=artifacts/eval/qwen35/hf-fp16 ./scripts/run_hf_modelopt.sh
MODEL_DIR="${MODEL_ROOT}/fp8"  DTYPE=auto     OUTPUT_DIR=artifacts/eval/qwen35/hf-fp8  ./scripts/run_hf_modelopt.sh
```

The HF path is mainly for checking whether checkpoint/tokenizer/config loading
works. NVFP4 and AWQ support on the HF path depends on the exported format and
the installed Transformers/ModelOpt integration.

## 4. vLLM Smoke Runs

```bash
MODEL_DIR="${MODEL_ROOT}/bf16"     VLLM_QUANTIZATION=             OUTPUT_DIR=artifacts/eval/qwen35/vllm-bf16     ./scripts/run_vllm_modelopt.sh
MODEL_DIR="${MODEL_ROOT}/fp16"     VLLM_QUANTIZATION=             OUTPUT_DIR=artifacts/eval/qwen35/vllm-fp16     ./scripts/run_vllm_modelopt.sh
MODEL_DIR="${MODEL_ROOT}/fp8"      VLLM_QUANTIZATION=modelopt     OUTPUT_DIR=artifacts/eval/qwen35/vllm-fp8      ./scripts/run_vllm_modelopt.sh
MODEL_DIR="${MODEL_ROOT}/int8_awq" VLLM_QUANTIZATION=awq          OUTPUT_DIR=artifacts/eval/qwen35/vllm-int8-awq ./scripts/run_vllm_modelopt.sh
MODEL_DIR="${MODEL_ROOT}/nvfp4"    VLLM_QUANTIZATION=modelopt_fp4 OUTPUT_DIR=artifacts/eval/qwen35/vllm-nvfp4    ./scripts/run_vllm_modelopt.sh
```

Start NVFP4 with a small sample:

```bash
MODEL_DIR="${MODEL_ROOT}/nvfp4" \
VLLM_QUANTIZATION=modelopt_fp4 \
LIMIT=10 \
OUTPUT_DIR=artifacts/eval/qwen35/vllm-nvfp4-smoke \
./scripts/run_vllm_modelopt.sh
```

## 5. SGLang Smoke Runs

```bash
MODEL_DIR="${MODEL_ROOT}/fp8"      SGLANG_QUANTIZATION=modelopt_fp8 OUTPUT_DIR=artifacts/eval/qwen35/sglang-fp8      ./scripts/run_sglang_modelopt.sh
MODEL_DIR="${MODEL_ROOT}/int8_awq" SGLANG_QUANTIZATION=awq          OUTPUT_DIR=artifacts/eval/qwen35/sglang-int8-awq ./scripts/run_sglang_modelopt.sh
MODEL_DIR="${MODEL_ROOT}/nvfp4"    SGLANG_QUANTIZATION=modelopt_fp4 OUTPUT_DIR=artifacts/eval/qwen35/sglang-nvfp4    ./scripts/run_sglang_modelopt.sh
```

If OOM occurs:

```bash
MEM_FRACTION_STATIC=0.60
MAX_MODEL_LEN=2048
BATCH_SIZE=8
```

## 6. One-Command Matrix

Default: `bf16 fp16 fp8 int8_awq nvfp4` and `hf vllm`:

```bash
MODEL_ROOT="${MODEL_ROOT}" \
TASKS="${TASKS}" \
LIMIT=100 \
OUTPUT_ROOT=artifacts/eval/qwen35 \
./scripts/run_qwen35_modelopt_scenario.sh
```

vLLM only:

```bash
MODEL_ROOT="${MODEL_ROOT}" \
BACKENDS="vllm" \
FORMATS="fp8 int8_awq nvfp4" \
LIMIT=100 \
OUTPUT_ROOT=artifacts/eval/qwen35-vllm \
./scripts/run_qwen35_modelopt_scenario.sh
```

HF/vLLM/SGLang:

```bash
MODEL_ROOT="${MODEL_ROOT}" \
BACKENDS="hf vllm sglang" \
FORMATS="bf16 fp16 fp8 int8_awq nvfp4" \
LIMIT=100 \
OUTPUT_ROOT=artifacts/eval/qwen35-all \
./scripts/run_qwen35_modelopt_scenario.sh
```

## 7. Evaluation Task Sets

CI smoke:

```bash
export TASKS=arc_challenge,hellaswag,piqa,winogrande
export LIMIT=100
```

Nightly:

```bash
export TASKS=arc_challenge,hellaswag,piqa,winogrande,boolq,mmlu
export LIMIT=200
```

Release:

```bash
export TASKS=mmlu,arc_challenge,hellaswag,winogrande,truthfulqa_mc2,gsm8k
export LIMIT=none
```

## 8. Compare Results

```bash
find artifacts/eval/qwen35 -name 'results_*.json' -print
```

Extract key metrics with `jq`:

```bash
for f in $(find artifacts/eval/qwen35 -name 'results_*.json' | sort); do
  echo "== ${f} =="
  jq -r '
    .results
    | to_entries[]
    | [.key, (.value["acc_norm,none"] // .value["acc,none"] // .value["exact_match,none"] // "NA")]
    | @tsv
  ' "$f"
done
```

