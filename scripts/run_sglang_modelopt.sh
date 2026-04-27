#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_DIR:?Set MODEL_DIR to an SGLang-supported model directory}"

TASKS="${TASKS:-arc_challenge,hellaswag,piqa,winogrande}"
LIMIT="${LIMIT:-100}"
SAMPLES_JSON="${SAMPLES_JSON:-}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
DTYPE="${DTYPE:-auto}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-True}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/eval/sglang}"

TP_SIZE="${TP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-}"
SGLANG_QUANTIZATION="${SGLANG_QUANTIZATION:-}"
EXTRA_MODEL_ARGS="${EXTRA_MODEL_ARGS:-}"

MODEL_ARGS="pretrained=${MODEL_DIR},dtype=${DTYPE},trust_remote_code=${TRUST_REMOTE_CODE},tp_size=${TP_SIZE},dp_size=${DP_SIZE},max_model_len=${MAX_MODEL_LEN}"
if [[ -n "${MEM_FRACTION_STATIC}" ]]; then
  MODEL_ARGS="${MODEL_ARGS},mem_fraction_static=${MEM_FRACTION_STATIC}"
fi
if [[ -n "${SGLANG_QUANTIZATION}" ]]; then
  MODEL_ARGS="${MODEL_ARGS},quantization=${SGLANG_QUANTIZATION}"
fi
if [[ -n "${EXTRA_MODEL_ARGS}" ]]; then
  MODEL_ARGS="${MODEL_ARGS},${EXTRA_MODEL_ARGS}"
fi

ARGS=(
  lm-eval run
  --model sglang
  --model_args "${MODEL_ARGS}"
  --tasks "${TASKS}"
  --batch_size "${BATCH_SIZE}"
  --num_fewshot "${NUM_FEWSHOT}"
  --output_path "${OUTPUT_DIR}"
)

if [[ -n "${SAMPLES_JSON}" ]]; then
  ARGS+=(--samples "${SAMPLES_JSON}")
elif [[ -n "${LIMIT}" && "${LIMIT}" != "none" ]]; then
  ARGS+=(--limit "${LIMIT}")
fi

echo "Running SGLang lm-eval"
echo "MODEL_ARGS=${MODEL_ARGS}"
echo "TASKS=${TASKS}"
"${ARGS[@]}"

