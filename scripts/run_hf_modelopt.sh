#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_DIR:?Set MODEL_DIR to a Hugging Face-compatible model directory}"

TASKS="${TASKS:-arc_challenge,hellaswag,piqa,winogrande}"
LIMIT="${LIMIT:-100}"
SAMPLES_JSON="${SAMPLES_JSON:-}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DTYPE="${DTYPE:-auto}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-True}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/eval/hf}"
EXTRA_MODEL_ARGS="${EXTRA_MODEL_ARGS:-}"

MODEL_ARGS="pretrained=${MODEL_DIR},dtype=${DTYPE},trust_remote_code=${TRUST_REMOTE_CODE}"
if [[ -n "${EXTRA_MODEL_ARGS}" ]]; then
  MODEL_ARGS="${MODEL_ARGS},${EXTRA_MODEL_ARGS}"
fi

ARGS=(
  lm-eval run
  --model hf
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

echo "Running HF lm-eval"
echo "MODEL_ARGS=${MODEL_ARGS}"
echo "TASKS=${TASKS}"
"${ARGS[@]}"

