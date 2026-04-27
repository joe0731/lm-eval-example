#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_DIR:?Set MODEL_DIR to a vLLM-supported model directory}"

TASKS="${TASKS:-arc_challenge,hellaswag,piqa,winogrande}"
LIMIT="${LIMIT:-100}"
SAMPLES_JSON="${SAMPLES_JSON:-}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
DTYPE="${DTYPE:-auto}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-True}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/eval/vllm}"

TP_SIZE="${TP_SIZE:-1}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-}"
EXTRA_MODEL_ARGS="${EXTRA_MODEL_ARGS:-}"

MODEL_ARGS="pretrained=${MODEL_DIR},dtype=${DTYPE},trust_remote_code=${TRUST_REMOTE_CODE},tensor_parallel_size=${TP_SIZE},data_parallel_size=${DATA_PARALLEL_SIZE},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},max_model_len=${MAX_MODEL_LEN}"
if [[ -n "${VLLM_QUANTIZATION}" ]]; then
  MODEL_ARGS="${MODEL_ARGS},quantization=${VLLM_QUANTIZATION}"
fi
if [[ -n "${MAX_NUM_SEQS}" ]]; then
  MODEL_ARGS="${MODEL_ARGS},max_num_seqs=${MAX_NUM_SEQS}"
fi
if [[ -n "${EXTRA_MODEL_ARGS}" ]]; then
  MODEL_ARGS="${MODEL_ARGS},${EXTRA_MODEL_ARGS}"
fi

ARGS=(
  lm-eval run
  --model vllm
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

echo "Running vLLM lm-eval"
echo "MODEL_ARGS=${MODEL_ARGS}"
echo "TASKS=${TASKS}"
"${ARGS[@]}"

