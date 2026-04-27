#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_ROOT:?Set MODEL_ROOT to a directory containing quantized model subdirectories}"

BACKENDS="${BACKENDS:-hf vllm sglang}"
QUANT_FORMATS="${QUANT_FORMATS:-bf16 fp16 fp8 int8_awq nvfp4}"
TASKS="${TASKS:-arc_challenge,hellaswag,piqa,winogrande}"
LIMIT="${LIMIT:-100}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/eval/matrix}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

vllm_quantization_for() {
  case "$1" in
    bf16|fp16|fp32) echo "" ;;
    fp8) echo "modelopt" ;;
    int8_awq|awq) echo "awq" ;;
    nvfp4|fp4) echo "modelopt_fp4" ;;
    mxfp8) echo "modelopt_mxfp8" ;;
    *) echo "" ;;
  esac
}

sglang_quantization_for() {
  case "$1" in
    bf16|fp16|fp32) echo "" ;;
    fp8) echo "modelopt_fp8" ;;
    int8_awq|awq) echo "awq" ;;
    nvfp4|fp4) echo "modelopt_fp4" ;;
    *) echo "" ;;
  esac
}

for quant in ${QUANT_FORMATS}; do
  model_dir="${MODEL_ROOT}/${quant}"
  if [[ ! -d "${model_dir}" ]]; then
    echo "Skipping ${quant}: ${model_dir} does not exist"
    continue
  fi

  for backend in ${BACKENDS}; do
    echo "=== backend=${backend} quant=${quant} model=${model_dir} ==="
    case "${backend}" in
      hf)
        MODEL_DIR="${model_dir}" \
        TASKS="${TASKS}" \
        LIMIT="${LIMIT}" \
        NUM_FEWSHOT="${NUM_FEWSHOT}" \
        OUTPUT_DIR="${OUTPUT_ROOT}/${backend}/${quant}" \
        "${script_dir}/run_hf_modelopt.sh"
        ;;
      vllm)
        MODEL_DIR="${model_dir}" \
        TASKS="${TASKS}" \
        LIMIT="${LIMIT}" \
        NUM_FEWSHOT="${NUM_FEWSHOT}" \
        OUTPUT_DIR="${OUTPUT_ROOT}/${backend}/${quant}" \
        VLLM_QUANTIZATION="$(vllm_quantization_for "${quant}")" \
        "${script_dir}/run_vllm_modelopt.sh"
        ;;
      sglang)
        MODEL_DIR="${model_dir}" \
        TASKS="${TASKS}" \
        LIMIT="${LIMIT}" \
        NUM_FEWSHOT="${NUM_FEWSHOT}" \
        OUTPUT_DIR="${OUTPUT_ROOT}/${backend}/${quant}" \
        SGLANG_QUANTIZATION="$(sglang_quantization_for "${quant}")" \
        "${script_dir}/run_sglang_modelopt.sh"
        ;;
      *)
        echo "Unknown backend: ${backend}" >&2
        exit 2
        ;;
    esac
  done
done
