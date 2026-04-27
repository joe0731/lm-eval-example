#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_ROOT:?Set MODEL_ROOT to a directory containing qwen3.5 quantized subdirectories}"

FORMATS="${FORMATS:-bf16 fp16 fp8 int8_awq nvfp4}"
BACKENDS="${BACKENDS:-hf vllm}"
TASKS="${TASKS:-arc_challenge,hellaswag,piqa,winogrande}"
LIMIT="${LIMIT:-100}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/eval/qwen35}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for format in ${FORMATS}; do
  model_dir="${MODEL_ROOT}/${format}"
  if [[ ! -d "${model_dir}" ]]; then
    echo "Skipping ${format}: ${model_dir} does not exist"
    continue
  fi

  echo "=== Inspecting ${format}: ${model_dir} ==="
  MODEL_DIR="${model_dir}" "${script_dir}/inspect_modelopt_checkpoint.sh"

  for backend in ${BACKENDS}; do
    case "${backend}" in
      hf)
        dtype="auto"
        if [[ "${format}" == "bf16" ]]; then
          dtype="bfloat16"
        elif [[ "${format}" == "fp16" ]]; then
          dtype="float16"
        fi

        MODEL_DIR="${model_dir}" \
        TASKS="${TASKS}" \
        LIMIT="${LIMIT}" \
        NUM_FEWSHOT="${NUM_FEWSHOT}" \
        DTYPE="${dtype}" \
        OUTPUT_DIR="${OUTPUT_ROOT}/hf-${format}" \
        "${script_dir}/run_hf_modelopt.sh"
        ;;
      vllm)
        case "${format}" in
          bf16|fp16) quantization="" ;;
          fp8) quantization="modelopt" ;;
          int8_awq) quantization="awq" ;;
          nvfp4) quantization="modelopt_fp4" ;;
          *) quantization="" ;;
        esac

        MODEL_DIR="${model_dir}" \
        TASKS="${TASKS}" \
        LIMIT="${LIMIT}" \
        NUM_FEWSHOT="${NUM_FEWSHOT}" \
        VLLM_QUANTIZATION="${quantization}" \
        OUTPUT_DIR="${OUTPUT_ROOT}/vllm-${format}" \
        "${script_dir}/run_vllm_modelopt.sh"
        ;;
      sglang)
        case "${format}" in
          bf16|fp16) quantization="" ;;
          fp8) quantization="modelopt_fp8" ;;
          int8_awq) quantization="awq" ;;
          nvfp4) quantization="modelopt_fp4" ;;
          *) quantization="" ;;
        esac

        MODEL_DIR="${model_dir}" \
        TASKS="${TASKS}" \
        LIMIT="${LIMIT}" \
        NUM_FEWSHOT="${NUM_FEWSHOT}" \
        SGLANG_QUANTIZATION="${quantization}" \
        OUTPUT_DIR="${OUTPUT_ROOT}/sglang-${format}" \
        "${script_dir}/run_sglang_modelopt.sh"
        ;;
      *)
        echo "Unknown backend: ${backend}" >&2
        exit 2
        ;;
    esac
  done
done

