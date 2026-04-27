# Qwen3.5 ModelOpt 多格式 Checkpoint 测试场景

这个场景假设你已经有本地 Qwen3.5 量化产物：

```text
/data/models/qwen3.5-modelopt/
  bf16/
  fp16/
  fp8/
  int8_awq/
  nvfp4/
```

目标是用 `lm-eval` 对这些产物做一致的质量回归测试，并对比 HF、vLLM、SGLang 三条路径。

## 版本有效性

这些示例使用的关键约定：

- vLLM ModelOpt FP8：`quantization=modelopt`
- vLLM ModelOpt NVFP4：`quantization=modelopt_fp4`
- vLLM AWQ：`quantization=awq`
- SGLang ModelOpt FP8：`quantization=modelopt_fp8`
- SGLang ModelOpt FP4/NVFP4：`quantization=modelopt_fp4`
- SGLang AWQ：`quantization=awq`

如果后端版本不接受这些参数，先用 `scripts/inspect_modelopt_checkpoint.sh` 查看 `hf_quant_config.json`，再按当前后端文档调整。

## 1. 设置公共变量

```bash
cd /localhome/swqa/workspace/aaa/lm-eval-example

export MODEL_ROOT=/data/models/qwen3.5-modelopt
export TASKS=arc_challenge,hellaswag,piqa,winogrande
export LIMIT=100
export NUM_FEWSHOT=0
```

## 2. 先检查每个 checkpoint 的量化元数据

```bash
for format in bf16 fp16 fp8 int8_awq nvfp4; do
  MODEL_DIR="${MODEL_ROOT}/${format}" ./scripts/inspect_modelopt_checkpoint.sh
done
```

这个脚本会输出建议的：

```bash
VLLM_QUANTIZATION=...
SGLANG_QUANTIZATION=...
```

## 3. HF backend 逐个 smoke

```bash
MODEL_DIR="${MODEL_ROOT}/bf16" DTYPE=bfloat16 OUTPUT_DIR=artifacts/eval/qwen35/hf-bf16 ./scripts/run_hf_modelopt.sh
MODEL_DIR="${MODEL_ROOT}/fp16" DTYPE=float16  OUTPUT_DIR=artifacts/eval/qwen35/hf-fp16 ./scripts/run_hf_modelopt.sh
MODEL_DIR="${MODEL_ROOT}/fp8"  DTYPE=auto     OUTPUT_DIR=artifacts/eval/qwen35/hf-fp8  ./scripts/run_hf_modelopt.sh
```

HF 路径主要用于验证 checkpoint/tokenizer/config 是否能正常工作。NVFP4 和 AWQ 是否能通过 HF 路径跑，取决于导出格式和 Transformers/ModelOpt 集成。

## 4. vLLM 逐个 smoke

```bash
MODEL_DIR="${MODEL_ROOT}/bf16"     VLLM_QUANTIZATION=             OUTPUT_DIR=artifacts/eval/qwen35/vllm-bf16     ./scripts/run_vllm_modelopt.sh
MODEL_DIR="${MODEL_ROOT}/fp16"     VLLM_QUANTIZATION=             OUTPUT_DIR=artifacts/eval/qwen35/vllm-fp16     ./scripts/run_vllm_modelopt.sh
MODEL_DIR="${MODEL_ROOT}/fp8"      VLLM_QUANTIZATION=modelopt     OUTPUT_DIR=artifacts/eval/qwen35/vllm-fp8      ./scripts/run_vllm_modelopt.sh
MODEL_DIR="${MODEL_ROOT}/int8_awq" VLLM_QUANTIZATION=awq          OUTPUT_DIR=artifacts/eval/qwen35/vllm-int8-awq ./scripts/run_vllm_modelopt.sh
MODEL_DIR="${MODEL_ROOT}/nvfp4"    VLLM_QUANTIZATION=modelopt_fp4 OUTPUT_DIR=artifacts/eval/qwen35/vllm-nvfp4    ./scripts/run_vllm_modelopt.sh
```

NVFP4 建议先小样本：

```bash
MODEL_DIR="${MODEL_ROOT}/nvfp4" \
VLLM_QUANTIZATION=modelopt_fp4 \
LIMIT=10 \
OUTPUT_DIR=artifacts/eval/qwen35/vllm-nvfp4-smoke \
./scripts/run_vllm_modelopt.sh
```

## 5. SGLang 逐个 smoke

```bash
MODEL_DIR="${MODEL_ROOT}/fp8"      SGLANG_QUANTIZATION=modelopt_fp8 OUTPUT_DIR=artifacts/eval/qwen35/sglang-fp8      ./scripts/run_sglang_modelopt.sh
MODEL_DIR="${MODEL_ROOT}/int8_awq" SGLANG_QUANTIZATION=awq          OUTPUT_DIR=artifacts/eval/qwen35/sglang-int8-awq ./scripts/run_sglang_modelopt.sh
MODEL_DIR="${MODEL_ROOT}/nvfp4"    SGLANG_QUANTIZATION=modelopt_fp4 OUTPUT_DIR=artifacts/eval/qwen35/sglang-nvfp4    ./scripts/run_sglang_modelopt.sh
```

如果 OOM：

```bash
MEM_FRACTION_STATIC=0.60
MAX_MODEL_LEN=2048
BATCH_SIZE=8
```

## 6. 一条命令跑矩阵

默认跑 `bf16 fp16 fp8 int8_awq nvfp4` 和 `hf vllm`：

```bash
MODEL_ROOT="${MODEL_ROOT}" \
TASKS="${TASKS}" \
LIMIT=100 \
OUTPUT_ROOT=artifacts/eval/qwen35 \
./scripts/run_qwen35_modelopt_scenario.sh
```

只跑 vLLM：

```bash
MODEL_ROOT="${MODEL_ROOT}" \
BACKENDS="vllm" \
FORMATS="fp8 int8_awq nvfp4" \
LIMIT=100 \
OUTPUT_ROOT=artifacts/eval/qwen35-vllm \
./scripts/run_qwen35_modelopt_scenario.sh
```

同时跑 HF/vLLM/SGLang：

```bash
MODEL_ROOT="${MODEL_ROOT}" \
BACKENDS="hf vllm sglang" \
FORMATS="bf16 fp16 fp8 int8_awq nvfp4" \
LIMIT=100 \
OUTPUT_ROOT=artifacts/eval/qwen35-all \
./scripts/run_qwen35_modelopt_scenario.sh
```

## 7. 正式评估任务集

CI smoke：

```bash
export TASKS=arc_challenge,hellaswag,piqa,winogrande
export LIMIT=100
```

Nightly：

```bash
export TASKS=arc_challenge,hellaswag,piqa,winogrande,boolq,mmlu
export LIMIT=200
```

Release：

```bash
export TASKS=mmlu,arc_challenge,hellaswag,winogrande,truthfulqa_mc2,gsm8k
export LIMIT=none
```

## 8. 结果比较

```bash
find artifacts/eval/qwen35 -name 'results_*.json' -print
```

用 `jq` 抽取核心指标：

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

