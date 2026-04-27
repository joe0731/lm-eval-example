# ModelOpt 量化 Checkpoint 的 lm-eval Bash 命令示例

这份文档用于研究不同情况下如何评估 ModelOpt 量化后的 checkpoint。命令默认在 `lm-eval-example` 仓库根目录执行。

## 版本有效性

这些命令基于本示例创建时的本地环境：

```bash
# lm-evaluation-harness
# package version: 0.4.12.dev0
# git describe: v0.4.11-24-g620262e0
# commit: 620262e0
# note date: 2026-04-27
```

vLLM、SGLang、ModelOpt 的量化参数可能随版本变化。建议每次 CI 记录版本：

```bash
python -m pip freeze | grep -E 'lm_eval|vllm|sglang|nvidia-modelopt|torch|transformers'
```

如果命令失败，优先用 `--limit 10` 缩小问题，再确认当前后端版本支持的 `quantization=` 名称。

## 0. 准备变量

```bash
export LMEVAL_REPO=/localhome/swqa/workspace/aaa/lm-evaluation-harness
export EXAMPLE_REPO=/localhome/swqa/workspace/aaa/lm-eval-example
export MODEL_ROOT=/path/to/modelopt/checkpoints

export TASKS=arc_challenge,hellaswag,piqa,winogrande
export LIMIT=100
export NUM_FEWSHOT=0
```

安装基础 HF backend：

```bash
cd "${LMEVAL_REPO}"
pip install -e ".[hf]"
```

安装 vLLM backend：

```bash
cd "${LMEVAL_REPO}"
pip install -e ".[vllm]"
```

SGLang 建议按官方文档安装 SGLang，再安装本仓：

```bash
cd "${LMEVAL_REPO}"
pip install -e .
```

## 1. 校验任务是否存在

```bash
cd "${EXAMPLE_REPO}"
TASKS="${TASKS}" ./scripts/validate_tasks.sh
```

查看可用任务：

```bash
lm-eval ls tasks
lm-eval ls groups
lm-eval ls subtasks
```

## 2. HF backend：先验证 checkpoint 是否可评估

适合做 reference/correctness 测试。

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

NVFP4 checkpoint：

```bash
MODEL_DIR="${MODEL_ROOT}/nvfp4" \
TASKS="${TASKS}" \
LIMIT="${LIMIT}" \
BATCH_SIZE=8 \
OUTPUT_DIR=artifacts/eval/hf-nvfp4 \
./scripts/run_hf_modelopt.sh
```

注意：HF backend 的 `DTYPE=auto/bfloat16/float16` 不是 ModelOpt 的 `fp8/nvfp4` 量化格式。量化格式应由 checkpoint 配置或自定义 loader 决定。

## 3. vLLM backend：测试 ModelOpt FP8

`lm-eval --model vllm` 使用 vLLM offline engine，不起 server。

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

如果 checkpoint 的 `hf_quant_config.json` 可被 vLLM 自动识别，可以先不传量化参数：

```bash
MODEL_DIR="${MODEL_ROOT}/fp8" \
VLLM_QUANTIZATION= \
OUTPUT_DIR=artifacts/eval/vllm-fp8-auto \
./scripts/run_vllm_modelopt.sh
```

## 4. vLLM backend：测试 ModelOpt NVFP4

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

Blackwell/NVFP4 场景通常更依赖硬件和后端版本。先用小样本确认：

```bash
MODEL_DIR="${MODEL_ROOT}/nvfp4" \
VLLM_QUANTIZATION=modelopt_fp4 \
LIMIT=10 \
OUTPUT_DIR=artifacts/eval/vllm-nvfp4-smoke \
./scripts/run_vllm_modelopt.sh
```

## 5. vLLM backend：多卡 tensor parallel

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

CI 初期不建议启用 `data_parallel_size>1`，因为它会走 Ray actor，依赖更多。

## 6. SGLang backend：测试 ModelOpt FP8

`lm-eval --model sglang` 使用 SGLang offline engine，不起 server。

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

如果当前 SGLang 版本不接受 `modelopt_fp8`，尝试：

```bash
MODEL_DIR="${MODEL_ROOT}/fp8" \
SGLANG_QUANTIZATION=modelopt \
OUTPUT_DIR=artifacts/eval/sglang-fp8-modelopt \
./scripts/run_sglang_modelopt.sh
```

或者不传，让后端读取 checkpoint 配置：

```bash
MODEL_DIR="${MODEL_ROOT}/fp8" \
SGLANG_QUANTIZATION= \
OUTPUT_DIR=artifacts/eval/sglang-fp8-auto \
./scripts/run_sglang_modelopt.sh
```

## 7. SGLang backend：测试 ModelOpt NVFP4/FP4

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

OOM 时优先调小：

```bash
MEM_FRACTION_STATIC=0.60
MAX_MODEL_LEN=2048
BATCH_SIZE=8
```

## 8. 固定样本，保证 CI 稳定

```bash
SAMPLES_JSON="$(cat config/sample_indices.json)" \
MODEL_DIR="${MODEL_ROOT}/fp8" \
TASKS="${TASKS}" \
LIMIT=none \
OUTPUT_DIR=artifacts/eval/hf-fp8-fixed \
./scripts/run_hf_modelopt.sh
```

`--samples` 和 `--limit` 互斥，所以这里设置 `LIMIT=none`。

## 9. 一次跑多种量化格式和后端

假设目录：

```text
/path/to/modelopt/checkpoints/
  bf16/
  fp8/
  nvfp4/
```

运行：

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

如果只研究 vLLM：

```bash
MODEL_ROOT="${MODEL_ROOT}" \
BACKENDS="vllm" \
QUANT_FORMATS="fp8 nvfp4" \
LIMIT=100 \
./scripts/run_modelopt_matrix.sh
```

## 10. baseline vs quantized 对比流程

先跑 BF16 baseline：

```bash
MODEL_DIR="${MODEL_ROOT}/bf16" \
TASKS="${TASKS}" \
LIMIT=200 \
OUTPUT_DIR=artifacts/eval/baseline-bf16 \
./scripts/run_hf_modelopt.sh
```

再跑量化 checkpoint：

```bash
MODEL_DIR="${MODEL_ROOT}/fp8" \
TASKS="${TASKS}" \
LIMIT=200 \
OUTPUT_DIR=artifacts/eval/quant-fp8 \
./scripts/run_vllm_modelopt.sh
```

查看结果文件：

```bash
find artifacts/eval -name 'results_*.json' -print
```

简单抽取指标：

```bash
jq '.results.arc_challenge["acc_norm,none"], .results.hellaswag["acc_norm,none"]' \
  "$(find artifacts/eval/quant-fp8 -name 'results_*.json' | sort | tail -1)"
```

## 11. CI 阈值门禁示例

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

## 12. Python API：需要自定义 ModelOpt restore 时

如果 checkpoint 不能直接被 HF/vLLM/SGLang 加载，先用 Python API 示例替换内部加载逻辑：

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

需要把脚本里的 `AutoModelForCausalLM.from_pretrained(...)` 替换成 ModelOpt restore 逻辑，但最终对象要满足：

```text
model(input_ids).logits
model.generate(...)
tokenizer.encode/decode 或 HF tokenizer 接口
```

## 13. 生成式任务研究

生成式任务更慢，且更受 chat template 影响，建议先小样本：

```bash
MODEL_DIR="${MODEL_ROOT}/fp8" \
TASKS=gsm8k_cot,ifeval \
LIMIT=20 \
BATCH_SIZE=auto \
EXTRA_MODEL_ARGS="max_gen_toks=512" \
OUTPUT_DIR=artifacts/eval/vllm-generative-smoke \
./scripts/run_vllm_modelopt.sh
```

如果是 chat/instruct 模型，增加 chat template：

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

## 14. 调试建议

最小化调试：

```bash
MODEL_DIR="${MODEL_ROOT}/fp8" \
TASKS=arc_challenge \
LIMIT=5 \
BATCH_SIZE=1 \
OUTPUT_DIR=artifacts/eval/debug \
./scripts/run_hf_modelopt.sh
```

显存不足：

```bash
MAX_MODEL_LEN=2048
BATCH_SIZE=4
GPU_MEMORY_UTILIZATION=0.75
MEM_FRACTION_STATIC=0.60
```

确认 tokenizer/chat template：

```bash
lm-eval run \
  --model hf \
  --model_args pretrained="${MODEL_ROOT}/fp8",dtype=auto,trust_remote_code=True \
  --tasks arc_challenge \
  --limit 1 \
  --write_out
```

