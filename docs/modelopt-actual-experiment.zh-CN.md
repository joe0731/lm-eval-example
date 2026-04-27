# ModelOpt 量化产物实际实验流程

这个流程的目标不是单纯证明 `lm-eval` 能启动，而是把 ModelOpt 量化后的 checkpoint 放进一个可重复、可比较、可进 CI 的评估链路：

1. 确认 checkpoint 元数据和量化格式。
2. 确认目标 backend 能真正加载并生成。
3. 用固定任务和固定样本比较 bf16/fp16 基线与量化产物。
4. 把结果落盘、汇总，并在 CI 中做阈值判断。

## 实验结论先行

对 ModelOpt 预量化产物，实际优先路径通常是：

| 产物格式 | 推荐 backend | 典型量化参数 |
| --- | --- | --- |
| bf16 | `hf` 或 `vllm` | 无 |
| fp16 | `hf` 或 `vllm` | 无 |
| fp8 | `vllm` | `quantization=modelopt` |
| int8_awq | `vllm` | `quantization=awq` |
| nvfp4 | `vllm` | `quantization=modelopt_fp4` |
| mxfp8 | `vllm` | `quantization=modelopt_mxfp8` |

`lm-eval --model vllm` 是单进程 offline engine 路径，会在当前进程里创建 `vllm.LLM(...)`，不需要先启动 server。

HF backend 只适合能被 Transformers 直接加载的产物。实际测试过的 `nvidia/Gemma-4-31B-IT-NVFP4` 在当前环境下不能通过 HF/lm-eval 路径加载，因为 Transformers 不识别 `quant_method=modelopt`，后续还会遇到量化权重 shape 和未量化模型 shape 不匹配。不要用 `ignore_mismatched_sizes=True` 做 benchmark，它会导致部分权重重新初始化。

SGLang 可以作为第二条高吞吐路径，但要按具体模型和版本验证。当前记录里的 Gemma4 NVFP4 场景中，SGLang 能加载 checkpoint，但生成阶段命中了模型实现/算子兼容问题，因此不能作为这个产物的可靠 benchmark backend。

## 0. 假设目录

假设本地已经有一组 Qwen3.5 ModelOpt 产物：

```text
/data/models/qwen3.5-modelopt/
  bf16/
  fp16/
  fp8/
  int8_awq/
  nvfp4/
```

后续命令在示例仓执行：

```bash
cd /localhome/swqa/workspace/aaa/lm-eval-example
export MODEL_ROOT=/data/models/qwen3.5-modelopt
```

## 1. 记录版本

每次正式实验先记录环境，后续结果才可解释：

```bash
python -m pip freeze | grep -E 'lm-eval|lm_eval|vllm|sglang|nvidia-modelopt|torch|transformers'
nvidia-smi
```

建议把这些输出保存到 `artifacts/eval/<run>/versions.txt`。

## 2. 检查 checkpoint

先不要直接跑 benchmark，先看导出的元数据是否符合预期：

```bash
for format in bf16 fp16 fp8 int8_awq nvfp4; do
  MODEL_DIR="${MODEL_ROOT}/${format}" ./scripts/inspect_modelopt_checkpoint.sh
done
```

重点看：

- `PRODUCER` 是否是 `modelopt`。
- `QUANT_ALGO` 是否符合目录名，例如 `NVFP4`、`FP8`、`AWQ`。
- `VLLM_QUANTIZATION` 和 `SGLANG_QUANTIZATION` 建议值。
- tokenizer/config/safetensors 是否齐全。

## 3. 单模型 smoke

smoke 的目标是确认能加载、能跑完、能写结果。不要在这一步追求统计意义。

bf16 基线：

```bash
MODEL_DIR="${MODEL_ROOT}/bf16" \
DTYPE=bfloat16 \
TASKS=arc_challenge \
LIMIT=1 \
BATCH_SIZE=1 \
OUTPUT_DIR=artifacts/eval/smoke/hf-bf16 \
./scripts/run_hf_modelopt.sh
```

NVFP4 量化产物：

```bash
MODEL_DIR="${MODEL_ROOT}/nvfp4" \
VLLM_QUANTIZATION=modelopt_fp4 \
TASKS=arc_challenge \
LIMIT=1 \
BATCH_SIZE=auto \
MAX_MODEL_LEN=2048 \
GPU_MEMORY_UTILIZATION=0.85 \
OUTPUT_DIR=artifacts/eval/smoke/vllm-nvfp4 \
./scripts/run_vllm_modelopt.sh
```

FP8 量化产物：

```bash
MODEL_DIR="${MODEL_ROOT}/fp8" \
VLLM_QUANTIZATION=modelopt \
TASKS=arc_challenge \
LIMIT=1 \
OUTPUT_DIR=artifacts/eval/smoke/vllm-fp8 \
./scripts/run_vllm_modelopt.sh
```

如果 smoke 失败，先修 backend/版本/量化参数，不要进入正式评估。

## 4. CI smoke 实验

CI smoke 建议用小样本、多任务，只判断明显回归和基础可用性：

```bash
MODEL_ROOT="${MODEL_ROOT}" \
FORMATS="bf16 fp16 fp8 int8_awq nvfp4" \
BACKENDS="vllm" \
TASKS="arc_challenge,hellaswag,piqa,winogrande" \
LIMIT=50 \
NUM_FEWSHOT=0 \
OUTPUT_ROOT=artifacts/eval/ci-smoke \
./scripts/run_qwen35_modelopt_scenario.sh
```

如果 GPU 时间更紧，可以先跑：

```bash
TASKS="arc_challenge,piqa" LIMIT=20
```

这一级不要设置过硬的精度阈值。更实际的 gate 是：

- 每个格式都能加载。
- 每个任务都有数值结果。
- 分数没有出现明显异常，例如接近 0 或明显低于历史区间。

## 5. Nightly 实验

Nightly 用于建立量化质量趋势，建议固定任务和样本数量：

```bash
MODEL_ROOT="${MODEL_ROOT}" \
FORMATS="bf16 fp16 fp8 int8_awq nvfp4" \
BACKENDS="vllm" \
TASKS="arc_challenge,hellaswag,piqa,winogrande,boolq,mmlu" \
LIMIT=200 \
NUM_FEWSHOT=0 \
OUTPUT_ROOT=artifacts/eval/nightly \
./scripts/run_qwen35_modelopt_scenario.sh
```

如果想做更稳定的回归比较，优先使用固定样本文件，而不是每次只写 `--limit`：

```bash
MODEL_ROOT="${MODEL_ROOT}" \
FORMATS="bf16 fp16 fp8 int8_awq nvfp4" \
BACKENDS="vllm" \
TASKS="arc_challenge,hellaswag,piqa,winogrande" \
SAMPLES_JSON="$(cat config/sample_indices.json)" \
OUTPUT_ROOT=artifacts/eval/nightly-fixed \
./scripts/run_qwen35_modelopt_scenario.sh
```

## 6. Release 实验

Release 才跑完整任务，不建议每个 PR 都跑：

```bash
MODEL_ROOT="${MODEL_ROOT}" \
FORMATS="bf16 fp16 fp8 int8_awq nvfp4" \
BACKENDS="vllm" \
TASKS="mmlu,arc_challenge,hellaswag,winogrande,truthfulqa_mc2,gsm8k" \
LIMIT=none \
NUM_FEWSHOT=0 \
OUTPUT_ROOT=artifacts/eval/release \
./scripts/run_qwen35_modelopt_scenario.sh
```

`gsm8k` 是生成式任务，更慢，也更受 prompt/chat template 影响。早期量化回归建议先看 multiple-choice/loglikelihood 类任务。

## 7. 汇总结果

`lm-eval` 会在 `OUTPUT_ROOT` 下生成 `results*.json`。用脚本汇总：

```bash
python scripts/summarize_lm_eval_results.py artifacts/eval/ci-smoke
```

输出 CSV：

```bash
python scripts/summarize_lm_eval_results.py artifacts/eval/ci-smoke --format csv \
  > artifacts/eval/ci-smoke/summary.csv
```

输出所有数值指标：

```bash
python scripts/summarize_lm_eval_results.py artifacts/eval/ci-smoke --all-metrics
```

## 8. CI 阈值建议

量化评估建议先按 bf16 或 fp16 同 backend 的结果做 baseline，不要把不同 backend 的差异直接归因于量化。

一个务实的初始策略：

| 阶段 | 判定方式 |
| --- | --- |
| PR smoke | 加载成功、任务完成、指标存在 |
| Nightly | 和 fp16/bf16 baseline 的 delta 做 warning |
| Release | delta 超阈值 fail |

阈值一开始不要拍死。先保留 3 到 5 次稳定实验结果，再按模型和任务设定，例如：

- FP8：多数任务相对 bf16/fp16 下降超过 1 到 2 个百分点时 warning。
- INT8 AWQ：多数任务下降超过 2 到 3 个百分点时 warning。
- NVFP4：多数任务下降超过 3 到 5 个百分点时 warning。

这些值不是通用标准，只是启动 CI 的经验上限。最终阈值应该来自你自己的模型、任务集和业务容忍度。

## 9. 实际记录里的发现

当前仓库的 `exp_record/` 记录了一个实际 ModelOpt NVFP4 产物测试：

- HF/lm-eval：失败。Transformers 不识别 `modelopt` 量化配置。
- vLLM offline：成功加载并生成，`quantization=modelopt_fp4` 正确。
- SGLang：能加载，但 Gemma4 当前路径生成失败，不适合作为这个产物的正式评估路径。

因此，对 ModelOpt NVFP4 产物，实际优先实验路径是：

```bash
inspect checkpoint -> vLLM smoke -> vLLM CI smoke -> vLLM nightly/release
```

HF 只作为能加载时的 reference，SGLang 只作为版本验证通过后的第二 backend。
