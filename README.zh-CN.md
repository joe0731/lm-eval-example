# lm-eval + ModelOpt 量化模型评估示例

这个目录给出一组围绕 ModelOpt 量化产物的 `lm-eval` 使用示例，目标是用于 CI/CD 中的单进程评估：

- HF backend：兼容性最好，适合作为 correctness/reference 路径。
- vLLM backend：吞吐更好，适合 vLLM 支持的 ModelOpt 量化 checkpoint。
- SGLang backend：吞吐更好，适合 SGLang 支持的 ModelOpt 量化 checkpoint。
- Python API：适合 ModelOpt 产物不能直接 `from_pretrained()`，需要自定义 restore 的场景。

## 目录结构

```text
lm-eval-example/
  README.md
  README.zh-CN.md
  README.en.md
  config/
    sample_indices.json
  docs/
    modelopt-checkpoint-eval-commands.zh-CN.md
    modelopt-checkpoint-eval-commands.en.md
    modelopt-actual-experiment.zh-CN.md
    modelopt-actual-experiment.en.md
    qwen35-modelopt-scenario.zh-CN.md
    qwen35-modelopt-scenario.en.md
  python/
    simple_eval_modelopt_hf.py
  scripts/
    run_hf_modelopt.sh
    run_vllm_modelopt.sh
    run_sglang_modelopt.sh
    run_modelopt_matrix.sh
    summarize_lm_eval_results.py
    validate_tasks.sh
```

命令 cookbook：

- [ModelOpt checkpoint eval bash 示例](docs/modelopt-checkpoint-eval-commands.zh-CN.md)
- [ModelOpt 量化产物实际实验流程](docs/modelopt-actual-experiment.zh-CN.md)
- [Qwen3.5 多格式 ModelOpt checkpoint 测试场景](docs/qwen35-modelopt-scenario.zh-CN.md)

## 版本有效性说明

这些示例基于当前本地仓库接口编写：

- `lm-evaluation-harness` package version: `0.4.12.dev0`
- local git describe: `v0.4.11-24-g620262e0`
- local commit: `620262e0`
- current date used for this note: `2026-04-27`

外部后端的量化参数会随版本变化，尤其是 vLLM、SGLang 和 ModelOpt。脚本把关键参数做成环境变量，不把某个后端版本的写法硬编码死：

- vLLM ModelOpt 文档当前说明 ModelOpt checkpoint 通过 `hf_quant_config.json` 识别，并支持 FP8、NVFP4、MXFP8 等格式。NVFP4 通常需要 `quantization=modelopt_fp4`，MXFP8 通常需要 `quantization=modelopt_mxfp8`。
- SGLang 文档当前对 ModelOpt 预量化模型常见写法是 `quantization=modelopt_fp8` 和 `quantization=modelopt_fp4`。
- 如果安装版本和文档不一致，以当前安装的后端版本为准，先用小样本 `--limit 10` 验证。

建议每个 CI image 固定这些版本：

```bash
python -m pip freeze | grep -E 'lm_eval|vllm|sglang|nvidia-modelopt|torch|transformers'
```

参考链接：

- lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness
- vLLM ModelOpt quantization: https://docs.vllm.ai/en/latest/features/quantization/modelopt/
- SGLang quantization: https://docs.sglang.io/docs/advanced_features/quantization
- NVIDIA ModelOpt: https://nvidia.github.io/TensorRT-Model-Optimizer/

## 安装

HF 路径：

```bash
cd /path/to/lm-evaluation-harness
pip install -e ".[hf]"
```

vLLM 路径：

```bash
cd /path/to/lm-evaluation-harness
pip install -e ".[vllm]"
```

SGLang 需要按 SGLang 官方方式安装，然后安装本仓：

```bash
cd /path/to/lm-evaluation-harness
pip install -e .
```

## 推荐任务集

CI smoke，主流且快：

```bash
arc_challenge,hellaswag,piqa,winogrande
```

CI 加强版：

```bash
arc_challenge,hellaswag,piqa,winogrande,boolq,mmlu
```

Release 或 nightly：

```bash
mmlu,arc_challenge,hellaswag,winogrande,truthfulqa_mc2,gsm8k
```

量化回归早期优先使用 multiple-choice/loglikelihood 类任务，因为它们更稳定。`gsm8k`、`ifeval`、`bbh_cot_zeroshot` 这类生成任务更慢，也更受 prompt/chat template 影响。

## 通用控制参数

```bash
--tasks        # 选择任务或 group
--limit        # 每个 task 跑多少条，CI smoke 常用
--samples      # 固定样本 index，和 --limit 互斥
--num_fewshot  # few-shot 示例数，不是测试次数
--output_path  # 保存结果
```

固定样本示例：

```bash
lm-eval run \
  --model hf \
  --model_args pretrained=/path/to/model,dtype=auto,trust_remote_code=True \
  --tasks arc_challenge,hellaswag,piqa,winogrande \
  --samples "$(cat lm-eval-example/config/sample_indices.json)" \
  --output_path artifacts/eval/fixed-samples
```

## 1. HF backend：参考路径

适合：

- ModelOpt 导出的模型目录能被 `transformers.AutoModelForCausalLM.from_pretrained()` 加载。
- 想先验证量化后精度，而不是追求吞吐。
- 作为 vLLM/SGLang 结果的参考基线。

运行：

```bash
MODEL_DIR=/path/to/modelopt_model \
OUTPUT_DIR=artifacts/eval/hf \
./lm-eval-example/scripts/run_hf_modelopt.sh
```

常用覆盖项：

```bash
TASKS=arc_challenge,hellaswag,piqa,winogrande \
LIMIT=100 \
BATCH_SIZE=8 \
DTYPE=auto \
MODEL_DIR=/path/to/modelopt_fp8 \
./lm-eval-example/scripts/run_hf_modelopt.sh
```

注意：不要把 `fp8`、`nvfp4` 当成 `dtype=` 传给 HF backend。`dtype=auto/bfloat16/float16` 控制加载和计算 dtype；ModelOpt 量化格式应该由导出的模型配置或自定义 loader 决定。

## 2. vLLM backend：ModelOpt 量化模型吞吐路径

`lm-eval --model vllm` 会在当前进程中创建 `vllm.LLM(...)`，不是起 server。

FP8 示例：

```bash
MODEL_DIR=/path/to/modelopt_fp8 \
VLLM_QUANTIZATION=modelopt \
OUTPUT_DIR=artifacts/eval/vllm-fp8 \
./lm-eval-example/scripts/run_vllm_modelopt.sh
```

NVFP4 示例：

```bash
MODEL_DIR=/path/to/modelopt_nvfp4 \
VLLM_QUANTIZATION=modelopt_fp4 \
OUTPUT_DIR=artifacts/eval/vllm-nvfp4 \
./lm-eval-example/scripts/run_vllm_modelopt.sh
```

MXFP8 示例：

```bash
MODEL_DIR=/path/to/modelopt_mxfp8 \
VLLM_QUANTIZATION=modelopt_mxfp8 \
OUTPUT_DIR=artifacts/eval/vllm-mxfp8 \
./lm-eval-example/scripts/run_vllm_modelopt.sh
```

多卡 tensor parallel：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL_DIR=/path/to/model \
TP_SIZE=4 \
./lm-eval-example/scripts/run_vllm_modelopt.sh
```

关键参数：

```bash
TP_SIZE=1
GPU_MEMORY_UTILIZATION=0.85
MAX_MODEL_LEN=4096
BATCH_SIZE=auto
VLLM_QUANTIZATION=modelopt_fp4
```

如果 vLLM 能从 checkpoint 的 `hf_quant_config.json` 自动识别格式，可以不传 `VLLM_QUANTIZATION`。如果自动识别失败，再显式传。

## 3. SGLang backend：ModelOpt 量化模型吞吐路径

`lm-eval --model sglang` 会在当前进程中创建 `sglang.Engine(...)`，不是起 server。

FP8 示例：

```bash
MODEL_DIR=/path/to/modelopt_fp8 \
SGLANG_QUANTIZATION=modelopt_fp8 \
OUTPUT_DIR=artifacts/eval/sglang-fp8 \
./lm-eval-example/scripts/run_sglang_modelopt.sh
```

NVFP4/FP4 示例：

```bash
MODEL_DIR=/path/to/modelopt_nvfp4 \
SGLANG_QUANTIZATION=modelopt_fp4 \
OUTPUT_DIR=artifacts/eval/sglang-nvfp4 \
./lm-eval-example/scripts/run_sglang_modelopt.sh
```

常见 OOM 调整：

```bash
MEM_FRACTION_STATIC=0.70
MAX_MODEL_LEN=4096
TP_SIZE=2
BATCH_SIZE=auto
```

SGLang 的 ModelOpt 量化参数在不同版本中可能有差异。如果 `modelopt_fp8/modelopt_fp4` 不适配你的安装版本，可以尝试 `SGLANG_QUANTIZATION=modelopt` 或直接不传，让后端按模型配置解析。

## 4. Python API：自定义 ModelOpt restore

如果 ModelOpt 产物不能被 HF/vLLM/SGLang 直接加载，走 Python API：

```bash
python lm-eval-example/python/simple_eval_modelopt_hf.py \
  --model-dir /path/to/modelopt_model \
  --tasks arc_challenge,hellaswag,piqa,winogrande \
  --limit 100 \
  --output-json artifacts/eval/python-api/results.json
```

你可以把脚本里的 HF 加载逻辑替换成公司内部的 ModelOpt restore 逻辑，只要最终对象满足：

- `model(input_ids).logits` 可用。
- `model.generate(...)` 可用。
- tokenizer 是 Hugging Face tokenizer 或兼容对象。

## 5. 一次跑多种量化格式

假设目录结构：

```text
artifacts/models/
  bf16/
  fp8/
  nvfp4/
```

运行：

```bash
MODEL_ROOT=artifacts/models \
BACKENDS="hf vllm sglang" \
QUANT_FORMATS="bf16 fp8 nvfp4" \
./lm-eval-example/scripts/run_modelopt_matrix.sh
```

默认映射：

| 格式 | HF | vLLM | SGLang |
| --- | --- | --- | --- |
| `bf16` | 无量化参数 | 无量化参数 | 无量化参数 |
| `fp8` | 由模型配置决定 | `modelopt` | `modelopt_fp8` |
| `nvfp4` | 由模型配置决定 | `modelopt_fp4` | `modelopt_fp4` |
| `mxfp8` | 由模型配置决定 | `modelopt_mxfp8` | 需要按安装版本确认 |

## 6. CI 建议

PR smoke：

```bash
TASKS=arc_challenge,hellaswag,piqa,winogrande \
LIMIT=100 \
NUM_FEWSHOT=0 \
MODEL_DIR=/path/to/modelopt_model \
./lm-eval-example/scripts/run_hf_modelopt.sh
```

Nightly：

```bash
TASKS=arc_challenge,hellaswag,piqa,winogrande,boolq,mmlu \
LIMIT=200 \
MODEL_ROOT=artifacts/models \
BACKENDS="hf vllm" \
QUANT_FORMATS="bf16 fp8 nvfp4" \
./lm-eval-example/scripts/run_modelopt_matrix.sh
```

Release：

```bash
TASKS=mmlu,arc_challenge,hellaswag,winogrande,truthfulqa_mc2,gsm8k \
LIMIT=none \
MODEL_DIR=/path/to/release_model \
./lm-eval-example/scripts/run_vllm_modelopt.sh
```

CI 门禁建议不要直接解析 stdout 表格。更稳的做法是读取 `--output_path` 下的 JSON，或者使用 Python API 脚本直接写固定路径 JSON，再用阈值脚本判断是否失败。
