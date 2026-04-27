# lm-eval + ModelOpt Quantized Model Evaluation Examples

This directory contains `lm-eval` examples focused on ModelOpt-style quantized
model artifacts and CI/CD usage.

- HF backend: best compatibility, good as a correctness/reference path.
- vLLM backend: higher throughput for vLLM-supported ModelOpt checkpoints.
- SGLang backend: higher throughput for SGLang-supported ModelOpt checkpoints.
- Python API: useful when your ModelOpt artifact requires a custom restore path.

## Layout

```text
lm-eval-example/
  README.md
  README.zh-CN.md
  README.en.md
  config/
    sample_indices.json
  python/
    simple_eval_modelopt_hf.py
  scripts/
    run_hf_modelopt.sh
    run_vllm_modelopt.sh
    run_sglang_modelopt.sh
    run_modelopt_matrix.sh
    validate_tasks.sh
```

## Version Validity

These examples are written against the current local repository interface:

- `lm-evaluation-harness` package version: `0.4.12.dev0`
- local git describe: `v0.4.11-24-g620262e0`
- local commit: `620262e0`
- note date: `2026-04-27`

Backend quantization arguments can change across vLLM, SGLang, and ModelOpt
versions. The scripts expose these values as environment variables instead of
hard-coding them into the workflow:

- vLLM ModelOpt documentation currently describes ModelOpt checkpoint detection
  through `hf_quant_config.json` and support for FP8, NVFP4, MXFP8, and related
  formats. NVFP4 commonly uses `quantization=modelopt_fp4`; MXFP8 commonly uses
  `quantization=modelopt_mxfp8`.
- SGLang documentation currently uses `quantization=modelopt_fp8` and
  `quantization=modelopt_fp4` for ModelOpt pre-quantized checkpoints.
- If your installed backend behaves differently, trust the installed version and
  validate first with a small `--limit 10` run.

Pin and record backend versions in CI:

```bash
python -m pip freeze | grep -E 'lm_eval|vllm|sglang|nvidia-modelopt|torch|transformers'
```

References:

- lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness
- vLLM ModelOpt quantization: https://docs.vllm.ai/en/latest/features/quantization/modelopt/
- SGLang quantization: https://docs.sglang.io/docs/advanced_features/quantization
- NVIDIA ModelOpt: https://nvidia.github.io/TensorRT-Model-Optimizer/

## Installation

HF path:

```bash
cd /path/to/lm-evaluation-harness
pip install -e ".[hf]"
```

vLLM path:

```bash
cd /path/to/lm-evaluation-harness
pip install -e ".[vllm]"
```

SGLang must be installed using the official SGLang instructions, then install
the harness:

```bash
cd /path/to/lm-evaluation-harness
pip install -e .
```

## Recommended Task Sets

Fast and mainstream CI smoke:

```bash
arc_challenge,hellaswag,piqa,winogrande
```

Stronger CI or nightly:

```bash
arc_challenge,hellaswag,piqa,winogrande,boolq,mmlu
```

Release or full nightly:

```bash
mmlu,arc_challenge,hellaswag,winogrande,truthfulqa_mc2,gsm8k
```

For early quantization regression, prefer multiple-choice/loglikelihood tasks
because they are more deterministic. `gsm8k`, `ifeval`, and `bbh_cot_zeroshot`
are slower and more sensitive to prompt/chat-template choices.

## Common Controls

```bash
--tasks        # task or group selection
--limit        # sample count per task for CI smoke
--samples      # fixed sample indices; incompatible with --limit
--num_fewshot  # number of few-shot examples, not the number of tests
--output_path  # result output location
```

Fixed-sample example:

```bash
lm-eval run \
  --model hf \
  --model_args pretrained=/path/to/model,dtype=auto,trust_remote_code=True \
  --tasks arc_challenge,hellaswag,piqa,winogrande \
  --samples "$(cat lm-eval-example/config/sample_indices.json)" \
  --output_path artifacts/eval/fixed-samples
```

## 1. HF Backend: Reference Path

Use this when:

- The ModelOpt exported model can be loaded by
  `transformers.AutoModelForCausalLM.from_pretrained()`.
- You want correctness before throughput.
- You need a reference baseline for vLLM or SGLang results.

Run:

```bash
MODEL_DIR=/path/to/modelopt_model \
OUTPUT_DIR=artifacts/eval/hf \
./lm-eval-example/scripts/run_hf_modelopt.sh
```

Common overrides:

```bash
TASKS=arc_challenge,hellaswag,piqa,winogrande \
LIMIT=100 \
BATCH_SIZE=8 \
DTYPE=auto \
MODEL_DIR=/path/to/modelopt_fp8 \
./lm-eval-example/scripts/run_hf_modelopt.sh
```

Do not pass `fp8` or `nvfp4` as the HF `dtype=` value. `dtype=auto`,
`bfloat16`, or `float16` controls model loading/computation dtype. The ModelOpt
quantization format should come from the exported model config or from your
custom loader.

## 2. vLLM Backend: Throughput Path

`lm-eval --model vllm` creates `vllm.LLM(...)` in the current process. It does
not start a separate server.

FP8 example:

```bash
MODEL_DIR=/path/to/modelopt_fp8 \
VLLM_QUANTIZATION=modelopt \
OUTPUT_DIR=artifacts/eval/vllm-fp8 \
./lm-eval-example/scripts/run_vllm_modelopt.sh
```

NVFP4 example:

```bash
MODEL_DIR=/path/to/modelopt_nvfp4 \
VLLM_QUANTIZATION=modelopt_fp4 \
OUTPUT_DIR=artifacts/eval/vllm-nvfp4 \
./lm-eval-example/scripts/run_vllm_modelopt.sh
```

MXFP8 example:

```bash
MODEL_DIR=/path/to/modelopt_mxfp8 \
VLLM_QUANTIZATION=modelopt_mxfp8 \
OUTPUT_DIR=artifacts/eval/vllm-mxfp8 \
./lm-eval-example/scripts/run_vllm_modelopt.sh
```

Tensor parallel example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL_DIR=/path/to/model \
TP_SIZE=4 \
./lm-eval-example/scripts/run_vllm_modelopt.sh
```

Important knobs:

```bash
TP_SIZE=1
GPU_MEMORY_UTILIZATION=0.85
MAX_MODEL_LEN=4096
BATCH_SIZE=auto
VLLM_QUANTIZATION=modelopt_fp4
```

If vLLM can auto-detect the format from `hf_quant_config.json`, leave
`VLLM_QUANTIZATION` empty. If auto-detection fails, pass it explicitly.

## 3. SGLang Backend: Throughput Path

`lm-eval --model sglang` creates `sglang.Engine(...)` in the current process. It
does not start a separate server.

FP8 example:

```bash
MODEL_DIR=/path/to/modelopt_fp8 \
SGLANG_QUANTIZATION=modelopt_fp8 \
OUTPUT_DIR=artifacts/eval/sglang-fp8 \
./lm-eval-example/scripts/run_sglang_modelopt.sh
```

NVFP4/FP4 example:

```bash
MODEL_DIR=/path/to/modelopt_nvfp4 \
SGLANG_QUANTIZATION=modelopt_fp4 \
OUTPUT_DIR=artifacts/eval/sglang-nvfp4 \
./lm-eval-example/scripts/run_sglang_modelopt.sh
```

Common OOM controls:

```bash
MEM_FRACTION_STATIC=0.70
MAX_MODEL_LEN=4096
TP_SIZE=2
BATCH_SIZE=auto
```

SGLang's ModelOpt quantization argument may vary across versions. If
`modelopt_fp8/modelopt_fp4` does not match your installed version, try
`SGLANG_QUANTIZATION=modelopt` or leave it empty and let the backend parse the
model config.

## 4. Python API: Custom ModelOpt Restore

Use the Python API if the ModelOpt artifact cannot be loaded directly by
HF/vLLM/SGLang:

```bash
python lm-eval-example/python/simple_eval_modelopt_hf.py \
  --model-dir /path/to/modelopt_model \
  --tasks arc_challenge,hellaswag,piqa,winogrande \
  --limit 100 \
  --output-json artifacts/eval/python-api/results.json
```

Replace the HF loading block with your internal ModelOpt restore logic. The
final objects must provide:

- `model(input_ids).logits`
- `model.generate(...)`
- a Hugging Face tokenizer or compatible tokenizer object

## 5. Running a Quantization Matrix

Expected directory layout:

```text
artifacts/models/
  bf16/
  fp8/
  nvfp4/
```

Run:

```bash
MODEL_ROOT=artifacts/models \
BACKENDS="hf vllm sglang" \
QUANT_FORMATS="bf16 fp8 nvfp4" \
./lm-eval-example/scripts/run_modelopt_matrix.sh
```

Default mapping:

| Format | HF | vLLM | SGLang |
| --- | --- | --- | --- |
| `bf16` | no quant arg | no quant arg | no quant arg |
| `fp8` | model config | `modelopt` | `modelopt_fp8` |
| `nvfp4` | model config | `modelopt_fp4` | `modelopt_fp4` |
| `mxfp8` | model config | `modelopt_mxfp8` | check installed version |

## 6. CI Suggestions

PR smoke:

```bash
TASKS=arc_challenge,hellaswag,piqa,winogrande \
LIMIT=100 \
NUM_FEWSHOT=0 \
MODEL_DIR=/path/to/modelopt_model \
./lm-eval-example/scripts/run_hf_modelopt.sh
```

Nightly:

```bash
TASKS=arc_challenge,hellaswag,piqa,winogrande,boolq,mmlu \
LIMIT=200 \
MODEL_ROOT=artifacts/models \
BACKENDS="hf vllm" \
QUANT_FORMATS="bf16 fp8 nvfp4" \
./lm-eval-example/scripts/run_modelopt_matrix.sh
```

Release:

```bash
TASKS=mmlu,arc_challenge,hellaswag,winogrande,truthfulqa_mc2,gsm8k \
LIMIT=none \
MODEL_DIR=/path/to/release_model \
./lm-eval-example/scripts/run_vllm_modelopt.sh
```

For gating, avoid parsing the stdout table. Prefer reading the JSON written
under `--output_path`, or use the Python API example to write a fixed JSON path
and apply thresholds from a small CI script.

