# Practical ModelOpt Quantized Checkpoint Experiment Flow

This flow is not just a "can lm-eval start" check. It turns ModelOpt quantized
checkpoints into a repeatable, comparable, CI-friendly evaluation workflow:

1. Verify checkpoint metadata and quantization format.
2. Verify that the target backend can load and generate.
3. Compare quantized artifacts against a bf16/fp16 baseline with fixed tasks and samples.
4. Save, summarize, and gate results in CI.

## Main Takeaways

The practical CI/CD conclusion is to use the vLLM offline backend as the primary
path for ModelOpt pre-quantized artifacts, keep HF as a reference path only when
it can directly load the artifact, and treat SGLang as a second backend after a
model/version smoke test. This keeps execution single-process, avoids a separate
server, and makes failures attributable to checkpoint metadata, backend support,
or the benchmark itself.

Recommended execution chain:

```text
inspect checkpoint -> one-sample smoke -> fixed-task CI smoke -> nightly/release comparison
```

For ModelOpt pre-quantized artifacts, the practical default path is usually:

| Artifact format | Recommended backend | Typical quantization argument |
| --- | --- | --- |
| bf16 | `hf` or `vllm` | none |
| fp16 | `hf` or `vllm` | none |
| fp8 | `vllm` | `quantization=modelopt` |
| int8_awq | `vllm` | `quantization=awq` |
| nvfp4 | `vllm` | `quantization=modelopt_fp4` |
| mxfp8 | `vllm` | `quantization=modelopt_mxfp8` |

`lm-eval --model vllm` uses the offline engine path. It creates `vllm.LLM(...)`
inside the current process and does not require a separate server.

The HF backend is only useful when Transformers can directly load the exported
artifact. In the recorded `nvidia/Gemma-4-31B-IT-NVFP4` test, HF/lm-eval failed
because Transformers did not recognize `quant_method=modelopt`, and pure
Transformers loading also hit shape mismatches between quantized tensors and the
unquantized model shape. Do not use `ignore_mismatched_sizes=True` for
benchmarking because it may reinitialize weights.

SGLang can be a second high-throughput path, but it must be validated per model
and version. In the recorded Gemma4 NVFP4 test, SGLang loaded the checkpoint but
failed during generation, so it was not a reliable benchmark backend for that
artifact.

## 0. Assumed Layout

Assume local Qwen3.5 ModelOpt artifacts:

```text
/data/models/qwen3.5-modelopt/
  bf16/
  fp16/
  fp8/
  int8_awq/
  nvfp4/
```

Run commands from this example repository:

```bash
cd /localhome/swqa/workspace/aaa/lm-eval-example
export MODEL_ROOT=/data/models/qwen3.5-modelopt
```

## 1. Record Versions

Record the environment before every formal experiment:

```bash
python -m pip freeze | grep -E 'lm-eval|lm_eval|vllm|sglang|nvidia-modelopt|torch|transformers'
nvidia-smi
```

Save this output to `artifacts/eval/<run>/versions.txt`.

## 2. Inspect Checkpoints

Inspect metadata before running benchmarks:

```bash
for format in bf16 fp16 fp8 int8_awq nvfp4; do
  MODEL_DIR="${MODEL_ROOT}/${format}" ./scripts/inspect_modelopt_checkpoint.sh
done
```

Check:

- `PRODUCER` is `modelopt`.
- `QUANT_ALGO` matches the directory name, such as `NVFP4`, `FP8`, or `AWQ`.
- Suggested `VLLM_QUANTIZATION` and `SGLANG_QUANTIZATION`.
- tokenizer/config/safetensors files exist.

## 3. Single-Model Smoke

The smoke phase only proves load, run, and result writing. It is not meant to be
statistically meaningful.

bf16 baseline:

```bash
MODEL_DIR="${MODEL_ROOT}/bf16" \
DTYPE=bfloat16 \
TASKS=arc_challenge \
LIMIT=1 \
BATCH_SIZE=1 \
OUTPUT_DIR=artifacts/eval/smoke/hf-bf16 \
./scripts/run_hf_modelopt.sh
```

NVFP4 artifact:

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

FP8 artifact:

```bash
MODEL_DIR="${MODEL_ROOT}/fp8" \
VLLM_QUANTIZATION=modelopt \
TASKS=arc_challenge \
LIMIT=1 \
OUTPUT_DIR=artifacts/eval/smoke/vllm-fp8 \
./scripts/run_vllm_modelopt.sh
```

If smoke fails, fix backend versions or quantization arguments before running
formal evaluation.

## 4. CI Smoke Experiment

For PR-level CI, use a few mainstream tasks and a small sample count:

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

If GPU time is tight, start with:

```bash
TASKS="arc_challenge,piqa" LIMIT=20
```

This layer should not use strict quality gates. Practical gates are:

- Each format loads.
- Each task has numeric results.
- Scores are not obviously abnormal, such as near zero or far outside the historical range.

## 5. Nightly Experiment

Nightly runs establish quantization quality trends:

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

For more stable regression comparison, prefer a fixed sample file over a plain
`--limit`:

```bash
MODEL_ROOT="${MODEL_ROOT}" \
FORMATS="bf16 fp16 fp8 int8_awq nvfp4" \
BACKENDS="vllm" \
TASKS="arc_challenge,hellaswag,piqa,winogrande" \
SAMPLES_JSON="$(cat config/sample_indices.json)" \
OUTPUT_ROOT=artifacts/eval/nightly-fixed \
./scripts/run_qwen35_modelopt_scenario.sh
```

## 6. Release Experiment

Run full tasks for release or scheduled full evaluation, not every PR:

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

`gsm8k` is generative, slower, and more sensitive to prompt/chat-template
choices. For early quantization regression, prefer
multiple-choice/loglikelihood tasks first.

## 7. Summarize Results

`lm-eval` writes `results*.json` files under `OUTPUT_ROOT`. Summarize them with:

```bash
python scripts/summarize_lm_eval_results.py artifacts/eval/ci-smoke
```

CSV output:

```bash
python scripts/summarize_lm_eval_results.py artifacts/eval/ci-smoke --format csv \
  > artifacts/eval/ci-smoke/summary.csv
```

All numeric metrics:

```bash
python scripts/summarize_lm_eval_results.py artifacts/eval/ci-smoke --all-metrics
```

## 8. CI Threshold Guidance

Use bf16 or fp16 on the same backend as the baseline. Do not attribute
cross-backend differences directly to quantization.

A practical starting policy:

| Stage | Gate |
| --- | --- |
| PR smoke | load succeeds, tasks finish, metrics exist |
| Nightly | quantized-vs-baseline delta warns |
| Release | delta beyond threshold fails |

Do not hard-code thresholds too early. Keep 3 to 5 stable experiment runs first,
then tune thresholds per model and task. Starting warning ranges:

- FP8: warn if most tasks drop more than 1 to 2 points vs bf16/fp16.
- INT8 AWQ: warn if most tasks drop more than 2 to 3 points.
- NVFP4: warn if most tasks drop more than 3 to 5 points.

These are not universal standards. Your own model, tasks, and product tolerance
should define the final CI gates.

## 9. Findings From the Recorded Experiment

The repository's `exp_record/` directory records one real ModelOpt NVFP4 artifact test:

- HF/lm-eval: failed because Transformers did not recognize the `modelopt` quantization config.
- vLLM offline: loaded and generated successfully with `quantization=modelopt_fp4`.
- SGLang: loaded the checkpoint, but Gemma4 generation failed in the tested environment.

For ModelOpt NVFP4 artifacts, the practical path is therefore:

```bash
inspect checkpoint -> vLLM smoke -> vLLM CI smoke -> vLLM nightly/release
```

Use HF only when it can directly load the artifact. Use SGLang only after a
version-specific smoke test passes.
