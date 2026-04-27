# vLLM NVFP4 Test Record

Date: 2026-04-27

Model: `nvidia/Gemma-4-31B-IT-NVFP4`

Source: https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4

Local path:

```bash
artifacts/models/Gemma-4-31B-IT-NVFP4
```

## Environment

- GPU: 8x NVIDIA RTX PRO 6000 Blackwell
- Driver: `580.82.07`
- CUDA shown by `nvidia-smi`: `13.0`
- Python: `3.12.13`
- vLLM: `0.19.1`
- Hugging Face user: `mymino`, org includes `nvidia`
- Disk before download: about `1.2T` available

Existing GPU process before and after this test:

```text
/usr/bin/python3 -c from _test_utils.deploy_utils import _run_vllm_deploy ...
model: nvidia/DeepSeek-R1-NVFP4
```

I did not stop that process. It used about 850 MiB per GPU and did not block the single-GPU Gemma smoke test.

## Download

Check remote files and sizes without downloading:

```bash
hf download nvidia/Gemma-4-31B-IT-NVFP4 --dry-run --format json
```

Important files reported by dry run:

```text
model-00001-of-00004.safetensors  10.0G
model-00002-of-00004.safetensors  10.0G
model-00003-of-00004.safetensors  10.0G
model-00004-of-00004.safetensors   2.7G
tokenizer.json                    32.2M
hf_quant_config.json               3.7K
config.json                        9.5K
```

Actual download:

```bash
hf download nvidia/Gemma-4-31B-IT-NVFP4 \
  --local-dir artifacts/models/Gemma-4-31B-IT-NVFP4 \
  --max-workers 8
```

Downloaded size:

```text
31G artifacts/models/Gemma-4-31B-IT-NVFP4
```

## Quantization Config

The model includes `hf_quant_config.json`.

Relevant fields:

```json
{
  "producer": {
    "name": "modelopt",
    "version": "0.37.0"
  },
  "quantization": {
    "quant_algo": "NVFP4",
    "kv_cache_quant_algo": "FP8",
    "group_size": 16
  }
}
```

For this vLLM build, the correct quantization argument is:

```text
modelopt_fp4
```

This was confirmed by searching the installed vLLM package. vLLM supports these ModelOpt names:

```text
modelopt
modelopt_fp4
modelopt_mxfp8
modelopt_mixed
```

## Successful Smoke Test

The following test loaded the model on GPU 0 and generated one answer:

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_LOGGING_LEVEL=INFO python -c '
import time
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

model = "artifacts/models/Gemma-4-31B-IT-NVFP4"
question = "用一句中文解释什么是量化模型。"
t0 = time.time()
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
prompt = tok.apply_chat_template(
    [{"role": "user", "content": question}],
    tokenize=False,
    add_generation_prompt=True,
)
print("PROMPT_CHARS", len(prompt), flush=True)
llm = LLM(
    model=model,
    quantization="modelopt_fp4",
    dtype="auto",
    trust_remote_code=True,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.85,
    max_model_len=4096,
)
print("LOAD_SECONDS", round(time.time() - t0, 2), flush=True)
params = SamplingParams(temperature=0.0, max_tokens=64)
t1 = time.time()
out = llm.generate([prompt], params)[0]
print("GEN_SECONDS", round(time.time() - t1, 2), flush=True)
print("OUTPUT_START")
print(out.outputs[0].text.strip())
print("OUTPUT_END")
'
```

Observed result:

```text
Resolved architecture: Gemma4ForConditionalGeneration
quantization=modelopt_fp4
Using fp8 data type to store kv cache
Using NvFp4LinearBackend.FLASHINFER_CUTLASS for NVFP4 GEMM
Loading weights took 9.24 seconds
Model loading took 31.04 GiB memory and 13.31 seconds
GPU KV cache size: 98,912 tokens
LOAD_SECONDS 172.58
GEN_SECONDS 4.20
```

Generated output:

```text
量化模型是将复杂的经济、金融或物理现象，通过数学公式和数据分析，转化为可量化、可计算的数学模型，以便进行客观预测和决策的工具。
```

The first prompt was ambiguous in Chinese, so the model interpreted "量化模型" as a quantitative model. A second prompt was used to test an LLM-specific answer.

## Second Prompt

Command differences:

- `question = "LLM 的 NVFP4 权重量化是什么？用一句中文回答。"`
- `max_model_len=2048`
- `enforce_eager=True`
- `VLLM_LOGGING_LEVEL=WARNING`

Observed result:

```text
LOAD_SECONDS 68.58
GEN_SECONDS 6.98
```

Generated output:

```text
NVFP4 是一种由 NVIDIA 引入的 4 位浮点数格式，通过将权重量化为极低精度的浮点数，在大幅降低模型显存占用和提升推理速度的同时，尽可能地保留
```

The answer was cut because `max_tokens=48`. Increase `max_tokens` for complete text.

## Useful vLLM Notes

Use `modelopt_fp4` for this checkpoint. `modelopt` is commonly used for ModelOpt FP8 checkpoints, while this one is NVFP4.

This checkpoint is multimodal and resolves to `Gemma4ForConditionalGeneration`. Text-only prompts work through the normal chat template.

The model card shows a server example with tensor parallel size 8:

```bash
vllm serve /models/gemma-4-31b-it-nvfp4 \
  --quantization modelopt \
  --tensor-parallel-size 8
```

In this local vLLM `0.19.1` environment, offline inference worked with single-GPU `quantization="modelopt_fp4"`.

The first run spent most time on profiling, torch compile, FlashInfer autotuning, and CUDA graph capture. Later runs can be faster after caches are created.

For quick debugging, `enforce_eager=True` avoids torch compile and CUDA graph capture. It reduces warmup complexity but can make generation slower.

The checkpoint uses FP8 KV cache according to the model config. vLLM logged:

```text
Checkpoint does not provide a q scaling factor. Setting it to k_scale.
Using KV cache scaling factor 1.0 for fp8_e4m3.
```

This is a warning, not a blocker for the smoke test.

After each smoke test the temporary Gemma engine shut down. Final `nvidia-smi` showed only the pre-existing DeepSeek vLLM workers.

## lm-eval Status

The repository contains scripts such as:

```bash
scripts/run_vllm_modelopt.sh
```

But the current Python environment does not have `lm_eval` installed:

```text
lm-eval: command not found
ModuleNotFoundError: No module named 'lm_eval'
```

So the completed validation is vLLM load plus generation smoke test, not an `lm-eval` benchmark run.

When `lm_eval` is available, a small smoke command should look like:

```bash
MODEL_DIR=artifacts/models/Gemma-4-31B-IT-NVFP4 \
VLLM_QUANTIZATION=modelopt_fp4 \
TASKS=arc_challenge \
LIMIT=1 \
TP_SIZE=1 \
MAX_MODEL_LEN=4096 \
OUTPUT_DIR=artifacts/eval/vllm-gemma4-nvfp4-smoke \
./scripts/run_vllm_modelopt.sh
```

## Checklist For Next Run

1. Confirm available disk:

   ```bash
   df -h . /tmp
   ```

2. Confirm GPU state:

   ```bash
   nvidia-smi
   pgrep -af vllm
   ```

3. Confirm vLLM version:

   ```bash
   vllm --version
   ```

4. Use local model path when possible to avoid repeated downloads:

   ```bash
   artifacts/models/Gemma-4-31B-IT-NVFP4
   ```

5. Start with a single-GPU smoke test before server or benchmark runs.

6. If a full server is needed, prefer a separate port and check existing services first.
