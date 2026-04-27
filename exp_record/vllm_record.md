# Gemma4 NVFP4 Backend Test Record

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
- Transformers: `5.5.4`
- PyTorch: `2.10.0+cu129`
- nvidia-modelopt: `0.44.0rc1`
- SGLang: not installed in the base Python environment
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

Full server command for the local checkpoint:

```bash
vllm serve artifacts/models/Gemma-4-31B-IT-NVFP4 \
  --quantization modelopt_fp4 \
  --tensor-parallel-size 8 \
  --dtype auto \
  --trust-remote-code \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --host 0.0.0.0 \
  --port 8000
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

## HF Backend Status

The pure Transformers/HF path was tested directly. It did not use vLLM, SGLang, `lm-eval`, or repository wrapper scripts.

Command:

```bash
CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_VERBOSITY=info python -c '
import time
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

model_dir = "artifacts/models/Gemma-4-31B-IT-NVFP4"
question = "LLM 的 NVFP4 权重量化是什么？用一句中文回答。"
print("HF_SMOKE_START", flush=True)
print("torch", torch.__version__, flush=True)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
prompt = tok.apply_chat_template(
    [{"role": "user", "content": question}],
    tokenize=False,
    add_generation_prompt=True,
)
print("PROMPT_CHARS", len(prompt), flush=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_dir,
    torch_dtype="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
).to("cuda:0")
model.eval()
print("LOAD_SECONDS", round(time.time() - t0, 2), flush=True)
inputs = tok(prompt, return_tensors="pt").to("cuda:0")
t1 = time.time()
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=48, do_sample=False)
print("GEN_SECONDS", round(time.time() - t1, 2), flush=True)
print("OUTPUT_START")
print(tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip())
print("OUTPUT_END")
'
```

Result: failed during model load.

Important log lines:

```text
Unknown quantization type, got modelopt - supported types are: [...]
Hence, we will skip the quantization.
```

Then Transformers reported ModelOpt/NVFP4 checkpoint tensors that do not match the unquantized Gemma4 module shapes:

```text
model.language_model.layers.{0...59}.mlp.up_proj.weight   | MISMATCH | ckpt: torch.Size([21504, 2688]) vs model: torch.Size([21504, 5376])
model.language_model.layers.{0...59}.mlp.down_proj.weight | MISMATCH | ckpt: torch.Size([5376, 10752]) vs model: torch.Size([5376, 21504])
model.language_model.layers.{0...59}.mlp.gate_proj.weight | MISMATCH | ckpt: torch.Size([21504, 2688]) vs model: torch.Size([21504, 5376])
RuntimeError: You set `ignore_mismatched_sizes` to `False`, thus raising an error.
```

Conclusion: in this environment, pure Transformers/HF cannot validly load this ModelOpt NVFP4 checkpoint directly. Setting `ignore_mismatched_sizes=True` would reinitialize mismatched weights, so it should not be used for a real benchmark. Use vLLM for this artifact, or use a ModelOpt/HF restore path that explicitly supports this checkpoint format.

If `lm_eval` is later installed, the expanded HF backend command would be:

```bash
lm-eval run \
  --model hf \
  --model_args pretrained=artifacts/models/Gemma-4-31B-IT-NVFP4,dtype=auto,trust_remote_code=True \
  --tasks arc_challenge \
  --batch_size 1 \
  --num_fewshot 0 \
  --limit 1 \
  --output_path artifacts/eval/hf-gemma4-nvfp4-smoke
```

Expected status for that command in the current environment: fail for the same ModelOpt quantization and MLP shape mismatch reasons, unless the HF loading path is changed.

## SGLang Backend Status

SGLang is not installed in the base Python environment.

Commands checked:

```bash
which sglang
python -c "import sglang; print(getattr(sglang, '__version__', 'unknown')); print(sglang.__file__)"
python -m sglang.launch_server --help
```

Observed result:

```text
ModuleNotFoundError: No module named 'sglang'
```

So no SGLang load/generate smoke was completed in the base environment.

Once SGLang is installed, use an expanded server command like this:

```bash
python -m sglang.launch_server \
  --model-path artifacts/models/Gemma-4-31B-IT-NVFP4 \
  --quantization modelopt_fp4 \
  --tp-size 1 \
  --dtype auto \
  --trust-remote-code \
  --context-length 4096 \
  --mem-fraction-static 0.85 \
  --host 0.0.0.0 \
  --port 30000
```

Expanded `lm-eval` command for the SGLang backend once both `sglang` and `lm_eval` are available:

```bash
lm-eval run \
  --model sglang \
  --model_args pretrained=artifacts/models/Gemma-4-31B-IT-NVFP4,dtype=auto,trust_remote_code=True,tp_size=1,dp_size=1,max_model_len=4096,quantization=modelopt_fp4 \
  --tasks arc_challenge \
  --batch_size auto \
  --num_fewshot 0 \
  --limit 1 \
  --output_path artifacts/eval/sglang-gemma4-nvfp4-smoke
```

## lm-eval Status

The current Python environment does not have `lm_eval` installed:

```text
lm-eval: command not found
ModuleNotFoundError: No module named 'lm_eval'
```

So the completed validation is vLLM load plus generation smoke test, not an `lm-eval` benchmark run.

When `lm_eval` is available, run the smoke test with the expanded command line below. This is the direct form of the vLLM backend invocation; it does not rely on repository wrapper scripts.

```bash
lm-eval run \
  --model vllm \
  --model_args pretrained=artifacts/models/Gemma-4-31B-IT-NVFP4,dtype=auto,trust_remote_code=True,tensor_parallel_size=1,data_parallel_size=1,gpu_memory_utilization=0.85,max_model_len=4096,quantization=modelopt_fp4 \
  --tasks arc_challenge \
  --batch_size auto \
  --num_fewshot 0 \
  --limit 1 \
  --output_path artifacts/eval/vllm-gemma4-nvfp4-smoke
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
