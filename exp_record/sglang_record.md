# SGLang Record: Gemma4 NVFP4

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
- Base environment SGLang: not installed
- Temporary venv: `/tmp/sglang-venv`
- SGLang in venv: `0.5.10.post1`
- Torch in venv: `2.9.1+cu128`
- Transformers in venv after upgrade: `5.5.4`

## Download

The model was downloaded once and reused locally:

```bash
hf download nvidia/Gemma-4-31B-IT-NVFP4 \
  --local-dir artifacts/models/Gemma-4-31B-IT-NVFP4 \
  --max-workers 8
```

Downloaded size:

```text
31G artifacts/models/Gemma-4-31B-IT-NVFP4
```

## Base Environment Check

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

SGLang was not available in the base Python environment.

## Temporary Install

I installed SGLang into an isolated temporary venv instead of changing the base vLLM environment.

Commands:

```bash
python -m venv --system-site-packages /tmp/sglang-venv
```

```bash
/tmp/sglang-venv/bin/python -m pip install -U pip setuptools wheel
```

```bash
/tmp/sglang-venv/bin/python -m pip install "sglang[all]"
```

Installed version:

```text
sglang 0.5.10.post1
torch 2.9.1+cu128
transformers 5.3.0
```

Two runtime fixes were needed before testing:

1. `torch` initially failed to import because `libcudnn.so.9` was not on the runtime library path.
2. `flashinfer` initially failed because global `flashinfer-jit-cache 0.6.6+cu129` did not match venv `flashinfer 0.6.7.post3`.

The matching `flashinfer-jit-cache==0.6.7.post3` wheel was not available:

```bash
/tmp/sglang-venv/bin/python -m pip install flashinfer-jit-cache==0.6.7.post3
```

Result:

```text
ERROR: No matching distribution found for flashinfer-jit-cache==0.6.7.post3
```

So the test used `FLASHINFER_DISABLE_VERSION_CHECK=1`, as suggested by the FlashInfer error message.

The SGLang install pinned `transformers==5.3.0`, which does not recognize `model_type=gemma4`. The first real SGLang model attempt failed with:

```text
ValueError: The checkpoint you are trying to load has model type `gemma4` but Transformers does not recognize this architecture.
```

I then upgraded Transformers inside the temporary venv:

```bash
/tmp/sglang-venv/bin/python -m pip install transformers==5.5.4
```

This violates SGLang's package pin (`sglang 0.5.10.post1 requires transformers==5.3.0`), but it was necessary to test the Gemma4 checkpoint at all.

## SGLang Engine Smoke Test

Runtime library path used for the venv:

```bash
LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusparselt/lib:/tmp/sglang-venv/lib/python3.12/site-packages/nvidia/cublas/lib:/tmp/sglang-venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/tmp/sglang-venv/lib/python3.12/site-packages/nvidia/cuda_cupti/lib:/tmp/sglang-venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:/tmp/sglang-venv/lib/python3.12/site-packages/nvidia/cufft/lib:/tmp/sglang-venv/lib/python3.12/site-packages/nvidia/curand/lib:/tmp/sglang-venv/lib/python3.12/site-packages/nvidia/cusolver/lib:/tmp/sglang-venv/lib/python3.12/site-packages/nvidia/cusparse/lib:/tmp/sglang-venv/lib/python3.12/site-packages/nvidia/cufile/lib:/tmp/sglang-venv/lib/python3.12/site-packages/nvidia/nvjitlink/lib:/tmp/sglang-venv/lib/python3.12/site-packages/nvidia/nvtx/lib:/tmp/sglang-venv/lib/python3.12/site-packages/nvidia/nvshmem/lib
```

Command:

```bash
CUDA_VISIBLE_DEVICES=0 FLASHINFER_DISABLE_VERSION_CHECK=1 LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusparselt/lib:/tmp/sglang-venv/lib/python3.12/site-packages/nvidia/cublas/lib:/tmp/sglang-venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/tmp/sglang-venv/lib/python3.12/site-packages/nvidia/cuda_cupti/lib:/tmp/sglang-venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:/tmp/sglang-venv/lib/python3.12/site-packages/nvidia/cufft/lib:/tmp/sglang-venv/lib/python3.12/site-packages/nvidia/curand/lib:/tmp/sglang-venv/lib/python3.12/site-packages/nvidia/cusolver/lib:/tmp/sglang-venv/lib/python3.12/site-packages/nvidia/cusparse/lib:/tmp/sglang-venv/lib/python3.12/site-packages/nvidia/cufile/lib:/tmp/sglang-venv/lib/python3.12/site-packages/nvidia/nvjitlink/lib:/tmp/sglang-venv/lib/python3.12/site-packages/nvidia/nvtx/lib:/tmp/sglang-venv/lib/python3.12/site-packages/nvidia/nvshmem/lib /tmp/sglang-venv/bin/python -c '
import time
import transformers
from transformers import AutoTokenizer
from sglang.srt.entrypoints.engine import Engine

model = "artifacts/models/Gemma-4-31B-IT-NVFP4"
question = "LLM 的 NVFP4 权重量化是什么？用一句中文回答。"
print("SGLANG_SMOKE_START", flush=True)
print("transformers", transformers.__version__, flush=True)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
prompt = tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)
print("PROMPT_CHARS", len(prompt), flush=True)
engine = None
try:
    engine = Engine(
        model_path=model,
        quantization="modelopt_fp4",
        dtype="auto",
        trust_remote_code=True,
        context_length=2048,
        mem_fraction_static=0.5,
        tp_size=1,
        disable_cuda_graph=True,
        log_level="info",
    )
    print("LOAD_SECONDS", round(time.time() - t0, 2), flush=True)
    t1 = time.time()
    out = engine.generate(prompt=prompt, sampling_params={"temperature": 0.0, "max_new_tokens": 48})
    print("GEN_SECONDS", round(time.time() - t1, 2), flush=True)
    print("OUTPUT_START")
    print(out)
    print("OUTPUT_END")
finally:
    if engine is not None:
        engine.shutdown()
'
```

Result: model load succeeded, generation failed.

Important load logs:

```text
Gemma4ForConditionalGeneration has no SGLang implementation, falling back to Transformers implementation.
Using ModelOptModelLoader due to ModelOpt quantization config.
Detected nvfp4 checkpoint. Please note that the format is experimental and subject to change.
Load weight end. elapsed=14.05 s, type=TransformersMultiModalForCausalLM, quant=modelopt_fp4, quant_algo=NVFP4, avail mem=60.00 GB, mem usage=33.43 GB.
Using KV cache dtype: torch.float8_e4m3fn
KV Cache is allocated. #tokens: 28938, K size: 6.62 GB, V size: 6.62 GB
LOAD_SECONDS 31.36
```

Generation failed in the scheduler:

```text
ValueError: Mismatched mW.shape[0] on argument #1 when calling:
`__call__(mX: Tensor([n0, 256], bfloat16), mW: Tensor([256], bfloat16), mY: Tensor([n0, 256], bfloat16), M: int32, eps: float32)`, expected to be 256
```

Conclusion: SGLang was actually installed and exercised in a temporary venv. It can load this Gemma4 NVFP4 checkpoint after upgrading Transformers, but it cannot complete generation in this environment because Gemma4 falls back to the Transformers backend and hits an RMSNorm/FlashInfer kernel shape mismatch.

After the failed run, two SGLang child processes remained as zombies:

```text
193125 [sglang::schedul] <defunct>
193126 [sglang::detoken] <defunct>
```

They were SIGKILLed but remained as `<defunct>` children of PID 1. They did not hold GPU memory; `nvidia-smi` returned to only the pre-existing DeepSeek vLLM workers.

## SGLang Server Command

Expanded server command for this installed SGLang version:

```bash
CUDA_VISIBLE_DEVICES=0 FLASHINFER_DISABLE_VERSION_CHECK=1 LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusparselt/lib:/tmp/sglang-venv/lib/python3.12/site-packages/nvidia/cublas/lib:/tmp/sglang-venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/tmp/sglang-venv/lib/python3.12/site-packages/nvidia/cuda_cupti/lib:/tmp/sglang-venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:/tmp/sglang-venv/lib/python3.12/site-packages/nvidia/cufft/lib:/tmp/sglang-venv/lib/python3.12/site-packages/nvidia/curand/lib:/tmp/sglang-venv/lib/python3.12/site-packages/nvidia/cusolver/lib:/tmp/sglang-venv/lib/python3.12/site-packages/nvidia/cusparse/lib:/tmp/sglang-venv/lib/python3.12/site-packages/nvidia/cufile/lib:/tmp/sglang-venv/lib/python3.12/site-packages/nvidia/nvjitlink/lib:/tmp/sglang-venv/lib/python3.12/site-packages/nvidia/nvtx/lib:/tmp/sglang-venv/lib/python3.12/site-packages/nvidia/nvshmem/lib /tmp/sglang-venv/bin/python -m sglang.launch_server \
  --model-path artifacts/models/Gemma-4-31B-IT-NVFP4 \
  --quantization modelopt_fp4 \
  --tensor-parallel-size 1 \
  --dtype auto \
  --trust-remote-code \
  --context-length 4096 \
  --mem-fraction-static 0.85 \
  --host 0.0.0.0 \
  --port 30000
```

The CLI help for `sglang 0.5.10.post1` lists `modelopt_fp4` as a supported quantization value.

## SGLang lm-eval Command

The current environment also does not have `lm_eval` installed:

```text
lm-eval: command not found
ModuleNotFoundError: No module named 'lm_eval'
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

Expected status in the base environment: fail before model loading because `sglang` is not installed. Expected status in `/tmp/sglang-venv`: model loading can succeed, but generation currently fails with the RMSNorm/FlashInfer shape mismatch described above.
