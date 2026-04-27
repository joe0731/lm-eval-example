# HF Record: Gemma4 NVFP4

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
- Transformers: `5.5.4`
- PyTorch: `2.10.0+cu129`
- nvidia-modelopt: `0.44.0rc1`

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

## Pure Transformers Smoke Test

This test used pure Transformers/HF. It did not use vLLM, SGLang, `lm-eval`, or repository wrapper scripts.

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

## HF lm-eval Command

The current environment does not have `lm_eval` installed:

```text
lm-eval: command not found
ModuleNotFoundError: No module named 'lm_eval'
```

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
