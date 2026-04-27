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

Initial status before this test:

```text
lm-eval: command not found
ModuleNotFoundError: No module named 'lm_eval'
```

The local `lm-evaluation-harness` tree was installed as an editable package:

```bash
cd /localhome/swqa/workspace/aaa/lm-evaluation-harness
python -m pip install -e . --no-deps
```

Missing runtime dependencies were then installed:

```bash
cd /localhome/swqa/workspace/aaa/lm-evaluation-harness
python -m pip install 'evaluate>=0.4.0' jsonlines pytablewriter 'rouge-score>=0.0.4' 'sacrebleu>=1.5.0' sqlitedict word2number zstandard
python -m pip install 'scikit-learn>=0.24.1'
```

Pip warning from the dependency install:

```text
colossus-cli 2.3.0 requires python-dateutil==2.8.1, but python-dateutil 2.9.0.post0 was installed.
colossus-cli 2.3.0 requires requests==2.25.1, but requests 2.33.1 was installed.
```

After the lm-eval HF smoke test, those two packages were restored to their original pinned versions:

```bash
python -m pip install requests==2.25.1 python-dateutil==2.8.1
```

Pip then reported the environment's opposite compatibility warnings:

```text
datasets 4.8.4 requires requests>=2.32.2, but requests 2.25.1 is installed.
pandas 3.0.2 requires python-dateutil>=2.8.2, but python-dateutil 2.8.1 is installed.
vllm 0.19.1 requires requests>=2.26.0, but requests 2.25.1 is installed.
tiktoken 0.12.0 requires requests>=2.26.0, but requests 2.25.1 is installed.
```

Expanded HF backend smoke command:

```bash
lm-eval run \
  --model hf \
  --model_args pretrained=artifacts/models/Gemma-4-31B-IT-NVFP4 dtype=auto trust_remote_code=True \
  --tasks arc_challenge \
  --batch_size 1 \
  --num_fewshot 0 \
  --limit 1 \
  --output_path artifacts/eval/hf-gemma4-nvfp4-lmeval-smoke
```

Actual result:

```text
2026-04-27:10:15:59 INFO [evaluator:238] Initializing hf model, with arguments: {'pretrained': 'artifacts/models/Gemma-4-31B-IT-NVFP4', 'dtype': 'auto', 'trust_remote_code': True}
2026-04-27:10:16:02 INFO [models.huggingface:256] Using device 'cuda:0'
ValueError: Unknown quantization type, got modelopt - supported types are: ['awq', 'bitsandbytes_4bit', 'bitsandbytes_8bit', 'gptq', 'aqlm', 'quanto', 'quark', 'fouroversix', 'fp_quant', 'eetq', 'higgs', 'hqq', 'compressed-tensors', 'fbgemm_fp8', 'torchao', 'bitnet', 'vptq', 'spqr', 'fp8', 'auto-round', 'mxfp4', 'metal', 'sinq']
```

This means `lm-eval --model hf` fails before safetensor weight loading. It stops while parsing `config.json` `quantization_config` through Transformers `AutoQuantizationConfig`.

Do not put `device=cpu` inside `--model_args` for this local CLI. It caused:

```text
TypeError: lm_eval.models.huggingface.HFLM() got multiple values for keyword argument 'device'
```

## lm-eval HF Backend Support Investigation

Question: how can `lm-eval --model hf` support a ModelOpt quantized model?

Local source inspected:

```bash
cd /localhome/swqa/workspace/aaa/lm-evaluation-harness
rg -n "quantization_config|AutoQuantizationConfig|from_pretrained|AUTO_MODEL_CLASS|HFLM" lm_eval/models/huggingface.py
sed -n '280,345p' lm_eval/models/huggingface.py
sed -n '713,770p' lm_eval/models/huggingface.py
```

Observed lm-eval HF loading path:

```text
lm_eval/models/huggingface.py reads config.quantization_config.
If it is a dict, it calls transformers.quantizers.AutoQuantizationConfig.from_dict(...).
Then it calls self.AUTO_MODEL_CLASS.from_pretrained(..., quantization_config=quantization_config, ...).
```

Transformers support check in this environment:

```bash
python -c "from transformers.quantizers import AutoQuantizationConfig; cfg={'quant_method':'modelopt','quant_algo':'NVFP4'}; print(AutoQuantizationConfig.from_dict(cfg))"
```

Result:

```text
ValueError: Unknown quantization type, got modelopt - supported types are: ['awq', 'bitsandbytes_4bit', 'bitsandbytes_8bit', 'gptq', 'aqlm', 'quanto', 'quark', 'fouroversix', 'fp_quant', 'eetq', 'higgs', 'hqq', 'compressed-tensors', 'fbgemm_fp8', 'torchao', 'bitnet', 'vptq', 'spqr', 'fp8', 'auto-round', 'mxfp4', 'metal', 'sinq']
```

The downloaded checkpoint layout was checked:

```bash
find /localhome/swqa/workspace/aaa/lm-eval-example/artifacts/models/Gemma-4-31B-IT-NVFP4 -maxdepth 1 -type f -printf '%f\n'
```

Relevant files:

```text
config.json
hf_quant_config.json
model-00001-of-00004.safetensors
model-00002-of-00004.safetensors
model-00003-of-00004.safetensors
model-00004-of-00004.safetensors
model.safetensors.index.json
```

Important absence:

```text
modelopt_state.pth is not present.
```

ModelOpt HF restore path inspected:

```bash
sed -n '1,220p' /usr/local/lib/python3.12/dist-packages/modelopt/torch/opt/plugins/huggingface.py
sed -n '1,140p' /usr/local/lib/python3.12/dist-packages/modelopt/torch/opt/plugins/transformers.py
sed -n '580,630p' /usr/local/lib/python3.12/dist-packages/modelopt/torch/quantization/conversion.py
```

Observed ModelOpt behavior:

```text
modelopt.torch.opt.enable_huggingface_checkpointing() patches HuggingFace from_pretrained/save_pretrained.
The restore trigger is pretrained_model_name_or_path/modelopt_state.pth.
If modelopt_state.pth exists, ModelOpt calls restore_from_modelopt_state(...).
For this checkpoint, modelopt_state.pth does not exist, so that restore path has nothing to load.
The installed ModelOpt package also has restore_export_quantized_model(...) as NotImplementedError.
```

Conclusion for this specific `nvidia/Gemma-4-31B-IT-NVFP4` artifact:

```text
lm-eval --model hf cannot currently load this exported ModelOpt NVFP4 checkpoint by arguments alone.
There are two blockers:
1. lm-eval/Transformers rejects quant_method=modelopt in config.quantization_config.
2. Even if that parse is bypassed, pure Transformers builds unquantized Gemma4 Linear modules, then the packed NVFP4 MLP tensors mismatch the unquantized shapes.
```

Valid support options:

```text
Option A: Use lm-eval vLLM backend for this artifact.
This is the working path already tested with quantization=modelopt_fp4.

Option B: Use lm-eval HF only for ModelOpt checkpoints saved with modelopt_state.pth.
That path requires enabling ModelOpt HuggingFace checkpointing before model load, then passing the restored model object to HFLM through the Python API or adding a small lm-eval HF hook that calls modelopt.torch.opt.enable_huggingface_checkpointing() before from_pretrained.

Option C: Add a new lm-eval backend/wrapper for exported ModelOpt NVFP4 checkpoints.
It would need a loader that understands hf_quant_config.json and packed NVFP4 tensors, equivalent in responsibility to vLLM/SGLang ModelOpt loaders. This is not a small model_args-only change to the existing HF backend.

Option D: Convert/export the checkpoint into a Transformers-supported quantization format, or evaluate an unquantized/bfloat16 checkpoint with the HF backend.
```

Do not use:

```text
ignore_mismatched_sizes=True
```

Reason: it would reinitialize mismatched MLP weights and the benchmark would no longer evaluate the downloaded quantized model.
