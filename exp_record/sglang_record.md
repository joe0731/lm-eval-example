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
- SGLang: not installed in the base Python environment

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

## Installed Package Check

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

## SGLang Server Command

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

If the installed SGLang version does not accept `modelopt_fp4`, try `modelopt` only after checking that version's quantization documentation. Do not assume parity with vLLM.

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

Expected status in the base environment: fail before model loading because `sglang` is not installed.
