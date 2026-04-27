#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_DIR:?Set MODEL_DIR to a local checkpoint directory}"

python - "$MODEL_DIR" <<'PY'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1]).resolve()
if not model_dir.is_dir():
    raise SystemExit(f"MODEL_DIR does not exist or is not a directory: {model_dir}")

quant_cfg_path = model_dir / "hf_quant_config.json"
config_path = model_dir / "config.json"

def load_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

quant_cfg = load_json(quant_cfg_path)
config = load_json(config_path)
dirname = model_dir.name.lower()

quant_algo = None
kv_cache_quant_algo = None
group_size = None
exclude_modules = []
producer = None
if quant_cfg:
    producer = (quant_cfg.get("producer") or {}).get("name")
    quantization = quant_cfg.get("quantization") or {}
    quant_algo = quantization.get("quant_algo")
    kv_cache_quant_algo = quantization.get("kv_cache_quant_algo")
    group_size = quantization.get("group_size")
    exclude_modules = quantization.get("exclude_modules") or []

if not quant_algo:
    if "nvfp4" in dirname or "fp4" in dirname:
        quant_algo = "NVFP4"
    elif "fp8" in dirname:
        quant_algo = "FP8"
    elif "awq" in dirname:
        quant_algo = "AWQ"
    elif "bf16" in dirname:
        quant_algo = "BF16"
    elif "fp16" in dirname:
        quant_algo = "FP16"
    else:
        quant_algo = "unknown"

algo_upper = str(quant_algo).upper()
if "NVFP4" in algo_upper or "FP4" in algo_upper:
    vllm_quant = "modelopt_fp4"
    sglang_quant = "modelopt_fp4"
elif "MXFP8" in algo_upper:
    vllm_quant = "modelopt_mxfp8"
    sglang_quant = ""
elif "FP8" in algo_upper:
    vllm_quant = "modelopt"
    sglang_quant = "modelopt_fp8"
elif "AWQ" in algo_upper:
    vllm_quant = "awq"
    sglang_quant = "awq"
else:
    vllm_quant = ""
    sglang_quant = ""

safetensors = sorted(model_dir.glob("*.safetensors"))
has_tokenizer = any((model_dir / name).exists() for name in (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
))

print(f"MODEL_DIR={model_dir}")
print(f"HAS_CONFIG_JSON={'yes' if config else 'no'}")
print(f"HAS_HF_QUANT_CONFIG={'yes' if quant_cfg else 'no'}")
print(f"PRODUCER={producer or ''}")
print(f"QUANT_ALGO={quant_algo}")
print(f"KV_CACHE_QUANT_ALGO={kv_cache_quant_algo or ''}")
print(f"GROUP_SIZE={group_size or ''}")
print(f"EXCLUDE_MODULES_COUNT={len(exclude_modules)}")
print(f"SAFETENSORS_FILES={len(safetensors)}")
print(f"HAS_TOKENIZER={'yes' if has_tokenizer else 'no'}")
print()
print("# Suggested backend arguments")
print(f"export MODEL_DIR='{model_dir}'")
print(f"export VLLM_QUANTIZATION='{vllm_quant}'")
print(f"export SGLANG_QUANTIZATION='{sglang_quant}'")
if config and config.get("model_type"):
    print(f"# model_type={config['model_type']}")
if quant_cfg and __import__("os").environ.get("PRINT_HF_QUANT_CONFIG") == "1":
    print("# hf_quant_config.json:")
    print(json.dumps(quant_cfg, indent=2, ensure_ascii=False))
PY
