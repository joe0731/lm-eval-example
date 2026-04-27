#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import handle_non_serializable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run lm-eval through the Python API for a ModelOpt-style HF checkpoint."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument(
        "--tasks",
        default="arc_challenge,hellaswag,piqa,winogrande",
        help="Comma-separated lm-eval task names.",
    )
    parser.add_argument("--limit", default="100")
    parser.add_argument("--samples-json", default=None)
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--batch-size", default="8")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def parse_limit(value: str):
    if value == "" or value == "none":
        return None
    parsed = float(value)
    if parsed.is_integer() and parsed >= 1:
        return int(parsed)
    return parsed


def torch_dtype(value: str):
    if value == "auto":
        return "auto"
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if value not in mapping:
        raise ValueError(
            f"Unsupported --dtype {value!r}. Use auto, float16, bfloat16, or float32."
        )
    return mapping[value]


def load_model_and_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=args.trust_remote_code,
    )

    # Replace this block with your internal ModelOpt restore path if the artifact
    # is not directly loadable by Transformers.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
    )

    if args.device and args.device != "auto":
        try:
            model = model.to(args.device)
        except ValueError:
            # Quantized or device-mapped models may already be placed correctly.
            pass

    model.eval()
    return model, tokenizer


def main() -> None:
    args = parse_args()
    samples = None
    if args.samples_json:
        samples = json.loads(Path(args.samples_json).read_text(encoding="utf-8"))

    model, tokenizer = load_model_and_tokenizer(args)
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
    )

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=[task.strip() for task in args.tasks.split(",") if task.strip()],
        num_fewshot=args.num_fewshot,
        limit=None if samples is not None else parse_limit(args.limit),
        samples=samples,
        log_samples=False,
        bootstrap_iters=0,
    )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2, default=handle_non_serializable),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

