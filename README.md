# lm-eval ModelOpt Examples

This directory contains examples for evaluating ModelOpt-style quantized models
with `lm-evaluation-harness`.

- Chinese guide: [README.zh-CN.md](README.zh-CN.md)
- English guide: [README.en.md](README.en.md)
- Shell scripts: [scripts/](scripts/)
- Python API example: [python/simple_eval_modelopt_hf.py](python/simple_eval_modelopt_hf.py)
- Fixed sample example: [config/sample_indices.json](config/sample_indices.json)
- Bash command cookbook:
  [中文](docs/modelopt-checkpoint-eval-commands.zh-CN.md) /
  [English](docs/modelopt-checkpoint-eval-commands.en.md)
- Qwen3.5 ModelOpt scenario:
  [中文](docs/qwen35-modelopt-scenario.zh-CN.md) /
  [English](docs/qwen35-modelopt-scenario.en.md)
- Practical experiment flow:
  [中文](docs/modelopt-actual-experiment.zh-CN.md) /
  [English](docs/modelopt-actual-experiment.en.md)

The examples focus on single-process CI/CD usage. The `vllm` and `sglang`
examples use their offline engines through `lm-eval`; they do not start a
separate OpenAI-compatible server.
