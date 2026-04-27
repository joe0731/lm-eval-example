#!/usr/bin/env bash
set -euo pipefail

TASKS="${TASKS:-arc_challenge,hellaswag,piqa,winogrande}"

lm-eval validate --tasks "${TASKS}"

