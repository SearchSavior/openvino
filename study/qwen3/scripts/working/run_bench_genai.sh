#!/usr/bin/env bash
# Sequence the genai VLM bench across configs, each in a FRESH process so
# there's no leftover Core / loaded-.so / weight-prepack state between runs
# (an in-process loop did not isolate cleanly). bench_genai.py wipes the
# compile cache at the start of each run, so every load time is a cold build.
#
# First pass is the stock baseline: VLMPipeline with no rewrites and no
# custom extension .so. Then the three fusions.
#
# Usage:
#   QWEN3_USE_C=1 bash scripts/working/run_bench_genai.sh [image_path]
set -euo pipefail

# cd to study/qwen3 (two levels up from scripts/working/).
cd "$(dirname "$0")/../.."

export QWEN3_USE_C=1
IMAGE="${1:-/tmp/llama.cpp/media/llama1-logo.png}"

for cfg in baseline v1 v2 v3; do
    echo
    echo "################################################################"
    echo "## $cfg"
    echo "################################################################"
    python3 scripts/working/bench_genai.py --config "$cfg" --image "$IMAGE"
done
