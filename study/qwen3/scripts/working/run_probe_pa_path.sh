#!/usr/bin/env bash
# Probe whether VLMPipeline's baseline-vs-v3 gap is the paged-attention
# fast path vs stateful-fallback path. Each row is a fresh process with a
# wiped compile cache.
set -euo pipefail
cd "$(dirname "$0")/../.."
export QWEN3_USE_C=1
for cfg in baseline_pa baseline_sdpa v3_pa v3_sdpa; do
    echo
    echo "###############################"
    echo "## $cfg"
    echo "###############################"
    python3 scripts/working/probe_pa_path.py --config "$cfg"
done
