#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export QWEN3_USE_C=1
for version in baseline v1 v2 v3; do
    for backend in pa sdpa; do
        echo
        echo "###############################"
        echo "## ${version}_${backend}"
        echo "###############################"
        python3 scripts/working/probe_pa_path.py --version "$version" --backend "$backend"
    done
done
