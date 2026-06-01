#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export QWEN3_USE_C=1
for version in baseline v1 v2 v3; do
    echo
    echo "###############################"
    echo "## raw  $version"
    echo "###############################"
    python3 scripts/working/probe_raw.py --version "$version"
done
