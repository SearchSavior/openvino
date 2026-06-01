#!/usr/bin/env bash
# Sweep OV compile_model properties for memory-vs-speed tradeoff on raw
# baseline. Each row is a fresh process.
set -euo pipefail
cd "$(dirname "$0")/../.."
export QWEN3_USE_C=1

run() {
    local label="$1"; shift
    local props="$*"
    echo
    echo "###############################"
    echo "## $label"
    echo "###############################"
    python3 scripts/working/probe_raw_lowmem.py --label "$label" --props "$props"
}

# Default knobs (matches probe_raw baseline)
run "default"           ""

# Hints
run "hint_latency"      "PERFORMANCE_HINT=LATENCY"
run "hint_throughput"   "PERFORMANCE_HINT=THROUGHPUT"

# inference precision
run "prec_f32"          "INFERENCE_PRECISION_HINT=f32"
run "prec_f16"          "INFERENCE_PRECISION_HINT=f16"
run "prec_bf16"         "INFERENCE_PRECISION_HINT=bf16"

# KV cache compression
run "kv_u8"             "KV_CACHE_PRECISION=u8"
run "kv_u8_grouped"     "KV_CACHE_PRECISION=u8,DYNAMIC_QUANTIZATION_GROUP_SIZE=32"

# Streams off / single executor
run "streams_1"         "NUM_STREAMS=1"

# Combine: latency + u8 KV + single stream
run "combo"             "PERFORMANCE_HINT=LATENCY,KV_CACHE_PRECISION=u8,NUM_STREAMS=1"

# Memory-leaning combos
run "bf16+lat+streams1"          "INFERENCE_PRECISION_HINT=bf16,PERFORMANCE_HINT=LATENCY,NUM_STREAMS=1"
run "bf16+u8kv+lat"              "INFERENCE_PRECISION_HINT=bf16,KV_CACHE_PRECISION=u8,PERFORMANCE_HINT=LATENCY"
run "ALL"                        "INFERENCE_PRECISION_HINT=bf16,KV_CACHE_PRECISION=u8,PERFORMANCE_HINT=LATENCY,NUM_STREAMS=1,DYNAMIC_QUANTIZATION_GROUP_SIZE=32"
