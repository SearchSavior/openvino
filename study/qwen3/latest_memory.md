# Latest memory work — `GatedDeltaRule` v1 → v2 → v3

A focused update on the most recent work: pushing the **linear-attention
layer's per-call activation footprint** down by progressively absorbing
the surrounding IR ops into the custom op's C kernel.

For the original context (KV cache, `QuantizedKVCache`, the llama.cpp
diff) see [`MEMORY.md`](MEMORY.md). This doc only covers the v1–v3
progression and the benchmarking pattern.

---

## Headline result

| variant | activation budget @ T_q=128 | per linear-attn layer | pp512 (tok/s) | tg32 (tok/s) |
|--------|--------------------------:|---------------------:|---------------:|--------------:|
| v1 baseline (only Loop→GDR) | 687 MiB | 25.8 MiB | 101 | 15.66 |
| v2 (+ split / L2-norm / Q-scale / Transpose) | 579 MiB | 19.8 MiB | 245 | 15.87 |
| **v3 (+ conv1d-with-state / SiLU / Transposes)** | **391 MiB** | **9.1 MiB** | **186** | **15.39** |
| *llama.cpp compute buffer (reference)* | *491 MiB* | — | — | — |

**v3 vs v1: −296 MiB of addressable activation memory (−43 %), −65 % per
linear-attn layer, +84 % prefill throughput, decode at parity.** v3's
total addressable activation budget is **below llama.cpp's measured
compute buffer** for the same workload.

End-to-end generation is identical to the baseline.

---

## Concepts

### Why this matters

Earlier work attacked the KV cache (`QuantizedKVCache`, see `MEMORY.md`).
It produced a real persistent-state reduction (44 MiB at T=2048) but the
peak RSS barely moved — **only 6 of the 24 layers are full attention**;
the other **18 are linear-attention** and were responsible for **71 % of
the activation budget**.

The attribution tool [`scripts/working/attribute_mem.py`](scripts/working/attribute_mem.py)
walks `get_runtime_model()` with `model.reshape({...})`-bound shapes and
sums bytes per node, bucketed by sub-architecture. It surfaced this:

```
                  linear_attn  self_attn  mlp  other   TOTAL
v1 baseline       818 MiB      66       42   227    1153 MiB
                  ^^^ 71 %
```

So v1→v3 is about cutting the linear-attn share.

### What each version absorbs

Each version replaces a longer chain of pre-existing IR ops with one C
kernel call. The intermediate tensors that used to be separate edges in
the IR become C-stack scratch inside the kernel.

| version | input(s) to the custom op | IR chain absorbed into the kernel |
|---------|---------------------------|------------------------------------|
| **v1** | `q, k, v, g, beta, state` (post-transpose, post-norm, post-scale) | just the gated-delta `Loop` body |
| **v2** | `mixed_qkv [B,T,key_dim*2+value_dim], g, beta, state` | + `VariadicSplit → Reshape → Multiply(L2) → Transpose → Divide(Q-scale)` for Q/K/V |
| **v3** | `mixed_in [B,T,C] (MatMul output), conv_w, prev_conv, g, beta, state` | + `Transpose → Concat(prev_conv) → GroupConvolution → Slice → Swish → Transpose` |

Concretely, v3 takes the in_proj_qkv MatMul output **before any
transpose**, does conv1d-with-state + SiLU + split + L2 norm + Q-scale +
the gated-delta-rule recurrence in one pass. New conv state is emitted
as a third output and wired to the original `Assign`.

### How the kernels relate

- All three live in [`kernels/kernels.c`](kernels/kernels.c):
  `gdr_kernel`, `gdr_kernel_v2`, `gdr_kernel_v3`.
- Each has matching declarations in [`kernels/kernels.h`](kernels/kernels.h)
  and ctypes bindings in [`kernels/kernels.py`](kernels/kernels.py)
  (`gdr`, `gdr_v2`).
- C++ extensions in [`cpp_ext/`](cpp_ext): `gated_delta_rule.cpp`,
  `gated_delta_rule_v2.cpp`, `gated_delta_rule_v3.cpp`, all registered in
  [`cpp_ext/ov_extension.cpp`](cpp_ext/ov_extension.cpp).
- Python op classes for IR construction in
  [`kernels/fused_linear_attn.py`](kernels/fused_linear_attn.py):
  `GatedDeltaRule` (handles 6-input v1 and 4-input v2),
  `GatedDeltaRuleV2`, `GatedDeltaRuleV3`.
- Graph rewrites: `replace_gated_delta_rule_loops`,
  `replace_gated_delta_rule_loops_v2`,
  `replace_gated_delta_rule_loops_v3`.

### The serialize/reload trick

When both a Python `Op` subclass and the `.so` register an op of the
same name, the Python class's `evaluate()` always wins — verified by
patching the Python evaluate to print and seeing it called once per
linear-attn layer per step, even though the `.so` was loaded via
`core.add_extension(SO)`.

To make the C++ extension actually run we:

1. Build the IR using the Python class (we need it for IR construction).
2. `ov.serialize(model, xml, bin)`.
3. Re-load with a **fresh `ov.Core()` that has only `add_extension(SO)`
   registered** — no Python class import in this Core's namespace.
4. Compile. The C++ implementation wins evaluate.

Both `bench_v2.py` and `bench_v3.py` use this pattern (see the
`build_and_serialize` / `measure` helpers).

---

## Usage

All scripts run from the repo root with `QWEN3_USE_C=1` set so the
ctypes path uses the C kernels:

```bash
cd study/qwen3

# 1. Build the kernels and the C++ extension (one-time setup).
bash kernels/build_kernels.sh
( cd cpp_ext && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j )

# 2. Memory attribution — what's actually using the bytes?
QWEN3_USE_C=1 python3 scripts/working/attribute_mem.py \
    --config baseline --T_q 128 --T_full 2048 --top 20

# 3. Bench v1 vs v2 (the smaller fusion).
QWEN3_USE_C=1 python3 scripts/working/bench_v2.py

# 4. Bench v1 vs v2 vs v3 (full fusion).
QWEN3_USE_C=1 python3 scripts/working/bench_v3.py
```

`attribute_mem.py` supports `--config baseline | int8_kv_dequant | int8_sdpa`
and `--dump-runtime <xml>` for off-line graph inspection. With
`ONEDNN_VERBOSE=all` it also prints every oneDNN primitive call with
memory descriptors during one inference (use `--infer`).

Both bench scripts:
- write the constructed IR to `/tmp/lm_v{1,2,3}.xml` + `.bin`,
- re-load it with a fresh `ov.Core() + add_extension(SO)`,
- measure activation budget via `get_runtime_model()` walks,
- run pp512 + tg32 with a 32-token warmup.

The last measured tables are baked into each script's docstring.

---

## Numbers in detail

### v3 bench output (chunk=128, INFERENCE_NUM_THREADS=4)

```
=== A. v1 (C++ ext) ===
  activation budget @ T_q=128: 687.4 MiB total
    linear_attn        464.2 MiB
    self_attn           33.1 MiB
    mlp                 21.0 MiB
    other              169.1 MiB
  pp512: 5.07s (100.90 tok/s)
  tg32:  2.04s (15.66 tok/s)

=== B. v2 (+ split/L2/scale/transpose) ===
  activation budget @ T_q=128: 579.1 MiB total
    linear_attn        355.9 MiB
    self_attn           33.1 MiB
    mlp                 21.0 MiB
    other              169.1 MiB
  pp512: 2.09s (245.17 tok/s)
  tg32:  2.02s (15.87 tok/s)

=== C. v3 (+ conv1d/SiLU/Transposes) ===
  activation budget @ T_q=128: 390.7 MiB total
    linear_attn        164.2 MiB
    self_attn           33.1 MiB
    mlp                 21.0 MiB
    other              172.5 MiB
  pp512: 2.76s (185.59 tok/s)
  tg32:  2.08s (15.39 tok/s)
```

### Trade-offs

- **v2 is the prefill winner** (245 tok/s, +143 % vs baseline). The
  removed Reshape / Transpose / Multiply / Divide chain was real CPU work
  the plugin doesn't need to do. Decode is flat at ~16 tok/s.
- **v3 absorbs more but gives back some prefill** (186 tok/s, +84 %
  vs baseline, −24 % vs v2). Our absorbed conv1d C code does not match
  the plugin's blocked-GEMM GroupConvolution, so the conv work itself is
  slower in C — but the eliminated IR-level edge buffers more than make
  up for it on the *memory* axis. Decode parity is maintained.
- The next obvious lever is to either re-introduce the plugin's
  GroupConvolution and absorb only what comes after, OR replace our
  per-channel scalar conv loop with a SIMD-blocked one. Both are
  follow-up work.

---

## Layout

```
study/qwen3/
├── latest_memory.md          # this doc
├── MEMORY.md                 # original full-history memory writeup
├── README.md                 # study top-level
├── kernels/                  # C kernels + Python ops + rewrites
│   ├── kernels.c kernels.h kernels.py
│   ├── fused_linear_attn.py  # GatedDeltaRule / V2 / V3 + rewrites
│   ├── quantized_kv.py
│   ├── quantized_int8_sdpa.py
│   └── fused_conv1d.py       # kept for reference, not in default path
├── cpp_ext/                  # C++ extension loaded into OV genai / Core
│   ├── gated_delta_rule.{hpp,cpp}      # v1
│   ├── gated_delta_rule_v2.{hpp,cpp}   # v2
│   ├── gated_delta_rule_v3.{hpp,cpp}   # v3 (NEW)
│   ├── quantized_kv_cache*.cpp
│   ├── quantized_int8_sdpa.cpp
│   └── ov_extension.cpp                # registers all ops
├── openvino/                 # stable / cleaned-up tooling
│   ├── lowmem_infer.py
│   ├── generate.py
│   ├── validate_fusion.py
│   ├── compare_kv.py
│   ├── memory_analysis.py
│   ├── benchmark_ops.py
│   ├── profile_prefill.py
│   └── profile_decode.py
├── genai/
│   └── genai_vlm_pipeline.py
└── scripts/working/          # in-progress / iteration scripts
    ├── bench_v2.py
    ├── bench_v3.py
    └── attribute_mem.py
```
