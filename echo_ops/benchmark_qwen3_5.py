# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""End-to-end inference + timing for Qwen3.5 with the echo_ops graphs.

Builds two compiled OV models from the torch checkpoint:

  * prefill: input_ids [B, T_dyn]
            -> logits [B, T_dyn, vocab]
               + conv_state, recurrent_state, k_cache, v_cache
                 (one set per linear/full layer)
  * decode:  input_ids [B, 1] + position_id [B, 1] + all caches
            -> logits [B, 1, vocab]
               + new caches (KV append along T axis)

Tokenizes a real prompt with the model's tokenizer, runs prefill on the
prompt, then loops decode N times sampling argmax. Reports per-stage
latency and KV-cache memory usage.

Usage:
    python -m echo_ops.benchmark_qwen3_5 \
        [--model HF_ID] [--prompt 'text'] [--steps N]
"""

import argparse
import time

import numpy as np
import torch

from openvino import Core, Model, PartialShape, Tensor, Type
import openvino.opset14 as opset

from .qwen3_5 import build_text_decode, build_text_prefill


DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"
DEFAULT_PROMPT = "The capital of France is"


def _patch_chunks_to_recurrent(model):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
    for layer in model.model.language_model.layers:
        if layer.layer_type == "linear_attention":
            layer.linear_attn.chunk_gated_delta_rule = mq.torch_recurrent_gated_delta_rule
            layer.linear_attn.recurrent_gated_delta_rule = mq.torch_recurrent_gated_delta_rule


def _layer_indices(text_model):
    linear, full = [], []
    for i, layer in enumerate(text_model.layers):
        (linear if layer.layer_type == "linear_attention" else full).append(i)
    return linear, full


def _bytes_human(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KiB"
    if n < 1024 ** 3:
        return f"{n / 1024 ** 2:.1f} MiB"
    return f"{n / 1024 ** 3:.2f} GiB"


def build_prefill_model(model, B: int, max_pos: int):
    """Dynamic-T prefill graph that emits logits + initial caches."""
    text_config = model.model.language_model.config
    head_dim = text_config.head_dim
    Nkv = text_config.num_key_value_heads

    ids_p = opset.parameter(
        PartialShape([B, -1]), dtype=Type.i64, name="input_ids")
    logits, conv_states, recur_states, k_caches, v_caches = build_text_prefill(
        ids_p, model, B, T=-1, max_pos=max_pos)

    results = [opset.result(logits, name="logits")]
    for i, n in enumerate(conv_states):
        results.append(opset.result(n, name=f"conv_state_{i}"))
    for i, n in enumerate(recur_states):
        results.append(opset.result(n, name=f"recur_state_{i}"))
    for i, n in enumerate(k_caches):
        results.append(opset.result(n, name=f"k_cache_{i}"))
    for i, n in enumerate(v_caches):
        results.append(opset.result(n, name=f"v_cache_{i}"))
    return Model(results, [ids_p], "Qwen3_5_TextPrefillDynamic")


def build_decode_model(model, B: int, max_pos: int):
    text_config = model.model.language_model.config
    Hk = text_config.linear_key_head_dim
    Hv = text_config.linear_value_head_dim
    Nv = text_config.linear_num_value_heads
    K  = text_config.linear_conv_kernel_dim
    Nkv = text_config.num_key_value_heads
    head_dim = text_config.head_dim
    conv_dim = (Hk * text_config.linear_num_key_heads) * 2 + (Hv * Nv)

    linear_idx, full_idx = _layer_indices(model.model.language_model)
    n_lin = len(linear_idx); n_full = len(full_idx)

    ids_p = opset.parameter([B, 1], dtype=Type.i64, name="input_ids")
    pos_p = opset.parameter([B, 1], dtype=Type.i64, name="position_id")
    conv_ps = [opset.parameter(
        [B, conv_dim, K], dtype=Type.f32, name=f"conv_state_{i}")
        for i in range(n_lin)]
    recur_ps = [opset.parameter(
        [B, Nv, Hk, Hv], dtype=Type.f32, name=f"recur_state_{i}")
        for i in range(n_lin)]
    k_ps = [opset.parameter(
        PartialShape([B, Nkv, -1, head_dim]), dtype=Type.f32, name=f"k_cache_{i}")
        for i in range(n_full)]
    v_ps = [opset.parameter(
        PartialShape([B, Nkv, -1, head_dim]), dtype=Type.f32, name=f"v_cache_{i}")
        for i in range(n_full)]

    logits, new_conv, new_recur, new_k, new_v = build_text_decode(
        ids_p, pos_p, model, B, conv_ps, recur_ps, k_ps, v_ps,
        max_pos=max_pos)

    results = [opset.result(logits, name="logits")]
    for i, n in enumerate(new_conv):
        results.append(opset.result(n, name=f"new_conv_state_{i}"))
    for i, n in enumerate(new_recur):
        results.append(opset.result(n, name=f"new_recur_state_{i}"))
    for i, n in enumerate(new_k):
        results.append(opset.result(n, name=f"new_k_cache_{i}"))
    for i, n in enumerate(new_v):
        results.append(opset.result(n, name=f"new_v_cache_{i}"))
    inputs = [ids_p, pos_p] + conv_ps + recur_ps + k_ps + v_ps
    return Model(results, inputs, "Qwen3_5_TextDecodeDynamic")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--prompt-tokens", type=int, default=0,
                    help="if > 0, ignore --prompt and use that many random "
                         "token ids (sampled away from special tokens) -- "
                         "useful for benchmarking long prefills / KV cache "
                         "size at a target sequence length without needing "
                         "a real prompt of that length")
    ap.add_argument("--steps", type=int, default=16,
                    help="number of decode steps to run")
    ap.add_argument("--max-position", type=int, default=4096)
    ap.add_argument("--device", default="CPU")
    args = ap.parse_args()

    print(f"loading {args.model} ...")
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    torch_t0 = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.float32).eval()
    _patch_chunks_to_recurrent(model)
    text_model = model.model.language_model
    text_config = text_model.config
    print(f"  loaded in {time.perf_counter()-torch_t0:.1f}s; "
          f"{text_config.num_hidden_layers} layers "
          f"({sum(1 for x in text_config.layer_types if x=='linear_attention')} linear / "
          f"{sum(1 for x in text_config.layer_types if x=='full_attention')} full)")
    linear_idx, full_idx = _layer_indices(text_model)
    n_lin = len(linear_idx); n_full = len(full_idx)
    B = 1

    # ---- Tokenize prompt (or synthesize random ids) --------------------------
    if args.prompt_tokens > 0:
        # Sample ids in [1000, 100000) to avoid both control tokens at the
        # bottom and vision/special tokens near the top of the vocab.
        rng = np.random.default_rng(0)
        input_ids_np = rng.integers(
            1000, 100000, size=(B, args.prompt_tokens), dtype=np.int64)
        T_prefill = args.prompt_tokens
        print(f"prompt: <synthetic, {T_prefill} random ids in [1000, 100000)>")
    else:
        enc = tok(args.prompt, return_tensors="np")
        input_ids_np = enc["input_ids"].astype(np.int64)
        T_prefill = input_ids_np.shape[1]
        print(f"prompt: {args.prompt!r}")
        print(f"  tokenized to {T_prefill} ids")

    # ---- Build & compile graphs ----------------------------------------------
    if T_prefill + args.steps > args.max_position:
        raise SystemExit(
            f"--max-position={args.max_position} too small for "
            f"T_prefill={T_prefill} + steps={args.steps}; raise it.")

    # Sequence the work to keep peak RAM bounded: build+compile+run prefill,
    # capture its outputs as numpy, free prefill, then build+compile+run
    # decode. Otherwise the f32 0.8B graphs (torch + 2 OV graphs + 2 compiled
    # models) will blow past 16 GiB.
    import gc

    core = Core()

    print(f"building prefill graph (dynamic T) ...")
    t0 = time.perf_counter()
    prefill_m = build_prefill_model(model, B, max_pos=args.max_position)
    print(f"  built in {time.perf_counter()-t0:.1f}s "
          f"({len(prefill_m.get_ordered_ops())} ops, "
          f"{len(prefill_m.outputs)} outputs)")

    print(f"compiling prefill on {args.device} ...")
    t0 = time.perf_counter()
    prefill = core.compile_model(prefill_m, args.device)
    print(f"  prefill compiled in {time.perf_counter()-t0:.1f}s")
    p_out = {p.get_any_name(): p for p in prefill.outputs}

    # Drop the graph object now that it's compiled (frees the constant-side copies).
    del prefill_m
    gc.collect()

    # ---- Prefill -------------------------------------------------------------
    t0 = time.perf_counter()
    out = prefill({"input_ids": Tensor(input_ids_np)})
    prefill_ms = (time.perf_counter() - t0) * 1000

    logits = out[p_out["logits"]]
    next_id = int(logits[0, -1].argmax())
    conv_states  = [np.array(out[p_out[f"conv_state_{i}"]])  for i in range(n_lin)]
    recur_states = [np.array(out[p_out[f"recur_state_{i}"]]) for i in range(n_lin)]
    k_caches     = [np.array(out[p_out[f"k_cache_{i}"]])     for i in range(n_full)]
    v_caches     = [np.array(out[p_out[f"v_cache_{i}"]])     for i in range(n_full)]

    # Free prefill's compiled model + any retained inference outputs before
    # we build the decode graph, to keep peak memory bounded.
    del prefill, p_out, out
    gc.collect()

    print(f"\nbuilding decode graph ...")
    t0 = time.perf_counter()
    decode_m = build_decode_model(model, B, max_pos=args.max_position)
    print(f"  built in {time.perf_counter()-t0:.1f}s "
          f"({len(decode_m.get_ordered_ops())} ops)")
    print(f"compiling decode on {args.device} ...")
    t0 = time.perf_counter()
    decode = core.compile_model(decode_m, args.device)
    print(f"  decode compiled in {time.perf_counter()-t0:.1f}s")
    d_out = {p.get_any_name(): p for p in decode.outputs}
    del decode_m, model
    gc.collect()

    print(f"\n--- prefill ---")
    print(f"  T = {T_prefill}    -> {prefill_ms:.1f} ms "
          f"({prefill_ms / T_prefill:.2f} ms / token)")

    conv_b  = sum(c.nbytes for c in conv_states)
    recur_b = sum(r.nbytes for r in recur_states)
    full_b  = sum(k.nbytes + v.nbytes for k, v in zip(k_caches, v_caches))
    full_per_tok = full_b / T_prefill if T_prefill > 0 else 0
    print(f"  cache after prefill:")
    print(f"    linear conv state ({n_lin} layers, fixed):       {_bytes_human(conv_b)}")
    print(f"    linear recur state ({n_lin} layers, fixed):      {_bytes_human(recur_b)}")
    print(f"    full attn KV ({n_full} layers, T={T_prefill}):  "
          f"{_bytes_human(full_b)} "
          f"({_bytes_human(int(full_per_tok))} / token / batch)")

    # ---- Decode loop ---------------------------------------------------------
    print(f"\n--- decode ({args.steps} steps) ---")
    decode_ms = []
    generated = [next_id]
    for step in range(args.steps):
        position = T_prefill + step
        ids_dec = np.array([[next_id]], dtype=np.int64)
        feed = {
            "input_ids": Tensor(ids_dec),
            "position_id": Tensor(np.array([[position]], dtype=np.int64)),
        }
        for i, c in enumerate(conv_states):
            feed[f"conv_state_{i}"] = Tensor(c)
        for i, r in enumerate(recur_states):
            feed[f"recur_state_{i}"] = Tensor(r)
        for i, k in enumerate(k_caches):
            feed[f"k_cache_{i}"] = Tensor(k)
        for i, v in enumerate(v_caches):
            feed[f"v_cache_{i}"] = Tensor(v)

        t0 = time.perf_counter()
        out = decode(feed)
        decode_ms.append((time.perf_counter() - t0) * 1000)

        logits = out[d_out["logits"]]
        next_id = int(logits[0, 0].argmax())
        generated.append(next_id)

        # roll caches forward
        conv_states  = [out[d_out[f"new_conv_state_{i}"]]  for i in range(n_lin)]
        recur_states = [out[d_out[f"new_recur_state_{i}"]] for i in range(n_lin)]
        k_caches     = [out[d_out[f"new_k_cache_{i}"]]     for i in range(n_full)]
        v_caches     = [out[d_out[f"new_v_cache_{i}"]]     for i in range(n_full)]

    decode_first = decode_ms[0]
    decode_warm = decode_ms[1:] if len(decode_ms) > 1 else decode_ms
    decode_warm_avg = sum(decode_warm) / len(decode_warm)
    decode_warm_med = sorted(decode_warm)[len(decode_warm) // 2]
    print(f"  step  0  (warmup):   {decode_first:6.1f} ms")
    print(f"  steps 1..{args.steps - 1}: avg {decode_warm_avg:6.1f} ms  "
          f"median {decode_warm_med:6.1f} ms  "
          f"({1000.0 / decode_warm_med:5.1f} tok/s)")

    print(f"\n--- final KV cache after {args.steps} decode steps ---")
    full_b  = sum(k.nbytes + v.nbytes for k, v in zip(k_caches, v_caches))
    print(f"  full attn KV (T={T_prefill + args.steps}):  {_bytes_human(full_b)}")

    print(f"\n--- generated continuation ---")
    text = tok.decode(generated, skip_special_tokens=True)
    print(f"  {text!r}")


if __name__ == "__main__":
    main()
