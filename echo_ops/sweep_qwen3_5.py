# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Sweep prefill T from 256 to 4096 (step 256) on Qwen3.5-0.8B.

Builds ONE unified prefill+decode graph and uses it for both modes:

  * prefill call:  input_ids = prompt[:T], position_ids = arange(T),
                   incoming caches are zero-shaped (T_past=0).
  * decode call:   input_ids = [next_id], position_ids = [T_past],
                   incoming caches = previous call's outputs.

Same compiled model serves both. fp16 weights by default to halve IR size.

For each T:
  * slice the first T tokens of a runtime-generated long prompt
  * run prefill, capture caches, time it, report cache sizes
  * run --steps-per-T decode steps and print the continuation

Usage:
    python -m echo_ops.sweep_qwen3_5 [--steps-per-T N] [--max-T 4096] [--no-fp16]
"""

import argparse
import gc
import time

import numpy as np
import torch

from openvino import Core, Model, PartialShape, Tensor, Type
import openvino.opset14 as opset

from . import qwen3_5
from .qwen3_5 import build_text_unified, set_weight_dtype


DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"


_BASE_PASSAGE = """\
The morning fog rolled across the bay, slow and dense, swallowing the small
fishing boats that bobbed near the harbor wall. Marie wrapped her woolen scarf
tighter and walked toward the lighthouse, careful of the slick cobblestones
underfoot. She had made this same walk every morning for thirty-two years, and
she knew the rhythm of the tides better than anyone in town. The lighthouse
keeper, an old friend, would already have the kettle on, and there would be
fresh bread from the baker on the corner. They would talk about the weather,
about the price of fish, about the new family that had moved into the empty
cottage by the cliffs. They would not talk about the storm, though everyone in
the village knew it was coming. The seabirds had been restless for two days,
and the air smelled wrong, metallic and sharp, the way it always did before
the sky turned over. Marie quickened her pace. There was wood to bring in, and
shutters to fasten, and the children would have to be kept indoors. The
lighthouse beam swept across the water, steady and unhurried, the way it had
for more than a hundred years. It had seen a great many storms before, and it
would see a great many more. Marie watched it turn, and she felt, as she
always did, a quiet kind of comfort that was older than she was, older than
the village, perhaps older than the sea itself.
"""


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
    if n < 1024: return f"{n} B"
    if n < 1024 ** 2: return f"{n / 1024:.1f} KiB"
    if n < 1024 ** 3: return f"{n / 1024 ** 2:.1f} MiB"
    return f"{n / 1024 ** 3:.2f} GiB"


def make_long_prompt(tokenizer, target_tokens: int) -> np.ndarray:
    text = _BASE_PASSAGE
    while len(tokenizer.encode(text)) < target_tokens + 64:
        text = text + "\n\n" + _BASE_PASSAGE
    ids = tokenizer.encode(text)
    return np.array(ids[: target_tokens + 64], dtype=np.int64)[None, :]


def build_unified_model(model, B: int, max_pos: int, cache_dtype):
    """Build ONE unified prefill+decode graph.

    Inputs (all Parameters):
        input_ids       [B, -1]        i64
        position_ids    [B, -1]        i64
        conv_state_<i>  [B, conv_dim, K]   f       (one per linear-attn layer)
        recur_state_<i> [B, Nv, Hk, Hv]    f
        k_cache_<i>     [B, Nkv, -1, head_dim]  f  (one per full-attn layer; T_past dynamic)
        v_cache_<i>     [B, Nkv, -1, head_dim]  f

    Caller drives prefill vs decode by sizing the cache inputs and
    position_ids appropriately:
        fresh prefill:  T_past=0 everywhere (zero-shaped k/v_cache),
                        conv/recur passed as zero tensors
        decode step:    feed previous call's cache outputs verbatim

    Returns the OV Model.
    """
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

    ids_p = opset.parameter(PartialShape([B, -1]), dtype=Type.i64, name="input_ids")
    pos_p = opset.parameter(PartialShape([B, -1]), dtype=Type.i64, name="position_ids")
    conv_ps = [opset.parameter([B, conv_dim, K], dtype=cache_dtype, name=f"conv_state_{i}")
               for i in range(n_lin)]
    recur_ps = [opset.parameter([B, Nv, Hk, Hv], dtype=cache_dtype, name=f"recur_state_{i}")
                for i in range(n_lin)]
    k_ps = [opset.parameter(PartialShape([B, Nkv, -1, head_dim]),
                            dtype=cache_dtype, name=f"k_cache_{i}")
            for i in range(n_full)]
    v_ps = [opset.parameter(PartialShape([B, Nkv, -1, head_dim]),
                            dtype=cache_dtype, name=f"v_cache_{i}")
            for i in range(n_full)]

    logits, new_conv, new_recur, new_k, new_v = build_text_unified(
        ids_p, pos_p, model, B,
        conv_ps, recur_ps, k_ps, v_ps,
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
    return Model(results, inputs, "Qwen3_5_TextUnified")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-T", type=int, default=4096)
    ap.add_argument("--T-step", type=int, default=256)
    ap.add_argument("--steps-per-T", type=int, default=12,
                    help="decode steps to run after each prefill")
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--no-fp16", action="store_true",
                    help="bake constants as fp32 instead of fp16")
    args = ap.parse_args()

    print(f"loading {args.model} ...")
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    t0 = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.float32).eval()
    _patch_chunks_to_recurrent(model)
    text_config = model.model.language_model.config
    print(f"  loaded in {time.perf_counter()-t0:.1f}s; "
          f"{text_config.num_hidden_layers} layers")

    linear_idx, full_idx = _layer_indices(model.model.language_model)
    n_lin = len(linear_idx); n_full = len(full_idx)
    Hk = text_config.linear_key_head_dim
    Hv = text_config.linear_value_head_dim
    Nv = text_config.linear_num_value_heads
    K  = text_config.linear_conv_kernel_dim
    Nkv = text_config.num_key_value_heads
    head_dim = text_config.head_dim
    conv_dim = (Hk * text_config.linear_num_key_heads) * 2 + (Hv * Nv)
    B = 1

    print(f"building prompt (target {args.max_T} tokens) ...")
    full_prompt = make_long_prompt(tok, args.max_T)
    print(f"  prompt has {full_prompt.shape[1]} tokens")

    use_fp16 = not args.no_fp16
    if use_fp16:
        set_weight_dtype(np.float16)
    np_dtype = np.float16 if use_fp16 else np.float32
    cache_dtype = Type.f16 if use_fp16 else Type.f32
    print(f"weight dtype: {'fp16' if use_fp16 else 'fp32'}")

    print(f"building unified graph ...")
    t0 = time.perf_counter()
    ov_model = build_unified_model(
        model, B, max_pos=args.max_T + args.steps_per_T + 16,
        cache_dtype=cache_dtype)
    print(f"  built in {time.perf_counter()-t0:.1f}s "
          f"({len(ov_model.get_ordered_ops())} ops, "
          f"{len(ov_model.inputs)} inputs, {len(ov_model.outputs)} outputs)")

    del model
    gc.collect()

    print(f"compiling on {args.device} ...")
    core = Core()
    t0 = time.perf_counter()
    compiled = core.compile_model(ov_model, args.device)
    print(f"  compiled in {time.perf_counter()-t0:.1f}s")
    out_by_name = {p.get_any_name(): p for p in compiled.outputs}
    del ov_model
    gc.collect()

    Ts = list(range(args.T_step, args.max_T + 1, args.T_step))
    print(f"\nsweeping T = {Ts[0]}..{Ts[-1]} step {args.T_step} "
          f"({args.steps_per_T} decode steps each)\n")

    rows = []
    feed = None
    out = None
    for T in Ts:
        # Free prior-iteration tensors before allocating the next prefill's
        # activations -- avoids monotonic RSS growth that OOMs on 16 GiB.
        feed = None
        out = None
        gc.collect()
        # ---- Prefill ---------------------------------------------------------
        ids = full_prompt[:, :T]
        positions = np.arange(T, dtype=np.int64)[None, :]
        feed = {
            "input_ids": Tensor(ids),
            "position_ids": Tensor(positions),
        }
        for i in range(n_lin):
            feed[f"conv_state_{i}"]  = Tensor(np.zeros((B, conv_dim, K), dtype=np_dtype))
            feed[f"recur_state_{i}"] = Tensor(np.zeros((B, Nv, Hk, Hv), dtype=np_dtype))
        for i in range(n_full):
            feed[f"k_cache_{i}"] = Tensor(np.zeros((B, Nkv, 0, head_dim), dtype=np_dtype))
            feed[f"v_cache_{i}"] = Tensor(np.zeros((B, Nkv, 0, head_dim), dtype=np_dtype))

        t0 = time.perf_counter()
        out = compiled(feed)
        prefill_ms = (time.perf_counter() - t0) * 1000

        logits = out[out_by_name["logits"]]
        next_id = int(logits[0, -1].argmax())
        conv_states  = [np.array(out[out_by_name[f"new_conv_state_{i}"]])  for i in range(n_lin)]
        recur_states = [np.array(out[out_by_name[f"new_recur_state_{i}"]]) for i in range(n_lin)]
        k_caches     = [np.array(out[out_by_name[f"new_k_cache_{i}"]])     for i in range(n_full)]
        v_caches     = [np.array(out[out_by_name[f"new_v_cache_{i}"]])     for i in range(n_full)]
        del out

        full_b  = sum(k.nbytes + v.nbytes for k, v in zip(k_caches, v_caches))
        conv_b  = sum(c.nbytes for c in conv_states)
        recur_b = sum(r.nbytes for r in recur_states)

        # ---- Decode loop -----------------------------------------------------
        decode_ms = []
        gen = [next_id]
        for step in range(args.steps_per_T):
            position = T + step
            d_feed = {
                "input_ids":     Tensor(np.array([[next_id]], dtype=np.int64)),
                "position_ids":  Tensor(np.array([[position]], dtype=np.int64)),
            }
            for i, c in enumerate(conv_states):  d_feed[f"conv_state_{i}"]  = Tensor(c)
            for i, r in enumerate(recur_states): d_feed[f"recur_state_{i}"] = Tensor(r)
            for i, k in enumerate(k_caches):     d_feed[f"k_cache_{i}"]     = Tensor(k)
            for i, v in enumerate(v_caches):     d_feed[f"v_cache_{i}"]     = Tensor(v)
            t0 = time.perf_counter()
            out = compiled(d_feed)
            decode_ms.append((time.perf_counter() - t0) * 1000)
            next_id = int(out[out_by_name["logits"]][0, 0].argmax())
            gen.append(next_id)
            conv_states  = [np.array(out[out_by_name[f"new_conv_state_{i}"]])  for i in range(n_lin)]
            recur_states = [np.array(out[out_by_name[f"new_recur_state_{i}"]]) for i in range(n_lin)]
            k_caches     = [np.array(out[out_by_name[f"new_k_cache_{i}"]])     for i in range(n_full)]
            v_caches     = [np.array(out[out_by_name[f"new_v_cache_{i}"]])     for i in range(n_full)]

        decode_avg = sum(decode_ms[1:]) / max(1, len(decode_ms) - 1)
        cont = tok.decode(gen, skip_special_tokens=True)
        rows.append((T, prefill_ms, decode_avg, full_b, conv_b, recur_b, cont))

        ctot = full_b + conv_b + recur_b
        print(f"T={T:>4}  prefill={prefill_ms / 1000:6.2f}s "
              f"({prefill_ms / T:5.1f} ms/tok)  "
              f"decode={decode_avg:6.1f} ms/tok  "
              f"cache={_bytes_human(ctot)} (full-attn={_bytes_human(full_b)})")
        print(f"      continuation: {cont!r}")

    # ---- Summary ------------------------------------------------------------
    print()
    print(f"{'T':>5}  {'prefill':>10}  {'decode/tok':>11}  "
          f"{'cache total':>13}  {'full-attn KV':>14}  {'linear fixed':>14}")
    print("-" * 80)
    for T, p_ms, d_ms, full_b, conv_b, recur_b, _cont in rows:
        ctot = full_b + conv_b + recur_b
        print(f"{T:>5}  {p_ms / 1000:>9.2f}s  {d_ms:>9.1f} ms  "
              f"{_bytes_human(ctot):>13}  {_bytes_human(full_b):>14}  "
              f"{_bytes_human(conv_b + recur_b):>14}")


if __name__ == "__main__":
    main()
