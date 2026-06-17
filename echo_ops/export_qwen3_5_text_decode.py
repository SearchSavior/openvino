# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Export the single-token decode graph for Qwen3.5 and verify parity.

Decode graph inputs (per layer, one set per layer-type):
    input_ids       [B, 1]                                int64
    position_id     [B, 1]                                int64
    conv_state_<i>  [B, conv_dim, conv_kernel_size]        f32   (linear layers)
    recur_state_<i> [B, Nv, Hk, Hv]                        f32   (linear layers)
    k_cache_<i>     [B, num_kv_heads, T_past, head_dim]    f32   (full layers; T_past dynamic)
    v_cache_<i>     [B, num_kv_heads, T_past, head_dim]    f32

Decode graph outputs:
    logits          [B, 1, vocab]
    new_conv_state_<i>, new_recur_state_<i>      (one per linear layer)
    new_k_cache_<i>, new_v_cache_<i>             (one per full layer; +1 token along T)

Test:
    1. Run torch prefill on T_prefill tokens with a `DynamicCache`.
    2. Capture per-layer cache state.
    3. Run T_gen single-token decode steps in torch and feed the same caches
       through the OV decode graph; compare logits and updated caches each step.

Usage:
    python -m echo_ops.export_qwen3_5_text_decode \
        [--prefill T_p] [--steps N] [--out PATH]
"""

import argparse
import os

import numpy as np
import torch

from openvino import Core, Model, PartialShape, Tensor, Type, save_model
import openvino.opset14 as opset

from .qwen3_5 import build_text_decode


DEFAULT_MODEL = "optimum-intel-internal-testing/tiny-random-qwen3.5"


def _patch_chunks_to_recurrent(model):
    """Pin every linear-attention layer's chunk fn to the recurrent variant
    so the torch reference uses the same algorithm our GatedDeltaRule op
    implements (eliminates the chunk-impl numerical drift at small T)."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
    for layer in model.model.language_model.layers:
        if layer.layer_type == "linear_attention":
            layer.linear_attn.chunk_gated_delta_rule = mq.torch_recurrent_gated_delta_rule
            layer.linear_attn.recurrent_gated_delta_rule = mq.torch_recurrent_gated_delta_rule


def _layer_indices(text_model):
    """Return (linear_idx, full_idx) lists into text_model.layers."""
    linear, full = [], []
    for i, layer in enumerate(text_model.layers):
        (linear if layer.layer_type == "linear_attention" else full).append(i)
    return linear, full


def _capture_cache(cache, linear_idx, full_idx):
    """Snapshot conv/recurrent/k/v tensors out of a DynamicCache as numpy."""
    out = {"conv": [], "recur": [], "k": [], "v": []}
    for i in linear_idx:
        out["conv"].append(cache.layers[i].conv_states.detach().numpy().copy())
        out["recur"].append(cache.layers[i].recurrent_states.detach().numpy().copy())
    for i in full_idx:
        out["k"].append(cache.layers[i].keys.detach().numpy().copy())
        out["v"].append(cache.layers[i].values.detach().numpy().copy())
    return out


def build_decode_model(model, B: int, max_pos: int | None = None):
    """Construct the OV decode Model with all per-layer cache I/O wired up.

    Returns (ov_model, input_names, output_names) where the *_names lists
    line up by index with model.model.language_model.layers' linear/full
    sub-orders.
    """
    text_model = model.model.language_model
    text_config = text_model.config
    Hk = text_config.linear_key_head_dim
    Hv = text_config.linear_value_head_dim
    Nv = text_config.linear_num_value_heads
    K  = text_config.linear_conv_kernel_dim
    Nk_attn = text_config.num_attention_heads
    Nkv = text_config.num_key_value_heads
    head_dim = text_config.head_dim
    conv_dim = (Hk * text_config.linear_num_key_heads) * 2 + (Hv * Nv)

    linear_idx, full_idx = _layer_indices(text_model)
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
    ov_model = Model(results, inputs, "Qwen3_5_TextDecode")

    return ov_model, n_lin, n_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prefill", type=int, default=8,
                    help="prefill seq length used to populate the caches")
    ap.add_argument("--steps", type=int, default=4, help="decode steps to verify")
    ap.add_argument("--out", default="qwen3_5_text_decode.xml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-position", type=int, default=4096,
                    help="size of the baked-in cos/sin table (caps the maximum "
                         "absolute position the IR can serve)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from transformers import AutoConfig, AutoModelForImageTextToText
    from transformers.cache_utils import DynamicCache

    config = AutoConfig.from_pretrained(args.model)
    text_config = config.text_config
    print(f"loaded {args.model}")
    print(f"  layers={text_config.layer_types}, hidden={text_config.hidden_size}, "
          f"vocab={text_config.vocab_size}")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.float32).eval()
    _patch_chunks_to_recurrent(model)
    text_model = model.model.language_model
    linear_idx, full_idx = _layer_indices(text_model)
    n_lin = len(linear_idx); n_full = len(full_idx)

    B = 1
    rng = np.random.default_rng(args.seed)
    vocab = text_config.vocab_size

    # ---- Prefill: populate the cache via torch, then snapshot it ----
    ids_prefill = torch.from_numpy(
        rng.integers(0, vocab, (B, args.prefill), dtype=np.int64))
    cache = DynamicCache(config=text_config)
    with torch.no_grad():
        text_model(input_ids=ids_prefill, past_key_values=cache, use_cache=True)
    snap = _capture_cache(cache, linear_idx, full_idx)
    print(f"prefill done: T={args.prefill}; "
          f"k_cache T_past={snap['k'][0].shape[2]}, conv K={snap['conv'][0].shape[-1]}, "
          f"recur shape={snap['recur'][0].shape}")

    # ---- Build OV decode graph and compile (single-thread, see prefill notes) ----
    if args.prefill + args.steps > args.max_position:
        raise SystemExit(
            f"--max-position={args.max_position} is too small for prefill "
            f"({args.prefill}) + steps ({args.steps}); raise it.")
    ov_model, _, _ = build_decode_model(model, B, max_pos=args.max_position)
    print(f"built OV decode model: {len(ov_model.get_ordered_ops())} ops, "
          f"{len(ov_model.inputs)} inputs, {len(ov_model.outputs)} outputs "
          f"(rope table covers {args.max_position} positions)")
    core = Core()
    core.set_property({"INFERENCE_NUM_THREADS": 1})
    compiled = core.compile_model(ov_model, "CPU")

    # output port -> name lookup
    out_by_name = {p.get_any_name(): p for p in compiled.outputs}

    # ---- Step through `args.steps` decode tokens, comparing every step ----
    max_logit_diff = 0.0
    max_cache_diff = 0.0
    for step in range(args.steps):
        position = args.prefill + step
        next_id = int(rng.integers(0, vocab))
        ids_dec = torch.tensor([[next_id]], dtype=torch.int64)

        # Torch decode step (cache mutated in place)
        with torch.no_grad():
            out = text_model(
                input_ids=ids_dec, past_key_values=cache, use_cache=True,
                cache_position=torch.tensor([position]))
            logits_torch = model.lm_head(out.last_hidden_state).numpy()
        snap_after = _capture_cache(cache, linear_idx, full_idx)

        # OV decode step using the PRE-step snapshot
        feed = {
            "input_ids": Tensor(ids_dec.numpy()),
            "position_id": Tensor(np.array([[position]], dtype=np.int64)),
        }
        for i, c in enumerate(snap["conv"]):
            feed[f"conv_state_{i}"] = Tensor(c)
        for i, r in enumerate(snap["recur"]):
            feed[f"recur_state_{i}"] = Tensor(r)
        for i, k in enumerate(snap["k"]):
            feed[f"k_cache_{i}"] = Tensor(k)
        for i, v in enumerate(snap["v"]):
            feed[f"v_cache_{i}"] = Tensor(v)
        ov_outs = compiled(feed)

        logits_ov = ov_outs[out_by_name["logits"]]
        d_log = float(np.abs(logits_ov - logits_torch).max())
        max_logit_diff = max(max_logit_diff, d_log)

        d_cache = 0.0
        for i in range(n_lin):
            c_ov = ov_outs[out_by_name[f"new_conv_state_{i}"]]
            r_ov = ov_outs[out_by_name[f"new_recur_state_{i}"]]
            d_cache = max(d_cache,
                          float(np.abs(c_ov - snap_after["conv"][i]).max()),
                          float(np.abs(r_ov - snap_after["recur"][i]).max()))
        for i in range(n_full):
            k_ov = ov_outs[out_by_name[f"new_k_cache_{i}"]]
            v_ov = ov_outs[out_by_name[f"new_v_cache_{i}"]]
            d_cache = max(d_cache,
                          float(np.abs(k_ov - snap_after["k"][i]).max()),
                          float(np.abs(v_ov - snap_after["v"][i]).max()))
        max_cache_diff = max(max_cache_diff, d_cache)

        print(f"  step {step}  pos={position}  next_id={next_id:>6}  "
              f"|logits|={d_log:.2e}  |cache|={d_cache:.2e}")

        # Advance: feed the OV-produced caches into the next step
        snap = {
            "conv":  [ov_outs[out_by_name[f"new_conv_state_{i}"]]  for i in range(n_lin)],
            "recur": [ov_outs[out_by_name[f"new_recur_state_{i}"]] for i in range(n_lin)],
            "k":     [ov_outs[out_by_name[f"new_k_cache_{i}"]]     for i in range(n_full)],
            "v":     [ov_outs[out_by_name[f"new_v_cache_{i}"]]     for i in range(n_full)],
        }

    print(f"overall max logit diff = {max_logit_diff:.3e}")
    print(f"overall max cache diff = {max_cache_diff:.3e}")
    if max(max_logit_diff, max_cache_diff) > 1e-3:
        raise SystemExit("FAIL: decode parity exceeded 1e-3")
    print("OK, decode parity within tolerance.")

    out_xml = os.path.abspath(args.out)
    save_model(ov_model, out_xml)
    print(f"saved IR to {out_xml} (+.bin)")


if __name__ == "__main__":
    main()
