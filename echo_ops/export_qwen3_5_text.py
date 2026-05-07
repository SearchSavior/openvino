# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""End-to-end prefill export of a tiny Qwen3.5 text model.

Embedding -> N decoder layers (linear-attention or full-attention) ->
final RMSNorm -> lm_head -> logits. No KV / recurrent / conv cache (so
this graph is for prefill only — it cannot generate token-by-token, but
it computes the same logits as a single-pass torch forward).

Usage:
    python -m echo_ops.export_qwen3_5_text \
        [--model HF_ID] [--batch B] [--seq-len T] [--out PATH]
"""

import argparse
import os

import numpy as np
import torch

from openvino import Model, Type, save_model, Core, Tensor
import openvino.opset14 as opset

from .qwen3_5 import build_text_prefill


DEFAULT_MODEL = "optimum-intel-internal-testing/tiny-random-qwen3.5"


def torch_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    """text_model(input_ids) -> hidden -> lm_head -> logits, no cache.

    Pins each linear-attention layer to `torch_recurrent_gated_delta_rule`
    so the torch reference uses the same algorithm our `GatedDeltaRule`
    op implements. The default chunk path is mathematically equivalent
    but, on this transformers build, drifts by ~1e-2 at certain seq_lens
    (e.g. T == chunk_size kernel boundary cases) due to padded-chunk
    arithmetic — that's a torch-internal artifact, not an OV mismatch.
    """
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    text_model = model.model.language_model
    for layer in text_model.layers:
        if layer.layer_type == "linear_attention":
            layer.linear_attn.chunk_gated_delta_rule = mq.torch_recurrent_gated_delta_rule
    with torch.no_grad():
        out = text_model(input_ids=input_ids, use_cache=False)
        return model.lm_head(out.last_hidden_state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--out", default="qwen3_5_text_prefill.xml")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from transformers import AutoConfig, AutoModelForImageTextToText

    config = AutoConfig.from_pretrained(args.model)
    text_config = config.text_config
    print(f"loaded {args.model}")
    print(f"  hidden_size={text_config.hidden_size}, vocab={text_config.vocab_size}, "
          f"layers={text_config.layer_types}")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.float32).eval()

    # Build random input_ids in [0, vocab)
    rng = np.random.default_rng(args.seed)
    input_ids_np = rng.integers(
        low=0, high=text_config.vocab_size,
        size=(args.batch, args.seq_len), dtype=np.int64)
    input_ids = torch.from_numpy(input_ids_np)

    # Torch reference
    ref = torch_logits(model, input_ids).numpy()
    print(f"torch logits: {ref.shape}, |ref|_max={np.abs(ref).max():.5f}")

    # Build OV graph (static T; dynamic-T variant lives in the benchmark script)
    ids_param = opset.parameter(
        [args.batch, args.seq_len], dtype=Type.i64, name="input_ids")
    logits, *_caches = build_text_prefill(
        ids_param, model, args.batch, args.seq_len)
    ov_model = Model([opset.result(logits, name="logits")], [ids_param],
                     "Qwen3_5_TextPrefill")
    print(f"built OV model with {len(ov_model.get_ordered_ops())} ops")

    # Pin to single-threaded inference for the parity check. The CPU plugin
    # exhibits non-deterministic numerics at small T (notably T=4) when the
    # full-attention block is run under multi-threading on this graph; the IR
    # is unaffected, so we compile fresh with one thread for a clean compare.
    core = Core()
    core.set_property({"INFERENCE_NUM_THREADS": 1})
    compiled = core.compile_model(ov_model, "CPU")
    got = compiled(Tensor(input_ids_np))[compiled.outputs[0]]

    diff = np.abs(got - ref).max()
    rel = diff / max(np.abs(ref).max(), 1e-9)
    print(f"max |torch - OV|     = {diff:.3e}")
    print(f"max relative diff    = {rel:.3e}")
    # logits magnitude scales with vocab*hidden; allow a slightly looser tolerance
    if diff > 1e-3:
        raise SystemExit(f"FAIL: tolerance exceeded ({diff:.3e} > 1e-3)")
    print("OK, prefill parity within tolerance.")

    out_xml = os.path.abspath(args.out)
    save_model(ov_model, out_xml)
    print(f"saved IR to {out_xml} (+.bin)")


if __name__ == "__main__":
    main()
