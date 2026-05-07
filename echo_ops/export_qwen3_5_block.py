# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Export the gated DeltaNet (linear-attention) block from a tiny Qwen3.5
checkpoint to OpenVINO IR using echo_ops.

Pipeline mirrored from `transformers.models.qwen3_5.modeling_qwen3_5.
Qwen3_5GatedDeltaNet.forward` (no-cache, seq_len > 1 path):

    h -> in_proj_qkv -> ShortConv1D -> SiLU -> split(q,k,v)
    h -> in_proj_z   (z, gate for output norm)
    h -> in_proj_b   ; beta = sigmoid(b)
    h -> in_proj_a   ; g = exp(-exp(A_log) * softplus(a + dt_bias))
    q,k -> L2Norm   ; q *= 1/sqrt(head_k_dim)
    -> GatedDeltaRule(q, k, v, g, beta, zero_state) -> core_attn_out
    -> GatedRMSNorm(core_attn_out, z, norm.weight) -> out_proj

By default loads `optimum-intel-internal-testing/tiny-random-qwen3.5`.

Usage:
    python -m echo_ops.export_qwen3_5_block \
        [--model HF_ID] [--layer N] [--seq-len T] [--out PATH]
"""

import argparse
import os

import numpy as np
import torch

from openvino import Model, Type, save_model, compile_model, Tensor
import openvino.opset14 as opset

from .qwen3_5 import build_gated_deltanet


DEFAULT_MODEL = "optimum-intel-internal-testing/tiny-random-qwen3.5"


def load_torch_layer(model_id: str, layer_idx: int):
    """Load a tiny Qwen3.5 model and return its `layer_idx`-th GatedDeltaNet."""
    from transformers import AutoConfig, AutoModelForImageTextToText

    config = AutoConfig.from_pretrained(model_id)
    print(f"loaded config from {model_id}")
    print(f"  layer_types = {config.text_config.layer_types}")

    if config.text_config.layer_types[layer_idx] != "linear_attention":
        raise ValueError(
            f"layer {layer_idx} is {config.text_config.layer_types[layer_idx]!r}, "
            "expected 'linear_attention'. Pick a different --layer.")

    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.float32).eval()
    layer = model.model.language_model.layers[layer_idx].linear_attn
    return config.text_config, layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--layer", type=int, default=0,
                    help="decoder layer index (must be a 'linear_attention' layer)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--out", default="qwen3_5_deltanet_block.xml")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    text_config, layer = load_torch_layer(args.model, args.layer)
    print(f"  hidden_size={text_config.hidden_size}, "
          f"num_v_heads={text_config.linear_num_value_heads}, "
          f"head_v_dim={text_config.linear_value_head_dim}, "
          f"K={text_config.linear_conv_kernel_dim}")

    hidden = torch.randn(args.batch, args.seq_len, text_config.hidden_size,
                         dtype=torch.float32)
    with torch.no_grad():
        ref = layer(hidden_states=hidden).numpy()
    print(f"torch reference output: {ref.shape}, |ref|_max={np.abs(ref).max():.5f}")

    h = opset.parameter([args.batch, args.seq_len, text_config.hidden_size],
                        dtype=Type.f32, name="hidden_states")
    out, _conv, _recur = build_gated_deltanet(
        h, layer, args.batch, args.seq_len, text_config)
    ov_model = Model([opset.result(out, name="output")], [h],
                     "Qwen3_5_GatedDeltaNet_Block")
    print(f"built OV model with {len(ov_model.get_ordered_ops())} ops")

    compiled = compile_model(ov_model, "CPU")
    got = compiled(Tensor(hidden.numpy()))[compiled.outputs[0]]

    diff = np.abs(got - ref).max()
    print(f"max |torch - OV| = {diff:.3e}")
    if diff > 1e-3:
        raise SystemExit(f"FAIL: tolerance exceeded ({diff:.3e} > 1e-3)")
    print("OK, parity within tolerance.")

    out_xml = os.path.abspath(args.out)
    save_model(ov_model, out_xml)
    print(f"saved IR to {out_xml} (+.bin)")


if __name__ == "__main__":
    main()
