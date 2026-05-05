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

By default loads `optimum-intel-internal-testing/tiny-random-qwen3.5`
(4 layers, hidden_size=32). Use --model to override and --layer to pick
which decoder layer's `linear_attn` to export.

Usage:
    python -m echo_ops.export_qwen3_5_block \
        [--model HF_ID] [--layer N] [--seq-len T] [--out PATH]
"""

import argparse
import math
import os

import numpy as np
import torch

from openvino import Model, Type, save_model, compile_model, Tensor
import openvino.opset14 as opset

from echo_ops import GatedDeltaRule, GatedRMSNorm, L2Norm, ShortConv1D


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


def torch_reference(layer, hidden: torch.Tensor) -> torch.Tensor:
    """Run the torch GatedDeltaNet (no cache) and return its output."""
    with torch.no_grad():
        return layer(hidden_states=hidden)


def _const(x: np.ndarray):
    """np.ndarray -> ov Constant of float32."""
    return opset.constant(np.ascontiguousarray(x).astype(np.float32))


def build_ov_graph(text_config, layer, batch: int, seq_len: int) -> Model:
    """Build an OV Model for a single GatedDeltaNet block, weights from `layer`."""
    H  = text_config.hidden_size
    Hk = text_config.linear_key_head_dim
    Hv = text_config.linear_value_head_dim
    Nk = text_config.linear_num_key_heads
    Nv = text_config.linear_num_value_heads
    K  = text_config.linear_conv_kernel_dim
    eps = text_config.rms_norm_eps
    key_dim = Hk * Nk
    value_dim = Hv * Nv
    conv_dim = key_dim * 2 + value_dim
    assert Nv == Nk, "this exporter assumes num_v_heads == num_k_heads"

    # ---- Pull weights out of the torch layer (numpy, float32) ----
    W_qkv = layer.in_proj_qkv.weight.detach().numpy()           # [conv_dim, H]
    W_z   = layer.in_proj_z.weight.detach().numpy()             # [value_dim, H]
    W_b   = layer.in_proj_b.weight.detach().numpy()             # [Nv, H]
    W_a   = layer.in_proj_a.weight.detach().numpy()             # [Nv, H]
    W_out = layer.out_proj.weight.detach().numpy()              # [H, value_dim]
    # nn.Conv1d weight is [conv_dim, 1, K]; ShortConv1D wants weight[c, 0]=current
    # step, so reverse along K (nn.Conv1d's weight[K-1] is the current step).
    conv_w = layer.conv1d.weight.detach().numpy().squeeze(1)    # [conv_dim, K]
    echo_conv_w = conv_w[:, ::-1].copy()
    A_log    = layer.A_log.detach().numpy()                     # [Nv]
    dt_bias  = layer.dt_bias.detach().numpy()                   # [Nv]
    norm_w   = layer.norm.weight.detach().numpy()               # [Hv]

    # ---- Graph ----
    h = opset.parameter([batch, seq_len, H], dtype=Type.f32, name="hidden_states")

    # in_proj_qkv -> [B, T, conv_dim]
    mixed_qkv = opset.matmul(h, _const(W_qkv), False, True)

    # depthwise causal conv1d along T  +  silu
    conv_out = ShortConv1D([mixed_qkv, _const(echo_conv_w)])
    mixed_qkv_act = opset.swish(conv_out)  # silu == swish(beta=1)

    # split into q, k, v along last dim with [key_dim, key_dim, value_dim]
    sizes = opset.constant(np.array([key_dim, key_dim, value_dim], dtype=np.int64))
    qkv_split = opset.variadic_split(mixed_qkv_act, opset.constant(np.int64(-1)), sizes)
    q_flat = qkv_split.output(0)
    k_flat = qkv_split.output(1)
    v_flat = qkv_split.output(2)

    # reshape to [B, T, num_heads, head_dim]
    def reshape_heads(t, num_heads, head_dim):
        target = opset.constant(np.array([batch, seq_len, num_heads, head_dim], dtype=np.int64))
        return opset.reshape(t, target, special_zero=False)

    q = reshape_heads(q_flat, Nk, Hk)
    k = reshape_heads(k_flat, Nk, Hk)
    v = reshape_heads(v_flat, Nv, Hv)

    # z gate: [B, T, value_dim] -> [B, T, Nv, Hv]
    z = opset.matmul(h, _const(W_z), False, True)
    z = reshape_heads(z, Nv, Hv)

    # beta = sigmoid(in_proj_b(h))    [B, T, Nv]
    b = opset.matmul(h, _const(W_b), False, True)
    beta = opset.sigmoid(b)

    # g = exp(-exp(A_log) * softplus(a + dt_bias))  -> [B, T, Nv]
    a = opset.matmul(h, _const(W_a), False, True)
    a_plus = opset.add(a, _const(dt_bias))
    sp = opset.softplus(a_plus)
    A_neg_exp = _const(-np.exp(A_log))                  # [Nv], broadcasts over [B,T,Nv]
    g_log = opset.multiply(A_neg_exp, sp)
    g_actual = opset.exp(g_log)

    # transpose q,k,v from [B,T,Nh,D] to [B,Nh,T,D]
    perm_4 = opset.constant(np.array([0, 2, 1, 3], dtype=np.int64))
    q_t = opset.transpose(q, perm_4)
    k_t = opset.transpose(k, perm_4)
    v_t = opset.transpose(v, perm_4)

    # transpose beta, g from [B,T,Nv] to [B,Nv,T]
    perm_3 = opset.constant(np.array([0, 2, 1], dtype=np.int64))
    beta_t = opset.transpose(beta, perm_3)
    g_t = opset.transpose(g_actual, perm_3)

    # L2 normalize q, k along last dim
    q_n = L2Norm([q_t], eps=1e-6)
    k_n = L2Norm([k_t], eps=1e-6)

    # q *= 1/sqrt(head_k_dim)   (matches torch_recurrent/chunk gated delta rule)
    q_scaled = opset.multiply(
        q_n, opset.constant(np.float32(1.0 / math.sqrt(Hk))))

    # zero initial state [B, Nv, Hk, Hv]
    s0 = _const(np.zeros((batch, Nv, Hk, Hv), dtype=np.float32))

    rule = GatedDeltaRule([q_scaled, k_n, v_t, g_t, beta_t, s0])
    core_attn_out = rule.output(0)              # [B, Nv, T, Hv]

    # back to [B, T, Nv, Hv]
    core_attn_out = opset.transpose(core_attn_out, perm_4)

    # GatedRMSNorm operates on the head_v_dim axis: flatten leading dims
    flat_shape = opset.constant(np.array([-1, Hv], dtype=np.int64))
    core_flat = opset.reshape(core_attn_out, flat_shape, special_zero=False)
    z_flat = opset.reshape(z, flat_shape, special_zero=False)
    post = GatedRMSNorm([core_flat, z_flat, _const(norm_w)], eps=eps)

    # back to [B, T, value_dim]
    target_back = opset.constant(np.array([batch, seq_len, value_dim], dtype=np.int64))
    post = opset.reshape(post, target_back, special_zero=False)

    # out_proj
    output = opset.matmul(post, _const(W_out), False, True)

    return Model([opset.result(output, name="output")], [h], "Qwen3_5_GatedDeltaNet_Block")


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

    # Reference forward
    hidden = torch.randn(args.batch, args.seq_len, text_config.hidden_size,
                         dtype=torch.float32)
    ref = torch_reference(layer, hidden).numpy()
    print(f"torch reference output: {ref.shape}, |ref|_max={np.abs(ref).max():.5f}")

    # Build OV model and run on CPU
    ov_model = build_ov_graph(text_config, layer, args.batch, args.seq_len)
    print(f"built OV model with {len(ov_model.get_ordered_ops())} ops")

    compiled = compile_model(ov_model, "CPU")
    out = compiled(Tensor(hidden.numpy()))
    got = out[compiled.outputs[0]]

    diff = np.abs(got - ref).max()
    print(f"max |torch - OV| = {diff:.3e}")
    if diff > 1e-3:
        raise SystemExit(f"FAIL: tolerance exceeded ({diff:.3e} > 1e-3)")
    print("OK, parity within tolerance.")

    # Serialize
    out_xml = os.path.abspath(args.out)
    save_model(ov_model, out_xml)
    print(f"saved IR to {out_xml} (+.bin)")


if __name__ == "__main__":
    main()
