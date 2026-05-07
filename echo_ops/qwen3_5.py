# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OpenVINO graph builders for the dense Qwen3.5 text-prefill path.

Each helper takes a `Node` for hidden states (and any auxiliary nodes such
as cos/sin) and a torch sub-module to read weights from, and returns a
`Node` that can be wired into a larger Model.

Implements all the pieces needed for prefill of a `Qwen3_5TextModel`:

    build_qwen_rmsnorm     - (1 + w) * x * rsqrt(mean(x^2) + eps)
    build_swiglu_mlp       - down(silu(gate(h)) * up(h))
    build_partial_rope     - apply rotary to first `rotary_dim` of head_dim
    build_repeat_kv        - GQA group expansion
    build_full_attention   - q/k/v + q/k norm + RoPE + GQA + SDPA + sigmoid gate
    build_gated_deltanet   - linear-attention block via echo_ops
    build_decoder_layer    - input-norm + mixer + post-norm + MLP
    build_text_prefill     - embed + N decoder layers + final norm + lm_head

Restrictions:
    * Static shape (B, T fixed at build time).
    * No KV / recurrent / conv cache (prefill only).
    * Assumes num_v_heads == num_k_heads in the linear path.
"""

import math

import numpy as np
import torch

from openvino import Type
import openvino.opset14 as opset

from .ops import GatedDeltaRule, GatedDeltaRuleStep, GatedRMSNorm, L2Norm, ShortConv1D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _const(arr) -> "opset.Node":
    """Return a float32 OV Constant from a numpy array or scalar."""
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    return opset.constant(a)


def _const_i64(arr) -> "opset.Node":
    """Return an int64 OV Constant for shapes/indices."""
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.int64))
    return opset.constant(a)


# ---------------------------------------------------------------------------
# Qwen3.5 RMSNorm:  (1 + weight) * x * rsqrt(mean(x*x) + eps)
# ---------------------------------------------------------------------------

def build_qwen_rmsnorm(x, weight: np.ndarray, eps: float):
    sq = opset.multiply(x, x)
    var = opset.reduce_mean(sq, _const_i64([-1]), keep_dims=True)
    rstd = opset.divide(_const(np.float32(1.0)),
                        opset.sqrt(opset.add(var, _const(np.float32(eps)))))
    normed = opset.multiply(x, rstd)
    return opset.multiply(normed,
                          _const((1.0 + np.asarray(weight)).astype(np.float32)))


# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------

def build_swiglu_mlp(h, mlp):
    Wg = mlp.gate_proj.weight.detach().numpy()
    Wu = mlp.up_proj.weight.detach().numpy()
    Wd = mlp.down_proj.weight.detach().numpy()
    g = opset.swish(opset.matmul(h, _const(Wg), False, True))   # silu(gate)
    u = opset.matmul(h, _const(Wu), False, True)
    return opset.matmul(opset.multiply(g, u), _const(Wd), False, True)


# ---------------------------------------------------------------------------
# Partial RoPE (non-interleaved / "GLM-style")
# ---------------------------------------------------------------------------

def _rotate_half(x, rotary_dim: int):
    half = rotary_dim // 2
    parts = opset.variadic_split(x, _const_i64(-1), _const_i64([half, half]))
    x1 = parts.output(0)
    x2 = parts.output(1)
    neg_x2 = opset.multiply(x2, _const(np.float32(-1.0)))
    return opset.concat([neg_x2, x1], axis=-1)


def build_partial_rope(q, k, cos, sin, rotary_dim: int, head_dim: int):
    """q, k:   [B, H, T, head_dim] ;  cos, sin: [B, T, rotary_dim]."""
    cos_e = opset.unsqueeze(cos, _const_i64(1))    # [B, 1, T, rotary_dim]
    sin_e = opset.unsqueeze(sin, _const_i64(1))

    pass_dim = head_dim - rotary_dim
    sizes = _const_i64([rotary_dim, pass_dim]) if pass_dim > 0 else None

    def apply(t):
        if pass_dim == 0:
            t_rot = t
            t_pass = None
        else:
            sp = opset.variadic_split(t, _const_i64(-1), sizes)
            t_rot = sp.output(0)
            t_pass = sp.output(1)
        rotated = opset.add(opset.multiply(t_rot, cos_e),
                            opset.multiply(_rotate_half(t_rot, rotary_dim), sin_e))
        return rotated if t_pass is None else opset.concat([rotated, t_pass], axis=-1)

    return apply(q), apply(k)


# ---------------------------------------------------------------------------
# GQA repeat (num_kv_heads -> num_attention_heads)
# ---------------------------------------------------------------------------

def build_repeat_kv(kv, group_size: int, B: int, T: int,
                    num_kv_heads: int, head_dim: int):
    if group_size == 1:
        return kv
    kv5 = opset.unsqueeze(kv, _const_i64(2))                       # [B, kvh, 1, T, hd]
    target = _const_i64([B, num_kv_heads, group_size, T, head_dim])
    kv5 = opset.broadcast(kv5, target)                             # tile by broadcast
    return opset.reshape(kv5,
                         _const_i64([B, num_kv_heads * group_size, T, head_dim]),
                         special_zero=False)


# ---------------------------------------------------------------------------
# Full attention (Qwen3.5 style, with sigmoid output gate from the q-proj)
# ---------------------------------------------------------------------------

def build_full_attention(h, attn, cos, sin, B: int, T: int,
                         num_heads: int, num_kv_heads: int,
                         head_dim: int, rotary_dim: int, eps: float):
    Wq = attn.q_proj.weight.detach().numpy()              # [num_heads * head_dim * 2, hidden]
    Wk = attn.k_proj.weight.detach().numpy()              # [num_kv_heads * head_dim, hidden]
    Wv = attn.v_proj.weight.detach().numpy()
    Wo = attn.o_proj.weight.detach().numpy()              # [hidden, num_heads * head_dim]
    qn = attn.q_norm.weight.detach().numpy()
    kn = attn.k_norm.weight.detach().numpy()

    # q_proj outputs (q || gate) per head, both of size head_dim
    q_full = opset.matmul(h, _const(Wq), False, True)
    q_full = opset.reshape(
        q_full, _const_i64([B, T, num_heads, head_dim * 2]), special_zero=False)
    sp = opset.variadic_split(q_full, _const_i64(-1), _const_i64([head_dim, head_dim]))
    q = sp.output(0)
    gate = sp.output(1)
    gate = opset.reshape(
        gate, _const_i64([B, T, num_heads * head_dim]), special_zero=False)

    k = opset.matmul(h, _const(Wk), False, True)
    k = opset.reshape(k, _const_i64([B, T, num_kv_heads, head_dim]), special_zero=False)
    v = opset.matmul(h, _const(Wv), False, True)
    v = opset.reshape(v, _const_i64([B, T, num_kv_heads, head_dim]), special_zero=False)

    # per-head q/k RMSNorm on head_dim axis
    q = build_qwen_rmsnorm(q, qn, eps)
    k = build_qwen_rmsnorm(k, kn, eps)

    # to [B, H, T, hd]
    perm = _const_i64([0, 2, 1, 3])
    q = opset.transpose(q, perm)
    k = opset.transpose(k, perm)
    v = opset.transpose(v, perm)

    q, k = build_partial_rope(q, k, cos, sin, rotary_dim, head_dim)

    # GQA: repeat k, v to num_heads
    group_size = num_heads // num_kv_heads
    k = build_repeat_kv(k, group_size, B, T, num_kv_heads, head_dim)
    v = build_repeat_kv(v, group_size, B, T, num_kv_heads, head_dim)

    scale = _const(np.float32(1.0 / math.sqrt(head_dim)))
    attn_out = opset.scaled_dot_product_attention(
        q, k, v, attention_mask=None, scale=scale, causal=True)

    # back to [B, T, num_heads * head_dim]
    attn_out = opset.transpose(attn_out, _const_i64([0, 2, 1, 3]))
    attn_out = opset.reshape(
        attn_out, _const_i64([B, T, num_heads * head_dim]), special_zero=False)

    # output gate from the q-proj second half
    attn_out = opset.multiply(attn_out, opset.sigmoid(gate))
    return opset.matmul(attn_out, _const(Wo), False, True)


# ---------------------------------------------------------------------------
# Gated DeltaNet linear-attention block
# ---------------------------------------------------------------------------

def build_gated_deltanet(h, layer, B: int, T: int, text_config):
    """Mirror of `Qwen3_5GatedDeltaNet.forward` (no cache, seq_len > 1 path)."""
    H  = text_config.hidden_size
    Hk = text_config.linear_key_head_dim
    Hv = text_config.linear_value_head_dim
    Nk = text_config.linear_num_key_heads
    Nv = text_config.linear_num_value_heads
    K  = text_config.linear_conv_kernel_dim
    eps = text_config.rms_norm_eps
    key_dim = Hk * Nk
    value_dim = Hv * Nv
    assert Nv == Nk, "this builder assumes num_v_heads == num_k_heads"
    del H, K  # unused locally; kept for documentation

    Wqkv = layer.in_proj_qkv.weight.detach().numpy()
    Wz   = layer.in_proj_z.weight.detach().numpy()
    Wb   = layer.in_proj_b.weight.detach().numpy()
    Wa   = layer.in_proj_a.weight.detach().numpy()
    Wout = layer.out_proj.weight.detach().numpy()

    # nn.Conv1d's kernel index K-1 is the "current step"; ShortConv1D wants
    # index 0 = current step, so reverse the kernel along K.
    conv_w = layer.conv1d.weight.detach().numpy().squeeze(1)     # [conv_dim, K]
    echo_conv_w = conv_w[:, ::-1].copy()
    A_log    = layer.A_log.detach().numpy()
    dt_bias  = layer.dt_bias.detach().numpy()
    norm_w   = layer.norm.weight.detach().numpy()

    mixed_qkv = opset.matmul(h, _const(Wqkv), False, True)
    conv_out = ShortConv1D([mixed_qkv, _const(echo_conv_w)])
    mixed_qkv_act = opset.swish(conv_out)

    sp = opset.variadic_split(
        mixed_qkv_act, _const_i64(-1), _const_i64([key_dim, key_dim, value_dim]))
    q_flat = sp.output(0)
    k_flat = sp.output(1)
    v_flat = sp.output(2)

    def reshape_heads(t, num_heads, hd):
        return opset.reshape(
            t, _const_i64([B, T, num_heads, hd]), special_zero=False)

    q = reshape_heads(q_flat, Nk, Hk)
    k = reshape_heads(k_flat, Nk, Hk)
    v = reshape_heads(v_flat, Nv, Hv)

    z = opset.matmul(h, _const(Wz), False, True)
    z = reshape_heads(z, Nv, Hv)

    b_node = opset.matmul(h, _const(Wb), False, True)
    beta = opset.sigmoid(b_node)

    a_node = opset.matmul(h, _const(Wa), False, True)
    a_plus = opset.add(a_node, _const(dt_bias))
    sp_a = opset.softplus(a_plus)
    g_log = opset.multiply(_const(-np.exp(A_log)), sp_a)
    g_actual = opset.exp(g_log)

    perm4 = _const_i64([0, 2, 1, 3])
    q_t = opset.transpose(q, perm4)
    k_t = opset.transpose(k, perm4)
    v_t = opset.transpose(v, perm4)

    perm3 = _const_i64([0, 2, 1])
    beta_t = opset.transpose(beta, perm3)
    g_t = opset.transpose(g_actual, perm3)

    q_n = L2Norm([q_t], eps=1e-6)
    k_n = L2Norm([k_t], eps=1e-6)
    q_scaled = opset.multiply(q_n, _const(np.float32(1.0 / math.sqrt(Hk))))

    s0 = _const(np.zeros((B, Nv, Hk, Hv), dtype=np.float32))
    rule = GatedDeltaRule([q_scaled, k_n, v_t, g_t, beta_t, s0])
    core_attn_out = rule.output(0)              # [B, Nv, T, Hv]
    core_attn_out = opset.transpose(core_attn_out, perm4)  # [B, T, Nv, Hv]

    flat = _const_i64([-1, Hv])
    core_flat = opset.reshape(core_attn_out, flat, special_zero=False)
    z_flat = opset.reshape(z, flat, special_zero=False)
    post = GatedRMSNorm([core_flat, z_flat, _const(norm_w)], eps=eps)
    post = opset.reshape(
        post, _const_i64([B, T, value_dim]), special_zero=False)
    return opset.matmul(post, _const(Wout), False, True)


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------

def build_decoder_layer(h, decoder, cos, sin, B: int, T: int, text_config):
    eps = text_config.rms_norm_eps

    # input_layernorm -> mixer -> residual
    h_n = build_qwen_rmsnorm(
        h, decoder.input_layernorm.weight.detach().numpy(), eps)
    if decoder.layer_type == "linear_attention":
        mixer_out = build_gated_deltanet(h_n, decoder.linear_attn, B, T, text_config)
    elif decoder.layer_type == "full_attention":
        head_dim = text_config.head_dim
        # rotary_dim from cos's last dim
        rope_pf = text_config.rope_parameters.get("partial_rotary_factor", 1.0)
        rotary_dim = int(head_dim * rope_pf)
        mixer_out = build_full_attention(
            h_n, decoder.self_attn, cos, sin, B, T,
            text_config.num_attention_heads,
            text_config.num_key_value_heads,
            head_dim, rotary_dim, eps)
    else:
        raise ValueError(f"unknown layer_type {decoder.layer_type!r}")
    h = opset.add(h, mixer_out)

    # post_attention_layernorm -> mlp -> residual
    h_n = build_qwen_rmsnorm(
        h, decoder.post_attention_layernorm.weight.detach().numpy(), eps)
    h = opset.add(h, build_swiglu_mlp(h_n, decoder.mlp))
    return h


# ---------------------------------------------------------------------------
# Top-level prefill
# ---------------------------------------------------------------------------

def precompute_rope(text_model, B: int, T: int):
    """Run torch's rotary_emb once to get cos/sin for [0..T-1] positions."""
    pos_ids = torch.arange(T, dtype=torch.long).unsqueeze(0).expand(B, -1)
    dummy = torch.zeros(B, 1, 1, dtype=torch.float32)
    with torch.no_grad():
        cos, sin = text_model.rotary_emb(dummy, pos_ids)
    return cos.detach().numpy(), sin.detach().numpy()


def build_text_prefill(input_ids_param, model, B: int, T: int):
    """Build the full prefill graph.

    Args:
        input_ids_param: int64 OV Parameter of shape [B, T].
        model: torch `Qwen3_5ForConditionalGeneration` (or anything with
               .model.language_model and .lm_head).
        B, T: static batch / sequence length.

    Returns: a Node producing logits of shape [B, T, vocab_size].
    """
    text_model = model.model.language_model
    text_config = text_model.config
    eps = text_config.rms_norm_eps

    embed_w = text_model.embed_tokens.weight.detach().numpy()       # [vocab, hidden]
    h = opset.gather(_const(embed_w), input_ids_param, _const_i64(0))

    cos_np, sin_np = precompute_rope(text_model, B, T)
    cos_c = _const(cos_np)
    sin_c = _const(sin_np)

    for decoder in text_model.layers:
        h = build_decoder_layer(h, decoder, cos_c, sin_c, B, T, text_config)

    h = build_qwen_rmsnorm(h, text_model.norm.weight.detach().numpy(), eps)
    return opset.matmul(h, _const(model.lm_head.weight.detach().numpy()),
                        False, True)


# ---------------------------------------------------------------------------
# Decode-mode builders (single-token, with caches)
# ---------------------------------------------------------------------------

def build_gated_deltanet_decode(h, layer, conv_state_in, recurrent_state_in,
                                B: int, text_config):
    """Single-token gated DeltaNet update with cached state.

    h:                 [B, 1, hidden]
    conv_state_in:     [B, conv_dim, K]                       (rolling conv window)
    recurrent_state_in:[B, num_v_heads, head_k_dim, head_v_dim]

    Returns: (output[B,1,hidden], new_conv_state, new_recurrent_state).
    """
    Hk = text_config.linear_key_head_dim
    Hv = text_config.linear_value_head_dim
    Nk = text_config.linear_num_key_heads
    Nv = text_config.linear_num_value_heads
    K  = text_config.linear_conv_kernel_dim
    eps = text_config.rms_norm_eps
    key_dim = Hk * Nk
    value_dim = Hv * Nv
    assert Nv == Nk, "this builder assumes num_v_heads == num_k_heads"

    Wqkv = layer.in_proj_qkv.weight.detach().numpy()
    Wz   = layer.in_proj_z.weight.detach().numpy()
    Wb   = layer.in_proj_b.weight.detach().numpy()
    Wa   = layer.in_proj_a.weight.detach().numpy()
    Wout = layer.out_proj.weight.detach().numpy()
    # nn.Conv1d weights: [conv_dim, 1, K]. For decode the convolution applies
    # at the right edge of [conv_state | new_token]; with that K-step window
    # taken in natural time-order, the dot product uses nn.Conv1d's weight
    # as-is (no flip): out[c] = sum_i nn_weight[c, i] * window[c, i].
    nn_weight = layer.conv1d.weight.detach().numpy().squeeze(1)        # [conv_dim, K]
    A_log    = layer.A_log.detach().numpy()
    dt_bias  = layer.dt_bias.detach().numpy()
    norm_w   = layer.norm.weight.detach().numpy()

    # in_proj_qkv -> [B, 1, conv_dim] -> [B, conv_dim, 1] for the conv pipeline
    mixed_qkv = opset.matmul(h, _const(Wqkv), False, True)
    mixed_qkv_t = opset.transpose(mixed_qkv, _const_i64([0, 2, 1]))

    # Window: [conv_state | mixed_qkv_t]   shape [B, conv_dim, K + 1]
    window = opset.concat([conv_state_in, mixed_qkv_t], axis=2)
    # New conv state = window[..., 1:K+1]   (drop oldest, keep last K)
    new_conv_state = opset.slice(
        window,
        _const_i64([1]), _const_i64([K + 1]), _const_i64([1]), _const_i64([2]))

    # Single-step depthwise conv at the right edge:
    #   conv_out[c] = sum_i nn_weight[c, i] * new_conv_state[c, i]
    weight_b = opset.unsqueeze(_const(nn_weight), _const_i64(0))       # [1, conv_dim, K]
    prod = opset.multiply(new_conv_state, weight_b)                    # [B, conv_dim, K]
    conv_out = opset.reduce_sum(prod, _const_i64([2]), keep_dims=True) # [B, conv_dim, 1]
    conv_out = opset.transpose(conv_out, _const_i64([0, 2, 1]))        # [B, 1, conv_dim]
    mixed_qkv_act = opset.swish(conv_out)                              # silu

    # Split into q, k, v and reshape per-head
    sp = opset.variadic_split(
        mixed_qkv_act, _const_i64(-1), _const_i64([key_dim, key_dim, value_dim]))
    q_flat = sp.output(0); k_flat = sp.output(1); v_flat = sp.output(2)

    def reshape_heads(t, num_heads, hd):
        return opset.reshape(
            t, _const_i64([B, 1, num_heads, hd]), special_zero=False)

    q = reshape_heads(q_flat, Nk, Hk)
    k = reshape_heads(k_flat, Nk, Hk)
    v = reshape_heads(v_flat, Nv, Hv)

    z = opset.matmul(h, _const(Wz), False, True)
    z = reshape_heads(z, Nv, Hv)

    b_node = opset.matmul(h, _const(Wb), False, True)
    beta = opset.sigmoid(b_node)                                       # [B, 1, Nv]

    a_node = opset.matmul(h, _const(Wa), False, True)
    a_plus = opset.add(a_node, _const(dt_bias))
    sp_a = opset.softplus(a_plus)
    g_log = opset.multiply(_const(-np.exp(A_log)), sp_a)
    g_actual = opset.exp(g_log)                                        # [B, 1, Nv]

    # Squeeze the T=1 axis for the Step op.
    one_axis = _const_i64([1])
    q_step    = opset.squeeze(q, one_axis)        # [B, Nk, Hk]
    k_step    = opset.squeeze(k, one_axis)
    v_step    = opset.squeeze(v, one_axis)        # [B, Nv, Hv]
    g_step    = opset.squeeze(g_actual, one_axis) # [B, Nv]
    beta_step = opset.squeeze(beta, one_axis)

    # L2 norm + 1/sqrt(Hk) scaling on q
    q_n = L2Norm([q_step], eps=1e-6)
    k_n = L2Norm([k_step], eps=1e-6)
    q_scaled = opset.multiply(q_n, _const(np.float32(1.0 / math.sqrt(Hk))))

    # Single-step recurrent update
    step = GatedDeltaRuleStep(
        [recurrent_state_in, q_scaled, k_n, v_step, g_step, beta_step])
    new_recurrent_state = step.output(0)        # [B, Nv, Hk, Hv]
    o_step = step.output(1)                     # [B, Nv, Hv]

    # Restore the T=1 axis and run GatedRMSNorm + out_proj
    o_unsq = opset.unsqueeze(o_step, one_axis)  # [B, 1, Nv, Hv]
    flat = _const_i64([-1, Hv])
    core_flat = opset.reshape(o_unsq, flat, special_zero=False)
    z_flat = opset.reshape(z, flat, special_zero=False)
    post = GatedRMSNorm([core_flat, z_flat, _const(norm_w)], eps=eps)
    post = opset.reshape(
        post, _const_i64([B, 1, value_dim]), special_zero=False)
    output = opset.matmul(post, _const(Wout), False, True)
    return output, new_conv_state, new_recurrent_state


def _repeat_kv_dynamic(kv, group_size: int, B: int, num_kv_heads: int,
                       head_dim: int):
    """GQA repeat where the T axis is dynamic. Equivalent to repeat_interleave
    on dim=1 by `group_size`."""
    if group_size == 1:
        return kv
    # [B, kvh, T, hd] -> [B, kvh, 1, T, hd] -> tile g -> [B, kvh*g, T, hd]
    kv5 = opset.unsqueeze(kv, _const_i64(2))
    kv5 = opset.tile(kv5, _const_i64([1, 1, group_size, 1, 1]))
    return opset.reshape(
        kv5,
        _const_i64([B, num_kv_heads * group_size, -1, head_dim]),
        special_zero=False)


def build_full_attention_decode(h, attn, cos_p, sin_p, k_cache_in, v_cache_in,
                                B: int, num_heads: int, num_kv_heads: int,
                                head_dim: int, rotary_dim: int, eps: float):
    """Full attention with KV cache for a single new token.

    h:           [B, 1, hidden]
    cos_p, sin_p:[B, 1, rotary_dim]   (cos/sin gathered at the new position)
    k/v_cache_in:[B, num_kv_heads, T_past, head_dim]  (T_past dynamic)

    Returns: (output[B,1,hidden], new_k_cache, new_v_cache) where the new
    caches have the new token appended along the T axis.
    """
    Wq = attn.q_proj.weight.detach().numpy()
    Wk = attn.k_proj.weight.detach().numpy()
    Wv = attn.v_proj.weight.detach().numpy()
    Wo = attn.o_proj.weight.detach().numpy()
    qn = attn.q_norm.weight.detach().numpy()
    kn = attn.k_norm.weight.detach().numpy()

    q_full = opset.matmul(h, _const(Wq), False, True)
    q_full = opset.reshape(
        q_full, _const_i64([B, 1, num_heads, head_dim * 2]), special_zero=False)
    sp = opset.variadic_split(q_full, _const_i64(-1), _const_i64([head_dim, head_dim]))
    q = sp.output(0)
    gate = sp.output(1)
    gate = opset.reshape(
        gate, _const_i64([B, 1, num_heads * head_dim]), special_zero=False)

    k_new = opset.matmul(h, _const(Wk), False, True)
    k_new = opset.reshape(
        k_new, _const_i64([B, 1, num_kv_heads, head_dim]), special_zero=False)
    v_new = opset.matmul(h, _const(Wv), False, True)
    v_new = opset.reshape(
        v_new, _const_i64([B, 1, num_kv_heads, head_dim]), special_zero=False)

    q = build_qwen_rmsnorm(q, qn, eps)
    k_new = build_qwen_rmsnorm(k_new, kn, eps)

    perm = _const_i64([0, 2, 1, 3])
    q = opset.transpose(q, perm)            # [B, num_heads, 1, hd]
    k_new = opset.transpose(k_new, perm)    # [B, num_kv_heads, 1, hd]
    v_new = opset.transpose(v_new, perm)

    q, k_new = build_partial_rope(q, k_new, cos_p, sin_p, rotary_dim, head_dim)

    # Append to caches along T axis
    new_k_cache = opset.concat([k_cache_in, k_new], axis=2)
    new_v_cache = opset.concat([v_cache_in, v_new], axis=2)

    # GQA expansion of the appended caches
    group_size = num_heads // num_kv_heads
    k_full = _repeat_kv_dynamic(new_k_cache, group_size, B, num_kv_heads, head_dim)
    v_full = _repeat_kv_dynamic(new_v_cache, group_size, B, num_kv_heads, head_dim)

    # SDPA: T_query = 1, so no causal mask needed (q sees all stored keys).
    scale = _const(np.float32(1.0 / math.sqrt(head_dim)))
    attn_out = opset.scaled_dot_product_attention(
        q, k_full, v_full, attention_mask=None, scale=scale, causal=False)

    attn_out = opset.transpose(attn_out, _const_i64([0, 2, 1, 3]))   # [B, 1, H, hd]
    attn_out = opset.reshape(
        attn_out, _const_i64([B, 1, num_heads * head_dim]), special_zero=False)
    attn_out = opset.multiply(attn_out, opset.sigmoid(gate))
    output = opset.matmul(attn_out, _const(Wo), False, True)
    return output, new_k_cache, new_v_cache


def precompute_rope_table(text_model, max_pos: int):
    """Precompute cos/sin for every position in [0, max_pos) as a 2D table.

    Returns numpy arrays of shape [max_pos, rotary_dim] each.
    """
    pos_ids = torch.arange(max_pos, dtype=torch.long).unsqueeze(0)
    dummy = torch.zeros(1, 1, 1, dtype=torch.float32)
    with torch.no_grad():
        cos, sin = text_model.rotary_emb(dummy, pos_ids)
    # cos shape [1, max_pos, rotary_dim]; collapse the leading batch dim.
    return cos[0].detach().numpy(), sin[0].detach().numpy()


def build_text_decode(input_ids, position_id, model, B: int,
                      conv_states, recurrent_states, k_caches, v_caches):
    """Build the single-token decode graph.

    input_ids:        int64 Parameter [B, 1]
    position_id:      int64 Parameter [B, 1]   (absolute position of the new token)
    conv_states:      list of f32 Parameters [B, conv_dim, K]               (one per linear-attn layer)
    recurrent_states: list of f32 Parameters [B, Nv, Hk, Hv]                (one per linear-attn layer)
    k_caches:         list of f32 Parameters [B, num_kv_heads, ?, head_dim] (one per full-attn layer; T_past dynamic)
    v_caches:         list of f32 Parameters [B, num_kv_heads, ?, head_dim]

    Returns:
        logits node [B, 1, vocab],
        new_conv_states (list), new_recurrent_states (list),
        new_k_caches (list), new_v_caches (list).
    """
    text_model = model.model.language_model
    text_config = text_model.config
    eps = text_config.rms_norm_eps
    head_dim = text_config.head_dim
    rope_pf = text_config.rope_parameters.get("partial_rotary_factor", 1.0)
    rotary_dim = int(head_dim * rope_pf)

    # Embed new token
    embed_w = text_model.embed_tokens.weight.detach().numpy()
    h = opset.gather(_const(embed_w), input_ids, _const_i64(0))         # [B, 1, hidden]

    # Gather cos/sin at position_id from a precomputed table covering all positions.
    cos_table, sin_table = precompute_rope_table(
        text_model, text_config.max_position_embeddings)
    cos_p = opset.gather(_const(cos_table), position_id, _const_i64(0)) # [B, 1, rotary_dim]
    sin_p = opset.gather(_const(sin_table), position_id, _const_i64(0))

    new_conv_states = []
    new_recurrent_states = []
    new_k_caches = []
    new_v_caches = []
    li = 0
    fi = 0
    for decoder in text_model.layers:
        h_n = build_qwen_rmsnorm(
            h, decoder.input_layernorm.weight.detach().numpy(), eps)
        if decoder.layer_type == "linear_attention":
            mix, new_conv, new_rec = build_gated_deltanet_decode(
                h_n, decoder.linear_attn,
                conv_states[li], recurrent_states[li], B, text_config)
            new_conv_states.append(new_conv)
            new_recurrent_states.append(new_rec)
            li += 1
        elif decoder.layer_type == "full_attention":
            mix, new_k, new_v = build_full_attention_decode(
                h_n, decoder.self_attn, cos_p, sin_p,
                k_caches[fi], v_caches[fi],
                B, text_config.num_attention_heads,
                text_config.num_key_value_heads,
                head_dim, rotary_dim, eps)
            new_k_caches.append(new_k)
            new_v_caches.append(new_v)
            fi += 1
        else:
            raise ValueError(f"unknown layer_type {decoder.layer_type!r}")
        h = opset.add(h, mix)

        h_n = build_qwen_rmsnorm(
            h, decoder.post_attention_layernorm.weight.detach().numpy(), eps)
        h = opset.add(h, build_swiglu_mlp(h_n, decoder.mlp))

    h = build_qwen_rmsnorm(h, text_model.norm.weight.detach().numpy(), eps)
    logits = opset.matmul(
        h, _const(model.lm_head.weight.detach().numpy()), False, True)
    return logits, new_conv_states, new_recurrent_states, new_k_caches, new_v_caches
