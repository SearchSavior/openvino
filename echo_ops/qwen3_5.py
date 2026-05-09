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
    """Return an OV Constant from a numpy array or scalar.

    By default, bakes f32. When `set_weight_dtype(np.float16)` has been
    called, the constant is stored directly as f16 -- and the rest of the
    graph (activations, custom-op inputs) flows in f16 too. This halves
    both IR bin size and the runtime weight footprint compared to fp32.
    Custom op evaluate() methods upcast f16 inputs to f32 internally for
    the reference math.

    Preserves rank-0 inputs as scalars (np.ascontiguousarray would
    upgrade them to rank-1).
    """
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim > 0 and not a.flags["C_CONTIGUOUS"]:
        a = np.ascontiguousarray(a)
    if _DTYPE_WEIGHTS == np.float16:
        a = a.astype(np.float16)
    return opset.constant(a)


def _const_i64(arr) -> "opset.Node":
    """Return an int64 OV Constant for shapes/indices, preserving rank-0."""
    a = np.asarray(arr, dtype=np.int64)
    if a.ndim > 0 and not a.flags["C_CONTIGUOUS"]:
        a = np.ascontiguousarray(a)
    return opset.constant(a)


def _silu(x):
    """SiLU / Swish-1 implemented as x * sigmoid(x). opset.swish defaults
    its beta to f32 and would fail with f16 inputs."""
    return opset.multiply(x, opset.sigmoid(x))


# Active dtype for weight constants baked into the OV graph. f32 by default;
# call set_weight_dtype(np.float16) before constructing graphs to switch.
_DTYPE_WEIGHTS = np.float32


def set_weight_dtype(dtype) -> None:
    global _DTYPE_WEIGHTS
    if dtype not in (np.float32, np.float16):
        raise ValueError(f"unsupported weight dtype {dtype!r}")
    _DTYPE_WEIGHTS = dtype


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
    g = _silu(opset.matmul(h, _const(Wg), False, True))   # silu(gate)
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

def build_repeat_kv(kv, group_size: int, B: int, T,
                    num_kv_heads: int, head_dim: int):
    """Repeat KV heads. T may be -1 to indicate a dynamic time dimension."""
    if group_size == 1:
        return kv
    kv5 = opset.unsqueeze(kv, _const_i64(2))                       # [B, kvh, 1, T, hd]
    kv5 = opset.tile(kv5, _const_i64([1, 1, group_size, 1, 1]))    # [B, kvh, g, T, hd]
    return opset.reshape(
        kv5,
        _const_i64([B, num_kv_heads * group_size, -1 if T == -1 else T, head_dim]),
        special_zero=False)


# ---------------------------------------------------------------------------
# Full attention (Qwen3.5 style, with sigmoid output gate from the q-proj)
# ---------------------------------------------------------------------------

def build_full_attention(h, attn, cos, sin, B: int, T,
                         num_heads: int, num_kv_heads: int,
                         head_dim: int, rotary_dim: int, eps: float,
                         k_cache_in, v_cache_in):
    """Unified full-attention block that consumes a KV cache and emits the
    appended cache. T may be -1 (dynamic).

    Caller is responsible for passing the right cache shapes:
        fresh prefill:  zero-length [B, num_kv_heads, 0, head_dim]
        decode/continue: prior call's cache outputs

    Returns (output, new_k_cache, new_v_cache); new caches have
    [B, num_kv_heads, T_past + T, head_dim].
    """
    Wq = attn.q_proj.weight.detach().numpy()              # [num_heads * head_dim * 2, hidden]
    Wk = attn.k_proj.weight.detach().numpy()              # [num_kv_heads * head_dim, hidden]
    Wv = attn.v_proj.weight.detach().numpy()
    Wo = attn.o_proj.weight.detach().numpy()              # [hidden, num_heads * head_dim]
    qn = attn.q_norm.weight.detach().numpy()
    kn = attn.k_norm.weight.detach().numpy()

    Td = -1 if T == -1 else T

    # q_proj outputs (q || gate) per head, both of size head_dim
    q_full = opset.matmul(h, _const(Wq), False, True)
    q_full = opset.reshape(
        q_full, _const_i64([B, Td, num_heads, head_dim * 2]), special_zero=False)
    sp = opset.variadic_split(q_full, _const_i64(-1), _const_i64([head_dim, head_dim]))
    q = sp.output(0)
    gate = sp.output(1)
    gate = opset.reshape(
        gate, _const_i64([B, Td, num_heads * head_dim]), special_zero=False)

    k_new = opset.matmul(h, _const(Wk), False, True)
    k_new = opset.reshape(k_new, _const_i64([B, Td, num_kv_heads, head_dim]), special_zero=False)
    v_new = opset.matmul(h, _const(Wv), False, True)
    v_new = opset.reshape(v_new, _const_i64([B, Td, num_kv_heads, head_dim]), special_zero=False)

    # per-head q/k RMSNorm on head_dim axis
    q = build_qwen_rmsnorm(q, qn, eps)
    k_new = build_qwen_rmsnorm(k_new, kn, eps)

    # to [B, H, T, hd]
    perm = _const_i64([0, 2, 1, 3])
    q = opset.transpose(q, perm)
    k_new = opset.transpose(k_new, perm)
    v_new = opset.transpose(v_new, perm)

    q, k_new = build_partial_rope(q, k_new, cos, sin, rotary_dim, head_dim)

    # Append new K, V to the incoming KV cache (along T axis = 2)
    new_k_cache = opset.concat([k_cache_in, k_new], axis=2)
    new_v_cache = opset.concat([v_cache_in, v_new], axis=2)

    # GQA expansion of the appended cache
    group_size = num_heads // num_kv_heads
    k_full = build_repeat_kv(new_k_cache, group_size, B, -1, num_kv_heads, head_dim)
    v_full = build_repeat_kv(new_v_cache, group_size, B, -1, num_kv_heads, head_dim)

    # Sliding-causal mask of shape [T_query, T_past + T_query]:
    #   mask[i, j] = -inf if j > T_past + i else 0
    # (== upper-triangular at offset T_past). Equivalent to causal=True for
    # T_past=0 (square prefill) and to "no mask" for T_query=1 (decode).
    # OV's built-in causal=True is wrong for T_query<T_key, so we build
    # the mask explicitly. Range/Greater/Select handle dynamic shapes.
    q_shape = opset.shape_of(q, output_type="i64")              # [4]
    k_shape = opset.shape_of(k_full, output_type="i64")
    Lq = opset.gather(q_shape, _const_i64(2), _const_i64(0))     # scalar T_query
    Lk = opset.gather(k_shape, _const_i64(2), _const_i64(0))     # scalar T_total
    Tpast = opset.subtract(Lk, Lq)                               # scalar T_past
    rows = opset.range(_const_i64(0), Lq, _const_i64(1), output_type=Type.i64)   # [Lq]
    cols = opset.range(_const_i64(0), Lk, _const_i64(1), output_type=Type.i64)   # [Lk]
    rows_offset = opset.add(rows, opset.unsqueeze(Tpast, _const_i64(0)))          # [Lq]
    rows_2d = opset.unsqueeze(rows_offset, _const_i64(1))         # [Lq, 1]
    cols_2d = opset.unsqueeze(cols, _const_i64(0))                # [1, Lk]
    mask_bool = opset.greater(cols_2d, rows_2d)                   # [Lq, Lk]
    neg_inf = _const(np.float32(-1e9))
    zero    = _const(np.float32(0.0))
    mask_f = opset.select(mask_bool, neg_inf, zero)               # [Lq, Lk]
    mask_4d = opset.unsqueeze(
        opset.unsqueeze(mask_f, _const_i64(0)), _const_i64(0))    # [1, 1, Lq, Lk]

    scale = _const(np.float32(1.0 / math.sqrt(head_dim)))
    attn_out = opset.scaled_dot_product_attention(
        q, k_full, v_full, attention_mask=mask_4d, scale=scale, causal=False)

    # back to [B, T, num_heads * head_dim]
    attn_out = opset.transpose(attn_out, _const_i64([0, 2, 1, 3]))
    attn_out = opset.reshape(
        attn_out, _const_i64([B, Td, num_heads * head_dim]), special_zero=False)

    # output gate from the q-proj second half
    attn_out = opset.multiply(attn_out, opset.sigmoid(gate))
    output = opset.matmul(attn_out, _const(Wo), False, True)
    return output, new_k_cache, new_v_cache


# ---------------------------------------------------------------------------
# Gated DeltaNet linear-attention block
# ---------------------------------------------------------------------------

def build_gated_deltanet(h, layer, B: int, T, text_config,
                         conv_state_in, recurrent_state_in):
    """Unified gated-DeltaNet block that consumes a (conv_state, recurrent_state)
    cache and emits its updated form. T may be -1 (dynamic).

    Caller passes:
        conv_state_in:      [B, conv_dim, K]                       (zeros for fresh prefill)
        recurrent_state_in: [B, num_v_heads, head_k_dim, head_v_dim]   (zeros for fresh prefill)

    Returns (output, new_conv_state, new_recurrent_state).
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

    Td = -1 if T == -1 else T

    Wqkv = layer.in_proj_qkv.weight.detach().numpy()
    Wz   = layer.in_proj_z.weight.detach().numpy()
    Wb   = layer.in_proj_b.weight.detach().numpy()
    Wa   = layer.in_proj_a.weight.detach().numpy()
    Wout = layer.out_proj.weight.detach().numpy()

    # nn.Conv1d weight is [conv_dim, 1, K]; ShortConv1D wants weight[c, 0]
    # to be the current step, so reverse along K.
    conv_w = layer.conv1d.weight.detach().numpy().squeeze(1)
    echo_conv_w = conv_w[:, ::-1].copy()
    A_log    = layer.A_log.detach().numpy()
    dt_bias  = layer.dt_bias.detach().numpy()
    norm_w   = layer.norm.weight.detach().numpy()

    mixed_qkv = opset.matmul(h, _const(Wqkv), False, True)         # [B, T, conv_dim]

    # Combine the K-step conv-state cache with the new T tokens along time:
    #   cache:    [B, conv_dim, K]    (cache convention)
    #   reshape:  [B, K, conv_dim]    (time-axis-1 form for concat)
    #   combined: [B, K + T, conv_dim]
    cache_BKC = opset.transpose(conv_state_in, _const_i64([0, 2, 1]))
    combined = opset.concat([cache_BKC, mixed_qkv], axis=1)

    # Causal depthwise conv across the combined window. ShortConv1D output
    # at index t reads combined[t-K+1..t] (zero-padded for negative). The
    # last T outputs (indices [K, K+T)) are exactly the conv values for
    # the new tokens, with their full K-step history coming from the
    # cache. (Matches torch's F.silu(self.conv1d(...)[:, :, :T]) for the
    # zero-cache case and torch's causal_conv1d_update for T == 1.)
    conv_full = ShortConv1D([combined, _const(echo_conv_w)])
    conv_out = opset.slice(
        conv_full,
        _const_i64([K]), _const_i64([2 ** 31 - 1]),
        _const_i64([1]), _const_i64([1]))
    mixed_qkv_act = _silu(conv_out)

    # New conv state seed for next call: last K values of `combined`,
    # transposed back into cache convention.
    new_conv_BKC = opset.slice(
        combined,
        _const_i64([-K]), _const_i64([2 ** 31 - 1]),
        _const_i64([1]), _const_i64([1]))
    new_conv_state = opset.transpose(new_conv_BKC, _const_i64([0, 2, 1]))

    sp = opset.variadic_split(
        mixed_qkv_act, _const_i64(-1), _const_i64([key_dim, key_dim, value_dim]))
    q_flat = sp.output(0)
    k_flat = sp.output(1)
    v_flat = sp.output(2)

    def reshape_heads(t, num_heads, hd):
        return opset.reshape(
            t, _const_i64([B, Td, num_heads, hd]), special_zero=False)

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

    rule = GatedDeltaRule([q_scaled, k_n, v_t, g_t, beta_t, recurrent_state_in])
    core_attn_out = rule.output(0)                              # [B, Nv, T, Hv]
    new_recurrent_state = rule.output(1)                        # [B, Nv, Hk, Hv]
    core_attn_out = opset.transpose(core_attn_out, perm4)       # [B, T, Nv, Hv]

    flat = _const_i64([-1, Hv])
    core_flat = opset.reshape(core_attn_out, flat, special_zero=False)
    z_flat = opset.reshape(z, flat, special_zero=False)
    post = GatedRMSNorm([core_flat, z_flat, _const(norm_w)], eps=eps)
    post = opset.reshape(
        post, _const_i64([B, Td, value_dim]), special_zero=False)
    output = opset.matmul(post, _const(Wout), False, True)
    return output, new_conv_state, new_recurrent_state


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------

def build_decoder_layer(h, decoder, cos, sin, B: int, T, text_config,
                        conv_state_in=None, recurrent_state_in=None,
                        k_cache_in=None, v_cache_in=None):
    """Run one decoder layer with cache I/O. T may be -1 (dynamic).

    Pass either (conv_state_in, recurrent_state_in) or (k_cache_in,
    v_cache_in) depending on `decoder.layer_type`. For fresh prefill the
    relevant inputs should be zero-shaped Constants; for decode they're
    Parameters carrying the prior call's cache outputs.

    Returns (output, layer_caches) where `layer_caches` is:
        ('linear', new_conv_state, new_recurrent_state)
        ('full',   new_k_cache,    new_v_cache)
    """
    eps = text_config.rms_norm_eps

    h_n = build_qwen_rmsnorm(
        h, decoder.input_layernorm.weight.detach().numpy(), eps)
    if decoder.layer_type == "linear_attention":
        if conv_state_in is None or recurrent_state_in is None:
            raise ValueError("linear_attention layer needs conv_state_in + recurrent_state_in")
        mixer_out, new_conv, new_recur = build_gated_deltanet(
            h_n, decoder.linear_attn, B, T, text_config,
            conv_state_in, recurrent_state_in)
        layer_caches = ("linear", new_conv, new_recur)
    elif decoder.layer_type == "full_attention":
        if k_cache_in is None or v_cache_in is None:
            raise ValueError("full_attention layer needs k_cache_in + v_cache_in")
        head_dim = text_config.head_dim
        rope_pf = text_config.rope_parameters.get("partial_rotary_factor", 1.0)
        rotary_dim = int(head_dim * rope_pf)
        mixer_out, new_k, new_v = build_full_attention(
            h_n, decoder.self_attn, cos, sin, B, T,
            text_config.num_attention_heads,
            text_config.num_key_value_heads,
            head_dim, rotary_dim, eps,
            k_cache_in, v_cache_in)
        layer_caches = ("full", new_k, new_v)
    else:
        raise ValueError(f"unknown layer_type {decoder.layer_type!r}")
    h = opset.add(h, mixer_out)

    h_n = build_qwen_rmsnorm(
        h, decoder.post_attention_layernorm.weight.detach().numpy(), eps)
    h = opset.add(h, build_swiglu_mlp(h_n, decoder.mlp))
    return h, layer_caches


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


def build_text_unified(input_ids, position_ids, model, B,
                       conv_states_in, recurrent_states_in,
                       k_caches_in, v_caches_in,
                       max_pos: int | None = None):
    """Build the unified prefill+decode text graph.

    Inputs (Nodes -- can be Parameters or Constants):
        input_ids:          [B, T_dyn] int64
        position_ids:       [B, T_dyn] int64  (absolute positions of each new token)
        conv_states_in:     list of [B, conv_dim, K]                 (one per linear layer)
        recurrent_states_in:list of [B, num_v_heads, head_k_dim, head_v_dim]
        k_caches_in:        list of [B, num_kv_heads, T_past, head_dim] (one per full layer)
        v_caches_in:        list of [B, num_kv_heads, T_past, head_dim]

    Caller drives the two modes:
        * fresh prefill:  zero-shaped Constants for every cache input,
                          position_ids = arange(T)
        * decode step:    Parameters carrying the prior call's outputs,
                          position_ids = [T_past] (length 1)

    Returns (logits, new_conv_states, new_recurrent_states,
             new_k_caches, new_v_caches).
    """
    text_model = model.model.language_model
    text_config = text_model.config
    eps = text_config.rms_norm_eps
    if max_pos is None:
        max_pos = text_config.max_position_embeddings

    embed_w = text_model.embed_tokens.weight.detach().numpy()       # [vocab, hidden]
    h = opset.gather(_const(embed_w), input_ids, _const_i64(0))

    cos_table, sin_table = precompute_rope_table(text_model, max_pos)
    cos_c = opset.gather(_const(cos_table), position_ids, _const_i64(0))
    sin_c = opset.gather(_const(sin_table), position_ids, _const_i64(0))

    new_conv = []
    new_recur = []
    new_k = []
    new_v = []
    li = 0
    fi = 0
    for decoder in text_model.layers:
        if decoder.layer_type == "linear_attention":
            h, caches = build_decoder_layer(
                h, decoder, cos_c, sin_c, B, -1, text_config,
                conv_state_in=conv_states_in[li],
                recurrent_state_in=recurrent_states_in[li])
            new_conv.append(caches[1])
            new_recur.append(caches[2])
            li += 1
        else:
            h, caches = build_decoder_layer(
                h, decoder, cos_c, sin_c, B, -1, text_config,
                k_cache_in=k_caches_in[fi],
                v_cache_in=v_caches_in[fi])
            new_k.append(caches[1])
            new_v.append(caches[2])
            fi += 1

    h = build_qwen_rmsnorm(h, text_model.norm.weight.detach().numpy(), eps)
    logits = opset.matmul(
        h, _const(model.lm_head.weight.detach().numpy()), False, True)
    return logits, new_conv, new_recur, new_k, new_v


def _layer_counts(text_model):
    n_lin = sum(1 for l in text_model.layers if l.layer_type == "linear_attention")
    n_full = sum(1 for l in text_model.layers if l.layer_type == "full_attention")
    return n_lin, n_full


def build_text_prefill(input_ids_param, model, B: int, T,
                       max_pos: int | None = None):
    """Backward-compat fresh-prefill entry. Builds the unified graph with
    in-graph zero caches and Range-derived positions.

    Returns (logits, conv_states, recurrent_states, k_caches, v_caches).
    """
    text_model = model.model.language_model
    text_config = text_model.config
    Hk = text_config.linear_key_head_dim
    Hv = text_config.linear_value_head_dim
    Nv = text_config.linear_num_value_heads
    K  = text_config.linear_conv_kernel_dim
    Nkv = text_config.num_key_value_heads
    head_dim = text_config.head_dim
    conv_dim = (Hk * text_config.linear_num_key_heads) * 2 + (Hv * Nv)
    n_lin, n_full = _layer_counts(text_model)

    if max_pos is None:
        max_pos = (text_config.max_position_embeddings
                   if T == -1 else max(T + 1, 8))

    if T == -1:
        ids_shape = opset.shape_of(input_ids_param, output_type="i64")
        T_scalar = opset.gather(ids_shape, _const_i64(1), _const_i64(0))
        positions = opset.range(
            _const_i64(0), T_scalar, _const_i64(1), output_type=Type.i64)
        position_ids = opset.unsqueeze(positions, _const_i64(0))    # [1, T]
    else:
        position_ids = _const_i64(np.arange(T)[None, :])             # [1, T]

    conv_states = [_const(np.zeros((B, conv_dim, K), dtype=np.float32))
                   for _ in range(n_lin)]
    recur_states = [_const(np.zeros((B, Nv, Hk, Hv), dtype=np.float32))
                    for _ in range(n_lin)]
    k_caches = [_const(np.zeros((B, Nkv, 0, head_dim), dtype=np.float32))
                for _ in range(n_full)]
    v_caches = [_const(np.zeros((B, Nkv, 0, head_dim), dtype=np.float32))
                for _ in range(n_full)]

    return build_text_unified(input_ids_param, position_ids, model, B,
                              conv_states, recur_states,
                              k_caches, v_caches, max_pos=max_pos)




def build_text_decode(input_ids, position_id, model, B: int,
                      conv_states, recurrent_states, k_caches, v_caches,
                      max_pos: int | None = None):
    """Thin compat wrapper that forwards to build_text_unified. Existing
    callers can pass `position_id` of shape [B, 1] and one cache Parameter
    per layer; this delegates to the unified graph.
    """
    return build_text_unified(input_ids, position_id, model, B,
                              conv_states, recurrent_states,
                              k_caches, v_caches, max_pos=max_pos)
