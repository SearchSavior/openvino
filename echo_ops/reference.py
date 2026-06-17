# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""NumPy reference implementations for the gated DeltaNet ops.

These are used by the custom-op `evaluate()` methods so the ops work
during CPU inference and constant folding. They are also handy as the
ground truth in tests.
"""

import numpy as np


def l2_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """L2-normalize x along its last dimension."""
    denom = np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + eps)
    return x / denom


def short_conv1d(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Causal depthwise 1-D convolution along the time axis.

    x:      [B, T, D]
    weight: [D, K]   (per-channel kernel; weight[:, 0] is the current step)
    returns [B, T, D]
    """
    B, T, D = x.shape
    K = weight.shape[1]
    y = np.zeros_like(x)
    for i in range(K):
        if i == 0:
            y += x * weight[None, None, :, 0].reshape(1, 1, D)
        else:
            shifted = np.zeros_like(x)
            shifted[:, i:, :] = x[:, : T - i, :]
            y += shifted * weight[None, None, :, i].reshape(1, 1, D)
    return y


def gated_delta_rule(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    g: np.ndarray,
    beta: np.ndarray,
    initial_state: np.ndarray,
):
    """Gated DeltaNet recurrence over a full sequence.

    Convention (matches Yang et al., "Gated Delta Networks"):
        S_t = g_t * S_{t-1} + beta_t * k_t (v_t - g_t * S_{t-1}^T k_t)^T
        o_t = S_t^T q_t

    Shapes:
        q, k:          [B, H, T, Dk]
        v:             [B, H, T, Dv]
        g, beta:       [B, H, T]
        initial_state: [B, H, Dk, Dv]
    Returns:
        o:           [B, H, T, Dv]
        final_state: [B, H, Dk, Dv]
    """
    B, H, T, Dk = q.shape
    Dv = v.shape[-1]
    dtype = q.dtype
    S = initial_state.astype(dtype, copy=True)
    o = np.zeros((B, H, T, Dv), dtype=dtype)
    for t in range(T):
        kt = k[:, :, t, :]
        vt = v[:, :, t, :]
        qt = q[:, :, t, :]
        gt = g[:, :, t]
        bt = beta[:, :, t]
        # v_pred = S^T @ kt  -> [B, H, Dv]
        v_pred = np.einsum("bhde,bhd->bhe", S, kt)
        # error gated by g_t
        err = vt - gt[..., None] * v_pred
        # outer(kt, bt * err) -> [B, H, Dk, Dv]
        upd = np.einsum("bhd,bhe->bhde", kt, bt[..., None] * err)
        S = gt[..., None, None] * S + upd
        o[:, :, t, :] = np.einsum("bhde,bhd->bhe", S, qt)
    return o, S


def gated_delta_rule_step(
    state: np.ndarray,
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    g: np.ndarray,
    beta: np.ndarray,
):
    """Single-step gated DeltaNet recurrence (for autoregressive decoding).

    Shapes:
        state: [B, H, Dk, Dv]
        q, k:  [B, H, Dk]
        v:     [B, H, Dv]
        g, b:  [B, H]
    Returns:
        new_state: [B, H, Dk, Dv]
        o:         [B, H, Dv]
    """
    dtype = q.dtype
    S = state.astype(dtype, copy=True)
    v_pred = np.einsum("bhde,bhd->bhe", S, k)
    err = v - g[..., None] * v_pred
    upd = np.einsum("bhd,bhe->bhde", k, beta[..., None] * err)
    S = g[..., None, None] * S + upd
    o = np.einsum("bhde,bhd->bhe", S, q)
    return S, o


def gated_rmsnorm(
    x: np.ndarray,
    gate: np.ndarray,
    weight: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """RMSNorm with a SiLU output gate.

        y = silu(gate) * weight * x / sqrt(mean(x^2, last) + eps)

    Shapes:
        x, gate: [..., D]
        weight:  [D]
    """
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    silu_gate = gate / (1.0 + np.exp(-gate))
    return silu_gate * weight * (x / rms)
