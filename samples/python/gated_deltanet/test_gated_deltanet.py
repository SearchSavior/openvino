# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the gated DeltaNet custom ops.

Each test wires the op into a tiny Model, compiles it on CPU and checks
the output matches the NumPy reference. Run from the repo root:

    python -m samples.python.gated_deltanet.test_gated_deltanet
"""

import sys
import numpy as np

import openvino.opset14 as ops
from openvino import Model, Shape, Tensor, compile_model

from . import reference
from .ops import (
    GatedDeltaRule,
    GatedDeltaRuleStep,
    GatedRMSNorm,
    L2Norm,
    ShortConv1D,
)


DEVICE = "CPU"


def _param(shape, name, dtype=np.float32):
    return ops.parameter(Shape(list(shape)), dtype=dtype, name=name)


def _run(model, feeds):
    compiled = compile_model(model, DEVICE)
    req = compiled.create_infer_request()
    out = req.infer({k: Tensor(v) for k, v in feeds.items()})
    return [out[i] for i in range(len(model.outputs))]


def test_l2norm():
    x = np.random.RandomState(0).randn(2, 4, 8).astype(np.float32)
    p = _param(x.shape, "x")
    node = L2Norm([p], eps=1e-6)
    model = Model([ops.result(node)], [p], "L2NormModel")

    (got,) = _run(model, {"x": x})
    ref = reference.l2_norm(x, eps=1e-6)
    assert np.allclose(got, ref, atol=1e-5), f"L2Norm mismatch: max={np.abs(got-ref).max()}"
    print("  L2Norm                ok")


def test_short_conv1d():
    rng = np.random.RandomState(1)
    x = rng.randn(2, 7, 5).astype(np.float32)
    w = rng.randn(5, 4).astype(np.float32)

    px = _param(x.shape, "x")
    pw = _param(w.shape, "w")
    node = ShortConv1D([px, pw])
    model = Model([ops.result(node)], [px, pw], "ShortConv1DModel")

    (got,) = _run(model, {"x": x, "w": w})
    ref = reference.short_conv1d(x, w)
    assert np.allclose(got, ref, atol=1e-5), f"ShortConv1D mismatch: max={np.abs(got-ref).max()}"
    print("  ShortConv1D           ok")


def test_gated_delta_rule():
    rng = np.random.RandomState(2)
    B, H, T, Dk, Dv = 1, 2, 5, 4, 6
    q = rng.randn(B, H, T, Dk).astype(np.float32) * 0.1
    k = rng.randn(B, H, T, Dk).astype(np.float32) * 0.1
    v = rng.randn(B, H, T, Dv).astype(np.float32) * 0.1
    g = rng.uniform(0.5, 1.0, (B, H, T)).astype(np.float32)
    beta = rng.uniform(0.0, 1.0, (B, H, T)).astype(np.float32)
    s0 = np.zeros((B, H, Dk, Dv), dtype=np.float32)

    pq = _param(q.shape, "q")
    pk = _param(k.shape, "k")
    pv = _param(v.shape, "v")
    pg = _param(g.shape, "g")
    pb = _param(beta.shape, "beta")
    ps = _param(s0.shape, "s0")
    node = GatedDeltaRule([pq, pk, pv, pg, pb, ps])
    model = Model(
        [ops.result(node.output(0)), ops.result(node.output(1))],
        [pq, pk, pv, pg, pb, ps],
        "GatedDeltaRuleModel",
    )

    o, s_final = _run(model, {"q": q, "k": k, "v": v, "g": g, "beta": beta, "s0": s0})
    o_ref, s_ref = reference.gated_delta_rule(q, k, v, g, beta, s0)
    assert np.allclose(o, o_ref, atol=1e-5), f"GatedDeltaRule o mismatch: max={np.abs(o-o_ref).max()}"
    assert np.allclose(s_final, s_ref, atol=1e-5), f"GatedDeltaRule state mismatch: max={np.abs(s_final-s_ref).max()}"
    print("  GatedDeltaRule        ok")


def test_gated_delta_rule_step():
    rng = np.random.RandomState(3)
    B, H, Dk, Dv = 2, 2, 3, 4
    s = rng.randn(B, H, Dk, Dv).astype(np.float32) * 0.1
    q = rng.randn(B, H, Dk).astype(np.float32) * 0.1
    k = rng.randn(B, H, Dk).astype(np.float32) * 0.1
    v = rng.randn(B, H, Dv).astype(np.float32) * 0.1
    g = rng.uniform(0.5, 1.0, (B, H)).astype(np.float32)
    beta = rng.uniform(0.0, 1.0, (B, H)).astype(np.float32)

    ps = _param(s.shape, "s")
    pq = _param(q.shape, "q")
    pk = _param(k.shape, "k")
    pv = _param(v.shape, "v")
    pg = _param(g.shape, "g")
    pb = _param(beta.shape, "beta")
    node = GatedDeltaRuleStep([ps, pq, pk, pv, pg, pb])
    model = Model(
        [ops.result(node.output(0)), ops.result(node.output(1))],
        [ps, pq, pk, pv, pg, pb],
        "GatedDeltaRuleStepModel",
    )

    s_new, o = _run(model, {"s": s, "q": q, "k": k, "v": v, "g": g, "beta": beta})
    s_ref, o_ref = reference.gated_delta_rule_step(s, q, k, v, g, beta)
    assert np.allclose(s_new, s_ref, atol=1e-5)
    assert np.allclose(o, o_ref, atol=1e-5)

    # Also check that unrolling Step matches the full sequence op.
    T = 6
    q_seq = rng.randn(B, H, T, Dk).astype(np.float32) * 0.1
    k_seq = rng.randn(B, H, T, Dk).astype(np.float32) * 0.1
    v_seq = rng.randn(B, H, T, Dv).astype(np.float32) * 0.1
    g_seq = rng.uniform(0.5, 1.0, (B, H, T)).astype(np.float32)
    b_seq = rng.uniform(0.0, 1.0, (B, H, T)).astype(np.float32)
    s_init = np.zeros((B, H, Dk, Dv), dtype=np.float32)

    s_state = s_init.copy()
    outs = []
    for t in range(T):
        s_state, o_t = reference.gated_delta_rule_step(
            s_state, q_seq[:, :, t], k_seq[:, :, t], v_seq[:, :, t],
            g_seq[:, :, t], b_seq[:, :, t])
        outs.append(o_t)
    o_unroll = np.stack(outs, axis=2)

    o_full, s_full = reference.gated_delta_rule(q_seq, k_seq, v_seq, g_seq, b_seq, s_init)
    assert np.allclose(o_unroll, o_full, atol=1e-5)
    assert np.allclose(s_state, s_full, atol=1e-5)
    print("  GatedDeltaRuleStep    ok")


def test_gated_rmsnorm():
    rng = np.random.RandomState(4)
    x = rng.randn(2, 3, 8).astype(np.float32)
    gate = rng.randn(2, 3, 8).astype(np.float32)
    w = rng.randn(8).astype(np.float32)

    px = _param(x.shape, "x")
    pg = _param(gate.shape, "gate")
    pw = _param(w.shape, "w")
    node = GatedRMSNorm([px, pg, pw], eps=1e-5)
    model = Model([ops.result(node)], [px, pg, pw], "GatedRMSNormModel")

    (got,) = _run(model, {"x": x, "gate": gate, "w": w})
    ref = reference.gated_rmsnorm(x, gate, w, eps=1e-5)
    assert np.allclose(got, ref, atol=1e-5), f"GatedRMSNorm mismatch: max={np.abs(got-ref).max()}"
    print("  GatedRMSNorm          ok")


def test_full_block():
    """Smoke test: a tiny gated DeltaNet block end-to-end.

    Pipeline:
        x   -> ShortConv1D  -> reshape to [B, H, T, D] -> L2Norm(q), L2Norm(k)
            -> GatedDeltaRule(q, k, v, g, beta, s0) -> o
            -> reshape back to [B, T, H*Dv] -> GatedRMSNorm with gate
    """
    rng = np.random.RandomState(5)
    B, T, H, Dk, Dv = 1, 4, 2, 3, 3
    D = H * Dk  # use Dk == Dv == D/H for simplicity

    x = rng.randn(B, T, D).astype(np.float32) * 0.1
    w_conv = rng.randn(D, 3).astype(np.float32) * 0.1
    g_seq = rng.uniform(0.7, 1.0, (B, H, T)).astype(np.float32)
    b_seq = rng.uniform(0.0, 1.0, (B, H, T)).astype(np.float32)
    gate = rng.randn(B, T, D).astype(np.float32) * 0.5
    rms_w = rng.randn(D).astype(np.float32)
    s0 = np.zeros((B, H, Dk, Dv), dtype=np.float32)

    # Reference path
    conv_x = reference.short_conv1d(x, w_conv)
    qkv = conv_x.reshape(B, T, H, Dk).transpose(0, 2, 1, 3)  # [B, H, T, Dk]
    q_ref = reference.l2_norm(qkv)
    k_ref = reference.l2_norm(qkv)
    v_ref = qkv  # share for simplicity
    o_seq, _ = reference.gated_delta_rule(q_ref, k_ref, v_ref, g_seq, b_seq, s0)
    o_flat = o_seq.transpose(0, 2, 1, 3).reshape(B, T, D)
    ref_out = reference.gated_rmsnorm(o_flat, gate, rms_w)

    # OV graph
    px = _param(x.shape, "x")
    pwc = _param(w_conv.shape, "w_conv")
    pg = _param(g_seq.shape, "g")
    pb = _param(b_seq.shape, "beta")
    pgate = _param(gate.shape, "gate")
    pw = _param(rms_w.shape, "rms_w")
    ps0 = _param(s0.shape, "s0")

    conv_node = ShortConv1D([px, pwc])
    # reshape to [B, T, H, Dk] then transpose to [B, H, T, Dk]
    reshape_pattern = ops.constant(np.array([B, T, H, Dk], dtype=np.int64))
    reshaped = ops.reshape(conv_node, reshape_pattern, special_zero=False)
    qkv_node = ops.transpose(reshaped, ops.constant(np.array([0, 2, 1, 3], dtype=np.int64)))
    q_node = L2Norm([qkv_node])
    k_node = L2Norm([qkv_node])
    rule = GatedDeltaRule([q_node, k_node, qkv_node, pg, pb, ps0])
    o_seq_node = rule.output(0)
    # back to [B, T, D]
    o_t = ops.transpose(o_seq_node, ops.constant(np.array([0, 2, 1, 3], dtype=np.int64)))
    o_flat_node = ops.reshape(o_t, ops.constant(np.array([B, T, D], dtype=np.int64)),
                              special_zero=False)
    rms = GatedRMSNorm([o_flat_node, pgate, pw])

    model = Model(
        [ops.result(rms)],
        [px, pwc, pg, pb, pgate, pw, ps0],
        "GatedDeltaNetBlock",
    )

    (got,) = _run(model, {
        "x": x, "w_conv": w_conv, "g": g_seq, "beta": b_seq,
        "gate": gate, "rms_w": rms_w, "s0": s0,
    })
    assert np.allclose(got, ref_out, atol=1e-4), \
        f"Full block mismatch: max={np.abs(got - ref_out).max()}"
    print("  full block            ok")


def main():
    print("running gated DeltaNet op tests on", DEVICE)
    test_l2norm()
    test_short_conv1d()
    test_gated_delta_rule()
    test_gated_delta_rule_step()
    test_gated_rmsnorm()
    test_full_block()
    print("all good.")


if __name__ == "__main__":
    sys.exit(main())
