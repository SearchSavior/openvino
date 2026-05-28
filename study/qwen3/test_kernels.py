"""Quick numerical-correctness check: C kernels vs the numpy reference.

Doesn't go through OpenVINO — just runs both implementations on random inputs
and checks max abs difference. Run after build_kernels.sh.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import kernels


def ref_gdr(q, k, v, g, beta, S_init):
    """Numpy reference matching GatedDeltaRule.evaluate() (numpy path)."""
    B, H, T, D = q.shape
    S = S_init.astype(np.float64).copy()
    out = np.zeros((B, H, T, D), dtype=np.float64)
    for t in range(T):
        q_t = q[:, :, t, :].astype(np.float64)
        k_t = k[:, :, t, :].astype(np.float64)
        v_t = v[:, :, t, :].astype(np.float64)
        g_t = np.exp(g[:, :, t].astype(np.float64))[..., None, None]
        beta_t = beta[:, :, t].astype(np.float64)[..., None]
        S *= g_t
        kv_mem = np.einsum("bhkv,bhk->bhv", S, k_t)
        delta = (v_t - kv_mem) * beta_t
        S += np.einsum("bhk,bhv->bhkv", k_t, delta)
        out[:, :, t, :] = np.einsum("bhkv,bhk->bhv", S, q_t)
    return out, S


def test_gdr():
    rng = np.random.default_rng(0)
    B, H, T, D = 1, 16, 8, 128
    q = rng.standard_normal((B, H, T, D)).astype(np.float32) * 0.1
    k = rng.standard_normal((B, H, T, D)).astype(np.float32) * 0.1
    v = rng.standard_normal((B, H, T, D)).astype(np.float32) * 0.1
    g = (rng.standard_normal((B, H, T)).astype(np.float32) * 0.1) - 0.1
    beta = rng.uniform(0.0, 1.0, (B, H, T)).astype(np.float32)
    S0 = rng.standard_normal((B, H, D, D)).astype(np.float32) * 0.01

    out_c = np.empty((B, H, T, D), dtype=np.float32)
    S_c = S0.copy()
    kernels.gdr(q, k, v, g, beta, S_c, out_c)

    out_ref, S_ref = ref_gdr(q, k, v, g, beta, S0)
    max_out_err = float(np.abs(out_c.astype(np.float64) - out_ref).max())
    max_S_err = float(np.abs(S_c.astype(np.float64) - S_ref).max())
    print(f"  gdr  out_max_err = {max_out_err:.3e}   S_max_err = {max_S_err:.3e}")
    assert max_out_err < 1e-3, f"gdr out diff too high: {max_out_err}"
    assert max_S_err < 1e-3, f"gdr S diff too high: {max_S_err}"


def ref_conv1d(prev, cur, wc):
    B, C, KS = prev.shape
    _, _, T = cur.shape
    K = wc.shape[1]
    out_len = KS + T - K + 1
    out = np.zeros((B, C, out_len), dtype=np.float64)
    for kk in range(K):
        cutover = max(0, min(KS - kk, out_len))
        wk = wc[:, kk][None, :, None].astype(np.float64)
        if cutover > 0:
            out[..., :cutover] += wk * prev[..., kk:kk + cutover].astype(np.float64)
        if cutover < out_len:
            out[..., cutover:] += wk * cur[..., :out_len - cutover].astype(np.float64)
    ns = np.empty((B, C, KS), dtype=np.float64)
    if T >= KS:
        ns[...] = cur[..., T - KS:T].astype(np.float64)
    else:
        ns[..., :KS - T] = prev[..., T:].astype(np.float64)
        ns[..., KS - T:] = cur.astype(np.float64)
    return out, ns


def test_conv1d():
    rng = np.random.default_rng(1)
    B, C, KS, T, K = 1, 6144, 3, 4, 4
    prev = rng.standard_normal((B, C, KS)).astype(np.float32) * 0.1
    cur = rng.standard_normal((B, C, T)).astype(np.float32) * 0.1
    w = rng.standard_normal((C, K)).astype(np.float32) * 0.1

    out_c = np.empty((B, C, KS + T - K + 1), dtype=np.float32)
    ns_c = np.empty((B, C, KS), dtype=np.float32)
    kernels.conv1d(prev, cur, w, out_c, ns_c)

    out_ref, ns_ref = ref_conv1d(prev, cur, w)
    max_out_err = float(np.abs(out_c.astype(np.float64) - out_ref).max())
    max_ns_err = float(np.abs(ns_c.astype(np.float64) - ns_ref).max())
    print(f"  conv1d  out_max_err = {max_out_err:.3e}   new_state_max_err = {max_ns_err:.3e}")
    assert max_out_err < 1e-4, f"conv1d out diff too high: {max_out_err}"
    assert max_ns_err < 1e-7, f"conv1d new_state diff too high: {max_ns_err}"


def test_conv1d_decode():
    """T=1 (decode step) — most common case."""
    rng = np.random.default_rng(2)
    B, C, KS, T, K = 1, 6144, 3, 1, 4
    prev = rng.standard_normal((B, C, KS)).astype(np.float32) * 0.1
    cur = rng.standard_normal((B, C, T)).astype(np.float32) * 0.1
    w = rng.standard_normal((C, K)).astype(np.float32) * 0.1

    out_c = np.empty((B, C, KS + T - K + 1), dtype=np.float32)
    ns_c = np.empty((B, C, KS), dtype=np.float32)
    kernels.conv1d(prev, cur, w, out_c, ns_c)
    out_ref, ns_ref = ref_conv1d(prev, cur, w)
    max_out_err = float(np.abs(out_c.astype(np.float64) - out_ref).max())
    max_ns_err = float(np.abs(ns_c.astype(np.float64) - ns_ref).max())
    print(f"  conv1d(T=1)  out_max_err = {max_out_err:.3e}   new_state_max_err = {max_ns_err:.3e}")
    assert max_out_err < 1e-4
    assert max_ns_err < 1e-7


if __name__ == "__main__":
    print("== test_gdr ==")
    test_gdr()
    print("== test_conv1d (T=4) ==")
    test_conv1d()
    print("== test_conv1d (T=1, decode) ==")
    test_conv1d_decode()
    print("OK")
