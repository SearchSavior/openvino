"""
ctypes bindings for kernels.so.

Toggle in custom Ops via env var:
    QWEN3_USE_C=1   → evaluate() dispatches to gdr_kernel / conv1d_kernel.
    unset / "0"     → stay on the numpy reference path.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np


_SO_PATH = Path(__file__).parent / "libqwen3_kernels.so"
_lib: ctypes.CDLL | None = None


def use_c() -> bool:
    return os.environ.get("QWEN3_USE_C", "0") == "1"


def get_lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        if not _SO_PATH.exists():
            raise FileNotFoundError(
                f"{_SO_PATH} not found — run study/qwen3/build_kernels.sh first")
        lib = ctypes.CDLL(str(_SO_PATH))
        lib.gdr_kernel.restype = None
        lib.gdr_kernel.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,    # q, k, v
            ctypes.c_void_p, ctypes.c_void_p,                      # g, beta
            ctypes.c_void_p, ctypes.c_void_p,                      # S, out
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        lib.gdr_kernel_v2.restype = None
        lib.gdr_kernel_v2.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,    # mixed_qkv, g, beta
            ctypes.c_void_p, ctypes.c_void_p,                      # S, out
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,  # B, H, T, D
            ctypes.c_int, ctypes.c_int,                            # key_dim, value_dim
        ]
        lib.conv1d_kernel.restype = None
        lib.conv1d_kernel.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,    # prev, cur, w
            ctypes.c_void_p, ctypes.c_void_p,                      # out, new_state
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
        ]
        lib.qmm_kernel.restype = None
        lib.qmm_kernel.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,                      # act, u8
            ctypes.c_void_p, ctypes.c_void_p,                      # scale, zp
            ctypes.c_void_p,                                       # out
            ctypes.c_int, ctypes.c_int, ctypes.c_int,              # M, N, K
        ]
        if hasattr(lib, "qmm_kernel_vnni"):
            lib.qmm_kernel_vnni.restype = None
            lib.qmm_kernel_vnni.argtypes = lib.qmm_kernel.argtypes
        lib.qkv_kernel.restype = None
        lib.qkv_kernel.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,    # prev_data, prev_scale, new_kv
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,    # new_data, new_scale, full_f32
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,  # B,H,T_prev,N,D
        ]
        _lib = lib
    return _lib


def _contig_f32(a):
    return np.ascontiguousarray(a, dtype=np.float32)


def gdr_v2(mixed_qkv, g, beta, S, out, key_dim, value_dim):
    """In-place: mutates S, writes out. All inputs must be C-contiguous float32.
    mixed_qkv  [B, T, key_dim*2 + value_dim]  pre-split Q|K|V
    g, beta    [B, T, H]
    S          [B, H, D, D]  in/out
    out        [B, T, H, D]
    """
    B, T, qkv_dim = mixed_qkv.shape
    assert qkv_dim == key_dim * 2 + value_dim, f"qkv_dim {qkv_dim} != {key_dim*2 + value_dim}"
    assert g.shape == (B, T, S.shape[1]) and beta.shape == g.shape, "g/beta shapes"
    H, D, _ = S.shape[1], S.shape[2], S.shape[3]
    assert out.shape == (B, T, H, D)
    for arr in (mixed_qkv, g, beta, S, out):
        assert arr.dtype == np.float32 and arr.flags["C_CONTIGUOUS"], "must be contiguous fp32"
    lib = get_lib()
    lib.gdr_kernel_v2(
        mixed_qkv.ctypes.data, g.ctypes.data, beta.ctypes.data,
        S.ctypes.data, out.ctypes.data,
        B, H, T, D, key_dim, value_dim,
    )


def gdr(q, k, v, g, beta, S, out):
    """In-place: mutates S, writes out. All inputs must be C-contiguous float32."""
    B, H, T, D = q.shape
    assert k.shape == (B, H, T, D) and v.shape == (B, H, T, D)
    assert g.shape == (B, H, T) and beta.shape == (B, H, T)
    assert S.shape == (B, H, D, D)
    assert out.shape == (B, H, T, D)
    for arr in (q, k, v, g, beta, S, out):
        assert arr.dtype == np.float32 and arr.flags["C_CONTIGUOUS"], "must be contiguous fp32"
    lib = get_lib()
    lib.gdr_kernel(
        q.ctypes.data, k.ctypes.data, v.ctypes.data,
        g.ctypes.data, beta.ctypes.data,
        S.ctypes.data, out.ctypes.data,
        B, H, T, D,
    )


def use_vnni() -> bool:
    return os.environ.get("QWEN3_USE_VNNI", "0") == "1"


def qmm(act, u8, scale, zp, out):
    """In-place: writes out. act, scale (f16 bits), zp (u8), u8 must be C-contiguous.
    If QWEN3_USE_VNNI=1 and the kernel has VNNI support, dispatches to qmm_kernel_vnni.
    """
    assert act.dtype == np.float32 and act.flags["C_CONTIGUOUS"]
    assert u8.dtype == np.uint8 and u8.flags["C_CONTIGUOUS"]
    assert scale.dtype == np.uint16 and scale.flags["C_CONTIGUOUS"]
    assert zp.dtype == np.uint8 and zp.flags["C_CONTIGUOUS"]
    assert out.dtype == np.float32 and out.flags["C_CONTIGUOUS"]
    M = int(np.prod(act.shape[:-1]))
    K = act.shape[-1]
    N = u8.shape[0]
    assert u8.shape[1] == K, f"u8.shape[1]={u8.shape[1]} != K={K}"
    assert scale.shape[0] == N, f"scale.shape={scale.shape} != N={N}"
    assert zp.shape[0] == N, f"zp.shape={zp.shape} != N={N}"
    lib = get_lib()
    fn = lib.qmm_kernel_vnni if (use_vnni() and hasattr(lib, "qmm_kernel_vnni")) else lib.qmm_kernel
    fn(act.ctypes.data, u8.ctypes.data,
       scale.ctypes.data, zp.ctypes.data,
       out.ctypes.data, M, N, K)


def qkv(prev_data, prev_scale, new_kv, new_data, new_scale, full_f32):
    """In-place: writes new_data, new_scale, full_f32. All inputs C-contiguous."""
    assert prev_data.dtype == np.int8 and prev_data.flags["C_CONTIGUOUS"]
    assert prev_scale.dtype == np.float32 and prev_scale.flags["C_CONTIGUOUS"]
    assert new_kv.dtype == np.float32 and new_kv.flags["C_CONTIGUOUS"]
    assert new_data.dtype == np.int8 and new_data.flags["C_CONTIGUOUS"]
    assert new_scale.dtype == np.float32 and new_scale.flags["C_CONTIGUOUS"]
    assert full_f32.dtype == np.float32 and full_f32.flags["C_CONTIGUOUS"]
    B, H, T_prev, D = prev_data.shape
    _, _, N, _ = new_kv.shape
    T_full = T_prev + N
    assert new_data.shape == (B, H, T_full, D)
    assert new_scale.shape == (B, H, T_full)
    assert full_f32.shape == (B, H, T_full, D)
    lib = get_lib()
    lib.qkv_kernel(
        prev_data.ctypes.data, prev_scale.ctypes.data, new_kv.ctypes.data,
        new_data.ctypes.data,  new_scale.ctypes.data,  full_f32.ctypes.data,
        B, H, T_prev, N, D)


def conv1d(prev, cur, w, out, new_state):
    """In-place: writes out and new_state. All inputs must be C-contiguous float32.

    `w` is interpreted as [C, K] — caller is expected to reshape from [C, 1, 1, K].
    """
    B, C, KS = prev.shape
    _, _, T = cur.shape
    K = w.shape[1] if w.ndim == 2 else w.shape[-1]
    assert out.shape == (B, C, KS + T - K + 1)
    assert new_state.shape == (B, C, KS)
    for arr in (prev, cur, w, out, new_state):
        assert arr.dtype == np.float32 and arr.flags["C_CONTIGUOUS"], "must be contiguous fp32"
    lib = get_lib()
    lib.conv1d_kernel(
        prev.ctypes.data, cur.ctypes.data, w.ctypes.data,
        out.ctypes.data, new_state.ctypes.data,
        B, C, KS, T, K,
    )
