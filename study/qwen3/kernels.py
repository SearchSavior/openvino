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
        lib.conv1d_kernel.restype = None
        lib.conv1d_kernel.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,    # prev, cur, w
            ctypes.c_void_p, ctypes.c_void_p,                      # out, new_state
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
        ]
        _lib = lib
    return _lib


def _contig_f32(a):
    return np.ascontiguousarray(a, dtype=np.float32)


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
