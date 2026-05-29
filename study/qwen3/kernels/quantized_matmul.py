"""
Streaming int8 matmul that avoids the plugin's bf16 weight pre-decompression.

The exported IR represents each linear layer as:
    Constant(u8)[N,K] -> Convert(f16) -> Subtract(zp f16)[N,1] -> Multiply(scale f16)[N,1] -> Convert(f32) -> MatMul(act, w, transpose_b=True)

The CPU plugin pre-decompresses the (u8, scale, zp) tuple into a bf16/f32
in-memory buffer for matmul throughput. Across the whole LM that buffer is
hundreds of MB on top of the file-backed .bin -- the main source of the
~865 MB anon spike on first infer.

Our `QuantizedMatMul` takes (act, u8_weight, scale, zp) as four inputs and
computes the matmul in row blocks, dequantizing each block into a small
scratch buffer just for that block. No persistent bf16/f32 weight buffer.

Math, exact match to the IR pre-dequant when computed in f16:
    w_f16[n, k] = (u8[n, k].astype(f16) - zp[n]) * scale[n]
    y[..., n]  = act[..., :] @ w_f16[n, :].astype(f32)

Inputs:
    0: act         [..., K]    f32
    1: u8_weight   [N, K]      u8
    2: scale       [N, 1]      f16
    3: zp          [N, 1]      u8

Output:
    0: y           [..., N]    f32
"""
from __future__ import annotations

import numpy as np
import openvino as ov
from openvino import Op

from kernels import use_c as _use_c, qmm as _qmm_c


_BLOCK = 4096  # rows of weight dequantized at a time


class QuantizedMatMul(Op):
    def __init__(self, inputs=None):
        super().__init__(self, inputs)

    def validate_and_infer_types(self):
        # Output rank = act rank; last dim = N (weight rows).
        et = self.get_input_element_type(0)
        act_ps = self.get_input_partial_shape(0)
        w_ps = self.get_input_partial_shape(1)
        out_ps = ov.PartialShape(list(act_ps) + [w_ps[0]])
        out_dims = list(act_ps)
        out_dims[-1] = w_ps[0]
        self.set_output_type(0, et, ov.PartialShape(out_dims))

    def clone_with_new_inputs(self, new_inputs):
        return QuantizedMatMul(list(new_inputs))

    def visit_attributes(self, visitor):
        return True

    def has_evaluate(self):
        return True

    def evaluate(self, outputs, inputs):
        act = np.asarray(inputs[0].data)           # [..., K] f32
        u8  = np.asarray(inputs[1].data)           # [N, K]   u8
        sc  = np.asarray(inputs[2].data)           # [N, 1]   f16
        zp  = np.asarray(inputs[3].data)           # [N, 1]   u8

        K = act.shape[-1]
        N = u8.shape[0]
        out_shape = list(act.shape[:-1]) + [N]
        outputs[0].shape = tuple(out_shape)
        out = np.asarray(outputs[0].data)

        # Flatten leading dims so act is [M, K]; reshape result at end.
        act_flat = np.ascontiguousarray(act.reshape(-1, K), dtype=np.float32)
        out_flat = out.reshape(-1, N)             # [M, N] f32

        if _use_c():
            sc_view = sc.reshape(-1)
            if sc_view.dtype == np.float16:
                sc_bits = np.ascontiguousarray(sc_view.view(np.uint16))
            else:
                sc_bits = np.ascontiguousarray(sc_view.astype(np.float16).view(np.uint16))
            zp_view = zp.reshape(-1)
            zp_u8 = np.ascontiguousarray(zp_view if zp_view.dtype == np.uint8
                                         else zp_view.astype(np.uint8))
            u8_c = np.ascontiguousarray(u8)
            out_c = np.empty((act_flat.shape[0], N), dtype=np.float32)
            _qmm_c(act_flat, u8_c, sc_bits, zp_u8, out_c)
            out_flat[...] = out_c
            return True

        # NumPy fallback: block over output rows so the dequant scratch is bounded.
        for n0 in range(0, N, _BLOCK):
            n1 = min(n0 + _BLOCK, N)
            block_u8 = u8[n0:n1]                  # [B, K] u8
            zp_blk = zp[n0:n1].astype(np.float16) # [B, 1]
            sc_blk = sc[n0:n1]                    # [B, 1] f16
            w_f16 = (block_u8.astype(np.float16) - zp_blk) * sc_blk   # [B, K] f16
            out_flat[:, n0:n1] = act_flat @ w_f16.astype(np.float32).T

        return True


# ---------------------------------------------------------------------------
# Graph rewrite: detect the dequant-into-MatMul pattern, replace with our op.
# ---------------------------------------------------------------------------
def _trace_dequant_inputs(matmul):
    """Return (u8_const, scale_const, zp_const) if matmul.input(1) matches the
    Convert -> Multiply -> Subtract -> Convert -> Constant(u8) pattern;
    otherwise return None.
    """
    n = matmul.input(1).get_source_output().get_node()
    if n.get_type_name() != "Convert":
        return None
    n_mul = n.input(0).get_source_output().get_node()
    if n_mul.get_type_name() != "Multiply":
        return None
    n_sub = n_mul.input(0).get_source_output().get_node()
    if n_sub.get_type_name() != "Subtract":
        return None
    n_cv = n_sub.input(0).get_source_output().get_node()
    if n_cv.get_type_name() != "Convert":
        return None
    n_u8 = n_cv.input(0).get_source_output().get_node()
    if n_u8.get_type_name() != "Constant":
        return None
    if n_u8.get_output_element_type(0).get_type_name() != "u8":
        return None

    # Scale = Multiply.input(1); follow through any Convert wrapper.
    def to_const(node):
        if node.get_type_name() == "Constant":
            return node
        if node.get_input_size() == 0:
            return None
        return to_const(node.input(0).get_source_output().get_node())

    scale_c = to_const(n_mul.input(1).get_source_output().get_node())
    zp_c    = to_const(n_sub.input(1).get_source_output().get_node())
    if scale_c is None or zp_c is None:
        return None
    if scale_c.get_output_element_type(0).get_type_name() != "f16":
        return None
    if zp_c.get_output_element_type(0).get_type_name() != "u8":
        return None
    return n_u8, scale_c, zp_c


def replace_quantized_matmuls(model: ov.Model, name_filter=None) -> int:
    """Replace MatMul(act, dequant(u8, sc, zp)) with our QuantizedMatMul.

    name_filter: optional callable(matmul_name) -> bool; only matmuls where
    this returns True are replaced. If None, every match is rewritten.
    """
    matmuls = [op for op in model.get_ops() if op.get_type_name() == "MatMul"]
    replaced = 0
    for mm in matmuls:
        if name_filter is not None and not name_filter(mm.get_friendly_name()):
            continue
        if not mm.get_transpose_b():
            # All linears here have transpose_b=True; skip anything else.
            continue
        parts = _trace_dequant_inputs(mm)
        if parts is None:
            continue
        u8_c, sc_c, zp_c = parts
        act_src = mm.input(0).get_source_output()
        fused = QuantizedMatMul([act_src, u8_c.output(0), sc_c.output(0), zp_c.output(0)])
        fused.set_friendly_name(mm.get_friendly_name() + "/QMM")
        mm.output(0).replace(fused.output(0))
        replaced += 1
    return replaced


def register(core: ov.Core) -> None:
    core.add_extension(QuantizedMatMul)
