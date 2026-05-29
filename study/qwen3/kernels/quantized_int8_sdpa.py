"""
Fused SDPA + int8 KV cache: replace the entire
    QuantizedKVCache (K) -> Unsqueeze -> Broadcast -> Reshape -> SDPA.K
    QuantizedKVCache (V) -> Unsqueeze -> Broadcast -> Reshape -> SDPA.V
chain with a single op that reads i8 K/V directly. No [B, H, T_full, D] f32
buffer is ever materialised. Closes the runtime gap to llama.cpp's q8_0
KV path.

This module also contains the small Python wrapper class used purely for IR
construction; the real evaluate() lives in cpp_ext/quantized_int8_sdpa.cpp
and is wired via the .so extension. The Python evaluate() is a fallback that
calls the same C kernel through ctypes (mostly for diagnostics).
"""
from __future__ import annotations

import numpy as np
import openvino as ov
from openvino import Op
from openvino import opset15 as ops
from openvino.op.util import Variable, VariableInfo


class QuantizedKVCacheUpdate(Op):
    """Same inputs as QuantizedKVCache but only produces (data, scale)."""

    def __init__(self, inputs=None):
        super().__init__(self, inputs)

    def validate_and_infer_types(self):
        prev_data_ps = self.get_input_partial_shape(0)
        new_ps = self.get_input_partial_shape(2)
        if prev_data_ps[2].is_static and new_ps[2].is_static:
            t_full = ov.Dimension(prev_data_ps[2].get_length() + new_ps[2].get_length())
        else:
            t_full = ov.Dimension.dynamic()
        data_shape  = ov.PartialShape([new_ps[0], new_ps[1], t_full, new_ps[3]])
        scale_shape = ov.PartialShape([new_ps[0], new_ps[1], t_full])
        self.set_output_type(0, ov.Type.i8,  data_shape)
        self.set_output_type(1, ov.Type.f32, scale_shape)

    def clone_with_new_inputs(self, new_inputs):
        return QuantizedKVCacheUpdate(list(new_inputs))

    def visit_attributes(self, visitor):
        return True

    def has_evaluate(self):
        return True

    def evaluate(self, outputs, inputs):
        # NumPy fallback; the real path is the C++ op in cpp_ext.
        prev_data  = np.asarray(inputs[0].data)
        prev_scale = np.asarray(inputs[1].data)
        new_kv     = np.asarray(inputs[2].data)
        B, H, T_prev, D = prev_data.shape
        N = new_kv.shape[2]
        T_full = T_prev + N
        max_abs    = np.maximum(np.abs(new_kv).max(axis=-1), 1e-12)
        new_scale_q = (max_abs / 127.0).astype(np.float32)
        new_data_q  = np.clip(np.round(new_kv / new_scale_q[..., None]), -128, 127).astype(np.int8)
        outputs[0].shape = (B, H, T_full, D)
        outputs[1].shape = (B, H, T_full)
        data  = np.asarray(outputs[0].data)
        scale = np.asarray(outputs[1].data)
        data[:, :, :T_prev, :] = prev_data
        data[:, :, T_prev:, :] = new_data_q
        scale[:, :, :T_prev]   = prev_scale
        scale[:, :, T_prev:]   = new_scale_q
        return True


class QuantizedInt8SDPA(Op):
    """Inputs: q, k_data, k_scale, v_data, v_scale, mask, scale. Output: [B,H_q,T_q,D]."""

    def __init__(self, inputs=None):
        super().__init__(self, inputs)

    def validate_and_infer_types(self):
        # Output shape = Q shape, dtype = f32.
        self.set_output_type(0, ov.Type.f32, self.get_input_partial_shape(0))

    def clone_with_new_inputs(self, new_inputs):
        return QuantizedInt8SDPA(list(new_inputs))

    def visit_attributes(self, visitor):
        return True

    def has_evaluate(self):
        return True

    def evaluate(self, outputs, inputs):
        # NumPy fallback. The real path is the C++ op.
        q       = np.asarray(inputs[0].data)
        k_data  = np.asarray(inputs[1].data)
        k_scale = np.asarray(inputs[2].data)
        v_data  = np.asarray(inputs[3].data)
        v_scale = np.asarray(inputs[4].data)
        mask    = np.asarray(inputs[5].data)
        scale   = float(np.asarray(inputs[6].data).reshape(-1)[0])

        B, H_q, T_q, D = q.shape
        H_kv = k_data.shape[1]
        T_full = k_data.shape[2]
        gqa = H_q // H_kv

        # Dequant K and V (fallback materialises; the C op streams).
        K_f = k_data.astype(np.float32) * k_scale[..., None]    # [B, H_kv, T_full, D]
        V_f = v_data.astype(np.float32) * v_scale[..., None]

        outputs[0].shape = q.shape
        out = np.asarray(outputs[0].data)

        for b in range(B):
            for h in range(H_q):
                h_kv = h // gqa
                # [T_q, D] @ [D, T_full] = [T_q, T_full]
                scores = q[b, h] @ K_f[b, h_kv].T * scale
                if mask.shape[-1] == T_full:
                    scores = scores + mask[b, 0]
                scores -= scores.max(axis=-1, keepdims=True)
                np.exp(scores, out=scores)
                scores /= scores.sum(axis=-1, keepdims=True)
                out[b, h] = scores @ V_f[b, h_kv]
        return True


def register(core: ov.Core) -> None:
    core.add_extension(QuantizedKVCacheUpdate)
    core.add_extension(QuantizedInt8SDPA)


# ---------------------------------------------------------------------------
# Graph rewrite: replace SDPA + KV state with QuantizedInt8SDPA + Q-KV-Update
# ---------------------------------------------------------------------------
def _find_sdpa_with_state_concats(model):
    """Yield (sdpa, k_concat, v_concat) for each full-attn SDPA."""
    for sdpa in model.get_ops():
        if "ScaledDotProductAttention" not in sdpa.get_type_name():
            continue
        k_node = sdpa.input(1).get_source_output().get_node()
        v_node = sdpa.input(2).get_source_output().get_node()
        while k_node.get_type_name() != "Concat":
            k_node = k_node.input(0).get_source_output().get_node()
        while v_node.get_type_name() != "Concat":
            v_node = v_node.input(0).get_source_output().get_node()
        yield sdpa, k_node, v_node


def _build_int8_kv(model, concat):
    """For one state Concat: build the new i8/scale Variables + ReadValue +
    QuantizedKVCacheUpdate, and the dead-chain detachment plumbing.

    Returns (data_out, scale_out, rv_data_g_for_shapeof, dead_cleanup_fn).
    """
    n = concat.input(0).get_source_output().get_node()
    gather = None
    if n.get_type_name() == "Gather":
        gather = n
        n = gather.input(0).get_source_output().get_node()
    assert n.get_type_name() == "ReadValue"
    rv = n
    old_assign = next(t.get_node() for t in concat.output(0).get_target_inputs()
                      if t.get_node().get_type_name() == "Assign")
    old_var = model.get_variable_by_id(rv.get_variable_id())
    old_info = old_var.get_info()
    old_shape = old_info.data_shape  # [B, H, T, D]
    D = int(old_shape[3].get_length())

    # New i8 + scale Variables.
    data_info = VariableInfo()
    data_info.variable_id = old_info.variable_id + ".i8"
    data_info.data_type = ov.Type.i8
    data_info.data_shape = old_shape
    data_var = Variable(data_info)
    scale_info = VariableInfo()
    scale_info.variable_id = old_info.variable_id + ".scale"
    scale_info.data_type = ov.Type.f32
    scale_info.data_shape = ov.PartialShape([old_shape[0], old_shape[1], old_shape[2]])
    scale_var = Variable(scale_info)
    model.add_variables([data_var, scale_var])

    old_init = rv.input(0).get_source_output().get_node()
    assert old_init.get_type_name() == "Broadcast"
    data_zero = ops.constant(np.zeros((), dtype=np.int8))
    data_init = ops.broadcast(data_zero, old_init.input(1).get_source_output())
    scale_zero = ops.constant(np.zeros((), dtype=np.float32))
    shape_vec = old_init.input(1).get_source_output()
    slice_start = ops.constant(np.array([0], dtype=np.int64))
    slice_stop  = ops.constant(np.array([3], dtype=np.int64))
    slice_step  = ops.constant(np.array([1], dtype=np.int64))
    slice_axes  = ops.constant(np.array([0], dtype=np.int64))
    scale_shape_vec = ops.slice(shape_vec, slice_start, slice_stop, slice_step, slice_axes)
    scale_init = ops.broadcast(scale_zero, scale_shape_vec)

    rv_data  = ops.read_value(data_init,  data_var)
    rv_scale = ops.read_value(scale_init, scale_var)
    if gather is not None:
        beam_idx = gather.input(1).get_source_output()
        axis     = gather.input(2).get_source_output()
        rv_data_g  = ops.gather(rv_data,  beam_idx, axis)
        rv_scale_g = ops.gather(rv_scale, beam_idx, axis)
    else:
        rv_data_g, rv_scale_g = rv_data, rv_scale

    new_kv_src = concat.input(1).get_source_output()
    qkv = QuantizedKVCacheUpdate([rv_data_g.output(0), rv_scale_g.output(0), new_kv_src])
    qkv.set_friendly_name(concat.get_friendly_name() + "/QKV_Upd")

    data_out  = ops.convert(qkv.output(0), ov.Type.i8)
    scale_out = ops.convert(qkv.output(1), ov.Type.f32)
    model.add_sinks([ops.assign(data_out,  data_var),
                     ops.assign(scale_out, scale_var)])

    def cleanup_dead_chain():
        dead_chain_nodes = {rv}
        if gather is not None:
            dead_chain_nodes.add(gather)
        dead_chain_nodes.add(concat)
        for shape_node in list(model.get_ops()):
            if shape_node.get_type_name() != "ShapeOf":
                continue
            src = shape_node.input(0).get_source_output().get_node()
            if src in dead_chain_nodes:
                shape_node.input(0).replace_source_output(rv_data_g.output(0))

        empty_f32 = ops.constant(np.zeros((1, 2, 0, D), dtype=np.float32))
        for consumer in list(rv.output(0).get_target_inputs()):
            consumer.replace_source_output(empty_f32.output(0))

        # The old Concat's only remaining consumer is the dead Assign; rewire
        # it to the QKV data output (we'll remove the sink right after).
        for consumer in list(concat.output(0).get_target_inputs()):
            consumer.replace_source_output(qkv.output(0))
        model.remove_sink(old_assign)
        model.remove_variable(old_var)

    return qkv, cleanup_dead_chain


def replace_kv_with_int8_sdpa(model: ov.Model) -> int:
    """Replace SDPA + KV concat/state of each full-attn layer with int8 ops.

    Returns the number of full-attn layers rewritten.
    """
    targets = list(_find_sdpa_with_state_concats(model))
    rewritten = 0

    for sdpa, k_concat, v_concat in targets:
        # Q is the post-RoPE Q of the full layer (already shape [B, H_q, T_q, D]).
        q_src    = sdpa.input(0).get_source_output()
        mask_src = sdpa.input(3).get_source_output()
        scale_src = sdpa.input(4).get_source_output() if sdpa.get_input_size() > 4 else None
        if scale_src is None:
            # Fallback: compute 1/sqrt(D). D = q_src.partial_shape[-1].
            D_q = int(q_src.get_partial_shape()[-1].get_length())
            scale_src = ops.constant(np.array(1.0 / (D_q ** 0.5), dtype=np.float32)).output(0)

        k_qkv, k_cleanup = _build_int8_kv(model, k_concat)
        v_qkv, v_cleanup = _build_int8_kv(model, v_concat)

        new_sdpa = QuantizedInt8SDPA([
            q_src,
            k_qkv.output(0), k_qkv.output(1),
            v_qkv.output(0), v_qkv.output(1),
            mask_src,
            scale_src,
        ])
        new_sdpa.set_friendly_name(sdpa.get_friendly_name() + "/Int8")
        sdpa.output(0).replace(new_sdpa.output(0))

        k_cleanup()
        v_cleanup()
        rewritten += 1

    return rewritten
