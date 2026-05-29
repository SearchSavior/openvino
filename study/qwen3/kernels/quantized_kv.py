"""
Int8 KV cache for the 6 full-attention layers of Qwen3.5.

In the stock IR each full-attn layer maintains f32 state for K and V:

    ReadValue(K_state) [B, 2, T_prev, 256] f32
      -> Gather (beam_idx) -> Concat(prev, new_K) -> SDPA
                              Concat -> Assign(K_state)

That's 4 bytes per (B, H, T, D) element of K and V. At T=2048, B=1, H=2, D=256:
6 layers × (K + V) × 4 bytes × 1 × 2 × 2048 × 256 = 48 MiB. llama.cpp at q8_0
uses 12.75 MiB for the same state -- 4× smaller because K and V live as int8
plus per-token scales instead of fp32.

This module mirrors that. The trick is one custom op that does:

    prev_K_i8  [B, 2, T_prev, 256] i8
    prev_K_sc  [B, 2, T_prev]     f32
    new_K_f32  [B, 2, N,     256] f32
        -> full_K_f32  [B, 2, T_prev+N, 256] f32   (for SDPA)
           new_K_i8    [B, 2, T_prev+N, 256] i8    (back to Assign)
           new_K_sc    [B, 2, T_prev+N]     f32    (back to Assign)

(per-token symmetric int8 quant: scale_t = max|K[..,t,:]| / 127).

Persistent state per K (or V) state: i8 data + f32 scale =
    T × D bytes + T × 4 bytes  =  T × (256 + 4)  =  T × 260 bytes
vs old fp32: T × 1024 bytes. ~4× reduction.

Memory accounting at T=2048 (6 layers × (K+V)):
    new state: 6 × 2 × 2048 × 260 = 6.5 MiB   (was 48 MiB)

Peak during compute is unchanged: the op still hands SDPA a full-T fp32 buffer.
"""
from __future__ import annotations

import numpy as np
import openvino as ov
from openvino import Op
from openvino import opset15 as ops
from openvino.op.util import Variable, VariableInfo

from kernels import use_c as _use_c, qkv as _qkv_c


class QuantizedKVCache(Op):
    """Dequant prev -> concat new -> requant new piece, all in one op."""

    def __init__(self, inputs=None):
        super().__init__(self, inputs)

    def validate_and_infer_types(self):
        # in[0]: prev_data  [B, H, T_prev, D] i8
        # in[1]: prev_scale [B, H, T_prev]    f32
        # in[2]: new_kv     [B, H, N,     D]  f32
        prev_data_ps = self.get_input_partial_shape(0)
        new_ps = self.get_input_partial_shape(2)

        # Time dim of output = T_prev + N. Both are dynamic in normal use -> stay dynamic.
        if prev_data_ps[2].is_static and new_ps[2].is_static:
            t_full = ov.Dimension(prev_data_ps[2].get_length() + new_ps[2].get_length())
        else:
            t_full = ov.Dimension.dynamic()

        full_shape  = ov.PartialShape([new_ps[0], new_ps[1], t_full, new_ps[3]])
        scale_shape = ov.PartialShape([new_ps[0], new_ps[1], t_full])

        self.set_output_type(0, ov.Type.f32, full_shape)   # for SDPA
        self.set_output_type(1, ov.Type.i8,  full_shape)   # to Assign(data)
        self.set_output_type(2, ov.Type.f32, scale_shape)  # to Assign(scale)

    def clone_with_new_inputs(self, new_inputs):
        return QuantizedKVCache(list(new_inputs))

    def visit_attributes(self, visitor):
        return True

    def has_evaluate(self):
        return True

    def evaluate(self, outputs, inputs):
        prev_data  = np.asarray(inputs[0].data)   # [B, H, T_prev, D] i8
        prev_scale = np.asarray(inputs[1].data)   # [B, H, T_prev]    f32
        new_kv     = np.asarray(inputs[2].data)   # [B, H, N,     D]  f32

        B, H, T_prev, D = prev_data.shape
        _, _, N, _ = new_kv.shape
        T_full = T_prev + N

        outputs[0].shape = (B, H, T_full, D)
        outputs[1].shape = (B, H, T_full, D)
        outputs[2].shape = (B, H, T_full)
        full_f32   = np.asarray(outputs[0].data)
        new_data   = np.asarray(outputs[1].data)
        new_scale  = np.asarray(outputs[2].data)

        if _use_c():
            prev_data_c  = np.ascontiguousarray(prev_data,  dtype=np.int8)
            prev_scale_c = np.ascontiguousarray(prev_scale, dtype=np.float32)
            new_kv_c     = np.ascontiguousarray(new_kv,     dtype=np.float32)
            _qkv_c(prev_data_c, prev_scale_c, new_kv_c, new_data, new_scale, full_f32)
            return True

        # NumPy reference
        max_abs    = np.maximum(np.abs(new_kv).max(axis=-1), 1e-12)
        new_scale_q = (max_abs / 127.0).astype(np.float32)
        new_data_q  = np.clip(np.round(new_kv / new_scale_q[..., None]), -128, 127).astype(np.int8)
        new_data [:, :, :T_prev, :] = prev_data
        new_data [:, :, T_prev:, :] = new_data_q
        new_scale[:, :, :T_prev] = prev_scale
        new_scale[:, :, T_prev:] = new_scale_q
        full_f32[...] = new_data.astype(np.float32) * new_scale[..., None]
        return True


def register(core: ov.Core) -> None:
    core.add_extension(QuantizedKVCache)


# ---------------------------------------------------------------------------
# Graph rewrite
# ---------------------------------------------------------------------------
def _find_state_concats(model):
    """Walk SDPA K/V inputs back to the state Concat. Returns list of Concat nodes."""
    sdpas = [op for op in model.get_ops() if "ScaledDotProductAttention" in op.get_type_name()]
    concats = []
    for sdpa in sdpas:
        for inp_idx in (1, 2):                  # K, V
            n = sdpa.input(inp_idx).get_source_output().get_node()
            while n.get_type_name() != "Concat":
                n = n.input(0).get_source_output().get_node()
            concats.append(n)
    return concats


def _trace_state_chain(concat):
    """For a state concat, return (read_value, gather_or_None, assign).

    Layout:  ReadValue -> Gather(beam_idx) -> Concat -> (SDPA chain, Assign)
    """
    n = concat.input(0).get_source_output().get_node()
    gather = None
    if n.get_type_name() == "Gather":
        gather = n
        n = gather.input(0).get_source_output().get_node()
    assert n.get_type_name() == "ReadValue", f"expected ReadValue, got {n.get_type_name()}"
    rv = n

    asgs = [t.get_node() for t in concat.output(0).get_target_inputs()
            if t.get_node().get_type_name() == "Assign"]
    assert len(asgs) == 1, f"expected 1 Assign, got {len(asgs)}"
    return rv, gather, asgs[0]


def replace_kv_with_int8(model: ov.Model) -> int:
    """Replace every full-attn K/V state Concat with QuantizedKVCache.

    For each state variable V_f32 [B, H, T, D]:
      1. Create V_i8   (same shape, i8) and V_scale ([B, H, T] f32)
      2. New ReadValue/Assign for each; init = zeros via Broadcast(Constant)
      3. Custom op consumes (prev_data, prev_scale, new_kv); outputs go to
         SDPA (f32) and the two new Assigns
      4. Remove the old f32 ReadValue/Concat/Assign, drop the old Variable.

    Returns count of state pairs rewritten (= 2 × n_full_attn_layers).
    """
    state_concats = _find_state_concats(model)
    rewritten = 0

    for concat in state_concats:
        rv, gather, old_assign = _trace_state_chain(concat)
        old_var = model.get_variable_by_id(rv.get_variable_id())
        old_info = old_var.get_info()
        old_shape = old_info.data_shape                          # [B, H, T, D]

        # New variables: data (i8), scale (f32). Same dyn rank semantics.
        data_info = VariableInfo()
        data_info.variable_id = old_info.variable_id + ".i8"
        data_info.data_type = ov.Type.i8
        data_info.data_shape = old_shape                          # same [B, H, T, D]
        data_var = Variable(data_info)

        scale_info = VariableInfo()
        scale_info.variable_id = old_info.variable_id + ".scale"
        scale_info.data_type = ov.Type.f32
        # [B, H, T] (drop last dim)
        scale_shape = ov.PartialShape([old_shape[0], old_shape[1], old_shape[2]])
        scale_info.data_shape = scale_shape
        scale_var = Variable(scale_info)

        model.add_variables([data_var, scale_var])

        # Initialisers (zeros, broadcast to dynamic shape). Mirror the existing
        # init style: Broadcast(Constant(...), shape_of(...)). We reuse the
        # old Broadcast's shape input.
        old_init = rv.input(0).get_source_output().get_node()
        assert old_init.get_type_name() == "Broadcast"
        # For the data init we use a zero i8 scalar; broadcast it to the same
        # ShapeOf input that drove the old f32 init.
        data_zero = ops.constant(np.zeros((), dtype=np.int8))
        data_init = ops.broadcast(data_zero, old_init.input(1).get_source_output())
        scale_zero = ops.constant(np.zeros((), dtype=np.float32))
        # Scale init shape needs to drop the last dim of the data shape.
        # Easier: take a Slice of the shape vector.
        shape_vec = old_init.input(1).get_source_output()         # [4] i64 (the data shape)
        # scale_shape_vec = shape_vec[:3]  via Slice
        slice_start = ops.constant(np.array([0], dtype=np.int64))
        slice_stop  = ops.constant(np.array([3], dtype=np.int64))
        slice_step  = ops.constant(np.array([1], dtype=np.int64))
        slice_axes  = ops.constant(np.array([0], dtype=np.int64))
        scale_shape_vec = ops.slice(shape_vec, slice_start, slice_stop, slice_step, slice_axes)
        scale_init = ops.broadcast(scale_zero, scale_shape_vec)

        rv_data  = ops.read_value(data_init,  data_var)
        rv_scale = ops.read_value(scale_init, scale_var)

        # Optional beam_idx gather. Mirror the original (it gathered along axis 0).
        if gather is not None:
            beam_idx = gather.input(1).get_source_output()
            axis     = gather.input(2).get_source_output()
            rv_data_g  = ops.gather(rv_data,  beam_idx, axis)
            rv_scale_g = ops.gather(rv_scale, beam_idx, axis)
        else:
            rv_data_g, rv_scale_g = rv_data, rv_scale

        new_kv_src = concat.input(1).get_source_output()
        qkv = QuantizedKVCache([rv_data_g.output(0), rv_scale_g.output(0), new_kv_src])
        qkv.set_friendly_name(concat.get_friendly_name() + "/QKV")

        # Rewire ALL consumers of the old Concat (SDPA chain *and* any ShapeOf
        # that downstream Broadcasts use for their target-shape vector) to the
        # QKV's f32 output. We later drop the dead Assign+ReadValue path.
        for consumer in list(concat.output(0).get_target_inputs()):
            consumer.replace_source_output(qkv.output(0))

        # opset.assign only accepts single-output Node, not Output. Wrap the
        # two other QKV outputs in identity Converts so they're standalone.
        data_out  = ops.convert(qkv.output(1), ov.Type.i8)
        scale_out = ops.convert(qkv.output(2), ov.Type.f32)
        new_asg_data  = ops.assign(data_out,  data_var)
        new_asg_scale = ops.assign(scale_out, scale_var)
        model.add_sinks([new_asg_data, new_asg_scale])

        # Detach the old f32 chain entirely.
        #
        # The IR computes the SDPA GQA Broadcast target T as
        #     Add(ShapeOf(old_Gather)[2], N)        # = T_prev + N
        # i.e. it reads the post-Gather *prev* tensor's T. With an empty f32
        # stub plugged into the old Gather, ShapeOf would say T=0 even when
        # the live path has T_prev>0, so the runtime Broadcast/Multiply
        # mismatches the input data dim. Redirect those ShapeOf nodes to read
        # from the new i8 Gather (`rv_data_g`) instead -- same T_prev, no
        # dtype impact since ShapeOf ignores element type.
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

        D = int(old_shape[3].get_length())
        empty_f32 = ops.constant(np.zeros((1, 2, 0, D), dtype=np.float32))
        for consumer in list(rv.output(0).get_target_inputs()):
            consumer.replace_source_output(empty_f32.output(0))

        model.remove_sink(old_assign)
        model.remove_variable(old_var)

        rewritten += 1

    return rewritten
