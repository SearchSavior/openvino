"""
Fused gated-delta-rule linear-attention kernel for Qwen3.5 / Qwen3-Next.

Replaces the unfused `Loop` subgraph (≈58% of decode time and ~30× memory
overhead in our measurements) with a single custom op `GatedDeltaRule` that
implements the recurrence in one Python/NumPy function. Registered with the
CPU plugin via the `add_extension()` path.

Reference math: transformers.models.qwen3_next.modeling_qwen3_next.
                torch_recurrent_gated_delta_rule (lines 545-556).

Inputs of GatedDeltaRule:
    0  q              [B, H, S, D]  fp32
    1  k              [B, H, S, D]  fp32
    2  v              [B, H, S, D]  fp32
    3  g              [B, H, S]     fp32     decay gate (raw; exp'd inside)
    4  beta           [B, H, S]     fp32     step size
    5  initial_state  [B, H, D, D]  fp32

Outputs:
    0  output         [B, H, S, D]  fp32     per-token output
    1  final_state    [B, H, D, D]  fp32     state after S steps

The op is registered as `extension::GatedDeltaRule v1`.
"""
from __future__ import annotations

import numpy as np
import openvino as ov
from openvino import Op

from kernels import use_c as _use_c, gdr as _gdr_c, gdr_v2 as _gdr_v2_c


class GatedDeltaRule(Op):
    """Fused linear-attention recurrence used in Qwen3-Next-family models.

    Two signatures:
      v1 (6 inputs): q [B,H,T,D], k, v, g [B,H,T], beta, state
                     -> out [B,H,T,D], new_state
      v2 (4 inputs): mixed_qkv [B,T,key_dim*2+value_dim], g [B,T,H], beta, state
                     -> out [B,T,H,D], new_state
                     Absorbs the upstream split / reshape / L2-norm / Q-scale /
                     transpose. Eliminates ~16 IR-level intermediate tensors
                     per linear-attn layer.
    """

    def __init__(self, inputs=None):
        super().__init__(self, inputs)

    def _is_v2(self):
        return self.get_input_size() == 4

    def validate_and_infer_types(self):
        et = self.get_input_element_type(0)
        if self._is_v2():
            # in[0] mixed_qkv  [B, T, key_dim*2 + value_dim]
            # in[1] g           [B, T, H]
            # in[2] beta        [B, T, H]
            # in[3] state       [B, H, D, D]
            mqkv = self.get_input_partial_shape(0)
            gps  = self.get_input_partial_shape(1)
            sps  = self.get_input_partial_shape(3)
            out_shape = ov.PartialShape([mqkv[0], mqkv[1], gps[2], sps[3]])  # [B, T, H, D]
            self.set_output_type(0, et, out_shape)
            self.set_output_type(1, et, sps)
        else:
            # v1: original signature.
            self.set_output_type(0, et, self.get_input_partial_shape(2))
            self.set_output_type(1, et, self.get_input_partial_shape(5))

    def clone_with_new_inputs(self, new_inputs):
        return GatedDeltaRule(list(new_inputs))

    def visit_attributes(self, visitor):
        return True

    def has_evaluate(self):
        return True

    def evaluate(self, outputs, inputs):
        if self._is_v2():
            return self._evaluate_v2(outputs, inputs)
        return self._evaluate_v1(outputs, inputs)

    def _evaluate_v2(self, outputs, inputs):
        mqkv = np.asarray(inputs[0].data)   # [B, T, qkv_dim]
        g    = np.asarray(inputs[1].data)   # [B, T, H]
        beta = np.asarray(inputs[2].data)
        S    = np.asarray(inputs[3].data).copy()
        B, T, qkv_dim = mqkv.shape
        H, D = g.shape[2], S.shape[3]
        # No GQA in linear-attn: key_dim = value_dim = H * D.
        key_dim = H * D
        value_dim = qkv_dim - 2 * key_dim
        outputs[0].shape = (B, T, H, D)
        outputs[1].shape = S.shape
        out = np.asarray(outputs[0].data)

        if _use_c():
            mqkv_c = np.ascontiguousarray(mqkv, dtype=np.float32)
            g_c    = np.ascontiguousarray(g,    dtype=np.float32)
            b_c    = np.ascontiguousarray(beta, dtype=np.float32)
            S_c    = np.ascontiguousarray(S,    dtype=np.float32)
            out_c  = np.empty((B, T, H, D), dtype=np.float32)
            _gdr_v2_c(mqkv_c, g_c, b_c, S_c, out_c, key_dim, value_dim)
            out[...] = out_c
            np.asarray(outputs[1].data)[...] = S_c
            return True

        # NumPy reference for v2.
        eps = 1e-6
        Q = mqkv[..., :key_dim].reshape(B, T, H, D).astype(np.float32)
        K = mqkv[..., key_dim:2*key_dim].reshape(B, T, H, D).astype(np.float32)
        V = mqkv[..., 2*key_dim:].reshape(B, T, H, D).astype(np.float32)
        Q = Q / np.sqrt((Q*Q).sum(-1, keepdims=True) + eps) / np.sqrt(D)
        K = K / np.sqrt((K*K).sum(-1, keepdims=True) + eps)
        for t in range(T):
            for h in range(H):
                qt = Q[:, t, h, :]
                kt = K[:, t, h, :]
                vt = V[:, t, h, :]
                gt = np.exp(g[:, t, h])[:, None, None]
                bt = beta[:, t, h][:, None]
                S[:, h] *= gt
                kv_mem = np.einsum("bkv,bk->bv", S[:, h], kt)
                delta  = (vt - kv_mem) * bt
                S[:, h] += np.einsum("bk,bv->bkv", kt, delta)
                out[:, t, h, :] = np.einsum("bkv,bk->bv", S[:, h], qt)
        np.asarray(outputs[1].data)[...] = S
        return True

    def _evaluate_v1(self, outputs, inputs):
        q = np.asarray(inputs[0].data)
        k = np.asarray(inputs[1].data)
        v = np.asarray(inputs[2].data)
        g = np.asarray(inputs[3].data)
        beta = np.asarray(inputs[4].data)
        S = np.asarray(inputs[5].data).copy()
        B, H, T, Dk = q.shape
        Dv = v.shape[-1]
        outputs[0].shape = (B, H, T, Dv)
        outputs[1].shape = S.shape
        out = np.asarray(outputs[0].data)

        if _use_c():
            q_c = np.ascontiguousarray(q, dtype=np.float32)
            k_c = np.ascontiguousarray(k, dtype=np.float32)
            v_c = np.ascontiguousarray(v, dtype=np.float32)
            g_c = np.ascontiguousarray(g, dtype=np.float32)
            b_c = np.ascontiguousarray(beta, dtype=np.float32)
            S_c = np.ascontiguousarray(S, dtype=np.float32)
            out_c = np.empty((B, H, T, Dv), dtype=np.float32)
            _gdr_c(q_c, k_c, v_c, g_c, b_c, S_c, out_c)
            out[...] = out_c
            np.asarray(outputs[1].data)[...] = S_c
            return True

        for t in range(T):
            q_t = q[:, :, t, :]
            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]
            g_t = np.exp(g[:, :, t])[..., None, None]
            beta_t = beta[:, :, t][..., None]
            S *= g_t
            kv_mem = np.einsum("bhkv,bhk->bhv", S, k_t)
            delta = (v_t - kv_mem) * beta_t
            S += np.einsum("bhk,bhv->bhkv", k_t, delta)
            out[:, :, t, :] = np.einsum("bhkv,bhk->bhv", S, q_t)
        np.asarray(outputs[1].data)[...] = S
        return True


# ---------------------------------------------------------------------------
# Graph rewrite: replace the gated-delta-rule `Loop` subgraph with the fused op
# ---------------------------------------------------------------------------

_BODY_FINGERPRINT = {
    "Exp": 1,
    "ScatterUpdate": 1,
    "ReduceSum": 2,
    "Subtract": 1,
    "Add": 1,
}


def _is_gated_delta_rule_loop(loop) -> bool:
    """Detect the gated-delta-rule Loop by body op-counts + parameter shapes."""
    body = loop.get_function()
    params = body.get_parameters()
    if len(params) != 8:
        return False
    from collections import Counter
    counts = Counter(op.get_type_name() for op in body.get_ops())
    for op_type, n in _BODY_FINGERPRINT.items():
        if counts.get(op_type, 0) < n:
            return False
    # Param shape sanity: q/k/v rank-4, gates rank-3, state rank-4 square.
    if params[6].get_partial_shape().rank.get_length() != 4:
        return False
    return True


def replace_gated_delta_rule_loops(model: ov.Model) -> int:
    """Walk `model`, replace each gated-delta-rule Loop with a fused op.

    Returns the number of Loops replaced.
    """
    targets = [n for n in model.get_ops()
               if n.get_type_name() == "Loop" and _is_gated_delta_rule_loop(n)]

    for loop in targets:
        q     = loop.input(2).get_source_output()
        k     = loop.input(3).get_source_output()
        v     = loop.input(4).get_source_output()
        g     = loop.input(5).get_source_output()
        beta  = loop.input(6).get_source_output()
        state = loop.input(7).get_source_output()
        gd_assign = _find_gd_state_assign(loop.output(1))
        fused = GatedDeltaRule([q, k, v, g, beta, state])
        fused.set_friendly_name(loop.get_friendly_name() + "/Fused")
        loop.output(0).replace(fused.output(0))
        loop.output(1).replace(fused.output(1))
        if gd_assign is not None:
            gd_assign.input(0).replace_source_output(fused.output(1))

    return len(targets)


# ---------------------------------------------------------------------------
# v2 rewrite: absorb the pre-GDR input prep (split, reshape, L2-norm, scale,
# transpose) for Q/K/V into the kernel. Same for g/beta (skip their final
# Transpose). This eliminates ~16 intermediate tensors per linear-attn layer.
# ---------------------------------------------------------------------------
def _walk_up(node, max_hops=8):
    """Yield (depth, node) walking input(0) until Constant/Param/depth limit."""
    yield 0, node
    for i in range(max_hops):
        if node.get_input_size() == 0:
            return
        node = node.input(0).get_source_output().get_node()
        yield i + 1, node


def _find_mixed_qkv_for_q(q_src_output):
    """The K/V/Q inputs to GDR are post: Transpose -> Multiply -> Reshape ->
    VariadicSplit -> Transpose([B,T,6144]) -> Swish.

    Walk up from q's source and return the Output handle of the Transpose
    node whose output is [B, T, qkv_dim] (= mixed_qkv pre-split).
    """
    n = q_src_output.get_node()
    for _, nn in _walk_up(n, 10):
        ps = nn.get_output_partial_shape(0)
        if ps.rank.get_length() == 3 and ps[2].is_static and ps[2].get_length() == 6144:
            return nn.output(0)
    return None


def _find_pre_transpose_3d(src_output, hops=6):
    """g and beta are reached via a final Transpose [?,?,16] -> [?,16,?].
    Return the Output handle of the node BEFORE that Transpose (still [B,T,H]).
    """
    n = src_output.get_node()
    # The first node is the Transpose itself (per the GDR.input(3)/(4)).
    if n.get_type_name() == "Transpose":
        return n.input(0).get_source_output()
    return src_output  # already pre-transpose


def replace_gated_delta_rule_loops_v2(model: ov.Model) -> int:
    """Same as replace_gated_delta_rule_loops, but use the v2 signature with
    absorbed input prep. The output of v2 is [B, T, H, D]; we insert a
    Transpose back to [B, H, T, D] so the downstream graph is unchanged.

    Returns the number of Loops replaced.
    """
    from openvino import opset15 as ops
    targets = [n for n in model.get_ops()
               if n.get_type_name() == "Loop" and _is_gated_delta_rule_loop(n)]

    replaced = 0
    for loop in targets:
        q_src    = loop.input(2).get_source_output()
        g_src    = loop.input(5).get_source_output()
        beta_src = loop.input(6).get_source_output()
        state    = loop.input(7).get_source_output()

        gd_assign = _find_gd_state_assign(loop.output(1))
        mixed_qkv = _find_mixed_qkv_for_q(q_src)
        if mixed_qkv is None:
            # couldn't find a [B, T, 6144] tensor above; fall back to v1.
            q     = loop.input(2).get_source_output()
            k     = loop.input(3).get_source_output()
            v     = loop.input(4).get_source_output()
            fused = GatedDeltaRule([q, k, v, g_src, beta_src, state])
            fused.set_friendly_name(loop.get_friendly_name() + "/Fused")
            loop.output(0).replace(fused.output(0))
            loop.output(1).replace(fused.output(1))
            if gd_assign is not None:
                gd_assign.input(0).replace_source_output(fused.output(1))
            replaced += 1
            continue

        g_pre    = _find_pre_transpose_3d(g_src)
        beta_pre = _find_pre_transpose_3d(beta_src)

        fused = GatedDeltaRuleV2([mixed_qkv, g_pre, beta_pre, state])
        fused.set_friendly_name(loop.get_friendly_name() + "/FusedV2")
        # v2 output is [B, T, H, D]. The original Loop.output(0) is [B, H, T, D].
        # Insert a Transpose([0, 2, 1, 3]) to bring it back.
        perm = ops.constant(np.array([0, 2, 1, 3], dtype=np.int64))
        out_BHTD = ops.transpose(fused.output(0), perm)
        loop.output(0).replace(out_BHTD.output(0))
        loop.output(1).replace(fused.output(1))
        if gd_assign is not None:
            gd_assign.input(0).replace_source_output(fused.output(1))
        replaced += 1

    return replaced


class GatedDeltaRuleV2(Op):
    """v2 op with absorbed split / L2-norm / Q-scale / transpose.

    Inputs: mixed_qkv [B,T,key_dim*2+value_dim], g [B,T,H], beta [B,T,H], state [B,H,D,D]
    Outputs: out [B,T,H,D], new_state [B,H,D,D]

    The C++ extension `Qwen3Ext::GatedDeltaRuleV2` matches this op name and
    takes over evaluate() at runtime when the .so is loaded via core.add_extension.
    The numpy fallback below is for diagnostics only.
    """

    def __init__(self, inputs=None):
        super().__init__(self, inputs)

    def validate_and_infer_types(self):
        et = self.get_input_element_type(0)
        mqkv = self.get_input_partial_shape(0)
        gps  = self.get_input_partial_shape(1)
        sps  = self.get_input_partial_shape(3)
        out_shape = ov.PartialShape([mqkv[0], mqkv[1], gps[2], sps[3]])
        self.set_output_type(0, et, out_shape)
        self.set_output_type(1, et, sps)

    def clone_with_new_inputs(self, new_inputs):
        return GatedDeltaRuleV2(list(new_inputs))

    def visit_attributes(self, visitor):
        return True

    def has_evaluate(self):
        return True

    def evaluate(self, outputs, inputs):
        mqkv = np.asarray(inputs[0].data)
        g    = np.asarray(inputs[1].data)
        beta = np.asarray(inputs[2].data)
        S    = np.asarray(inputs[3].data).copy()
        B, T, qkv_dim = mqkv.shape
        H, D = g.shape[2], S.shape[3]
        key_dim = H * D
        value_dim = qkv_dim - 2 * key_dim
        outputs[0].shape = (B, T, H, D)
        outputs[1].shape = S.shape
        out = np.asarray(outputs[0].data)
        if _use_c():
            mqkv_c = np.ascontiguousarray(mqkv, dtype=np.float32)
            g_c    = np.ascontiguousarray(g,    dtype=np.float32)
            b_c    = np.ascontiguousarray(beta, dtype=np.float32)
            S_c    = np.ascontiguousarray(S,    dtype=np.float32)
            out_c  = np.empty((B, T, H, D), dtype=np.float32)
            _gdr_v2_c(mqkv_c, g_c, b_c, S_c, out_c, key_dim, value_dim)
            out[...] = out_c
            np.asarray(outputs[1].data)[...] = S_c
            return True
        # NumPy fallback (slow; for the C++ op path this should never run).
        return GatedDeltaRule._evaluate_v2(self, outputs, inputs)


class GatedDeltaRuleV3(Op):
    """v3 op: v2 + absorbed conv1d-with-state + SiLU + Transposes.

    Inputs: mixed_in [B,T,C], conv_w [C,1,1,K], prev_conv [B,C,K-1],
            g [B,T,H], beta [B,T,H], prev_state [B,H,D,D]
    Outputs: out [B,T,H,D], new_state [B,H,D,D], new_conv [B,C,K-1]
    """

    def __init__(self, inputs=None):
        super().__init__(self, inputs)

    def validate_and_infer_types(self):
        et = self.get_input_element_type(0)
        mi = self.get_input_partial_shape(0)   # [B, T, C]
        pc = self.get_input_partial_shape(2)   # [B, C, K-1]
        gs = self.get_input_partial_shape(3)   # [B, T, H]
        ss = self.get_input_partial_shape(5)   # [B, H, D, D]
        self.set_output_type(0, et, ov.PartialShape([mi[0], mi[1], gs[2], ss[3]]))
        self.set_output_type(1, et, ss)
        self.set_output_type(2, et, pc)

    def clone_with_new_inputs(self, new_inputs):
        return GatedDeltaRuleV3(list(new_inputs))

    def visit_attributes(self, visitor):
        return True

    def has_evaluate(self):
        return True

    def evaluate(self, outputs, inputs):
        # NumPy fallback for diagnostics; the real implementation is the C++ op.
        mi    = np.asarray(inputs[0].data)
        cw    = np.asarray(inputs[1].data)
        pc    = np.asarray(inputs[2].data)
        g     = np.asarray(inputs[3].data)
        beta  = np.asarray(inputs[4].data)
        state = np.asarray(inputs[5].data).copy()
        B, T, C = mi.shape
        H, D = g.shape[2], state.shape[3]
        K_conv = cw.shape[-1]
        KS = K_conv - 1
        key_dim = H * D
        # Output shapes
        outputs[0].shape = (B, T, H, D)
        outputs[1].shape = state.shape
        outputs[2].shape = pc.shape
        out = np.asarray(outputs[0].data)
        new_conv = np.asarray(outputs[2].data)

        # Conv1d-with-state + SiLU + Transpose to NLC
        cw_2d = cw.reshape(C, K_conv)
        # Build padded sequence: (prev_conv ++ mixed_in transposed to NCL) -> [B, C, KS+T]
        mi_NCT = np.ascontiguousarray(mi.transpose(0, 2, 1))  # [B, C, T]
        padded = np.concatenate([pc, mi_NCT], axis=-1)        # [B, C, KS+T]
        conv_out = np.empty((B, C, T), dtype=np.float32)
        for k in range(K_conv):
            conv_out += cw_2d[:, k:k+1] * padded[:, :, k:k+T] if k > 0 else cw_2d[:, k:k+1] * padded[:, :, k:k+T]
        # Above is buggy (overwrites); redo cleanly
        conv_out = np.zeros((B, C, T), dtype=np.float32)
        for k in range(K_conv):
            conv_out += cw_2d[None, :, k:k+1] * padded[:, :, k:k+T]
        # SiLU
        silu = conv_out * (1.0 / (1.0 + np.exp(-conv_out)))
        # Transpose to NLC
        post = silu.transpose(0, 2, 1)  # [B, T, C]

        # new_conv = last KS of (prev ++ cur), per channel
        if T >= KS:
            new_conv[...] = mi_NCT[:, :, T - KS:T]
        else:
            new_conv[..., :KS - T] = pc[..., T:]
            new_conv[..., KS - T:] = mi_NCT

        # Then v2-style: split, L2 norm, scale, GDR step
        eps = 1e-6
        Q = post[..., :key_dim].reshape(B, T, H, D).astype(np.float32)
        K = post[..., key_dim:2*key_dim].reshape(B, T, H, D).astype(np.float32)
        V = post[..., 2*key_dim:].reshape(B, T, H, D).astype(np.float32)
        Q = Q / np.sqrt((Q*Q).sum(-1, keepdims=True) + eps) / np.sqrt(D)
        K = K / np.sqrt((K*K).sum(-1, keepdims=True) + eps)
        for t in range(T):
            for h in range(H):
                qt = Q[:, t, h, :]
                kt = K[:, t, h, :]
                vt = V[:, t, h, :]
                gt = np.exp(g[:, t, h])[:, None, None]
                bt = beta[:, t, h][:, None]
                state[:, h] *= gt
                kv_mem = np.einsum("bkv,bk->bv", state[:, h], kt)
                delta  = (vt - kv_mem) * bt
                state[:, h] += np.einsum("bk,bv->bkv", kt, delta)
                out[:, t, h, :] = np.einsum("bkv,bk->bv", state[:, h], qt)
        np.asarray(outputs[1].data)[...] = state
        return True


def _find_groupconv_above(node_output, max_hops=10):
    """Walk node_output backward to find a GroupConvolution. Returns the node or None."""
    n = node_output.get_node()
    for _ in range(max_hops):
        if n.get_type_name() == "GroupConvolution":
            return n
        if n.get_input_size() == 0:
            return None
        n = n.input(0).get_source_output().get_node()
    return None


def _find_conv_state_assign(concat):
    """The conv state Assign is reached from the Concat via a Slice (the
    'last KS' slice). Return (slice_node, assign_node) or (None, None)."""
    for inp in concat.output(0).get_target_inputs():
        n = inp.get_node()
        if n.get_type_name() == "Slice":
            for inp2 in n.output(0).get_target_inputs():
                n2 = inp2.get_node()
                if n2.get_type_name() == "Assign":
                    return n, n2
    return None, None


def _find_gd_state_assign(loop_output1):
    """The gated-delta state Assign is reached from the Loop's state output via
    a PyTorch-export-pattern chain:
        Loop.output(1) -> Reshape (flatten to 1D)
                       -> Concat (combine with the per-T output reshape)
                       -> Slice (drop the per-T prefix)
                       -> Reshape (restore [B,H,D,D])
                       -> Assign
    Return the Assign node or None."""
    visited = set()
    def walk(out, depth):
        if depth > 8: return None
        for inp in out.get_target_inputs():
            n = inp.get_node()
            if n.get_friendly_name() in visited: continue
            visited.add(n.get_friendly_name())
            if n.get_type_name() == "Assign":
                return n
            if n.get_type_name() in {"Reshape", "Concat", "Slice"}:
                for j in range(n.get_output_size()):
                    res = walk(n.output(j), depth + 1)
                    if res is not None:
                        return res
        return None
    return walk(loop_output1, 0)


def replace_gated_delta_rule_loops_v3(model: ov.Model) -> int:
    """Replace each gated-delta Loop with a v3 op that absorbs the conv1d-
    with-state + SiLU + Transposes + split / norm / scale / transpose chain.

    Returns the number of Loops replaced.
    """
    from openvino import opset15 as ops
    targets = [n for n in model.get_ops()
               if n.get_type_name() == "Loop" and _is_gated_delta_rule_loop(n)]

    replaced = 0
    for loop in targets:
        # Walk q input back to find Swish -> Slice -> GroupConv -> Concat chain.
        # Q chain: Divide -> Transpose -> Multiply -> Reshape -> VariadicSplit
        #          -> Transpose -> Swish -> Slice -> GroupConv -> Concat
        q_node = loop.input(2).get_source_output().get_node()
        gc = _find_groupconv_above(q_node.output(0), max_hops=10)
        if gc is None:
            continue
        concat = gc.input(0).get_source_output().get_node()
        if concat.get_type_name() != "Concat":
            continue

        # Sources for v3
        conv_w_src    = gc.input(1).get_source_output()           # [C, 1, 1, K] (via Convert)
        prev_conv_src = concat.input(0).get_source_output()       # [B, C, KS] (post-Gather)
        current_node  = concat.input(1).get_source_output().get_node()  # Transpose to [B, C, T]
        if current_node.get_type_name() != "Transpose":
            continue
        # The MatMul output feeds the Transpose
        mixed_in_src  = current_node.input(0).get_source_output()  # [B, T, C]

        g_src    = loop.input(5).get_source_output()
        beta_src = loop.input(6).get_source_output()
        state    = loop.input(7).get_source_output()
        g_pre    = _find_pre_transpose_3d(g_src)
        beta_pre = _find_pre_transpose_3d(beta_src)

        v3 = GatedDeltaRuleV3([mixed_in_src, conv_w_src, prev_conv_src,
                                g_pre, beta_pre, state])
        v3.set_friendly_name(loop.get_friendly_name() + "/FusedV3")

        # Find the gated-delta state Assign BEFORE replacing loop.output(1),
        # since we walk forward from the loop's output to reach it.
        gd_assign = _find_gd_state_assign(loop.output(1))

        # out [B, T, H, D] -> Transpose to [B, H, T, D] for the downstream
        perm = ops.constant(np.array([0, 2, 1, 3], dtype=np.int64))
        out_BHTD = ops.transpose(v3.output(0), perm)
        loop.output(0).replace(out_BHTD.output(0))
        loop.output(1).replace(v3.output(1))

        # Bypass the PyTorch-export Reshape->Concat->Slice->Reshape chain on
        # the new-state path: wire the Assign directly to v3.output(1).
        if gd_assign is not None:
            gd_assign.input(0).replace_source_output(v3.output(1))

        # Rewire the conv-state Assign to take v3.output(2)
        slice_node, conv_assign = _find_conv_state_assign(concat)
        if conv_assign is not None:
            conv_assign.input(0).replace_source_output(v3.output(2))

        replaced += 1

    return replaced


def register(core: ov.Core) -> None:
    """Register the custom op classes with an OV Core."""
    core.add_extension(GatedDeltaRule)
    core.add_extension(GatedDeltaRuleV2)
    core.add_extension(GatedDeltaRuleV3)
