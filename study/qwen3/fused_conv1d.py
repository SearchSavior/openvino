"""
Fused causal conv1d with explicit prev-state.

Replaces the per-layer
    Concat(prev_state, current_input, axis=-1) ─┬─ GroupConvolution → ...
                                                └─ Slice(last K positions) → Assign

subgraph with a single custom op that computes the depthwise conv directly
against (prev_state, current_input) — without materialising the [B, C, T+K]
concatenated tensor — and emits the new state alongside.

Inputs:
    0: prev_state    [B, C, KS]         fp32
    1: current_input [B, C, T]          fp32   (already in NCL layout)
    2: weight        [C, 1, 1, K]       fp32   depthwise conv weights
Outputs:
    0: conv_output   [B, C, KS+T-K+1]   fp32   matches GroupConvolution output shape
    1: new_state     [B, C, KS]         fp32   last KS positions of (prev_state ++ current_input)
"""
from __future__ import annotations

import numpy as np
import openvino as ov
from openvino import Op

from kernels import use_c as _use_c, conv1d as _conv1d_c


class FusedCausalConv1d(Op):
    def __init__(self, inputs=None):
        super().__init__(self, inputs)

    def validate_and_infer_types(self):
        et = self.get_input_element_type(0)
        prev_shape = self.get_input_partial_shape(0)     # [B, C, KS]
        cur_shape = self.get_input_partial_shape(1)      # [B, C, T]
        w_shape = self.get_input_partial_shape(2)        # [C, 1, 1, K]
        # conv_output: same B, C; seq = KS + T - K + 1
        out_shape = ov.PartialShape([
            cur_shape[0], cur_shape[1],
            ov.Dimension.dynamic(),                       # KS + T - K + 1
        ])
        self.set_output_type(0, et, out_shape)
        # new_state: same as prev_state shape
        self.set_output_type(1, et, prev_shape)

    def clone_with_new_inputs(self, new_inputs):
        return FusedCausalConv1d(list(new_inputs))

    def visit_attributes(self, visitor):
        return True

    def has_evaluate(self):
        return True

    def evaluate(self, outputs, inputs):
        prev = np.asarray(inputs[0].data)         # [B, C, KS]
        cur = np.asarray(inputs[1].data)          # [B, C, T]
        w = np.asarray(inputs[2].data)            # [C, 1, 1, K]

        B, C, KS = prev.shape
        _, _, T = cur.shape
        K = w.shape[-1]
        out_len = KS + T - K + 1

        # Depthwise weights as [C, K]
        wc = w[:, 0, 0, :]                        # [C, K]

        outputs[0].shape = (B, C, out_len)
        out = np.asarray(outputs[0].data)
        outputs[1].shape = prev.shape
        ns = np.asarray(outputs[1].data)

        if _use_c():
            prev_c = np.ascontiguousarray(prev, dtype=np.float32)
            cur_c = np.ascontiguousarray(cur, dtype=np.float32)
            wc_c = np.ascontiguousarray(wc, dtype=np.float32)
            out_c = np.empty((B, C, out_len), dtype=np.float32)
            ns_c = np.empty((B, C, KS), dtype=np.float32)
            _conv1d_c(prev_c, cur_c, wc_c, out_c, ns_c)
            out[...] = out_c
            ns[...] = ns_c
            return True

        out.fill(0.0)

        # For each kernel position k, the source index in the implicit padded sequence
        # is (t + k) for output position t. Source < KS → prev_state[t+k];
        # otherwise → current_input[t+k-KS]. Compute the two contiguous ranges
        # without materialising the concat.
        for k in range(K):
            cutover = max(0, min(KS - k, out_len))    # t in [0, cutover) reads prev_state
            wk = wc[:, k][None, :, None]              # broadcast [1, C, 1]
            if cutover > 0:
                out[..., :cutover] += wk * prev[..., k:k + cutover]
            if cutover < out_len:
                out[..., cutover:] += wk * cur[..., : out_len - cutover]

        # new_state = last KS positions of (prev ++ cur).
        if T >= KS:
            ns[...] = cur[..., T - KS: T]
        else:
            # T < KS — spans both
            ns[..., : KS - T] = prev[..., T:]
            ns[..., KS - T:] = cur
        return True


# ---------------------------------------------------------------------------
# Graph rewrite: replace each linear-attn Concat→{GroupConv, Slice(state)} with the fused op
# ---------------------------------------------------------------------------

def _matches_conv_state_concat(concat) -> bool:
    if concat.get_type_name() != "Concat":
        return False
    attrs = concat.get_attributes()
    if attrs.get("axis") != -1:
        return False
    if "linear_attn/aten::cat/Concat" not in concat.get_friendly_name():
        return False
    return True


def replace_causal_conv1d_chains(model: ov.Model) -> int:
    """Walk model, replace each linear-attn conv1d state-concat with FusedCausalConv1d."""
    concats = [op for op in model.get_ops() if _matches_conv_state_concat(op)]
    replaced = 0

    for concat in concats:
        prev_state = concat.input(0).get_source_output()    # Gather of ReadValue
        current_input = concat.input(1).get_source_output() # Transpose

        # Find downstream GroupConvolution + state-Slice consumers of this Concat.
        consumers = list(concat.output(0).get_target_inputs())
        gconv = next((t.get_node() for t in consumers
                      if t.get_node().get_type_name() == "GroupConvolution"), None)
        state_slice = next((t.get_node() for t in consumers
                            if t.get_node().get_type_name() == "Slice"), None)
        if gconv is None or state_slice is None:
            continue

        weight = gconv.input(1).get_source_output()         # [C, 1, 1, K]
        fused = FusedCausalConv1d([prev_state, current_input, weight])
        fused.set_friendly_name(concat.get_friendly_name() + "/Fused")

        gconv.output(0).replace(fused.output(0))
        state_slice.output(0).replace(fused.output(1))
        replaced += 1

    return replaced


def register(core: ov.Core) -> None:
    core.add_extension(FusedCausalConv1d)
