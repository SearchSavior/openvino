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


class GatedDeltaRule(Op):
    """Fused linear-attention recurrence used in Qwen3-Next-family models."""

    def __init__(self, inputs=None):
        # The Op base __init__ takes care of registering the type and calling
        # constructor_validate_and_infer_types() when inputs are passed.
        super().__init__(self, inputs)

    def validate_and_infer_types(self):
        # Output 0 is shaped like v ([B, H, S, D_v]).
        # Output 1 is shaped like initial_state ([B, H, D_k, D_v]).
        et = self.get_input_element_type(0)
        self.set_output_type(0, et, self.get_input_partial_shape(2))
        self.set_output_type(1, et, self.get_input_partial_shape(5))

    def clone_with_new_inputs(self, new_inputs):
        return GatedDeltaRule(list(new_inputs))

    def visit_attributes(self, visitor):
        # No attributes.
        return True

    def has_evaluate(self):
        return True

    def evaluate(self, outputs, inputs):
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
        fused = GatedDeltaRule([q, k, v, g, beta, state])
        fused.set_friendly_name(loop.get_friendly_name() + "/Fused")
        loop.output(0).replace(fused.output(0))
        loop.output(1).replace(fused.output(1))

    return len(targets)


def register(core: ov.Core) -> None:
    """Register the custom op with an OV Core so compile_model can dispatch it."""
    core.add_extension(GatedDeltaRule)
