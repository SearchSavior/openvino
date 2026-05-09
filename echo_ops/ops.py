# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Custom Gated DeltaNet ops built with the OpenVINO Python extension API.

The set covers what a typical Gated DeltaNet block needs:

    L2Norm              - L2 normalize along the last dim (q, k pre-attn).
    ShortConv1D         - causal depthwise 1-D conv along time (pre-mixer).
    GatedDeltaRule      - sequence-level gated delta-rule recurrence.
    GatedDeltaRuleStep  - single-step variant for autoregressive decoding.
    GatedRMSNorm        - RMSNorm with a SiLU output gate (post-mixer).

Each op overrides validate_and_infer_types / clone_with_new_inputs and
implements evaluate() in NumPy so the ops can run on CPU and survive
constant folding. Attributes are exposed via visit_attributes so the
ops round-trip through OV IR.
"""

import numpy as np

from openvino import DiscreteTypeInfo, Op, PartialShape

from . import reference


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_outputs_from_tensors(outputs, arrays):
    """Copy a list of NumPy arrays into a list of OV Tensors (resizing them)."""
    for i, arr in enumerate(arrays):
        outputs[i].shape = arr.shape
        np.asarray(outputs[i].data)[...] = arr


def _np(tensor, dtype=None):
    """Read an OV Tensor into a NumPy array."""
    arr = np.asarray(tensor.data)
    if dtype is not None and arr.dtype != dtype:
        arr = arr.astype(dtype)
    return arr


def _f32_in(tensor):
    """Read an OV Tensor as f32 (upcasting from f16/bf16 if needed). Also
    returns the original dtype so callers can downcast outputs back."""
    arr = np.asarray(tensor.data)
    orig = arr.dtype
    if orig != np.float32:
        arr = arr.astype(np.float32)
    return arr, orig


def _set_outputs_dtyped(outputs, arrays, target_dtype):
    """Copy numpy arrays into OV Tensor outputs, casting to `target_dtype`."""
    for i, arr in enumerate(arrays):
        if arr.dtype != target_dtype:
            arr = arr.astype(target_dtype)
        outputs[i].shape = arr.shape
        np.asarray(outputs[i].data)[...] = arr


# ---------------------------------------------------------------------------
# L2Norm
# ---------------------------------------------------------------------------

class L2Norm(Op):
    """L2-normalize along the last dimension: x / sqrt(sum(x*x, -1) + eps)."""

    class_type_info = DiscreteTypeInfo("L2Norm", "gated_deltanet")

    def __init__(self, inputs=None, eps: float = 1e-6):
        super().__init__(self)
        self._attrs = {"eps": float(eps)}
        if inputs is not None:
            self.set_arguments(list(inputs))
            self.constructor_validate_and_infer_types()

    def validate_and_infer_types(self):
        self.set_output_type(0, self.get_input_element_type(0),
                             self.get_input_partial_shape(0))

    def clone_with_new_inputs(self, new_inputs):
        return L2Norm(new_inputs, eps=self._attrs["eps"])

    def get_type_info(self):
        return L2Norm.class_type_info

    def visit_attributes(self, visitor):
        visitor.on_attributes(self._attrs)
        return True

    def evaluate(self, outputs, inputs):
        x, in_dtype = _f32_in(inputs[0])
        y = reference.l2_norm(x, eps=self._attrs["eps"])
        _set_outputs_dtyped(outputs, [y], in_dtype)
        return True

    def has_evaluate(self):
        return True


# ---------------------------------------------------------------------------
# ShortConv1D
# ---------------------------------------------------------------------------

class ShortConv1D(Op):
    """Causal depthwise 1-D convolution along the time axis.

    Inputs:
        x:      [B, T, D]
        weight: [D, K]   (weight[:, 0] multiplies the current time step)
    Output:
        y:      [B, T, D]
    """

    class_type_info = DiscreteTypeInfo("ShortConv1D", "gated_deltanet")

    def __init__(self, inputs=None):
        super().__init__(self)
        if inputs is not None:
            self.set_arguments(list(inputs))
            self.constructor_validate_and_infer_types()

    def validate_and_infer_types(self):
        # Output shape == x's shape
        self.set_output_type(0, self.get_input_element_type(0),
                             self.get_input_partial_shape(0))

    def clone_with_new_inputs(self, new_inputs):
        return ShortConv1D(new_inputs)

    def get_type_info(self):
        return ShortConv1D.class_type_info

    def evaluate(self, outputs, inputs):
        x, in_dtype = _f32_in(inputs[0])
        w = _np(inputs[1], dtype=np.float32)
        y = reference.short_conv1d(x, w)
        _set_outputs_dtyped(outputs, [y], in_dtype)
        return True

    def has_evaluate(self):
        return True


# ---------------------------------------------------------------------------
# GatedDeltaRule (full sequence)
# ---------------------------------------------------------------------------

class GatedDeltaRule(Op):
    """Gated delta-rule recurrence over a full sequence.

    Inputs (in order):
        q             : [B, H, T, Dk]
        k             : [B, H, T, Dk]
        v             : [B, H, T, Dv]
        g             : [B, H, T]              (forget gate, in (0, 1])
        beta          : [B, H, T]              (write strength)
        initial_state : [B, H, Dk, Dv]         (pass zeros if you don't have one)
    Outputs:
        o           : [B, H, T, Dv]
        final_state : [B, H, Dk, Dv]
    """

    class_type_info = DiscreteTypeInfo("GatedDeltaRule", "gated_deltanet")

    def __init__(self, inputs=None):
        super().__init__(self)
        if inputs is not None:
            if len(list(inputs)) != 6:
                raise ValueError(
                    "GatedDeltaRule expects 6 inputs (q, k, v, g, beta, initial_state); "
                    f"got {len(list(inputs))}")
            self.set_arguments(list(inputs))
            self.constructor_validate_and_infer_types()

    def validate_and_infer_types(self):
        et = self.get_input_element_type(0)
        q_ps = self.get_input_partial_shape(0)        # [B, H, T, Dk]
        v_ps = self.get_input_partial_shape(2)        # [B, H, T, Dv]
        s_ps = self.get_input_partial_shape(5)        # [B, H, Dk, Dv]
        # o: [B, H, T, Dv]
        if q_ps.rank.is_static and v_ps.rank.is_static:
            o_ps = PartialShape([q_ps[0], q_ps[1], q_ps[2], v_ps[-1]])
        else:
            o_ps = PartialShape.dynamic(4)
        self.set_output_type(0, et, o_ps)
        # final_state: same shape as initial_state
        self.set_output_type(1, et, s_ps)

    def clone_with_new_inputs(self, new_inputs):
        return GatedDeltaRule(new_inputs)

    def get_type_info(self):
        return GatedDeltaRule.class_type_info

    def evaluate(self, outputs, inputs):
        q, in_dtype = _f32_in(inputs[0])
        k = _np(inputs[1], dtype=np.float32)
        v = _np(inputs[2], dtype=np.float32)
        g = _np(inputs[3], dtype=np.float32)
        beta = _np(inputs[4], dtype=np.float32)
        s0 = _np(inputs[5], dtype=np.float32)
        o, s_final = reference.gated_delta_rule(q, k, v, g, beta, s0)
        _set_outputs_dtyped(outputs, [o, s_final], in_dtype)
        return True

    def has_evaluate(self):
        return True


# ---------------------------------------------------------------------------
# GatedDeltaRuleStep (single token)
# ---------------------------------------------------------------------------

class GatedDeltaRuleStep(Op):
    """Single-step gated delta-rule update (for autoregressive decoding).

    Inputs (in order):
        state : [B, H, Dk, Dv]
        q     : [B, H, Dk]
        k     : [B, H, Dk]
        v     : [B, H, Dv]
        g     : [B, H]
        beta  : [B, H]
    Outputs:
        new_state : [B, H, Dk, Dv]
        o         : [B, H, Dv]
    """

    class_type_info = DiscreteTypeInfo("GatedDeltaRuleStep", "gated_deltanet")

    def __init__(self, inputs=None):
        super().__init__(self)
        if inputs is not None:
            if len(list(inputs)) != 6:
                raise ValueError(
                    "GatedDeltaRuleStep expects 6 inputs (state, q, k, v, g, beta); "
                    f"got {len(list(inputs))}")
            self.set_arguments(list(inputs))
            self.constructor_validate_and_infer_types()

    def validate_and_infer_types(self):
        et = self.get_input_element_type(0)
        s_ps = self.get_input_partial_shape(0)     # [B, H, Dk, Dv]
        v_ps = self.get_input_partial_shape(3)     # [B, H, Dv]
        # new_state has the same shape as state
        self.set_output_type(0, et, s_ps)
        # o has the same shape as v
        self.set_output_type(1, et, v_ps)

    def clone_with_new_inputs(self, new_inputs):
        return GatedDeltaRuleStep(new_inputs)

    def get_type_info(self):
        return GatedDeltaRuleStep.class_type_info

    def evaluate(self, outputs, inputs):
        s, in_dtype = _f32_in(inputs[0])
        q = _np(inputs[1], dtype=np.float32)
        k = _np(inputs[2], dtype=np.float32)
        v = _np(inputs[3], dtype=np.float32)
        g = _np(inputs[4], dtype=np.float32)
        beta = _np(inputs[5], dtype=np.float32)
        s_new, o = reference.gated_delta_rule_step(s, q, k, v, g, beta)
        _set_outputs_dtyped(outputs, [s_new, o], in_dtype)
        return True

    def has_evaluate(self):
        return True


# ---------------------------------------------------------------------------
# GatedRMSNorm
# ---------------------------------------------------------------------------

class GatedRMSNorm(Op):
    """RMSNorm along the last dim, multiplied by a learnable weight and a
    SiLU-gated input.

    Inputs:
        x      : [..., D]
        gate   : [..., D]
        weight : [D]
    Output:
        y      : [..., D]
    """

    class_type_info = DiscreteTypeInfo("GatedRMSNorm", "gated_deltanet")

    def __init__(self, inputs=None, eps: float = 1e-6):
        super().__init__(self)
        self._attrs = {"eps": float(eps)}
        if inputs is not None:
            if len(list(inputs)) != 3:
                raise ValueError(
                    "GatedRMSNorm expects 3 inputs (x, gate, weight); "
                    f"got {len(list(inputs))}")
            self.set_arguments(list(inputs))
            self.constructor_validate_and_infer_types()

    def validate_and_infer_types(self):
        self.set_output_type(0, self.get_input_element_type(0),
                             self.get_input_partial_shape(0))

    def clone_with_new_inputs(self, new_inputs):
        return GatedRMSNorm(new_inputs, eps=self._attrs["eps"])

    def get_type_info(self):
        return GatedRMSNorm.class_type_info

    def visit_attributes(self, visitor):
        visitor.on_attributes(self._attrs)
        return True

    def evaluate(self, outputs, inputs):
        x, in_dtype = _f32_in(inputs[0])
        gate = _np(inputs[1], dtype=np.float32)
        weight = _np(inputs[2], dtype=np.float32)
        y = reference.gated_rmsnorm(x, gate, weight, eps=self._attrs["eps"])
        _set_outputs_dtyped(outputs, [y], in_dtype)
        return True

    def has_evaluate(self):
        return True
