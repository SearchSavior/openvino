/*
 * GatedDeltaRuleV3 - v2 plus absorbed conv1d-with-state + SiLU + Transposes.
 *
 * Inputs:
 *   0: mixed_in       [B, T, C]            fp32  in_proj_qkv MatMul output
 *   1: conv_w         [C, 1, 1, K]         fp32  depthwise conv weights
 *                     (memory-equivalent to [C, K] in row-major)
 *   2: prev_conv      [B, C, K-1]          fp32  previous conv state
 *   3: g              [B, T, H]            fp32  pre-transpose decay gate
 *   4: beta           [B, T, H]            fp32  pre-transpose step size
 *   5: prev_state     [B, H, D, D]         fp32  recurrent state
 *
 * Outputs:
 *   0: out            [B, T, H, D]         fp32
 *   1: new_state      [B, H, D, D]         fp32  final recurrent state
 *   2: new_conv       [B, C, K-1]          fp32  last K-1 inputs of (prev ++ cur)
 *
 * C = key_dim * 2 + value_dim (= 6144 for Qwen3.5-0.8B). No GQA in
 * linear-attn so key_dim = value_dim = H * D.
 *
 * Absorbs the entire IR chain
 *     Transpose -> Concat(state) -> GroupConv -> Slice -> Swish ->
 *     Transpose -> VariadicSplit -> Reshape -> Multiply(L2) -> Transpose ->
 *     Divide(scale)
 * into one kernel call -- ~21 IR-level edge buffers per linear-attn layer.
 */
#pragma once

#include <openvino/op/op.hpp>

namespace Qwen3Ext {

class GatedDeltaRuleV3 : public ov::op::Op {
public:
    OPENVINO_OP("GatedDeltaRuleV3");

    GatedDeltaRuleV3() = default;
    GatedDeltaRuleV3(const ov::OutputVector& args);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;
};

}  // namespace Qwen3Ext
