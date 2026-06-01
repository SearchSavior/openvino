/*
 * FusedCausalConv1d custom op — C++ wrapper around conv1d_kernel().
 *
 * Inputs:
 *   0: prev_state    [B, C, KS]         fp32
 *   1: current_input [B, C, T]          fp32
 *   2: weight        [C, 1, 1, K]       fp32   depthwise weights
 *
 * Outputs:
 *   0: conv_output   [B, C, KS+T-K+1]   fp32
 *   1: new_state     [B, C, KS]         fp32
 */
#pragma once

#include <openvino/op/op.hpp>

namespace Qwen3Ext {

class FusedCausalConv1d : public ov::op::Op {
public:
    OPENVINO_OP("FusedCausalConv1d");

    FusedCausalConv1d() = default;
    FusedCausalConv1d(const ov::OutputVector& args);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;
};

}  // namespace Qwen3Ext
