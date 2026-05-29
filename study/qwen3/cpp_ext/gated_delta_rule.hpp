/*
 * GatedDeltaRule custom op — C++ wrapper around gdr_kernel().
 *
 * Inputs:
 *   0: q              [B, H, T, D]   fp32
 *   1: k              [B, H, T, D]   fp32
 *   2: v              [B, H, T, D]   fp32
 *   3: g              [B, H, T]      fp32 (decay gate, exp'd inside)
 *   4: beta           [B, H, T]      fp32
 *   5: initial_state  [B, H, D, D]   fp32
 *
 * Outputs:
 *   0: output         [B, H, T, D]   fp32 (shape = v)
 *   1: final_state    [B, H, D, D]   fp32 (shape = initial_state)
 */
#pragma once

#include <openvino/op/op.hpp>

namespace Qwen3Ext {

class GatedDeltaRule : public ov::op::Op {
public:
    OPENVINO_OP("GatedDeltaRule");

    GatedDeltaRule() = default;
    GatedDeltaRule(const ov::OutputVector& args);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;
};

}  // namespace Qwen3Ext
