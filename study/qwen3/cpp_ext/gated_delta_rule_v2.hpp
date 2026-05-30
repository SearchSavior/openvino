/*
 * GatedDeltaRuleV2 - same recurrence, with the upstream split / reshape /
 * L2 norm / Q-scale / transpose absorbed into the kernel. Eliminates ~16
 * IR-level intermediate tensors per linear-attn layer.
 *
 * Inputs:
 *   0: mixed_qkv  [B, T, key_dim*2 + value_dim]   fp32
 *   1: g          [B, T, H]                        fp32   (pre-transpose)
 *   2: beta       [B, T, H]                        fp32   (pre-transpose)
 *   3: state      [B, H, D, D]                     fp32   (in/out)
 *
 * Outputs:
 *   0: out        [B, T, H, D]                     fp32   (no transpose)
 *   1: final_state [B, H, D, D]                    fp32
 *
 * Layout convention: Q [B,T,H,D] occupies mixed_qkv[..., 0 : H*D]; K occupies
 * [H*D : 2*H*D]; V occupies [2*H*D : qkv_dim]. No GQA in this model's
 * linear-attn so key_dim == value_dim == H * D.
 */
#pragma once

#include <openvino/op/op.hpp>

namespace Qwen3Ext {

class GatedDeltaRuleV2 : public ov::op::Op {
public:
    OPENVINO_OP("GatedDeltaRuleV2");

    GatedDeltaRuleV2() = default;
    GatedDeltaRuleV2(const ov::OutputVector& args);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;
};

}  // namespace Qwen3Ext
