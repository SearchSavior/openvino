/*
 * GatedDeltaRule custom op — subclass of ov::op::internal::GatedDeltaNet.
 *
 * Subtle: the parent's args-taking constructors call constructor_validate_
 * and_infer_types() inside the base-class body, where dynamic_type is still
 * the base class — so virtual dispatch lands on the parent's strict
 * [B,T,H,D] shape inference instead of our [B,H,T,D] override. To bypass,
 * we call the parent's DEFAULT constructor (no validate), then do
 * set_arguments() + constructor_validate_and_infer_types() ourselves from
 * the derived constructor body, where dynamic_type is now our class.
 *
 * Consequence: the parent's private m_fuse_qk_l2norm / m_q_l2_norm_eps /
 * m_k_l2_norm_eps fields stay default-initialized — we shadow them with
 * our own m_* copies and override the getters / visit_attributes.
 */
#pragma once

#include <openvino/op/gated_delta_net.hpp>

namespace Qwen3Ext {

class GatedDeltaRule : public ov::op::internal::GatedDeltaNet {
public:
    OPENVINO_OP("GatedDeltaRule", "extension", ov::op::internal::GatedDeltaNet);

    GatedDeltaRule() = default;
    GatedDeltaRule(const ov::OutputVector& args,
                   bool fuse_qk_l2norm = false,
                   float q_l2_norm_eps = 1e-6F,
                   float k_l2_norm_eps = 1e-6F);

    void validate_and_infer_types() override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;

private:
    bool m_l2norm_flag = false;
    float m_q_eps = 1e-6F;
    float m_k_eps = 1e-6F;
};

}  // namespace Qwen3Ext
