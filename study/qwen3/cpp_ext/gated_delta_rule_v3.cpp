#include "gated_delta_rule_v3.hpp"
#include "kernels.h"

#include <cstring>

using namespace Qwen3Ext;

GatedDeltaRuleV3::GatedDeltaRuleV3(const ov::OutputVector& args) : Op(args) {
    constructor_validate_and_infer_types();
}

void GatedDeltaRuleV3::validate_and_infer_types() {
    const auto et = get_input_element_type(0);
    const auto mi = get_input_partial_shape(0);  // [B, T, C]
    const auto pc = get_input_partial_shape(2);  // [B, C, K-1]
    const auto gs = get_input_partial_shape(3);  // [B, T, H]
    const auto ss = get_input_partial_shape(5);  // [B, H, D, D]

    // out: [B, T, H, D]
    set_output_type(0, et, ov::PartialShape{mi[0], mi[1], gs[2], ss[3]});
    // new recurrent state: [B, H, D, D]
    set_output_type(1, et, ss);
    // new conv state: [B, C, K-1]
    set_output_type(2, et, pc);
}

std::shared_ptr<ov::Node> GatedDeltaRuleV3::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    return std::make_shared<GatedDeltaRuleV3>(new_args);
}

bool GatedDeltaRuleV3::visit_attributes(ov::AttributeVisitor&) { return true; }

bool GatedDeltaRuleV3::has_evaluate() const { return true; }

bool GatedDeltaRuleV3::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    const auto& mi    = inputs[0];
    const auto& cw    = inputs[1];
    const auto& pc    = inputs[2];
    const auto& g     = inputs[3];
    const auto& beta  = inputs[4];
    const auto& state = inputs[5];

    const auto mis = mi.get_shape();
    const auto cws = cw.get_shape();
    const auto pcs = pc.get_shape();
    const auto gs  = g.get_shape();
    const auto ss  = state.get_shape();
    OPENVINO_ASSERT(mis.size() == 3, "GatedDeltaRuleV3: mixed_in rank != 3");
    OPENVINO_ASSERT(gs.size() == 3, "GatedDeltaRuleV3: g rank != 3");
    OPENVINO_ASSERT(ss.size() == 4, "GatedDeltaRuleV3: state rank != 4");

    const int B = static_cast<int>(mis[0]);
    const int T = static_cast<int>(mis[1]);
    const int C = static_cast<int>(mis[2]);
    const int H = static_cast<int>(gs[2]);
    const int D = static_cast<int>(ss[3]);
    // conv kernel: weights are [C, 1, 1, K] in OV GroupConvolution layout.
    // Last dim of the shape is K. Memory is row-major so [C, K] is fine.
    const int K_conv = static_cast<int>(cws[cws.size() - 1]);
    const int key_dim   = H * D;
    const int value_dim = C - 2 * key_dim;

    auto& out  = outputs[0];
    auto& fst  = outputs[1];
    auto& fcv  = outputs[2];
    out.set_shape({(size_t)B, (size_t)T, (size_t)H, (size_t)D});
    fst.set_shape(ss);
    fcv.set_shape(pcs);

    // Initialize new state from prev (kernel mutates it in place).
    std::memcpy(fst.data(), state.data(), state.get_byte_size());

    gdr_kernel_v3(
        static_cast<const float*>(mi.data()),
        static_cast<const float*>(cw.data()),
        static_cast<const float*>(pc.data()),
        static_cast<const float*>(g.data()),
        static_cast<const float*>(beta.data()),
        static_cast<float*>(fst.data()),
        static_cast<float*>(out.data()),
        static_cast<float*>(fcv.data()),
        B, H, T, D, C, K_conv, key_dim, value_dim);
    return true;
}
