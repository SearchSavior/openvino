#include "gated_delta_rule_v2.hpp"
#include "kernels.h"

#include <cstring>

using namespace Qwen3Ext;

GatedDeltaRuleV2::GatedDeltaRuleV2(const ov::OutputVector& args) : Op(args) {
    constructor_validate_and_infer_types();
}

void GatedDeltaRuleV2::validate_and_infer_types() {
    const auto et = get_input_element_type(0);
    const auto mqkv = get_input_partial_shape(0);   // [B, T, qkv_dim]
    const auto gps  = get_input_partial_shape(1);   // [B, T, H]
    const auto sps  = get_input_partial_shape(3);   // [B, H, D, D]
    // out: [B, T, H, D]
    set_output_type(0, et, ov::PartialShape{mqkv[0], mqkv[1], gps[2], sps[3]});
    set_output_type(1, et, sps);
}

std::shared_ptr<ov::Node> GatedDeltaRuleV2::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    return std::make_shared<GatedDeltaRuleV2>(new_args);
}

bool GatedDeltaRuleV2::visit_attributes(ov::AttributeVisitor&) { return true; }

bool GatedDeltaRuleV2::has_evaluate() const { return true; }

bool GatedDeltaRuleV2::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    const auto& mqkv  = inputs[0];
    const auto& g     = inputs[1];
    const auto& beta  = inputs[2];
    const auto& state = inputs[3];

    const auto mqs = mqkv.get_shape();
    const auto gs  = g.get_shape();
    const auto ss  = state.get_shape();
    OPENVINO_ASSERT(mqs.size() == 3 && gs.size() == 3 && ss.size() == 4,
                    "GatedDeltaRuleV2 shape rank mismatch");

    const int B  = static_cast<int>(mqs[0]);
    const int T  = static_cast<int>(mqs[1]);
    const int H  = static_cast<int>(gs[2]);
    const int D  = static_cast<int>(ss[3]);
    const int qkv_dim = static_cast<int>(mqs[2]);
    const int key_dim   = H * D;
    const int value_dim = qkv_dim - 2 * key_dim;
    OPENVINO_ASSERT(value_dim > 0, "value_dim must be > 0");

    auto& out  = outputs[0];
    auto& fst  = outputs[1];
    out.set_shape({(size_t)B, (size_t)T, (size_t)H, (size_t)D});
    fst.set_shape(ss);

    // Copy initial state -> final_state in-place, then kernel mutates it.
    std::memcpy(fst.data(), state.data(), state.get_byte_size());

    gdr_kernel_v2(
        static_cast<const float*>(mqkv.data()),
        static_cast<const float*>(g.data()),
        static_cast<const float*>(beta.data()),
        static_cast<float*>(fst.data()),
        static_cast<float*>(out.data()),
        B, H, T, D, key_dim, value_dim);
    return true;
}
