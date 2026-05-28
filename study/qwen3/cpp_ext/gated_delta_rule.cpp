#include "gated_delta_rule.hpp"
#include "kernels.h"

#include <cstring>

using namespace Qwen3Ext;

GatedDeltaRule::GatedDeltaRule(const ov::OutputVector& args) : Op(args) {
    constructor_validate_and_infer_types();
}

void GatedDeltaRule::validate_and_infer_types() {
    const auto et = get_input_element_type(0);
    // Output 0 has the shape of v (input 2); output 1 has the shape of initial_state (input 5).
    set_output_type(0, et, get_input_partial_shape(2));
    set_output_type(1, et, get_input_partial_shape(5));
}

std::shared_ptr<ov::Node> GatedDeltaRule::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    return std::make_shared<GatedDeltaRule>(new_args);
}

bool GatedDeltaRule::visit_attributes(ov::AttributeVisitor&) {
    return true;
}

bool GatedDeltaRule::has_evaluate() const {
    return true;
}

bool GatedDeltaRule::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    const auto& q_t = inputs[0];
    const auto& k_t = inputs[1];
    const auto& v_t = inputs[2];
    const auto& g_t = inputs[3];
    const auto& beta_t = inputs[4];
    const auto& state_t = inputs[5];

    const auto qs = q_t.get_shape();
    OPENVINO_ASSERT(qs.size() == 4, "GatedDeltaRule expects rank-4 q");
    const int B = static_cast<int>(qs[0]);
    const int H = static_cast<int>(qs[1]);
    const int T = static_cast<int>(qs[2]);
    const int D = static_cast<int>(qs[3]);

    const auto ss = state_t.get_shape();
    OPENVINO_ASSERT(ss.size() == 4 && static_cast<int>(ss[2]) == D && static_cast<int>(ss[3]) == D,
                    "GatedDeltaRule: state shape mismatch");

    auto& out = outputs[0];
    auto& final_state = outputs[1];
    out.set_shape({static_cast<size_t>(B), static_cast<size_t>(H),
                   static_cast<size_t>(T), static_cast<size_t>(D)});
    final_state.set_shape(state_t.get_shape());

    // Copy initial state -> final_state, then mutate final_state in place.
    std::memcpy(final_state.data(), state_t.data(), state_t.get_byte_size());

    gdr_kernel(
        static_cast<const float*>(q_t.data()),
        static_cast<const float*>(k_t.data()),
        static_cast<const float*>(v_t.data()),
        static_cast<const float*>(g_t.data()),
        static_cast<const float*>(beta_t.data()),
        static_cast<float*>(final_state.data()),
        static_cast<float*>(out.data()),
        B, H, T, D);
    return true;
}
