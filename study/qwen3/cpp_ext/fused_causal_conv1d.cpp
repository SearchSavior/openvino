#include "fused_causal_conv1d.hpp"
#include "kernels.h"

using namespace Qwen3Ext;

FusedCausalConv1d::FusedCausalConv1d(const ov::OutputVector& args) : Op(args) {
    constructor_validate_and_infer_types();
}

void FusedCausalConv1d::validate_and_infer_types() {
    const auto et = get_input_element_type(0);
    const auto cur_shape = get_input_partial_shape(1);    // [B, C, T]
    const auto prev_shape = get_input_partial_shape(0);   // [B, C, KS]
    // conv_output: [B, C, dynamic]; we don't know KS+T-K+1 statically at all dim positions.
    ov::PartialShape out_shape{cur_shape[0], cur_shape[1], ov::Dimension::dynamic()};
    set_output_type(0, et, out_shape);
    set_output_type(1, et, prev_shape);
}

std::shared_ptr<ov::Node> FusedCausalConv1d::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    return std::make_shared<FusedCausalConv1d>(new_args);
}

bool FusedCausalConv1d::visit_attributes(ov::AttributeVisitor&) {
    return true;
}

bool FusedCausalConv1d::has_evaluate() const {
    return true;
}

bool FusedCausalConv1d::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    const auto& prev = inputs[0];
    const auto& cur = inputs[1];
    const auto& w = inputs[2];

    const auto pshape = prev.get_shape();
    const auto cshape = cur.get_shape();
    const auto wshape = w.get_shape();
    OPENVINO_ASSERT(pshape.size() == 3 && cshape.size() == 3 && wshape.size() == 4,
                    "FusedCausalConv1d shape rank mismatch");

    const int B = static_cast<int>(pshape[0]);
    const int C = static_cast<int>(pshape[1]);
    const int KS = static_cast<int>(pshape[2]);
    const int T = static_cast<int>(cshape[2]);
    const int K = static_cast<int>(wshape[3]);

    auto& out = outputs[0];
    auto& new_state = outputs[1];
    const int out_len = KS + T - K + 1;
    out.set_shape({static_cast<size_t>(B), static_cast<size_t>(C), static_cast<size_t>(out_len)});
    new_state.set_shape(pshape);

    // weight layout [C, 1, 1, K] is contiguous and bit-identical to [C, K] in row-major memory.
    conv1d_kernel(
        static_cast<const float*>(prev.data()),
        static_cast<const float*>(cur.data()),
        static_cast<const float*>(w.data()),
        static_cast<float*>(out.data()),
        static_cast<float*>(new_state.data()),
        B, C, KS, T, K);
    return true;
}
