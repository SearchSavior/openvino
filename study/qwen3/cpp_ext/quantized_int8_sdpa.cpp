#include "quantized_int8_sdpa.hpp"
#include "kernels.h"

#include <cmath>

using namespace Qwen3Ext;

QuantizedInt8SDPA::QuantizedInt8SDPA(const ov::OutputVector& args) : Op(args) {
    constructor_validate_and_infer_types();
}

void QuantizedInt8SDPA::validate_and_infer_types() {
    // Output: [B, H_q, T_q, D] f32  (= shape of Q).
    set_output_type(0, ov::element::f32, get_input_partial_shape(0));
}

std::shared_ptr<ov::Node> QuantizedInt8SDPA::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    return std::make_shared<QuantizedInt8SDPA>(new_args);
}

bool QuantizedInt8SDPA::visit_attributes(ov::AttributeVisitor&) {
    return true;
}

bool QuantizedInt8SDPA::has_evaluate() const {
    return true;
}

bool QuantizedInt8SDPA::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    const auto& q       = inputs[0];
    const auto& k_data  = inputs[1];
    const auto& k_scale = inputs[2];
    const auto& v_data  = inputs[3];
    const auto& v_scale = inputs[4];
    const auto& mask    = inputs[5];
    const auto& scale   = inputs[6];

    const auto qs  = q.get_shape();
    const auto kds = k_data.get_shape();
    OPENVINO_ASSERT(qs.size() == 4 && kds.size() == 4, "QuantizedInt8SDPA rank mismatch");

    const int B      = static_cast<int>(qs[0]);
    const int H_q    = static_cast<int>(qs[1]);
    const int T_q    = static_cast<int>(qs[2]);
    const int D      = static_cast<int>(qs[3]);
    const int H_kv   = static_cast<int>(kds[1]);
    const int T_full = static_cast<int>(kds[2]);

    // scalar f32 in tensor of shape {} or {1}; just read the first element.
    const float kq_scale = *static_cast<const float*>(scale.data());

    // mask may be a "no-op" 1-element tensor; treat as null if numel < T_full.
    const float *mask_ptr = nullptr;
    if (mask.get_size() >= static_cast<size_t>(T_q) * (size_t)T_full) {
        mask_ptr = static_cast<const float*>(mask.data());
    }

    auto& out = outputs[0];
    out.set_shape({(size_t)B, (size_t)H_q, (size_t)T_q, (size_t)D});

    int8_sdpa_kernel(
        static_cast<const float*>      (q      .data()),
        static_cast<const signed char*>(k_data .data()),
        static_cast<const float*>      (k_scale.data()),
        static_cast<const signed char*>(v_data .data()),
        static_cast<const float*>      (v_scale.data()),
        mask_ptr,
        kq_scale,
        static_cast<float*>            (out    .data()),
        B, H_q, H_kv, T_q, T_full, D);
    return true;
}
