#include "quantized_kv_cache.hpp"
#include "kernels.h"

using namespace Qwen3Ext;

QuantizedKVCache::QuantizedKVCache(const ov::OutputVector& args) : Op(args) {
    constructor_validate_and_infer_types();
}

void QuantizedKVCache::validate_and_infer_types() {
    const auto prev_data_ps = get_input_partial_shape(0);   // [B, H, T_prev, D] i8
    const auto new_ps       = get_input_partial_shape(2);   // [B, H, N,     D] f32

    // Time dim of output = T_prev + N. Both are dynamic in normal use.
    ov::Dimension t_full = ov::Dimension::dynamic();
    if (prev_data_ps[2].is_static() && new_ps[2].is_static()) {
        t_full = ov::Dimension(prev_data_ps[2].get_length() + new_ps[2].get_length());
    }
    ov::PartialShape full_shape{new_ps[0], new_ps[1], t_full, new_ps[3]};
    ov::PartialShape scale_shape{new_ps[0], new_ps[1], t_full};

    set_output_type(0, ov::element::f32, full_shape);   // for SDPA
    set_output_type(1, ov::element::i8,  full_shape);   // to Assign(data)
    set_output_type(2, ov::element::f32, scale_shape);  // to Assign(scale)
}

std::shared_ptr<ov::Node> QuantizedKVCache::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    return std::make_shared<QuantizedKVCache>(new_args);
}

bool QuantizedKVCache::visit_attributes(ov::AttributeVisitor&) {
    return true;
}

bool QuantizedKVCache::has_evaluate() const {
    return true;
}

bool QuantizedKVCache::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    const auto& prev_data  = inputs[0];
    const auto& prev_scale = inputs[1];
    const auto& new_kv     = inputs[2];

    const auto pds = prev_data.get_shape();
    const auto pss = prev_scale.get_shape();
    const auto nks = new_kv.get_shape();
    OPENVINO_ASSERT(pds.size() == 4 && pss.size() == 3 && nks.size() == 4,
                    "QuantizedKVCache shape rank mismatch");

    const int B      = static_cast<int>(pds[0]);
    const int H      = static_cast<int>(pds[1]);
    const int T_prev = static_cast<int>(pds[2]);
    const int D      = static_cast<int>(pds[3]);
    const int N      = static_cast<int>(nks[2]);
    const int T_full = T_prev + N;

    auto& full_f32  = outputs[0];
    auto& new_data  = outputs[1];
    auto& new_scale = outputs[2];

    full_f32 .set_shape({(size_t)B, (size_t)H, (size_t)T_full, (size_t)D});
    new_data .set_shape({(size_t)B, (size_t)H, (size_t)T_full, (size_t)D});
    new_scale.set_shape({(size_t)B, (size_t)H, (size_t)T_full});

    qkv_kernel(
        static_cast<const signed char*>(prev_data .data()),
        static_cast<const float*>      (prev_scale.data()),
        static_cast<const float*>      (new_kv    .data()),
        static_cast<signed char*>      (new_data  .data()),
        static_cast<float*>            (new_scale .data()),
        static_cast<float*>            (full_f32  .data()),
        B, H, T_prev, N, D);
    return true;
}
