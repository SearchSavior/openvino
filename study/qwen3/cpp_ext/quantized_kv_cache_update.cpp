#include "quantized_kv_cache_update.hpp"
#include "kernels.h"

using namespace Qwen3Ext;

QuantizedKVCacheUpdate::QuantizedKVCacheUpdate(const ov::OutputVector& args) : Op(args) {
    constructor_validate_and_infer_types();
}

void QuantizedKVCacheUpdate::validate_and_infer_types() {
    const auto prev_data_ps = get_input_partial_shape(0);
    const auto new_ps       = get_input_partial_shape(2);

    ov::Dimension t_full = ov::Dimension::dynamic();
    if (prev_data_ps[2].is_static() && new_ps[2].is_static()) {
        t_full = ov::Dimension(prev_data_ps[2].get_length() + new_ps[2].get_length());
    }
    ov::PartialShape data_shape{new_ps[0], new_ps[1], t_full, new_ps[3]};
    ov::PartialShape scale_shape{new_ps[0], new_ps[1], t_full};

    set_output_type(0, ov::element::i8,  data_shape);
    set_output_type(1, ov::element::f32, scale_shape);
}

std::shared_ptr<ov::Node> QuantizedKVCacheUpdate::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    return std::make_shared<QuantizedKVCacheUpdate>(new_args);
}

bool QuantizedKVCacheUpdate::visit_attributes(ov::AttributeVisitor&) { return true; }

bool QuantizedKVCacheUpdate::has_evaluate() const { return true; }

bool QuantizedKVCacheUpdate::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    const auto& prev_data  = inputs[0];
    const auto& prev_scale = inputs[1];
    const auto& new_kv     = inputs[2];

    const auto pds = prev_data.get_shape();
    const auto nks = new_kv.get_shape();

    const int B      = static_cast<int>(pds[0]);
    const int H      = static_cast<int>(pds[1]);
    const int T_prev = static_cast<int>(pds[2]);
    const int D      = static_cast<int>(pds[3]);
    const int N      = static_cast<int>(nks[2]);
    const int T_full = T_prev + N;

    auto& new_data  = outputs[0];
    auto& new_scale = outputs[1];
    new_data .set_shape({(size_t)B, (size_t)H, (size_t)T_full, (size_t)D});
    new_scale.set_shape({(size_t)B, (size_t)H, (size_t)T_full});

    /* Pass NULL for full_f32: qkv_kernel skips the dequant pass. */
    qkv_kernel(
        static_cast<const signed char*>(prev_data .data()),
        static_cast<const float*>      (prev_scale.data()),
        static_cast<const float*>      (new_kv    .data()),
        static_cast<signed char*>      (new_data  .data()),
        static_cast<float*>            (new_scale .data()),
        /*full_f32=*/nullptr,
        B, H, T_prev, N, D);
    return true;
}
