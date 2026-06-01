/*
 * QuantizedKVCache custom op — C++ wrapper around qkv_kernel().
 *
 * Per-token symmetric int8 quant of the new tokens, concatenated with the
 * existing i8 state, and dequantised in full for SDPA consumption.
 *
 * Inputs:
 *   0: prev_data    [B, H, T_prev, D]   i8     existing quantised state
 *   1: prev_scale   [B, H, T_prev]      f32    existing per-token scales
 *   2: new_kv       [B, H, N,     D]    f32    new K (or V) for this chunk
 *
 * Outputs:
 *   0: full_f32     [B, H, T_full, D]   f32    dequantised KV for SDPA
 *   1: new_data     [B, H, T_full, D]   i8     concatenated quantised state
 *   2: new_scale    [B, H, T_full]      f32    concatenated scales
 *
 *   T_full = T_prev + N
 */
#pragma once

#include <openvino/op/op.hpp>

namespace Qwen3Ext {

class QuantizedKVCache : public ov::op::Op {
public:
    OPENVINO_OP("QuantizedKVCache");

    QuantizedKVCache() = default;
    QuantizedKVCache(const ov::OutputVector& args);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;
};

}  // namespace Qwen3Ext
