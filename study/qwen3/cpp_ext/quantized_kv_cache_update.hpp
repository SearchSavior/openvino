/*
 * QuantizedKVCacheUpdate - 2-output variant of QuantizedKVCache that skips
 * the f32 dequant. Used by the int8 SDPA path where the dequant is fused
 * into the attention kernel.
 *
 * Inputs:
 *   0: prev_data    [B, H, T_prev, D]   i8
 *   1: prev_scale   [B, H, T_prev]      f32
 *   2: new_kv       [B, H, N,     D]    f32
 *
 * Outputs:
 *   0: new_data     [B, H, T_full, D]   i8
 *   1: new_scale    [B, H, T_full]      f32
 *
 *   T_full = T_prev + N
 */
#pragma once

#include <openvino/op/op.hpp>

namespace Qwen3Ext {

class QuantizedKVCacheUpdate : public ov::op::Op {
public:
    OPENVINO_OP("QuantizedKVCacheUpdate");

    QuantizedKVCacheUpdate() = default;
    QuantizedKVCacheUpdate(const ov::OutputVector& args);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;
};

}  // namespace Qwen3Ext
