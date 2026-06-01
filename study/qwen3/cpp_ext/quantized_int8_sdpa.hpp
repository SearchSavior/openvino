/*
 * QuantizedInt8SDPA custom op — fused scaled dot-product attention that reads
 * i8 K and V directly. No f32 dequant buffer for K or V is ever materialised.
 *
 * Replaces the canonical pattern:
 *      QuantizedKVCache (K) -> Unsqueeze -> Broadcast -> Reshape -> SDPA
 *      QuantizedKVCache (V) -> Unsqueeze -> Broadcast -> Reshape -> SDPA
 * with a single op that owns the attention loop and dequants per row inside
 * the score / weighted-sum kernels.
 *
 * Inputs:
 *   0: q          [B, H_q,  T_q,    D]    fp32  (post-RoPE Q, not GQA-broadcast)
 *   1: k_data     [B, H_kv, T_full, D]    i8
 *   2: k_scale    [B, H_kv, T_full]       fp32
 *   3: v_data     [B, H_kv, T_full, D]    i8
 *   4: v_scale    [B, H_kv, T_full]       fp32
 *   5: mask       [B, 1, T_q, T_full]     fp32  (additive mask; -inf for masked)
 *   6: scale_attr scalar f32              (1 / sqrt(D))
 *
 * Output:
 *   0: out        [B, H_q, T_q, D]        fp32
 */
#pragma once

#include <openvino/op/op.hpp>

namespace Qwen3Ext {

class QuantizedInt8SDPA : public ov::op::Op {
public:
    OPENVINO_OP("QuantizedInt8SDPA");

    QuantizedInt8SDPA() = default;
    QuantizedInt8SDPA(const ov::OutputVector& args);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;
};

}  // namespace Qwen3Ext
