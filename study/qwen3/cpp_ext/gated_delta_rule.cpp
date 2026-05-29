#include "gated_delta_rule.hpp"
#include "kernels.h"

#include <cmath>
#include <cstring>
#include <vector>

using namespace Qwen3Ext;

GatedDeltaRule::GatedDeltaRule(const ov::OutputVector& args,
                               bool fuse_qk_l2norm,
                               float q_l2_norm_eps,
                               float k_l2_norm_eps)
    : ov::op::internal::GatedDeltaNet(),  // default ctor — no shape validation
      m_l2norm_flag(fuse_qk_l2norm),
      m_q_eps(q_l2_norm_eps),
      m_k_eps(k_l2_norm_eps) {
    // Set inputs after base is constructed; then trigger validate via OUR override
    // (dynamic_type is GatedDeltaRule once we are in the derived ctor body).
    set_arguments(args);
    constructor_validate_and_infer_types();
}

void GatedDeltaRule::validate_and_infer_types() {
    // We accept the original Qwen3-Next IR layout [B, H, T, D] for q/k/v and
    // [B, H, T] for g/beta. Recurrent state is [B, H, D_k, D_v].
    // The parent's strict checks assume [B, T, H, D] (FuseGDNLoop inserts
    // Transposes) and would fail on our shapes, so we don't call the parent.
    set_output_type(0, get_input_element_type(2), get_input_partial_shape(2));
    set_output_type(1, get_input_element_type(3), get_input_partial_shape(3));
}

bool GatedDeltaRule::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("fuse_qk_l2norm", m_l2norm_flag);
    visitor.on_attribute("q_l2_norm_eps", m_q_eps);
    visitor.on_attribute("k_l2_norm_eps", m_k_eps);
    return true;
}

std::shared_ptr<ov::Node> GatedDeltaRule::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    return std::make_shared<GatedDeltaRule>(new_args, m_l2norm_flag, m_q_eps, m_k_eps);
}

bool GatedDeltaRule::has_evaluate() const {
    return true;
}

static void l2_normalize(const float* in, float* out, int n, int d, float eps) {
    for (int i = 0; i < n; ++i) {
        const float* xi = in + (size_t)i * d;
        float* yi = out + (size_t)i * d;
        float sum_sq = 0.0f;
        for (int j = 0; j < d; ++j) sum_sq += xi[j] * xi[j];
        const float scale = 1.0f / std::sqrt(sum_sq + eps);
        for (int j = 0; j < d; ++j) yi[j] = xi[j] * scale;
    }
}

bool GatedDeltaRule::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    // OV GatedDeltaNet input order: (query, key, value, recurrent_state, gate, beta)
    const auto& q_t = inputs[0];
    const auto& k_t = inputs[1];
    const auto& v_t = inputs[2];
    const auto& s_t = inputs[3];
    const auto& g_t = inputs[4];
    const auto& b_t = inputs[5];

    const auto qs = q_t.get_shape();  // [B, H, T, D]
    OPENVINO_ASSERT(qs.size() == 4, "GatedDeltaRule expects rank-4 query");
    const int B = static_cast<int>(qs[0]);
    const int H = static_cast<int>(qs[1]);
    const int T = static_cast<int>(qs[2]);
    const int D = static_cast<int>(qs[3]);

    auto& out = outputs[0];
    auto& final_state = outputs[1];
    out.set_shape(v_t.get_shape());
    final_state.set_shape(s_t.get_shape());

    std::memcpy(final_state.data(), s_t.data(), s_t.get_byte_size());

    const float* q_ptr = static_cast<const float*>(q_t.data());
    const float* k_ptr = static_cast<const float*>(k_t.data());

    std::vector<float> q_norm, k_norm;
    if (m_l2norm_flag) {
        const size_t n = static_cast<size_t>(B) * H * T;
        q_norm.resize(n * D);
        k_norm.resize(n * D);
        l2_normalize(q_ptr, q_norm.data(), static_cast<int>(n), D, m_q_eps);
        l2_normalize(k_ptr, k_norm.data(), static_cast<int>(n), D, m_k_eps);
        q_ptr = q_norm.data();
        k_ptr = k_norm.data();
    }

    gdr_kernel(q_ptr,
               k_ptr,
               static_cast<const float*>(v_t.data()),
               static_cast<const float*>(g_t.data()),
               static_cast<const float*>(b_t.data()),
               static_cast<float*>(final_state.data()),
               static_cast<float*>(out.data()),
               B, H, T, D);
    return true;
}
