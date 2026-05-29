#include "quantized_int8_sdpa.hpp"
#include "kernels.h"

#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

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

    // cpp_ext is built without libgomp (TBB conflict). Parallelise the SDPA
    // by splitting (b, h_q) across std::threads. Each thread runs the kernel
    // slice for its (bh_start, bh_end) range and writes its own output rows.
    const float       *qp  = static_cast<const float*>      (q      .data());
    const signed char *kdp = static_cast<const signed char*>(k_data .data());
    const float       *ksp = static_cast<const float*>      (k_scale.data());
    const signed char *vdp = static_cast<const signed char*>(v_data .data());
    const float       *vsp = static_cast<const float*>      (v_scale.data());
    float             *op  = static_cast<float*>            (out    .data());

    const int total_bh = B * H_q;
    int n_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (n_threads < 1) n_threads = 1;
    if (n_threads > total_bh) n_threads = total_bh;

    if (n_threads <= 1) {
        int8_sdpa_kernel_slice(qp, kdp, ksp, vdp, vsp, mask_ptr,
                               kq_scale, op,
                               B, H_q, H_kv, T_q, T_full, D,
                               0, total_bh);
    } else {
        std::vector<std::thread> workers;
        workers.reserve(n_threads);
        int chunk = (total_bh + n_threads - 1) / n_threads;
        for (int t = 0; t < n_threads; ++t) {
            int s = t * chunk;
            int e = std::min(s + chunk, total_bh);
            if (s >= e) break;
            workers.emplace_back([=]() {
                int8_sdpa_kernel_slice(qp, kdp, ksp, vdp, vsp, mask_ptr,
                                       kq_scale, op,
                                       B, H_q, H_kv, T_q, T_full, D, s, e);
            });
        }
        for (auto& th : workers) th.join();
    }
    return true;
}
