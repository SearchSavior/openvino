/* Registration entry point — exposes both ops to ov::Core::add_extension(). */
#include <openvino/core/extension.hpp>
#include <openvino/core/op_extension.hpp>

#include "gated_delta_rule.hpp"
#include "fused_causal_conv1d.hpp"

OPENVINO_CREATE_EXTENSIONS(
    std::vector<ov::Extension::Ptr>({
        std::make_shared<ov::OpExtension<Qwen3Ext::GatedDeltaRule>>(),
        std::make_shared<ov::OpExtension<Qwen3Ext::FusedCausalConv1d>>(),
    }));
