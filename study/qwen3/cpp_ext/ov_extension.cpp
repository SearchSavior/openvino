/* Registration entry point — exposes all custom ops to ov::Core::add_extension(). */
#include <openvino/core/extension.hpp>
#include <openvino/core/op_extension.hpp>

#include "gated_delta_rule.hpp"
#include "gated_delta_rule_v2.hpp"
#include "gated_delta_rule_v3.hpp"
#include "fused_causal_conv1d.hpp"
#include "quantized_kv_cache.hpp"
#include "quantized_kv_cache_update.hpp"
#include "quantized_int8_sdpa.hpp"

OPENVINO_CREATE_EXTENSIONS(
    std::vector<ov::Extension::Ptr>({
        std::make_shared<ov::OpExtension<Qwen3Ext::GatedDeltaRule>>(),
        std::make_shared<ov::OpExtension<Qwen3Ext::GatedDeltaRuleV2>>(),
        std::make_shared<ov::OpExtension<Qwen3Ext::GatedDeltaRuleV3>>(),
        std::make_shared<ov::OpExtension<Qwen3Ext::FusedCausalConv1d>>(),
        std::make_shared<ov::OpExtension<Qwen3Ext::QuantizedKVCache>>(),
        std::make_shared<ov::OpExtension<Qwen3Ext::QuantizedKVCacheUpdate>>(),
        std::make_shared<ov::OpExtension<Qwen3Ext::QuantizedInt8SDPA>>(),
    }));
