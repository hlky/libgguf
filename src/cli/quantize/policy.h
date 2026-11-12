#pragma once

#include "cli/quantize/options.h"
#include "conversion/types.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

bool qtype_matches_source(native_source_dtype dtype, ggml_type qtype);
std::string qtype_name(ggml_type qtype);
bool is_quantized_qtype(ggml_type qtype);
std::pair<std::string, std::string> parse_qtype(const std::string & input);
uint32_t file_type_value(const std::string & file_type_name);
model_template detect_arch(const std::vector<tensor_meta> & tensors);
std::vector<tensor_plan> prepare_plans(
    const std::vector<tensor_meta> & tensors,
    const model_template & model,
    const std::string & file_type_name,
    const std::string & base_qtype_name,
    const cli_options & options,
    std::vector<kv_value> & kv,
    std::vector<tensor_info> & infos,
    conversion_result & result
);
