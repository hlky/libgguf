#pragma once

#include "conversion/types.h"

#include <cstdint>
#include <cstdio>
#include <vector>

constexpr uint64_t GGUF_DEFAULT_ALIGNMENT = 32;
constexpr uint32_t GGML_QUANT_VERSION = 2;

void write_all_file(FILE * file, const void * data, uint64_t length);
void write_pre_tensor_padding(FILE * file, uint64_t alignment);
void write_post_tensor_padding(FILE * file, uint64_t tensor_nbytes, uint64_t alignment);
void write_gguf_metadata(
    FILE * output,
    const std::vector<kv_value> & kv,
    const std::vector<tensor_info> & infos
);
