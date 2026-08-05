#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <libgguf/types.h>

constexpr uint64_t SAFETENSORS_PAYLOAD_DEFAULT_SCRATCH_BYTES = 32ull * 1024ull * 1024ull;

enum payload_source_dtype {
    PAYLOAD_DTYPE_F32,
    PAYLOAD_DTYPE_F16,
    PAYLOAD_DTYPE_BF16,
};

struct safetensors_payload_plan {
    std::string key;
    payload_source_dtype source_dtype = PAYLOAD_DTYPE_F32;
    ggml_type qtype = GGML_TYPE_F32;
    uint64_t data_begin = 0;
    uint64_t data_end = 0;
    uint64_t n_values = 0;
    uint64_t n_rows = 0;
    uint64_t n_per_row = 0;
    uint64_t expected_nbytes = 0;
    const float * imatrix = nullptr;
    std::vector<float> auto_imatrix;
};

bool parse_payload_source_dtype(const char * name, payload_source_dtype * output);
size_t payload_source_dtype_size(payload_source_dtype dtype);

bool write_safetensors_payload(
    const std::string & path,
    int output_fd,
    std::vector<safetensors_payload_plan> & plans,
    uint64_t alignment,
    uint64_t scratch_bytes,
    uint64_t * data_bytes_written,
    std::string * error
);
