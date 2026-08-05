#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <libgguf/types.h>

enum gguf_value_type : uint32_t {
    GGUF_TYPE_UINT8 = 0,
    GGUF_TYPE_INT8 = 1,
    GGUF_TYPE_UINT16 = 2,
    GGUF_TYPE_INT16 = 3,
    GGUF_TYPE_UINT32 = 4,
    GGUF_TYPE_INT32 = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL = 7,
    GGUF_TYPE_STRING = 8,
    GGUF_TYPE_ARRAY = 9,
    GGUF_TYPE_UINT64 = 10,
    GGUF_TYPE_INT64 = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

enum native_source_dtype {
    NATIVE_DTYPE_F32,
    NATIVE_DTYPE_F16,
    NATIVE_DTYPE_BF16,
};

struct tensor_meta {
    std::string source_key;
    std::string key;
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t data_begin = 0;
    uint64_t data_end = 0;
};

struct tensor_plan {
    std::string key;
    native_source_dtype source_dtype = NATIVE_DTYPE_F32;
    ggml_type qtype = GGML_TYPE_F32;
    std::vector<int64_t> source_shape;
    std::vector<int64_t> write_shape;
    uint64_t data_begin = 0;
    uint64_t data_end = 0;
    uint64_t n_values = 0;
    uint64_t n_rows = 0;
    uint64_t n_per_row = 0;
    uint64_t expected_nbytes = 0;
    std::vector<float> auto_imatrix;
};

struct model_template {
    std::string arch;
    bool shape_fix = false;
    std::vector<std::vector<std::string>> keys_detect;
    std::vector<std::string> keys_banned;
    std::vector<std::string> keys_hiprec;
    std::vector<std::string> keys_ignore;
};

struct tensor_info {
    std::string key;
    std::vector<int64_t> gguf_shape;
    ggml_type qtype = GGML_TYPE_F32;
    uint64_t nbytes = 0;
};

struct kv_value {
    std::string key;
    gguf_value_type type = GGUF_TYPE_UINT32;
    uint32_t u32 = 0;
    std::string str;
    std::vector<int64_t> arr_i64;
};

struct conversion_result {
    std::string output_path;
    std::string arch;
    std::string file_type_name;
    std::map<std::string, int> tensor_type_counts;
    std::map<std::string, int> fallback_counts;
};
