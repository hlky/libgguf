#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

constexpr uint64_t NATIVE_DEFAULT_SCRATCH_BYTES = 32ull * 1024ull * 1024ull;
constexpr uint64_t VERIFY_GPU_ALL_TENSORS = std::numeric_limits<uint64_t>::max();

enum converter_backend {
    CONVERTER_BACKEND_AUTO,
    CONVERTER_BACKEND_CPU,
    CONVERTER_BACKEND_GPU,
};

struct cli_options {
    std::string src;
    std::string dst;
    std::string qtype = "Q4_K_S";
    std::string policy = "comfy";
    std::string imatrix;
    std::vector<std::pair<std::string, std::string>> tensor_overrides;
    std::vector<std::string> include;
    std::vector<std::string> exclude;
    uint64_t scratch_bytes = NATIVE_DEFAULT_SCRATCH_BYTES;
    unsigned int threads = 0;
    bool timings = false;
    bool overwrite = false;
    converter_backend backend = CONVERTER_BACKEND_AUTO;
    bool gpu_fallback_cpu = false;
    uint64_t verify_gpu_tensors = 0;
    uint64_t verify_gpu_large_tensors = 0;
    uint64_t verify_gpu_random_tensors = 0;
    uint64_t seed = 0;
    uint64_t gpu_vram_bytes = 0;
    bool gpu_pipeline = true;
};

bool is_supported_quant_qtype_name(const std::string & name);
std::string parse_tensor_qtype_name(const std::string & input);
bool file_exists(const std::string & path);
unsigned int parse_positive_thread_count(const char * value);
cli_options parse_args(int argc, char ** argv);
