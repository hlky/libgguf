#include "cli/quantize/options.h"

#include "conversion/common.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <stdexcept>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/stat.h>
#endif

bool file_exists(const std::string & path) {
#if defined(_WIN32)
    int wide_len = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    if (wide_len <= 0) {
        return false;
    }
    std::vector<wchar_t> wide_path((size_t)wide_len);
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wide_path.data(), wide_len);
    DWORD attrs = GetFileAttributesW(wide_path.data());
    return attrs != INVALID_FILE_ATTRIBUTES && !(attrs & FILE_ATTRIBUTE_DIRECTORY);
#else
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
#endif
}

unsigned int parse_positive_thread_count(const char * value) {
    if (value && value[0]) {
        errno = 0;
        char * end = nullptr;
        const long long parsed = std::strtoll(value, &end, 10);
        if (errno == 0 && end != value && *end == '\0' && parsed > 0) {
            return (unsigned int)std::min<long long>(parsed, (long long)std::numeric_limits<unsigned int>::max());
        }
    }
    return 0;
}

namespace {

bool is_supported_storage_qtype_name(const std::string & name) {
    return is_supported_quant_qtype_name(name) || name == "F32" || name == "F16" || name == "BF16";
}

void parse_tensor_type_arg(
    const std::string & value,
    std::vector<std::pair<std::string, std::string>> & out
) {
    const size_t eq = value.find('=');
    if (eq == std::string::npos || eq == 0 || eq + 1 == value.size()) {
        fail("expected PATTERN=QTYPE for --tensor-type");
    }
    out.push_back({value.substr(0, eq), parse_tensor_qtype_name(value.substr(eq + 1))});
}

void print_help() {
    std::puts("Usage: libgguf_quantize_gguf --src PATH --qtype QTYPE [options]");
    std::puts("");
    std::puts("Native safetensors-only Q/K quantization of diffusion models to GGUF.");
    std::puts("");
    std::puts("Options:");
    std::puts("  --src PATH                 Source .safetensors model");
    std::puts("  --qtype QTYPE              Output file type, e.g. Q4_K_S, Q4_K_M, Q4_K, Q8_0");
    std::puts("  --dst PATH                 Output GGUF path");
    std::puts("  --overwrite                Overwrite an existing output file");
    std::puts("  --policy comfy|dynamic|uniform");
    std::puts("                           Tensor selection policy (default: comfy)");
    std::puts("  --imatrix PATH             Accepted for CLI parity; Q/K quantizers do not require it");
    std::puts("  --tensor-type PATTERN=TYPE Override matching tensor storage/quant type");
    std::puts("  --include PATTERN          Force matching 2D tensors into quantization when possible");
    std::puts("  --exclude PATTERN          Keep matching tensors unquantized");
    std::puts("  --scratch-bytes N          Native scratch and direct-copy buffer target in bytes");
    std::puts("  --cpu-ram-bytes N          Alias for --scratch-bytes");
    std::puts("  --threads N                Worker thread count (default: hardware or LIBGGUF_NUM_THREADS)");
    std::puts("  --backend cpu|gpu|auto    Conversion backend for quantized tensors (default: auto)");
    std::puts("  --gpu-fallback cpu        Use CPU for tensors unsupported by the GPU converter");
    std::puts("  --verify-gpu-tensors N|all");
    std::puts("                           Compare GPU tensors against CPU bytes");
    std::puts("  --verify-gpu-large-tensors N");
    std::puts("                           Compare the N largest GPU tensors against CPU bytes");
    std::puts("  --verify-gpu-random-tensors N");
    std::puts("                           Compare N seeded random GPU tensors against CPU bytes");
    std::puts("  --seed N                  Seed for random GPU tensor verification (default: 0)");
    std::puts("  --gpu-vram-bytes N        GPU device chunk budget in bytes (0 uses --scratch-bytes)");
    std::puts("  --gpu-batch-mb N          GPU device chunk budget in MiB (alias for --gpu-vram-bytes)");
    std::puts("  --gpu-pipeline 0|1        Use two-slot GPU host pipeline for unverified tensors (default: 1)");
    std::puts("  --timings                  Print conversion timing breakdown to stderr");
    std::puts("  --help                     Show this help");
}

} // namespace

bool is_supported_quant_qtype_name(const std::string & name) {
    static const std::set<std::string> supported = {
        "Q1_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"};
    return supported.count(name) != 0;
}

std::string parse_tensor_qtype_name(const std::string & input) {
    std::string name = upper_ascii(input);
    if (name == "Q8_K") {
        fail("Q8_K tensor output is not supported");
    }
    if (name == "Q3_K" || name == "Q3_K_S" || name == "Q3_K_M" || name == "Q3_K_L") name = "Q3_K";
    else if (name == "Q4_K" || name == "Q4_K_S" || name == "Q4_K_M") name = "Q4_K";
    else if (name == "Q5_K" || name == "Q5_K_S" || name == "Q5_K_M") name = "Q5_K";
    else if (name == "Q2_K_S") name = "Q2_K";
    if (!is_supported_storage_qtype_name(name)) {
        fail("unsupported GGML tensor type: " + input);
    }
    return name;
}

cli_options parse_args(int argc, char ** argv) {
    cli_options options;
    auto parse_positive_bytes = [](const std::string & value, const char * name) -> uint64_t {
        char * end = nullptr;
        errno = 0;
        unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
        if (errno != 0 || end == value.c_str() || *end != '\0' || parsed == 0) {
            fail(std::string(name) + " must be positive");
        }
        return (uint64_t)parsed;
    };
    auto parse_non_negative_integer = [](const std::string & value, const char * name) -> uint64_t {
        char * end = nullptr;
        errno = 0;
        unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
        if (errno != 0 || end == value.c_str() || *end != '\0' || value[0] == '-') {
            fail(std::string(name) + " must be a non-negative integer");
        }
        return (uint64_t)parsed;
    };
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need_value = [&](const char * name) -> std::string {
            if (i + 1 >= argc) {
                fail(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (arg == "--help" || arg == "-h") {
            print_help();
            std::exit(0);
        } else if (arg == "--src") {
            options.src = need_value("--src");
        } else if (arg == "--qtype") {
            options.qtype = need_value("--qtype");
        } else if (arg == "--dst") {
            options.dst = need_value("--dst");
        } else if (arg == "--overwrite") {
            options.overwrite = true;
        } else if (arg == "--policy") {
            options.policy = need_value("--policy");
            if (options.policy != "comfy" && options.policy != "dynamic" && options.policy != "uniform") {
                fail("--policy must be 'comfy', 'dynamic', or 'uniform'");
            }
        } else if (arg == "--imatrix") {
            options.imatrix = need_value("--imatrix");
        } else if (arg == "--tensor-type") {
            parse_tensor_type_arg(need_value("--tensor-type"), options.tensor_overrides);
        } else if (arg == "--include") {
            options.include.push_back(need_value("--include"));
        } else if (arg == "--exclude") {
            options.exclude.push_back(need_value("--exclude"));
        } else if (arg == "--scratch-bytes") {
            options.scratch_bytes = parse_positive_bytes(need_value("--scratch-bytes"), "--scratch-bytes");
        } else if (arg == "--cpu-ram-bytes") {
            options.scratch_bytes = parse_positive_bytes(need_value("--cpu-ram-bytes"), "--cpu-ram-bytes");
        } else if (arg == "--threads") {
            std::string value = need_value("--threads");
            const unsigned int parsed = parse_positive_thread_count(value.c_str());
            if (parsed == 0) {
                fail("--threads must be a positive integer");
            }
            options.threads = parsed;
        } else if (arg == "--backend") {
            std::string value = need_value("--backend");
            if (value == "auto") {
                options.backend = CONVERTER_BACKEND_AUTO;
            } else if (value == "cpu") {
                options.backend = CONVERTER_BACKEND_CPU;
            } else if (value == "gpu") {
                options.backend = CONVERTER_BACKEND_GPU;
            } else {
                fail("--backend must be 'cpu', 'gpu', or 'auto'");
            }
        } else if (arg == "--gpu-fallback") {
            std::string value = need_value("--gpu-fallback");
            if (value != "cpu") {
                fail("--gpu-fallback must be 'cpu'");
            }
            options.gpu_fallback_cpu = true;
        } else if (arg == "--verify-gpu-tensors") {
            std::string value = need_value("--verify-gpu-tensors");
            if (value == "all") {
                options.verify_gpu_tensors = VERIFY_GPU_ALL_TENSORS;
            } else {
                try {
                    options.verify_gpu_tensors = parse_non_negative_integer(value, "--verify-gpu-tensors");
                } catch (...) {
                    fail("--verify-gpu-tensors must be a non-negative integer or 'all'");
                }
            }
        } else if (arg == "--verify-gpu-large-tensors") {
            options.verify_gpu_large_tensors =
                parse_non_negative_integer(need_value("--verify-gpu-large-tensors"), "--verify-gpu-large-tensors");
        } else if (arg == "--verify-gpu-random-tensors") {
            options.verify_gpu_random_tensors =
                parse_non_negative_integer(need_value("--verify-gpu-random-tensors"), "--verify-gpu-random-tensors");
        } else if (arg == "--seed") {
            options.seed = parse_non_negative_integer(need_value("--seed"), "--seed");
        } else if (arg == "--gpu-vram-bytes") {
            options.gpu_vram_bytes = parse_non_negative_integer(need_value("--gpu-vram-bytes"), "--gpu-vram-bytes");
        } else if (arg == "--gpu-batch-mb") {
            const uint64_t mib = parse_non_negative_integer(need_value("--gpu-batch-mb"), "--gpu-batch-mb");
            if (mib > std::numeric_limits<uint64_t>::max() / (1024ULL * 1024ULL)) {
                fail("--gpu-batch-mb is too large");
            }
            options.gpu_vram_bytes = mib * 1024ULL * 1024ULL;
        } else if (arg == "--gpu-pipeline") {
            const std::string value = need_value("--gpu-pipeline");
            if (value == "0") {
                options.gpu_pipeline = false;
            } else if (value == "1") {
                options.gpu_pipeline = true;
            } else {
                fail("--gpu-pipeline must be 0 or 1");
            }
        } else if (arg == "--timings") {
            options.timings = true;
        } else {
            fail("unknown argument: " + arg);
        }
    }
    if (options.src.empty()) {
        fail("--src is required");
    }
    if (options.qtype.empty()) {
        fail("--qtype is required");
    }
    if (!file_exists(options.src)) {
        fail("invalid source file: " + options.src);
    }
    if (!ends_with(upper_ascii(options.src), ".SAFETENSORS")) {
        fail("native quantize_gguf only supports .safetensors inputs");
    }
    if (options.verify_gpu_tensors > 0 && options.backend == CONVERTER_BACKEND_CPU) {
        fail("--verify-gpu-tensors requires --backend gpu or auto");
    }
    if (options.verify_gpu_large_tensors > 0 && options.backend == CONVERTER_BACKEND_CPU) {
        fail("--verify-gpu-large-tensors requires --backend gpu or auto");
    }
    if (options.verify_gpu_random_tensors > 0 && options.backend == CONVERTER_BACKEND_CPU) {
        fail("--verify-gpu-random-tensors requires --backend gpu or auto");
    }
    if (options.gpu_vram_bytes > 0 && options.backend == CONVERTER_BACKEND_CPU) {
        fail("--gpu-vram-bytes requires --backend gpu or auto");
    }
    return options;
}
