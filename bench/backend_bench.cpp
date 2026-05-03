#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <cstdlib>
#endif

#include "libgguf.h"

extern "C" const char *libgguf_common_quant_backend(void);
extern "C" int libgguf_common_quant_cpu_supports_backend(const char *backend);
extern "C" int libgguf_common_quant_set_backend(const char *backend);
extern "C" const char *libgguf_storage_backend(void);
extern "C" int libgguf_storage_cpu_supports_backend(const char *backend);
extern "C" int libgguf_storage_set_backend(const char *backend);
extern "C" const char *libgguf_dequant_backend(int type);
extern "C" int libgguf_dequant_cpu_supports_backend(const char *backend);
extern "C" size_t libgguf_dequantize_for_backend(
    int type,
    const char *backend,
    const void *src,
    float *dst,
    int64_t nrows,
    int64_t n_per_row);

extern "C" int libgguf_q1_0_cpu_supports_backend(const char *backend);
extern "C" size_t libgguf_quantize_q1_0_for_backend(
    const char *backend,
    const float *src,
    void *dst,
    int64_t nrows,
    int64_t n_per_row);
extern "C" int libgguf_q4_0_cpu_supports_backend(const char *backend);
extern "C" size_t libgguf_quantize_q4_0_for_backend(
    const char *backend,
    const float *src,
    void *dst,
    int64_t nrows,
    int64_t n_per_row);
extern "C" int libgguf_q4_1_cpu_supports_backend(const char *backend);
extern "C" size_t libgguf_quantize_q4_1_for_backend(
    const char *backend,
    const float *src,
    void *dst,
    int64_t nrows,
    int64_t n_per_row);
extern "C" int libgguf_q5_0_cpu_supports_backend(const char *backend);
extern "C" size_t libgguf_quantize_q5_0_for_backend(
    const char *backend,
    const float *src,
    void *dst,
    int64_t nrows,
    int64_t n_per_row);
extern "C" int libgguf_q5_1_cpu_supports_backend(const char *backend);
extern "C" size_t libgguf_quantize_q5_1_for_backend(
    const char *backend,
    const float *src,
    void *dst,
    int64_t nrows,
    int64_t n_per_row);
extern "C" int libgguf_q8_0_cpu_supports_backend(const char *backend);
extern "C" size_t libgguf_quantize_q8_0_for_backend(
    const char *backend,
    const float *src,
    void *dst,
    int64_t nrows,
    int64_t n_per_row);
extern "C" int libgguf_q2_k_cpu_supports_backend(const char *backend);
extern "C" size_t libgguf_quantize_q2_k_for_backend(
    const char *backend,
    const float *src,
    void *dst,
    int64_t nrows,
    int64_t n_per_row);
extern "C" int libgguf_q3_k_cpu_supports_backend(const char *backend);
extern "C" size_t libgguf_quantize_q3_k_for_backend(
    const char *backend,
    const float *src,
    void *dst,
    int64_t nrows,
    int64_t n_per_row);
extern "C" int libgguf_q4_k_cpu_supports_backend(const char *backend);
extern "C" size_t libgguf_quantize_q4_k_for_backend(
    const char *backend,
    const float *src,
    void *dst,
    int64_t nrows,
    int64_t n_per_row);
extern "C" int libgguf_q5_k_cpu_supports_backend(const char *backend);
extern "C" size_t libgguf_quantize_q5_k_for_backend(
    const char *backend,
    const float *src,
    void *dst,
    int64_t nrows,
    int64_t n_per_row);
extern "C" int libgguf_q6_k_cpu_supports_backend(const char *backend);
extern "C" size_t libgguf_quantize_q6_k_for_backend(
    const char *backend,
    const float *src,
    void *dst,
    int64_t nrows,
    int64_t n_per_row);
extern "C" int libgguf_iq4_nl_cpu_supports_backend(const char *backend);
extern "C" size_t libgguf_quantize_iq4_nl_for_backend(
    const char *backend,
    const float *src,
    void *dst,
    int64_t nrows,
    int64_t n_per_row);
extern "C" int libgguf_iq4_xs_cpu_supports_backend(const char *backend);
extern "C" size_t libgguf_quantize_iq4_xs_for_backend(
    const char *backend,
    const float *src,
    void *dst,
    int64_t nrows,
    int64_t n_per_row);
extern "C" int libgguf_tq1_0_cpu_supports_backend(const char *backend);
extern "C" size_t libgguf_quantize_tq1_0_for_backend(
    const char *backend,
    const float *src,
    void *dst,
    int64_t nrows,
    int64_t n_per_row);
extern "C" int libgguf_tq2_0_cpu_supports_backend(const char *backend);
extern "C" size_t libgguf_quantize_tq2_0_for_backend(
    const char *backend,
    const float *src,
    void *dst,
    int64_t nrows,
    int64_t n_per_row);
extern "C" int libgguf_mxfp4_cpu_supports_backend(const char *backend);
extern "C" size_t libgguf_quantize_mxfp4_for_backend(
    const char *backend,
    const float *src,
    void *dst,
    int64_t nrows,
    int64_t n_per_row);
extern "C" int libgguf_nvfp4_cpu_supports_backend(const char *backend);
extern "C" size_t libgguf_quantize_nvfp4_for_backend(
    const char *backend,
    const float *src,
    void *dst,
    int64_t nrows,
    int64_t n_per_row);

namespace
{

using Clock = std::chrono::steady_clock;
using QuantizeForBackendFn = size_t (*)(const char *, const float *, void *, int64_t, int64_t);
using SupportsBackendFn = int (*)(const char *);

enum class TypeKind
{
  storage,
  quantized,
};

struct DirectQuantFns
{
  SupportsBackendFn supports = nullptr;
  QuantizeForBackendFn quantize = nullptr;
};

struct TypeCase
{
  ggml_type type;
  const char *name;
  TypeKind kind;
  DirectQuantFns direct;
};

struct ThreadSpec
{
  std::string mode;
  int threads;
};

struct Options
{
  std::vector<int64_t> sizes = {256, 1024, 4096, 8192};
  std::vector<int> thread_tokens = {1, 4, 0};
  int64_t rows = 1024;
  std::vector<std::string> backends = {"ref", "sse2", "sse4_1", "avx2"};
  std::vector<std::string> ops = {"quantize", "dequantize"};
  std::vector<std::string> types;
  int samples = 5;
  double min_ms = 10.0;
  std::string csv_path;
  std::string json_path;
  bool progress = true;
};

struct BenchStats
{
  int samples = 0;
  double median_ns = 0.0;
  double best_ns = 0.0;
};

struct BenchResult
{
  std::string op;
  std::string type;
  int64_t row_width = 0;
  int64_t n_rows = 0;
  std::string backend;
  std::string thread_mode;
  int threads = 0;
  bool supported = false;
  std::string reason;
  int samples = 0;
  double median_ns = 0.0;
  double best_ns = 0.0;
  double rows_per_s = 0.0;
  double elements_per_s = 0.0;
  double input_mb_per_s = 0.0;
  double output_mb_per_s = 0.0;
  size_t encoded_bytes_per_row = 0;
};

volatile std::uint64_t g_sink = 0;

const std::vector<TypeCase> &all_types()
{
  static const std::vector<TypeCase> types = {
      {GGML_TYPE_F32, "F32", TypeKind::storage, {}},
      {GGML_TYPE_F16, "F16", TypeKind::storage, {}},
      {GGML_TYPE_BF16, "BF16", TypeKind::storage, {}},
      {GGML_TYPE_Q1_0, "Q1_0", TypeKind::quantized, {libgguf_q1_0_cpu_supports_backend, libgguf_quantize_q1_0_for_backend}},
      {GGML_TYPE_Q4_0, "Q4_0", TypeKind::quantized, {libgguf_q4_0_cpu_supports_backend, libgguf_quantize_q4_0_for_backend}},
      {GGML_TYPE_Q4_1, "Q4_1", TypeKind::quantized, {libgguf_q4_1_cpu_supports_backend, libgguf_quantize_q4_1_for_backend}},
      {GGML_TYPE_Q5_0, "Q5_0", TypeKind::quantized, {libgguf_q5_0_cpu_supports_backend, libgguf_quantize_q5_0_for_backend}},
      {GGML_TYPE_Q5_1, "Q5_1", TypeKind::quantized, {libgguf_q5_1_cpu_supports_backend, libgguf_quantize_q5_1_for_backend}},
      {GGML_TYPE_Q8_0, "Q8_0", TypeKind::quantized, {libgguf_q8_0_cpu_supports_backend, libgguf_quantize_q8_0_for_backend}},
      {GGML_TYPE_Q2_K, "Q2_K", TypeKind::quantized, {libgguf_q2_k_cpu_supports_backend, libgguf_quantize_q2_k_for_backend}},
      {GGML_TYPE_Q3_K, "Q3_K", TypeKind::quantized, {libgguf_q3_k_cpu_supports_backend, libgguf_quantize_q3_k_for_backend}},
      {GGML_TYPE_Q4_K, "Q4_K", TypeKind::quantized, {libgguf_q4_k_cpu_supports_backend, libgguf_quantize_q4_k_for_backend}},
      {GGML_TYPE_Q5_K, "Q5_K", TypeKind::quantized, {libgguf_q5_k_cpu_supports_backend, libgguf_quantize_q5_k_for_backend}},
      {GGML_TYPE_Q6_K, "Q6_K", TypeKind::quantized, {libgguf_q6_k_cpu_supports_backend, libgguf_quantize_q6_k_for_backend}},
      {GGML_TYPE_IQ2_XXS, "IQ2_XXS", TypeKind::quantized, {}},
      {GGML_TYPE_IQ2_XS, "IQ2_XS", TypeKind::quantized, {}},
      {GGML_TYPE_IQ2_S, "IQ2_S", TypeKind::quantized, {}},
      {GGML_TYPE_IQ3_XXS, "IQ3_XXS", TypeKind::quantized, {}},
      {GGML_TYPE_IQ3_S, "IQ3_S", TypeKind::quantized, {}},
      {GGML_TYPE_IQ1_S, "IQ1_S", TypeKind::quantized, {}},
      {GGML_TYPE_IQ1_M, "IQ1_M", TypeKind::quantized, {}},
      {GGML_TYPE_IQ4_NL, "IQ4_NL", TypeKind::quantized, {libgguf_iq4_nl_cpu_supports_backend, libgguf_quantize_iq4_nl_for_backend}},
      {GGML_TYPE_IQ4_XS, "IQ4_XS", TypeKind::quantized, {libgguf_iq4_xs_cpu_supports_backend, libgguf_quantize_iq4_xs_for_backend}},
      {GGML_TYPE_TQ1_0, "TQ1_0", TypeKind::quantized, {libgguf_tq1_0_cpu_supports_backend, libgguf_quantize_tq1_0_for_backend}},
      {GGML_TYPE_TQ2_0, "TQ2_0", TypeKind::quantized, {libgguf_tq2_0_cpu_supports_backend, libgguf_quantize_tq2_0_for_backend}},
      {GGML_TYPE_MXFP4, "MXFP4", TypeKind::quantized, {libgguf_mxfp4_cpu_supports_backend, libgguf_quantize_mxfp4_for_backend}},
      {GGML_TYPE_NVFP4, "NVFP4", TypeKind::quantized, {libgguf_nvfp4_cpu_supports_backend, libgguf_quantize_nvfp4_for_backend}},
  };
  return types;
}

std::string upper_ascii(std::string value)
{
  for (char &ch : value)
  {
    if (ch >= 'a' && ch <= 'z')
    {
      ch = char(ch - 'a' + 'A');
    }
  }
  return value;
}

std::string lower_ascii(std::string value)
{
  for (char &ch : value)
  {
    if (ch >= 'A' && ch <= 'Z')
    {
      ch = char(ch - 'A' + 'a');
    }
  }
  return value;
}

std::vector<std::string> split_csv(const std::string &value)
{
  std::vector<std::string> out;
  std::stringstream ss(value);
  std::string item;
  while (std::getline(ss, item, ','))
  {
    if (!item.empty())
    {
      out.push_back(item);
    }
  }
  return out;
}

int64_t parse_i64(const std::string &value, const char *name)
{
  size_t consumed = 0;
  long long parsed = 0;
  try
  {
    parsed = std::stoll(value, &consumed, 10);
  }
  catch (const std::exception &)
  {
    throw std::runtime_error(std::string("invalid ") + name + ": " + value);
  }
  if (consumed != value.size() || parsed <= 0)
  {
    throw std::runtime_error(std::string("invalid ") + name + ": " + value);
  }
  return (int64_t)parsed;
}

int parse_int(const std::string &value, const char *name)
{
  const int64_t parsed = parse_i64(value, name);
  if (parsed > std::numeric_limits<int>::max())
  {
    throw std::runtime_error(std::string("invalid ") + name + ": " + value);
  }
  return (int)parsed;
}

double parse_double(const std::string &value, const char *name)
{
  size_t consumed = 0;
  double parsed = 0.0;
  try
  {
    parsed = std::stod(value, &consumed);
  }
  catch (const std::exception &)
  {
    throw std::runtime_error(std::string("invalid ") + name + ": " + value);
  }
  if (consumed != value.size() || parsed <= 0.0)
  {
    throw std::runtime_error(std::string("invalid ") + name + ": " + value);
  }
  return parsed;
}

std::string require_arg(int argc, char **argv, int *index, const std::string &name)
{
  if (*index + 1 >= argc)
  {
    throw std::runtime_error("missing value for " + name);
  }
  ++(*index);
  return argv[*index];
}

bool parse_option_value(const std::string &arg, const std::string &name, std::string *value)
{
  const std::string prefix = name + "=";
  if (arg.rfind(prefix, 0) == 0)
  {
    *value = arg.substr(prefix.size());
    return true;
  }
  return false;
}

void print_help(std::ostream &out)
{
  out << "Usage: libgguf_backend_bench [options]\n"
      << "\n"
      << "Options:\n"
      << "  --sizes LIST       Row widths. Default: 256,1024,4096,8192\n"
      << "  --threads LIST     Thread modes. Use positive integers or default. Default: 1,4,default\n"
      << "  --rows N           Rows per benchmark case. Default: 1024\n"
      << "  --backends LIST    Backends. Default: ref,sse2,sse4_1,avx2\n"
      << "  --ops LIST         Operations: quantize,store,dequantize. Default: quantize,dequantize\n"
      << "  --types LIST       Type names. Default: all supported row-operation types\n"
      << "  --samples N        Samples per case. Default: 5\n"
      << "  --min-ms N         Minimum milliseconds per sample. Default: 10\n"
      << "  --csv PATH         Write CSV output to PATH. Use - for stdout\n"
      << "  --json PATH        Write JSON output to PATH. Use - for stdout\n"
      << "  --no-progress     Disable case-level progress output\n"
      << "  --help             Show this help\n";
}

Options parse_options(int argc, char **argv)
{
  Options options;
  for (int i = 1; i < argc; ++i)
  {
    const std::string arg = argv[i];
    std::string value;
    if (arg == "--help" || arg == "-h")
    {
      print_help(std::cout);
      std::exit(0);
    }
    if (arg == "--sizes" || parse_option_value(arg, "--sizes", &value))
    {
      if (value.empty())
      {
        value = require_arg(argc, argv, &i, "--sizes");
      }
      options.sizes.clear();
      for (const std::string &item : split_csv(value))
      {
        options.sizes.push_back(parse_i64(item, "size"));
      }
    }
    else if (arg == "--threads" || parse_option_value(arg, "--threads", &value))
    {
      if (value.empty())
      {
        value = require_arg(argc, argv, &i, "--threads");
      }
      options.thread_tokens.clear();
      for (std::string item : split_csv(value))
      {
        item = lower_ascii(item);
        if (item == "default")
        {
          options.thread_tokens.push_back(0);
        }
        else
        {
          options.thread_tokens.push_back(parse_int(item, "thread count"));
        }
      }
    }
    else if (arg == "--rows" || parse_option_value(arg, "--rows", &value))
    {
      if (value.empty())
      {
        value = require_arg(argc, argv, &i, "--rows");
      }
      options.rows = parse_i64(value, "rows");
    }
    else if (arg == "--backends" || parse_option_value(arg, "--backends", &value))
    {
      if (value.empty())
      {
        value = require_arg(argc, argv, &i, "--backends");
      }
      options.backends.clear();
      for (std::string item : split_csv(value))
      {
        options.backends.push_back(lower_ascii(item));
      }
    }
    else if (arg == "--ops" || parse_option_value(arg, "--ops", &value))
    {
      if (value.empty())
      {
        value = require_arg(argc, argv, &i, "--ops");
      }
      options.ops.clear();
      for (std::string item : split_csv(value))
      {
        item = lower_ascii(item);
        if (item == "store" || item == "quantize/store")
        {
          item = "quantize";
        }
        if (item != "quantize" && item != "dequantize")
        {
          throw std::runtime_error("invalid op: " + item);
        }
        options.ops.push_back(item);
      }
    }
    else if (arg == "--types" || parse_option_value(arg, "--types", &value))
    {
      if (value.empty())
      {
        value = require_arg(argc, argv, &i, "--types");
      }
      options.types.clear();
      for (std::string item : split_csv(value))
      {
        item = upper_ascii(item);
        if (item.rfind("GGML_TYPE_", 0) == 0)
        {
          item = item.substr(std::strlen("GGML_TYPE_"));
        }
        options.types.push_back(item);
      }
    }
    else if (arg == "--samples" || parse_option_value(arg, "--samples", &value))
    {
      if (value.empty())
      {
        value = require_arg(argc, argv, &i, "--samples");
      }
      options.samples = parse_int(value, "samples");
    }
    else if (arg == "--min-ms" || parse_option_value(arg, "--min-ms", &value))
    {
      if (value.empty())
      {
        value = require_arg(argc, argv, &i, "--min-ms");
      }
      options.min_ms = parse_double(value, "min-ms");
    }
    else if (arg == "--csv" || parse_option_value(arg, "--csv", &value))
    {
      if (value.empty())
      {
        value = require_arg(argc, argv, &i, "--csv");
      }
      options.csv_path = value;
    }
    else if (arg == "--json" || parse_option_value(arg, "--json", &value))
    {
      if (value.empty())
      {
        value = require_arg(argc, argv, &i, "--json");
      }
      options.json_path = value;
    }
    else if (arg == "--no-progress")
    {
      options.progress = false;
    }
    else
    {
      throw std::runtime_error("unknown option: " + arg);
    }
  }

  if (options.sizes.empty())
  {
    throw std::runtime_error("--sizes cannot be empty");
  }
  if (options.thread_tokens.empty())
  {
    throw std::runtime_error("--threads cannot be empty");
  }
  if (options.backends.empty())
  {
    throw std::runtime_error("--backends cannot be empty");
  }
  if (options.ops.empty())
  {
    throw std::runtime_error("--ops cannot be empty");
  }
  return options;
}

bool stdout_is_data_output(const Options &options)
{
  return options.csv_path == "-" ||
         options.json_path == "-" ||
         (options.csv_path.empty() && options.json_path.empty());
}

std::vector<ThreadSpec> resolve_threads(const std::vector<int> &tokens)
{
  unsigned int hardware = std::thread::hardware_concurrency();
  if (hardware == 0)
  {
    hardware = 1;
  }

  std::vector<ThreadSpec> threads;
  for (int token : tokens)
  {
    if (token == 0)
    {
      threads.push_back({"default", (int)hardware});
    }
    else
    {
      threads.push_back({std::to_string(token), token});
    }
  }
  return threads;
}

const TypeCase *find_type(const std::string &name)
{
  for (const TypeCase &type_case : all_types())
  {
    if (upper_ascii(type_case.name) == upper_ascii(name))
    {
      return &type_case;
    }
  }
  return nullptr;
}

std::vector<const TypeCase *> resolve_types(const std::vector<std::string> &names)
{
  std::vector<const TypeCase *> types;
  if (names.empty())
  {
    for (const TypeCase &type_case : all_types())
    {
      types.push_back(&type_case);
    }
    return types;
  }

  for (const std::string &name : names)
  {
    if (lower_ascii(name) == "all")
    {
      for (const TypeCase &type_case : all_types())
      {
        types.push_back(&type_case);
      }
      continue;
    }
    const TypeCase *type_case = find_type(name);
    if (!type_case)
    {
      throw std::runtime_error("unsupported benchmark type: " + name);
    }
    types.push_back(type_case);
  }
  return types;
}

bool set_process_env_threads_one()
{
#ifdef _WIN32
  return _putenv_s("LIBGGUF_NUM_THREADS", "1") == 0;
#else
  return setenv("LIBGGUF_NUM_THREADS", "1", 1) == 0;
#endif
}

std::uint32_t mix32(std::uint32_t x)
{
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

std::vector<float> build_rows(const TypeCase &type_case, int64_t rows, int64_t width)
{
  std::vector<float> data((size_t)rows * (size_t)width);
  const std::uint32_t type_seed = (std::uint32_t)type_case.type * 2654435761U;
  for (int64_t row = 0; row < rows; ++row)
  {
    for (int64_t col = 0; col < width; ++col)
    {
      const std::uint32_t seed = mix32(type_seed ^ (std::uint32_t)(row * 1315423911ULL) ^ (std::uint32_t)col);
      const float random_part = ((float)(seed & 0xffffU) / 32768.0f) - 1.0f;
      const float wave_part = (float)((col % 31) - 15) / 7.0f;
      const float row_part = (float)((row % 17) - 8) * 0.03125f;
      float value = 0.85f * random_part + 0.20f * wave_part + row_part;
      if (((row + col) % 257) == 0)
      {
        value = 0.0f;
      }
      data[(size_t)row * (size_t)width + (size_t)col] = value;
    }
  }
  return data;
}

std::vector<float> build_imatrix(const std::vector<float> &rows, int64_t nrows, int64_t width)
{
  std::vector<float> imatrix((size_t)width, 0.0f);
  for (int64_t row = 0; row < nrows; ++row)
  {
    const float *src = rows.data() + (size_t)row * (size_t)width;
    for (int64_t col = 0; col < width; ++col)
    {
      imatrix[(size_t)col] += src[(size_t)col] * src[(size_t)col];
    }
  }
  for (float &value : imatrix)
  {
    value = std::max(value, 1.0e-6f);
  }
  return imatrix;
}

template <class Fn>
void run_partitioned(int64_t rows, int threads, Fn fn)
{
  if (rows <= 0)
  {
    return;
  }
  const int workers = std::max<int>(1, std::min<int64_t>((int64_t)std::max(threads, 1), rows));
  if (workers == 1)
  {
    fn(0, rows);
    return;
  }

  std::vector<std::thread> pool;
  pool.reserve((size_t)workers);
  for (int worker = 0; worker < workers; ++worker)
  {
    const int64_t row_begin = rows * worker / workers;
    const int64_t row_end = rows * (worker + 1) / workers;
    pool.emplace_back([=]() {
      if (row_end > row_begin)
      {
        fn(row_begin, row_end - row_begin);
      }
    });
  }
  for (std::thread &thread : pool)
  {
    thread.join();
  }
}

void mix_output_bytes(const std::vector<std::uint8_t> &bytes)
{
  if (bytes.empty())
  {
    return;
  }
  const size_t mid = bytes.size() / 2;
  g_sink ^= (std::uint64_t)bytes.front();
  g_sink ^= (std::uint64_t)bytes[mid] << 8;
  g_sink ^= (std::uint64_t)bytes.back() << 16;
}

void mix_output_floats(const std::vector<float> &values)
{
  if (values.empty())
  {
    return;
  }
  std::uint32_t first = 0;
  std::uint32_t mid = 0;
  std::uint32_t last = 0;
  std::memcpy(&first, &values.front(), sizeof(first));
  std::memcpy(&mid, &values[values.size() / 2], sizeof(mid));
  std::memcpy(&last, &values.back(), sizeof(last));
  g_sink ^= first;
  g_sink ^= (std::uint64_t)mid << 16;
  g_sink ^= (std::uint64_t)last << 32;
}

BenchStats measure(int samples, double min_ms, const std::function<void()> &run_once)
{
  std::vector<double> sample_ns;
  sample_ns.reserve((size_t)samples);
  const auto min_duration = std::chrono::duration<double, std::milli>(min_ms);

  for (int sample = 0; sample < samples; ++sample)
  {
    int64_t reps = 0;
    const auto begin = Clock::now();
    auto elapsed = Clock::duration::zero();
    do
    {
      run_once();
      ++reps;
      elapsed = Clock::now() - begin;
    } while (elapsed < min_duration);

    const double elapsed_ns = std::chrono::duration<double, std::nano>(elapsed).count();
    sample_ns.push_back(elapsed_ns / (double)reps);
  }

  std::sort(sample_ns.begin(), sample_ns.end());
  BenchStats stats;
  stats.samples = (int)sample_ns.size();
  stats.best_ns = sample_ns.empty() ? 0.0 : sample_ns.front();
  stats.median_ns = sample_ns.empty() ? 0.0 : sample_ns[sample_ns.size() / 2];
  return stats;
}

bool storage_quantize_supported(const TypeCase &type_case, const std::string &backend, std::string *reason)
{
  if (type_case.type == GGML_TYPE_F32 || type_case.type == GGML_TYPE_F16)
  {
    if (backend == "ref")
    {
      return true;
    }
    *reason = std::string(type_case.name) + " storage conversion has no backend-specific kernel";
    return false;
  }
  if (type_case.type == GGML_TYPE_BF16)
  {
    if (libgguf_storage_cpu_supports_backend(backend.c_str()))
    {
      return true;
    }
    *reason = "storage backend is not supported by this CPU";
    return false;
  }
  *reason = "unsupported storage type";
  return false;
}

bool quantize_supported(const TypeCase &type_case, const std::string &backend, std::string *reason)
{
  if (type_case.kind == TypeKind::storage)
  {
    return storage_quantize_supported(type_case, backend, reason);
  }
  if (type_case.direct.supports)
  {
    if (type_case.direct.supports(backend.c_str()))
    {
      return true;
    }
    *reason = "direct quantize backend is not supported by this CPU";
    return false;
  }
  if (libgguf_common_quant_cpu_supports_backend(backend.c_str()))
  {
    return true;
  }
  *reason = "common quant backend is not supported by this CPU";
  return false;
}

bool dequantize_supported(const TypeCase &type_case, const std::string &backend, std::string *reason)
{
  if (type_case.kind != TypeKind::quantized)
  {
    *reason = "dequantize applies only to quantized row types";
    return false;
  }
  if (libgguf_dequant_cpu_supports_backend(backend.c_str()))
  {
    return true;
  }
  *reason = "dequant backend is not supported by this CPU";
  return false;
}

std::string quantize_backend_label(const TypeCase &type_case, const std::string &backend)
{
  if (type_case.kind == TypeKind::quantized && !type_case.direct.quantize)
  {
    return "common:" + backend;
  }
  return backend;
}

std::vector<std::string> backends_for_case(
    const std::string &op,
    const TypeCase &type_case,
    const std::vector<std::string> &requested_backends)
{
  if (op == "quantize")
  {
    if (type_case.kind == TypeKind::storage && type_case.type != GGML_TYPE_BF16)
    {
      const auto found = std::find(requested_backends.begin(), requested_backends.end(), "ref");
      if (found != requested_backends.end())
      {
        return {"ref"};
      }
      return requested_backends.empty() ? std::vector<std::string>() : std::vector<std::string>{requested_backends.front()};
    }

    if (type_case.kind == TypeKind::quantized && !type_case.direct.quantize)
    {
      const auto found = std::find(requested_backends.begin(), requested_backends.end(), "ref");
      if (found != requested_backends.end())
      {
        return {"ref"};
      }
      return requested_backends.empty() ? std::vector<std::string>() : std::vector<std::string>{requested_backends.front()};
    }
  }

  return requested_backends;
}

size_t count_cases(
    const Options &options,
    const std::vector<const TypeCase *> &type_cases,
    const std::vector<ThreadSpec> &thread_specs)
{
  size_t total = 0;
  for (const std::string &op : options.ops)
  {
    for (const TypeCase *type_case : type_cases)
    {
      const size_t backend_count = backends_for_case(op, *type_case, options.backends).size();
      total += options.sizes.size() * backend_count * thread_specs.size();
    }
  }
  return total;
}

void prepare_quantize_backend(const TypeCase &type_case, const std::string &backend)
{
  if (type_case.kind == TypeKind::storage)
  {
    if (type_case.type == GGML_TYPE_BF16)
    {
      if (!libgguf_storage_set_backend(backend.c_str()))
      {
        throw std::runtime_error("failed to select storage backend: " + backend);
      }
    }
    return;
  }

  if (libgguf_common_quant_cpu_supports_backend(backend.c_str()))
  {
    if (!libgguf_common_quant_set_backend(backend.c_str()))
    {
      throw std::runtime_error("failed to select common quant backend: " + backend);
    }
  }
  else if (!libgguf_common_quant_set_backend("ref"))
  {
    throw std::runtime_error("failed to select common quant ref backend");
  }
}

size_t public_quantize_chunk(
    const TypeCase &type_case,
    const float *src,
    void *dst,
    int64_t rows,
    int64_t width,
    const float *imatrix)
{
  return libgguf_quantize_chunk(type_case.type, src, dst, 0, rows, width, imatrix);
}

BenchResult make_skipped_result(
    const std::string &op,
    const TypeCase &type_case,
    int64_t width,
    int64_t rows,
    const std::string &backend,
    const ThreadSpec &thread_spec,
    size_t row_size,
    std::string reason)
{
  BenchResult result;
  result.op = op;
  result.type = type_case.name;
  result.row_width = width;
  result.n_rows = rows;
  result.backend = backend;
  result.thread_mode = thread_spec.mode;
  result.threads = thread_spec.threads;
  result.supported = false;
  result.reason = std::move(reason);
  result.encoded_bytes_per_row = row_size;
  return result;
}

void fill_metrics(BenchResult *result, size_t input_bytes, size_t output_bytes)
{
  if (result->median_ns <= 0.0)
  {
    return;
  }
  const double seconds = result->median_ns / 1.0e9;
  result->rows_per_s = (double)result->n_rows / seconds;
  result->elements_per_s = (double)result->n_rows * (double)result->row_width / seconds;
  result->input_mb_per_s = ((double)input_bytes / 1.0e6) / seconds;
  result->output_mb_per_s = ((double)output_bytes / 1.0e6) / seconds;
}

BenchResult run_quantize_case(
    const Options &options,
    const TypeCase &type_case,
    int64_t width,
    const std::string &backend,
    const ThreadSpec &thread_spec)
{
  const size_t row_size = libgguf_row_size(type_case.type, width);
  const std::string display_backend = quantize_backend_label(type_case, backend);
  std::string reason;
  if (row_size == 0)
  {
    return make_skipped_result("quantize", type_case, width, options.rows, display_backend, thread_spec, 0, "row width is invalid for this type");
  }
  if (!quantize_supported(type_case, backend, &reason))
  {
    return make_skipped_result("quantize", type_case, width, options.rows, display_backend, thread_spec, row_size, reason);
  }

  std::vector<float> rows = build_rows(type_case, options.rows, width);
  std::vector<float> imatrix_storage = libgguf_quantize_requires_imatrix(type_case.type)
                                           ? build_imatrix(rows, options.rows, width)
                                           : std::vector<float>();
  const float *imatrix = imatrix_storage.empty() ? nullptr : imatrix_storage.data();
  std::vector<std::uint8_t> encoded((size_t)options.rows * row_size);

  prepare_quantize_backend(type_case, backend);

  auto run_once = [&]() {
    run_partitioned(options.rows, thread_spec.threads, [&](int64_t row_begin, int64_t row_count) {
      const float *src = rows.data() + (size_t)row_begin * (size_t)width;
      void *dst = encoded.data() + (size_t)row_begin * row_size;
      size_t written = 0;
      if (type_case.kind == TypeKind::quantized && type_case.direct.quantize)
      {
        written = type_case.direct.quantize(backend.c_str(), src, dst, row_count, width);
      }
      else
      {
        written = public_quantize_chunk(type_case, src, dst, row_count, width, imatrix);
      }
      if (written != (size_t)row_count * row_size)
      {
        throw std::runtime_error("quantize returned an unexpected byte count");
      }
    });
    mix_output_bytes(encoded);
  };

  BenchStats stats = measure(options.samples, options.min_ms, run_once);
  BenchResult result;
  result.op = "quantize";
  result.type = type_case.name;
  result.row_width = width;
  result.n_rows = options.rows;
  result.backend = display_backend;
  result.thread_mode = thread_spec.mode;
  result.threads = thread_spec.threads;
  result.supported = true;
  result.reason = "";
  result.samples = stats.samples;
  result.median_ns = stats.median_ns;
  result.best_ns = stats.best_ns;
  result.encoded_bytes_per_row = row_size;
  fill_metrics(&result, rows.size() * sizeof(float), encoded.size());
  return result;
}

bool build_encoded_for_dequant(
    const TypeCase &type_case,
    int64_t rows_count,
    int64_t width,
    std::vector<std::uint8_t> *encoded,
    std::string *reason)
{
  const size_t row_size = libgguf_row_size(type_case.type, width);
  if (row_size == 0)
  {
    *reason = "row width is invalid for this type";
    return false;
  }

  std::vector<float> rows = build_rows(type_case, rows_count, width);
  std::vector<float> imatrix_storage = libgguf_quantize_requires_imatrix(type_case.type)
                                           ? build_imatrix(rows, rows_count, width)
                                           : std::vector<float>();
  const float *imatrix = imatrix_storage.empty() ? nullptr : imatrix_storage.data();

  if (!libgguf_common_quant_set_backend("ref"))
  {
    *reason = "failed to select ref common quant backend";
    return false;
  }
  encoded->assign((size_t)rows_count * row_size, 0);
  const size_t written = libgguf_quantize_chunk(
      type_case.type,
      rows.data(),
      encoded->data(),
      0,
      rows_count,
      width,
      imatrix);
  if (written != encoded->size())
  {
    *reason = "failed to prepare encoded input";
    return false;
  }
  return true;
}

BenchResult run_dequantize_case(
    const Options &options,
    const TypeCase &type_case,
    int64_t width,
    const std::string &backend,
    const ThreadSpec &thread_spec)
{
  const size_t row_size = libgguf_row_size(type_case.type, width);
  std::string reason;
  if (!dequantize_supported(type_case, backend, &reason))
  {
    return make_skipped_result("dequantize", type_case, width, options.rows, backend, thread_spec, row_size, reason);
  }
  if (row_size == 0)
  {
    return make_skipped_result("dequantize", type_case, width, options.rows, backend, thread_spec, 0, "row width is invalid for this type");
  }

  std::vector<std::uint8_t> encoded;
  if (!build_encoded_for_dequant(type_case, options.rows, width, &encoded, &reason))
  {
    return make_skipped_result("dequantize", type_case, width, options.rows, backend, thread_spec, row_size, reason);
  }
  std::vector<float> decoded((size_t)options.rows * (size_t)width);

  auto run_once = [&]() {
    run_partitioned(options.rows, thread_spec.threads, [&](int64_t row_begin, int64_t row_count) {
      const void *src = encoded.data() + (size_t)row_begin * row_size;
      float *dst = decoded.data() + (size_t)row_begin * (size_t)width;
      const size_t written = libgguf_dequantize_for_backend(
          type_case.type,
          backend.c_str(),
          src,
          dst,
          row_count,
          width);
      if (written != (size_t)row_count * (size_t)width * sizeof(float))
      {
        throw std::runtime_error("dequantize returned an unexpected byte count");
      }
    });
    mix_output_floats(decoded);
  };

  BenchStats stats = measure(options.samples, options.min_ms, run_once);
  BenchResult result;
  result.op = "dequantize";
  result.type = type_case.name;
  result.row_width = width;
  result.n_rows = options.rows;
  result.backend = backend;
  result.thread_mode = thread_spec.mode;
  result.threads = thread_spec.threads;
  result.supported = true;
  result.reason = "";
  result.samples = stats.samples;
  result.median_ns = stats.median_ns;
  result.best_ns = stats.best_ns;
  result.encoded_bytes_per_row = row_size;
  fill_metrics(&result, encoded.size(), decoded.size() * sizeof(float));
  return result;
}

std::string csv_escape(const std::string &value)
{
  if (value.find_first_of(",\"\n\r") == std::string::npos)
  {
    return value;
  }
  std::string out = "\"";
  for (char ch : value)
  {
    if (ch == '"')
    {
      out += "\"\"";
    }
    else
    {
      out += ch;
    }
  }
  out += '"';
  return out;
}

void write_csv(std::ostream &out, const std::vector<BenchResult> &results)
{
  out << "op,type,row_width,n_rows,backend,thread_mode,threads,supported,reason,samples,median_ns,best_ns,rows_per_s,elements_per_s,input_mb_per_s,output_mb_per_s,encoded_bytes_per_row\n";
  out << std::setprecision(17);
  for (const BenchResult &result : results)
  {
    out << csv_escape(result.op) << ','
        << csv_escape(result.type) << ','
        << result.row_width << ','
        << result.n_rows << ','
        << csv_escape(result.backend) << ','
        << csv_escape(result.thread_mode) << ','
        << result.threads << ','
        << (result.supported ? "true" : "false") << ','
        << csv_escape(result.reason) << ','
        << result.samples << ','
        << result.median_ns << ','
        << result.best_ns << ','
        << result.rows_per_s << ','
        << result.elements_per_s << ','
        << result.input_mb_per_s << ','
        << result.output_mb_per_s << ','
        << result.encoded_bytes_per_row << '\n';
  }
}

std::string json_escape(const std::string &value)
{
  std::string out;
  out.reserve(value.size() + 8);
  for (char ch : value)
  {
    switch (ch)
    {
    case '\\':
      out += "\\\\";
      break;
    case '"':
      out += "\\\"";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      out += ch;
      break;
    }
  }
  return out;
}

void write_json(std::ostream &out, const std::vector<BenchResult> &results)
{
  out << std::setprecision(17);
  out << "[\n";
  for (size_t i = 0; i < results.size(); ++i)
  {
    const BenchResult &r = results[i];
    out << "  {"
        << "\"op\":\"" << json_escape(r.op) << "\","
        << "\"type\":\"" << json_escape(r.type) << "\","
        << "\"row_width\":" << r.row_width << ','
        << "\"n_rows\":" << r.n_rows << ','
        << "\"backend\":\"" << json_escape(r.backend) << "\","
        << "\"thread_mode\":\"" << json_escape(r.thread_mode) << "\","
        << "\"threads\":" << r.threads << ','
        << "\"supported\":" << (r.supported ? "true" : "false") << ','
        << "\"reason\":\"" << json_escape(r.reason) << "\","
        << "\"samples\":" << r.samples << ','
        << "\"median_ns\":" << r.median_ns << ','
        << "\"best_ns\":" << r.best_ns << ','
        << "\"rows_per_s\":" << r.rows_per_s << ','
        << "\"elements_per_s\":" << r.elements_per_s << ','
        << "\"input_mb_per_s\":" << r.input_mb_per_s << ','
        << "\"output_mb_per_s\":" << r.output_mb_per_s << ','
        << "\"encoded_bytes_per_row\":" << r.encoded_bytes_per_row
        << "}";
    if (i + 1 != results.size())
    {
      out << ',';
    }
    out << '\n';
  }
  out << "]\n";
}

void write_path_or_stdout(
    const std::string &path,
    const std::function<void(std::ostream &)> &writer)
{
  if (path.empty() || path == "-")
  {
    writer(std::cout);
    return;
  }
  std::ofstream file(path);
  if (!file)
  {
    throw std::runtime_error("failed to open output path: " + path);
  }
  writer(file);
}

} // namespace

int main(int argc, char **argv)
{
  try
  {
    Options options = parse_options(argc, argv);
    if (!set_process_env_threads_one())
    {
      throw std::runtime_error("failed to set LIBGGUF_NUM_THREADS=1");
    }

    const std::vector<ThreadSpec> thread_specs = resolve_threads(options.thread_tokens);
    const std::vector<const TypeCase *> type_cases = resolve_types(options.types);
    std::vector<BenchResult> results;
    const size_t total_cases = count_cases(options, type_cases, thread_specs);
    size_t case_index = 0;
    std::ostream *progress = nullptr;
    if (options.progress)
    {
      progress = stdout_is_data_output(options) ? &std::cerr : &std::cout;
      *progress << "libgguf_backend_bench: " << total_cases << " cases"
                << ", rows=" << options.rows
                << ", samples=" << options.samples
                << ", min_ms=" << options.min_ms
                << '\n';
    }

    for (const std::string &op : options.ops)
    {
      for (const TypeCase *type_case : type_cases)
      {
        for (const int64_t width : options.sizes)
        {
          const std::vector<std::string> case_backends = backends_for_case(op, *type_case, options.backends);
          for (const std::string &backend : case_backends)
          {
            for (const ThreadSpec &thread_spec : thread_specs)
            {
              ++case_index;
              if (progress)
              {
                const std::string display_backend = op == "quantize"
                                                        ? quantize_backend_label(*type_case, backend)
                                                        : backend;
                *progress << "[case " << case_index << '/' << total_cases << "] "
                          << op
                          << " type=" << type_case->name
                          << " row_width=" << width
                          << " backend=" << display_backend
                          << " thread_mode=" << thread_spec.mode
                          << " threads=" << thread_spec.threads
                          << '\n'
                          << std::flush;
              }
              if (op == "quantize")
              {
                results.push_back(run_quantize_case(options, *type_case, width, backend, thread_spec));
              }
              else if (op == "dequantize")
              {
                results.push_back(run_dequantize_case(options, *type_case, width, backend, thread_spec));
              }
            }
          }
        }
      }
    }

    if (!options.csv_path.empty())
    {
      write_path_or_stdout(options.csv_path, [&](std::ostream &out) { write_csv(out, results); });
    }
    if (!options.json_path.empty())
    {
      write_path_or_stdout(options.json_path, [&](std::ostream &out) { write_json(out, results); });
    }
    if (options.csv_path.empty() && options.json_path.empty())
    {
      write_csv(std::cout, results);
    }

    libgguf_storage_set_backend("auto");
    libgguf_common_quant_set_backend("ref");
    libgguf_quantize_free();
    return 0;
  }
  catch (const std::exception &exc)
  {
    std::cerr << "error: " << exc.what() << '\n';
    return 1;
  }
}
