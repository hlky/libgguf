#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <limits>
#include <map>
#include <mutex>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <emmintrin.h>
#define LIBGGUF_NATIVE_SSE2 1
#endif
#include "common/libgguf_common.h"
#include "common/libgguf_cpu.h"
#include "common/libgguf_internal.h"
#include "libgguf.h"

#if defined(LIBGGUF_HAS_CUDA_NATIVE)
#include "libgguf_cuda_native.h"
#endif

extern "C" void libgguf_quantize_gguf_f16_to_f32_f16c(const ggml_fp16_t *src, uint64_t count, float *dst);

namespace {

constexpr uint32_t GGUF_MAGIC = 0x46554747u;
constexpr uint32_t GGUF_VERSION = 3;
constexpr uint32_t GGML_QUANT_VERSION = 2;
constexpr uint64_t GGUF_DEFAULT_ALIGNMENT = 32;
constexpr uint64_t NATIVE_DEFAULT_SCRATCH_BYTES = 32ull * 1024ull * 1024ull;
constexpr int64_t QUANTIZATION_THRESHOLD = 1024;
constexpr int64_t REARRANGE_THRESHOLD = 512;
constexpr size_t MAX_TENSOR_NAME_LENGTH = 127;
constexpr uint64_t VERIFY_CUDA_ALL_TENSORS = std::numeric_limits<uint64_t>::max();

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

enum converter_backend {
  CONVERTER_BACKEND_AUTO,
  CONVERTER_BACKEND_CPU,
  CONVERTER_BACKEND_CUDA,
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
  bool cuda_fallback_cpu = false;
  uint64_t verify_cuda_tensors = 0;
  uint64_t verify_cuda_large_tensors = 0;
  uint64_t verify_cuda_random_tensors = 0;
  uint64_t seed = 0;
  uint64_t cuda_vram_bytes = 0;
  bool cuda_pipeline = true;
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

struct timing_totals {
  double metadata_s = 0.0;
  double read_s = 0.0;
  double cpu_convert_s = 0.0;
  double h2d_s = 0.0;
  double cuda_quant_s = 0.0;
  double d2h_s = 0.0;
  double write_s = 0.0;
  double total_s = 0.0;
  uint64_t tensors = 0;
  uint64_t cuda_tensors = 0;
  uint64_t cuda_verified_tensors = 0;
  uint64_t cuda_chunks = 0;
  uint64_t cuda_max_input_bytes = 0;
  uint64_t cuda_max_output_bytes = 0;
};

using steady_clock = std::chrono::steady_clock;

double elapsed_seconds(steady_clock::time_point begin, steady_clock::time_point end) {
  return std::chrono::duration<double>(end - begin).count();
}

struct input_file {
#if defined(_WIN32)
  HANDLE file = INVALID_HANDLE_VALUE;
#else
  int fd = -1;
#endif
  uint64_t size = 0;

  bool open(const std::string &path, std::string *error) {
#if defined(_WIN32)
    int wide_len = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    if (wide_len <= 0) {
      *error = "failed to convert path to UTF-16";
      return false;
    }
    std::vector<wchar_t> wide_path((size_t)wide_len);
    if (MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wide_path.data(), wide_len) <= 0) {
      *error = "failed to convert path to UTF-16";
      return false;
    }
    file = CreateFileW(wide_path.data(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
      *error = "failed to open input file";
      return false;
    }
    LARGE_INTEGER file_size;
    if (!GetFileSizeEx(file, &file_size) || file_size.QuadPart < 0) {
      *error = "failed to stat input file";
      return false;
    }
    size = (uint64_t)file_size.QuadPart;
    return true;
#else
    fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
      *error = std::string("failed to open input file: ") + std::strerror(errno);
      return false;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 0) {
      *error = std::string("failed to stat input file: ") + std::strerror(errno);
      return false;
    }
    size = (uint64_t)st.st_size;
    return true;
#endif
  }

  bool read_at(uint64_t offset, void *dst, uint64_t len, std::string *error) const {
    if (offset > size || len > size - offset) {
      *error = "read range exceeds input file size";
      return false;
    }
    uint8_t *cursor = (uint8_t *)dst;
    uint64_t remaining = len;
#if defined(_WIN32)
    LARGE_INTEGER pos;
    while (remaining > 0) {
      pos.QuadPart = (LONGLONG)offset;
      OVERLAPPED ov{};
      ov.Offset = pos.LowPart;
      ov.OffsetHigh = pos.HighPart;
      DWORD chunk = (DWORD)std::min<uint64_t>(remaining, (uint64_t)std::numeric_limits<DWORD>::max());
      DWORD got = 0;
      if (!ReadFile(file, cursor, chunk, &got, &ov)) {
        *error = "failed to read input file";
        return false;
      }
      if (got == 0) {
        *error = "unexpected end of input file";
        return false;
      }
      cursor += got;
      offset += got;
      remaining -= got;
    }
#else
    while (remaining > 0) {
      const size_t chunk = (size_t)std::min<uint64_t>(remaining, (uint64_t)std::numeric_limits<size_t>::max());
      const ssize_t got = pread(fd, cursor, chunk, (off_t)offset);
      if (got < 0) {
        if (errno == EINTR) {
          continue;
        }
        *error = std::string("failed to read input file: ") + std::strerror(errno);
        return false;
      }
      if (got == 0) {
        *error = "unexpected end of input file";
        return false;
      }
      cursor += got;
      offset += (uint64_t)got;
      remaining -= (uint64_t)got;
    }
#endif
    return true;
  }

  void close() {
#if defined(_WIN32)
    if (file != INVALID_HANDLE_VALUE) {
      CloseHandle(file);
      file = INVALID_HANDLE_VALUE;
    }
#else
    if (fd >= 0) {
      ::close(fd);
      fd = -1;
    }
#endif
    size = 0;
  }

  ~input_file() {
    close();
  }
};

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error(message);
}

bool ends_with(const std::string &value, const std::string &suffix) {
  return value.size() >= suffix.size() && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool starts_with(const std::string &value, const std::string &prefix) {
  return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

std::string upper_ascii(std::string value) {
  for (char &ch : value) {
    if (ch >= 'a' && ch <= 'z') {
      ch = char(ch - 'a' + 'A');
    }
  }
  return value;
}

bool file_exists(const std::string &path) {
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

std::string default_output_path(const std::string &src, const std::string &file_type_name) {
  size_t slash = src.find_last_of("/\\");
  size_t dot = src.find_last_of('.');
  if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) {
    dot = src.size();
  }
  return src.substr(0, dot) + "-" + file_type_name + ".gguf";
}

bool checked_mul_u64(uint64_t a, uint64_t b, uint64_t *out) {
  if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
    return false;
  }
  *out = a * b;
  return true;
}

uint64_t product_shape(const std::vector<int64_t> &shape) {
  uint64_t total = 1;
  for (int64_t dim : shape) {
    if (dim <= 0) {
      fail("shape dimensions must be positive");
    }
    uint64_t next = 0;
    if (!checked_mul_u64(total, (uint64_t)dim, &next)) {
      fail("shape element count is too large");
    }
    total = next;
  }
  return total;
}

uint64_t gguf_pad(uint64_t n, uint64_t alignment) {
  return ((n + alignment - 1) / alignment) * alignment;
}

template <typename T>
void append_le(std::vector<uint8_t> &out, T value) {
  uint8_t *ptr = reinterpret_cast<uint8_t *>(&value);
  out.insert(out.end(), ptr, ptr + sizeof(T));
}

void append_string(std::vector<uint8_t> &out, const std::string &value) {
  append_le<uint64_t>(out, (uint64_t)value.size());
  out.insert(out.end(), value.begin(), value.end());
}

void write_all_file(FILE *f, const void *data, uint64_t len) {
  const uint8_t *cursor = (const uint8_t *)data;
  while (len > 0) {
    const size_t chunk = (size_t)std::min<uint64_t>(len, (uint64_t)std::numeric_limits<size_t>::max());
    const size_t written = std::fwrite(cursor, 1, chunk, f);
    if (written == 0) {
      fail("failed to write output file");
    }
    cursor += written;
    len -= written;
  }
}

void write_zeros_file(FILE *f, uint64_t len) {
  static const uint8_t zeros[4096] = {};
  while (len > 0) {
    const uint64_t chunk = std::min<uint64_t>(len, sizeof(zeros));
    write_all_file(f, zeros, chunk);
    len -= chunk;
  }
}

uint64_t tell_file(FILE *f) {
#if defined(_WIN32)
  __int64 pos = _ftelli64(f);
#else
  long pos = std::ftell(f);
#endif
  if (pos < 0) {
    fail("failed to query output file position");
  }
  return (uint64_t)pos;
}

void write_pre_tensor_padding(FILE *f, uint64_t alignment) {
  const uint64_t pos = tell_file(f);
  write_zeros_file(f, gguf_pad(pos, alignment) - pos);
}

void write_post_tensor_padding(FILE *f, uint64_t tensor_nbytes, uint64_t alignment) {
  write_zeros_file(f, gguf_pad(tensor_nbytes, alignment) - tensor_nbytes);
}

size_t source_dtype_size(native_source_dtype dtype) {
  switch (dtype) {
  case NATIVE_DTYPE_F32:
    return sizeof(float);
  case NATIVE_DTYPE_F16:
  case NATIVE_DTYPE_BF16:
    return sizeof(uint16_t);
  }
  return 0;
}

native_source_dtype parse_source_dtype(const std::string &dtype) {
  if (dtype == "F32") {
    return NATIVE_DTYPE_F32;
  }
  if (dtype == "F16") {
    return NATIVE_DTYPE_F16;
  }
  if (dtype == "BF16") {
    return NATIVE_DTYPE_BF16;
  }
  fail("unsupported source dtype: " + dtype);
}

bool qtype_matches_source(native_source_dtype dtype, ggml_type qtype) {
  return (dtype == NATIVE_DTYPE_F32 && qtype == GGML_TYPE_F32) ||
         (dtype == NATIVE_DTYPE_F16 && qtype == GGML_TYPE_F16) ||
         (dtype == NATIVE_DTYPE_BF16 && qtype == GGML_TYPE_BF16);
}

std::string qtype_name(ggml_type qtype) {
  switch (qtype) {
  case GGML_TYPE_F32:
    return "F32";
  case GGML_TYPE_F16:
    return "F16";
  case GGML_TYPE_BF16:
    return "BF16";
  case GGML_TYPE_Q1_0:
    return "Q1_0";
  case GGML_TYPE_Q4_0:
    return "Q4_0";
  case GGML_TYPE_Q4_1:
    return "Q4_1";
  case GGML_TYPE_Q5_0:
    return "Q5_0";
  case GGML_TYPE_Q5_1:
    return "Q5_1";
  case GGML_TYPE_Q8_0:
    return "Q8_0";
  case GGML_TYPE_Q2_K:
    return "Q2_K";
  case GGML_TYPE_Q3_K:
    return "Q3_K";
  case GGML_TYPE_Q4_K:
    return "Q4_K";
  case GGML_TYPE_Q5_K:
    return "Q5_K";
  case GGML_TYPE_Q6_K:
    return "Q6_K";
  default:
    return "UNKNOWN";
  }
}

bool is_quantized_qtype(ggml_type qtype) {
  return qtype != GGML_TYPE_F32 && qtype != GGML_TYPE_F16 && qtype != GGML_TYPE_BF16;
}

bool is_supported_quant_qtype_name(const std::string &name) {
  static const std::set<std::string> supported = {
      "Q1_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"};
  return supported.count(name) != 0;
}

bool is_supported_storage_qtype_name(const std::string &name) {
  return is_supported_quant_qtype_name(name) || name == "F32" || name == "F16" || name == "BF16";
}

uint64_t qtype_block_size(ggml_type qtype) {
  switch (qtype) {
  case GGML_TYPE_F32:
  case GGML_TYPE_F16:
  case GGML_TYPE_BF16:
    return 1;
  case GGML_TYPE_Q4_0:
  case GGML_TYPE_Q4_1:
  case GGML_TYPE_Q5_0:
  case GGML_TYPE_Q5_1:
  case GGML_TYPE_Q8_0:
    return 32;
  case GGML_TYPE_Q1_0:
    return 128;
  case GGML_TYPE_Q2_K:
  case GGML_TYPE_Q3_K:
  case GGML_TYPE_Q4_K:
  case GGML_TYPE_Q5_K:
  case GGML_TYPE_Q6_K:
    return 256;
  default:
    return 0;
  }
}

ggml_type qtype_from_name(const std::string &name) {
  if (name == "F32") return GGML_TYPE_F32;
  if (name == "F16") return GGML_TYPE_F16;
  if (name == "BF16") return GGML_TYPE_BF16;
  if (name == "Q1_0") return GGML_TYPE_Q1_0;
  if (name == "Q4_0") return GGML_TYPE_Q4_0;
  if (name == "Q4_1") return GGML_TYPE_Q4_1;
  if (name == "Q5_0") return GGML_TYPE_Q5_0;
  if (name == "Q5_1") return GGML_TYPE_Q5_1;
  if (name == "Q8_0") return GGML_TYPE_Q8_0;
  if (name == "Q2_K") return GGML_TYPE_Q2_K;
  if (name == "Q3_K") return GGML_TYPE_Q3_K;
  if (name == "Q4_K") return GGML_TYPE_Q4_K;
  if (name == "Q5_K") return GGML_TYPE_Q5_K;
  if (name == "Q6_K") return GGML_TYPE_Q6_K;
  fail("unsupported GGML tensor type: " + name);
}

uint32_t file_type_value(const std::string &file_type_name) {
  if (file_type_name == "Q4_0") return 2;
  if (file_type_name == "Q4_1") return 3;
  if (file_type_name == "Q8_0") return 7;
  if (file_type_name == "Q5_0") return 8;
  if (file_type_name == "Q5_1") return 9;
  if (file_type_name == "Q2_K") return 10;
  if (file_type_name == "Q3_K_S") return 11;
  if (file_type_name == "Q3_K_M") return 12;
  if (file_type_name == "Q3_K_L") return 13;
  if (file_type_name == "Q4_K_S") return 14;
  if (file_type_name == "Q4_K_M") return 15;
  if (file_type_name == "Q5_K_S") return 16;
  if (file_type_name == "Q5_K_M") return 17;
  if (file_type_name == "Q6_K") return 18;
  if (file_type_name == "Q2_K_S") return 21;
  if (file_type_name == "Q1_0") return 27;
  fail("unsupported output file type: " + file_type_name);
}

std::pair<std::string, std::string> parse_qtype(const std::string &input) {
  std::string file_type = upper_ascii(input);
  if (file_type == "Q8_K") {
    fail("Q8_K output is not supported: this repo has metadata for Q8_K but no output quantizer");
  }
  if (file_type.rfind("IQ", 0) == 0 || file_type.rfind("TQ", 0) == 0 || file_type == "MXFP4" || file_type == "NVFP4") {
    fail("non-Q/K quantization families are not supported by this native executable yet");
  }
  if (file_type == "Q3_K") file_type = "Q3_K_M";
  if (file_type == "Q4_K") file_type = "Q4_K_M";
  if (file_type == "Q5_K") file_type = "Q5_K_M";

  std::string tensor_qtype = file_type;
  if (file_type == "Q3_K_S" || file_type == "Q3_K_M" || file_type == "Q3_K_L") tensor_qtype = "Q3_K";
  if (file_type == "Q4_K_S" || file_type == "Q4_K_M") tensor_qtype = "Q4_K";
  if (file_type == "Q5_K_S" || file_type == "Q5_K_M") tensor_qtype = "Q5_K";
  if (file_type == "Q2_K_S") tensor_qtype = "Q2_K";

  if (!is_supported_quant_qtype_name(tensor_qtype)) {
    fail("unsupported direct quantization type: " + input);
  }
  (void)file_type_value(file_type);
  return {file_type, tensor_qtype};
}

std::string parse_tensor_qtype_name(const std::string &input) {
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

uint64_t qtype_nbytes(const std::vector<int64_t> &shape, ggml_type qtype) {
  const uint64_t values = product_shape(shape);
  const uint64_t block_size = qtype_block_size(qtype);
  if (block_size == 0) {
    fail("unsupported qtype: " + qtype_name(qtype));
  }
  if (values % block_size != 0) {
    fail("Tensor with incompatible shape cannot be stored as " + qtype_name(qtype));
  }
  if (shape.empty()) {
    const size_t row_size = libgguf_row_size(qtype, 1);
    if (row_size == 0) {
      fail("unsupported qtype or scalar shape for " + qtype_name(qtype));
    }
    return (uint64_t)row_size;
  }
  const int64_t n_per_row = shape.back();
  const size_t row_size = libgguf_row_size(qtype, n_per_row);
  if (row_size == 0) {
    fail("unsupported qtype or row width for " + qtype_name(qtype));
  }
  return (values / (uint64_t)n_per_row) * (uint64_t)row_size;
}

bool fnmatch_case(const char *pattern, const char *text) {
  while (*pattern) {
    if (*pattern == '*') {
      while (*pattern == '*') {
        ++pattern;
      }
      if (!*pattern) {
        return true;
      }
      while (*text) {
        if (fnmatch_case(pattern, text)) {
          return true;
        }
        ++text;
      }
      return false;
    }
    if (*pattern == '?') {
      if (!*text) {
        return false;
      }
      ++pattern;
      ++text;
      continue;
    }
    if (*pattern != *text) {
      return false;
    }
    ++pattern;
    ++text;
  }
  return *text == '\0';
}

bool matches_any(const std::string &name, const std::vector<std::string> &patterns) {
  for (const std::string &pattern : patterns) {
    if (fnmatch_case(pattern.c_str(), name.c_str())) {
      return true;
    }
  }
  return false;
}

bool is_digits(const std::string &value) {
  if (value.empty()) {
    return false;
  }
  for (char ch : value) {
    if (ch < '0' || ch > '9') {
      return false;
    }
  }
  return true;
}

std::vector<std::string> split_dot(const std::string &value) {
  std::vector<std::string> parts;
  size_t start = 0;
  while (start <= value.size()) {
    size_t pos = value.find('.', start);
    if (pos == std::string::npos) {
      parts.push_back(value.substr(start));
      break;
    }
    parts.push_back(value.substr(start, pos - start));
    start = pos + 1;
  }
  return parts;
}

int dynamic_layer_index(const std::string &key) {
  static const std::set<std::string> block_names = {
      "blocks", "double_blocks", "double_layers", "double_stream_blocks", "input_blocks", "joint_blocks",
      "layers", "output_blocks", "single_blocks", "single_stream_blocks", "transformer_blocks"};
  const std::vector<std::string> parts = split_dot(key);
  for (size_t i = 0; i + 1 < parts.size(); ++i) {
    if (block_names.count(parts[i]) && is_digits(parts[i + 1])) {
      return std::stoi(parts[i + 1]);
    }
  }
  for (size_t i = 0; i + 1 < parts.size(); ++i) {
    if ((parts[i] == "context_refiner" || parts[i] == "noise_refiner") && is_digits(parts[i + 1])) {
      return std::stoi(parts[i + 1]);
    }
  }
  return -1;
}

std::string promote_k_qtype(const std::string &qtype, int steps = 1) {
  static const std::vector<std::string> order = {"Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0"};
  for (size_t i = 0; i < order.size(); ++i) {
    if (order[i] == qtype) {
      return order[std::min<size_t>(i + (size_t)steps, order.size() - 1)];
    }
  }
  return qtype;
}

bool contains_any(const std::string &name, const std::vector<std::string> &markers) {
  for (const std::string &marker : markers) {
    if (name.find(marker) != std::string::npos) {
      return true;
    }
  }
  return false;
}

struct json_value {
  enum kind_t { NIL, STRING, NUMBER, ARRAY, OBJECT } kind = NIL;
  std::string str;
  uint64_t num = 0;
  std::vector<json_value> array;
  std::vector<std::pair<std::string, json_value>> object;
};

struct json_parser {
  const char *p;
  const char *end;

  void skip_ws() {
    while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) {
      ++p;
    }
  }

  json_value parse() {
    skip_ws();
    if (p >= end) {
      fail("invalid JSON: unexpected end");
    }
    if (*p == '"') {
      json_value v;
      v.kind = json_value::STRING;
      v.str = parse_string();
      return v;
    }
    if (*p == '{') {
      return parse_object();
    }
    if (*p == '[') {
      return parse_array();
    }
    if (*p >= '0' && *p <= '9') {
      json_value v;
      v.kind = json_value::NUMBER;
      v.num = parse_number();
      return v;
    }
    fail("invalid JSON: unsupported value");
  }

  std::string parse_string() {
    if (*p != '"') {
      fail("invalid JSON: expected string");
    }
    ++p;
    std::string out;
    while (p < end && *p != '"') {
      unsigned char ch = (unsigned char)*p++;
      if (ch == '\\') {
        if (p >= end) {
          fail("invalid JSON escape");
        }
        char esc = *p++;
        switch (esc) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        default:
          fail("invalid JSON escape");
        }
      } else {
        out.push_back((char)ch);
      }
    }
    if (p >= end || *p != '"') {
      fail("invalid JSON: unterminated string");
    }
    ++p;
    return out;
  }

  uint64_t parse_number() {
    uint64_t value = 0;
    if (p >= end || *p < '0' || *p > '9') {
      fail("invalid JSON: expected number");
    }
    while (p < end && *p >= '0' && *p <= '9') {
      const uint64_t digit = (uint64_t)(*p - '0');
      if (value > (std::numeric_limits<uint64_t>::max() - digit) / 10) {
        fail("invalid JSON: number is too large");
      }
      value = value * 10 + digit;
      ++p;
    }
    return value;
  }

  json_value parse_array() {
    json_value v;
    v.kind = json_value::ARRAY;
    ++p;
    skip_ws();
    if (p < end && *p == ']') {
      ++p;
      return v;
    }
    while (true) {
      v.array.push_back(parse());
      skip_ws();
      if (p >= end) {
        fail("invalid JSON array");
      }
      if (*p == ']') {
        ++p;
        return v;
      }
      if (*p != ',') {
        fail("invalid JSON array: expected comma");
      }
      ++p;
    }
  }

  json_value parse_object() {
    json_value v;
    v.kind = json_value::OBJECT;
    ++p;
    skip_ws();
    if (p < end && *p == '}') {
      ++p;
      return v;
    }
    while (true) {
      skip_ws();
      std::string key = parse_string();
      skip_ws();
      if (p >= end || *p != ':') {
        fail("invalid JSON object: expected colon");
      }
      ++p;
      json_value value = parse();
      v.object.push_back({key, std::move(value)});
      skip_ws();
      if (p >= end) {
        fail("invalid JSON object");
      }
      if (*p == '}') {
        ++p;
        return v;
      }
      if (*p != ',') {
        fail("invalid JSON object: expected comma");
      }
      ++p;
    }
  }
};

const json_value *object_get(const json_value &object, const std::string &key) {
  if (object.kind != json_value::OBJECT) {
    return nullptr;
  }
  for (const auto &item : object.object) {
    if (item.first == key) {
      return &item.second;
    }
  }
  return nullptr;
}

std::vector<tensor_meta> parse_safetensors_header(const input_file &input, uint64_t *data_start) {
  if (input.size < 8) {
    fail("invalid safetensors file");
  }
  uint64_t header_len = 0;
  std::string error;
  if (!input.read_at(0, &header_len, sizeof(header_len), &error)) {
    fail(error);
  }
  if (header_len > input.size - 8) {
    fail("invalid safetensors header length");
  }
  std::vector<uint8_t> header((size_t)header_len);
  if (!input.read_at(8, header.data(), header_len, &error)) {
    fail(error);
  }
  *data_start = 8 + header_len;
  json_parser parser{(const char *)header.data(), (const char *)(header.data() + header.size())};
  json_value root = parser.parse();
  if (root.kind != json_value::OBJECT) {
    fail("invalid safetensors header");
  }

  std::vector<tensor_meta> tensors;
  for (const auto &entry : root.object) {
    if (entry.first == "__metadata__") {
      continue;
    }
    const json_value &info = entry.second;
    const json_value *dtype = object_get(info, "dtype");
    const json_value *shape = object_get(info, "shape");
    const json_value *offsets = object_get(info, "data_offsets");
    if (!dtype || dtype->kind != json_value::STRING || !shape || shape->kind != json_value::ARRAY || !offsets || offsets->kind != json_value::ARRAY || offsets->array.size() != 2) {
      fail("invalid safetensors tensor entry for " + entry.first);
    }
    tensor_meta meta;
    meta.source_key = entry.first;
    meta.key = entry.first;
    meta.dtype = dtype->str;
    for (const json_value &dim : shape->array) {
      if (dim.kind != json_value::NUMBER || dim.num > (uint64_t)std::numeric_limits<int64_t>::max()) {
        fail("invalid safetensors shape for " + entry.first);
      }
      meta.shape.push_back((int64_t)dim.num);
    }
    if (offsets->array[0].kind != json_value::NUMBER || offsets->array[1].kind != json_value::NUMBER) {
      fail("invalid safetensors data_offsets for " + entry.first);
    }
    meta.data_begin = *data_start + offsets->array[0].num;
    meta.data_end = *data_start + offsets->array[1].num;
    const uint64_t values = product_shape(meta.shape);
    const native_source_dtype source_dtype = parse_source_dtype(meta.dtype);
    uint64_t expected_source_bytes = 0;
    if (!checked_mul_u64(values, (uint64_t)source_dtype_size(source_dtype), &expected_source_bytes)) {
      fail("source byte count is too large for " + entry.first);
    }
    if (meta.data_end < meta.data_begin || meta.data_end - meta.data_begin != expected_source_bytes) {
      fail("safetensors byte length does not match shape/dtype for " + entry.first);
    }
    if (meta.data_end > input.size) {
      fail("safetensors byte range exceeds file size for " + entry.first);
    }
    tensors.push_back(std::move(meta));
  }
  return tensors;
}

void strip_prefix(std::vector<tensor_meta> &tensors) {
  const std::vector<std::string> prefixes = {"model.diffusion_model.", "model."};
  for (const std::string &prefix : prefixes) {
    bool any = false;
    for (const tensor_meta &tensor : tensors) {
      if (starts_with(tensor.key, prefix)) {
        any = true;
        break;
      }
    }
    if (any) {
      std::vector<tensor_meta> stripped;
      for (tensor_meta tensor : tensors) {
        if (starts_with(tensor.key, prefix)) {
          tensor.key = tensor.key.substr(prefix.size());
          stripped.push_back(std::move(tensor));
        }
      }
      tensors = std::move(stripped);
      return;
    }
  }

  bool all_net = !tensors.empty();
  for (const tensor_meta &tensor : tensors) {
    if (!starts_with(tensor.key, "net.")) {
      all_net = false;
      break;
    }
  }
  if (all_net) {
    for (tensor_meta &tensor : tensors) {
      tensor.key = tensor.key.substr(4);
    }
  }
}

std::vector<model_template> model_templates() {
  return {
      {"flux", false, {{"transformer_blocks.0.attn.norm_added_k.weight"}, {"double_blocks.0.img_attn.proj.weight"}}, {"transformer_blocks.0.attn.norm_added_k.weight"}, {}, {}},
      {"sd3", false, {{"transformer_blocks.0.attn.add_q_proj.weight"}, {"joint_blocks.0.x_block.attn.qkv.weight"}}, {"transformer_blocks.0.attn.add_q_proj.weight"}, {}, {}},
      {"aura", false, {{"double_layers.3.modX.1.weight"}, {"joint_transformer_blocks.3.ff_context.out_projection.weight"}}, {"joint_transformer_blocks.3.ff_context.out_projection.weight"}, {}, {}},
      {"hidream", false, {{"caption_projection.0.linear.weight", "double_stream_blocks.0.block.ff_i.shared_experts.w3.weight"}}, {}, {".ff_i.gate.weight", "img_emb.emb_pos"}, {}},
      {"cosmos", false, {{"blocks.0.mlp.layer1.weight", "blocks.0.adaln_modulation_cross_attn.1.weight"}}, {}, {"pos_embedder"}, {"_extra_state", "accum_"}},
      {"hyvid", false, {{"double_blocks.0.img_attn_proj.weight", "txt_in.individual_token_refiner.blocks.1.self_attn_qkv.weight"}}, {}, {}, {}},
      {"wan", false, {{"blocks.0.self_attn.norm_q.weight", "text_embedding.2.weight", "head.modulation"}}, {}, {".modulation"}, {}},
      {"ltxv", false, {{"adaln_single.emb.timestep_embedder.linear_2.weight", "transformer_blocks.27.scale_shift_table", "caption_projection.linear_2.weight"}}, {}, {"scale_shift_table"}, {}},
      {"sdxl", true, {{"down_blocks.0.downsamplers.0.conv.weight", "add_embedding.linear_1.weight"}, {"input_blocks.3.0.op.weight", "input_blocks.6.0.op.weight", "output_blocks.2.2.conv.weight", "output_blocks.5.2.conv.weight"}, {"label_emb.0.0.weight"}}, {}, {}, {}},
      {"sd1", true, {{"down_blocks.0.downsamplers.0.conv.weight"}, {"input_blocks.3.0.op.weight", "input_blocks.6.0.op.weight", "input_blocks.9.0.op.weight", "output_blocks.2.1.conv.weight", "output_blocks.5.2.conv.weight", "output_blocks.8.2.conv.weight"}}, {}, {}, {}},
      {"lumina2", false, {{"cap_embedder.1.weight", "context_refiner.0.attention.qkv.weight"}}, {}, {}, {}},
  };
}

model_template detect_arch(const std::vector<tensor_meta> &tensors) {
  std::set<std::string> keys;
  for (const tensor_meta &tensor : tensors) {
    keys.insert(tensor.key);
  }

  for (const model_template &model : model_templates()) {
    bool matched = false;
    for (const auto &match_list : model.keys_detect) {
      bool all = true;
      for (const std::string &key : match_list) {
        if (!keys.count(key)) {
          all = false;
          break;
        }
      }
      if (all) {
        matched = true;
        break;
      }
    }
    if (matched) {
      for (const std::string &key : model.keys_banned) {
        if (keys.count(key)) {
          fail("Model architecture not allowed for conversion; use the reference checkpoint key format");
        }
      }
      return model;
    }
  }
  fail("Unknown model architecture");
}

const std::vector<std::string> &arch_skip_patterns(const std::string &arch) {
  static const std::vector<std::string> empty;
  static const std::map<std::string, std::vector<std::string>> patterns = {
      {"flux", {"txt_in.*", "img_in.*", "time_in.*", "vector_in.*", "guidance_in.*", "final_layer.*"}},
      {"sd1", {"class_embedding.*", "time_embedding.*", "add_embedding.*", "time_embed.*", "label_emb.*", "conv_in.*", "conv_out.*", "input_blocks.0.0.weight", "out.2.weight"}},
      {"sdxl", {"class_embedding.*", "time_embedding.*", "add_embedding.*", "time_embed.*", "label_emb.*", "conv_in.*", "conv_out.*", "input_blocks.0.0.weight", "out.2.weight"}},
      {"sd3", {"final_layer.*", "time_text_embed.*", "context_embedder.*", "t_embedder.*", "y_embedder.*", "x_embedder.*", "proj_out.weight", "pos_embed"}},
      {"aura", {"t_embedder.*", "init_x_linear.*", "modF.1.weight", "cond_seq_linear.weight", "final_linear.weight", "positional_encoding", "register_tokens"}},
      {"ltxv", {"adaln_single.*", "caption_projection.*", "patchify_proj.*", "proj_out.*", "*scale_shift_table*"}},
      {"hyvid", {"txt_in.*", "img_in.*", "time_in.*", "vector_in.*", "guidance_in.*", "final_layer.*"}},
      {"wan", {"*modulation.*", "patch_embedding.*", "text_embedding.*", "time_projection.*", "time_embedding.*", "img_emb.*", "head.*"}},
      {"hidream", {"p_embedder.*", "t_embedder.*", "x_embedder.*", "final_layer.*", "*.ff_i.gate.weight", "caption_projection.*"}},
      {"cosmos", {"p_embedder.*", "t_embedder.*", "t_embedding_norm.*", "x_embedder.*", "pos_embedder.*", "final_layer.*"}},
      {"lumina2", {"t_embedder.*", "x_embedder.*", "final_layer.*", "cap_embedder.*", "context_refiner.*", "noise_refiner.*"}},
  };
  auto it = patterns.find(arch);
  return it == patterns.end() ? empty : it->second;
}

bool policy_allows_quant_shape(const std::string &key, const std::vector<int64_t> &shape, const model_template &model, const std::string &policy) {
  if (shape.size() != 2) {
    return false;
  }
  if (!ends_with(key, "weight")) {
    return false;
  }
  if (policy == "uniform") {
    return true;
  }
  if (matches_any(key, arch_skip_patterns(model.arch))) {
    return false;
  }
  return true;
}

std::string mixed_policy_qtype(const std::string &file_type, const std::string &base_qtype, const std::string &key, const model_template &model, std::map<std::string, int> &counters) {
  static const std::vector<std::string> attention_value = {
      "*attn_v.weight*", "*.to_v.weight*", "*.v.weight*", "*.attn.w1v.weight*", "*.attn.w2v.weight*", "*_attn.v_proj.weight*"};
  static const std::vector<std::string> fused_qkv = {"*attn_qkv.weight*", "*attn.qkv.weight*", "*attention.qkv.weight*"};
  static const std::vector<std::string> ffn_down = {
      "*ffn_down*", "*experts.*.w2.weight*", "*shared_experts.w2.weight*", "*.ffn.2.weight*", "*.ff.net.2.weight*", "*.mlp.layer2.weight*", "*.adaln_modulation_mlp.2.weight*", "*.feed_forward.w2.weight*"};

  std::string qtype = base_qtype;
  if (matches_any(key, attention_value)) {
    if (file_type == "Q2_K") qtype = "Q3_K";
    else if (file_type == "Q3_K_M") qtype = counters["attention_value"] < 2 ? "Q5_K" : "Q4_K";
    else if (file_type == "Q3_K_L") qtype = "Q5_K";
    else if (file_type == "Q4_K_M" || file_type == "Q5_K_M") qtype = "Q6_K";
    else if (file_type == "Q4_K_S" && counters["attention_value"] < 4) qtype = "Q5_K";
    counters["attention_value"] += 1;
  } else if (matches_any(key, fused_qkv)) {
    if (file_type == "Q3_K_M" || file_type == "Q3_K_L") qtype = "Q4_K";
    else if (file_type == "Q4_K_M") qtype = "Q5_K";
    else if (file_type == "Q5_K_M") qtype = "Q6_K";
  } else if (matches_any(key, ffn_down)) {
    if (file_type == "Q3_K_M") qtype = "Q4_K";
    else if (file_type == "Q3_K_L") qtype = "Q5_K";
    else if (file_type == "Q4_K_S") qtype = "Q5_K";
    else if (file_type == "Q4_K_M" || file_type == "Q5_K_M") qtype = "Q6_K";
    else if (file_type == "Q4_0") qtype = "Q4_1";
    else if (file_type == "Q5_0") qtype = "Q5_1";
    counters["ffn_down"] += 1;
  } else if (model.arch == "aura" && file_type == "Q3_K_M" && fnmatch_case("double_layers.*.mlp?.c_proj.weight", key.c_str())) {
    qtype = "Q4_K";
  }
  return qtype;
}

std::string dynamic_policy_qtype(const std::string &file_type, const std::string &comfy_qtype, const std::string &key, int max_layer) {
  static const std::vector<std::string> attention_value = {
      "*attn_v.weight*", "*.to_v.weight*", "*.v.weight*", "*.attn.w1v.weight*", "*.attn.w2v.weight*", "*_attn.v_proj.weight*"};
  static const std::vector<std::string> fused_qkv = {"*attn_qkv.weight*", "*attn.qkv.weight*", "*attention.qkv.weight*"};
  static const std::vector<std::string> attention_qko = {
      "*.to_q.weight", "*.q_proj.weight", "*.query_proj.weight", "*_attn.q_proj.weight", "*attn_q.weight",
      "*.to_k.weight", "*.k_proj.weight", "*.key_proj.weight", "*_attn.k_proj.weight", "*attn_k.weight",
      "*.to_out.0.weight", "*.to_out.weight", "*.o_proj.weight", "*.out_proj.weight", "*.proj_out.weight", "*_attn.proj.weight"};
  static const std::vector<std::string> ffn_up_down = {
      "*ffn_down*", "*experts.*.w2.weight*", "*shared_experts.w2.weight*", "*.ffn.2.weight*", "*.ff.net.2.weight*",
      "*.mlp.layer2.weight*", "*.adaln_modulation_mlp.2.weight*", "*.feed_forward.w2.weight*",
      "*.mlp.linear_fc2.weight", "*.linear_fc2.weight", "*.down_proj.weight", "*.c_proj.weight", "*.out_projection.weight",
      "*.mlp.up_proj.weight", "*.up_proj.weight", "*.w3.weight", "*.fc1.weight", "*.linear_fc1.weight", "*.mlp.layer1.weight"};
  static const std::vector<std::string> ffn_gate = {"*.mlp.gate_proj.weight", "*.gate_proj.weight", "*.w1.weight"};
  static const std::vector<std::string> ffn = {
      "*ffn_down*", "*experts.*.w2.weight*", "*shared_experts.w2.weight*", "*.ffn.2.weight*", "*.ff.net.2.weight*",
      "*.mlp.layer2.weight*", "*.adaln_modulation_mlp.2.weight*", "*.feed_forward.w2.weight*",
      "*.mlp.linear_fc2.weight", "*.linear_fc2.weight", "*.down_proj.weight", "*.c_proj.weight", "*.out_projection.weight",
      "*.mlp.up_proj.weight", "*.up_proj.weight", "*.w3.weight", "*.fc1.weight", "*.linear_fc1.weight", "*.mlp.layer1.weight",
      "*.mlp.gate_proj.weight", "*.gate_proj.weight", "*.w1.weight"};

  const int layer = dynamic_layer_index(key);
  if (layer < 0) {
    return comfy_qtype;
  }
  const bool early = layer <= 1;
  const bool final_tail = max_layer >= 0 && layer >= max_layer - 1;
  if (matches_any(key, attention_value)) {
    if (file_type == "Q2_K" || file_type == "Q3_K_M" || file_type == "Q3_K_L") return "Q6_K";
    if (file_type == "Q4_K_S" || file_type == "Q4_K_M" || file_type == "Q5_K_S" || file_type == "Q5_K_M") return "Q8_0";
    return promote_k_qtype(comfy_qtype);
  }
  if (matches_any(key, fused_qkv)) {
    return promote_k_qtype(comfy_qtype, early ? 2 : 1);
  }
  if (matches_any(key, attention_qko)) {
    std::string qtype = promote_k_qtype(comfy_qtype, early ? 2 : 1);
    if (matches_any(key, {"*.to_out.0.weight", "*.to_out.weight", "*.o_proj.weight", "*.out_proj.weight", "*.proj_out.weight", "*_attn.proj.weight"})) {
      if ((file_type == "Q2_K" || file_type == "Q3_K_M" || file_type == "Q4_K_M") && ((layer >= 15 && layer <= 24) || (layer >= 28 && !final_tail))) {
        qtype = promote_k_qtype(qtype);
      } else if (file_type == "Q5_K_M" && (layer == 3 || layer == 24 || (layer >= 31 && !final_tail))) {
        qtype = "Q8_0";
      }
    }
    return qtype;
  }
  if (matches_any(key, ffn)) {
    std::string qtype = promote_k_qtype(comfy_qtype, early && (file_type == "Q4_K_M" || file_type == "Q5_K_M") ? 2 : 1);
    if (matches_any(key, ffn_up_down)) {
      if ((file_type == "Q2_K" || file_type == "Q3_K_M" || file_type == "Q4_K_M") && (layer == 7 || layer == 21 || (layer >= 23 && !final_tail))) {
        qtype = promote_k_qtype(qtype);
      } else if (file_type == "Q5_K_M" && (layer == 7 || (layer >= 27 && !final_tail))) {
        qtype = "Q8_0";
      }
    }
    (void)ffn_gate;
    return qtype;
  }
  return comfy_qtype;
}

ggml_type base_storage_type_for_meta(const tensor_meta &tensor, const model_template &model) {
  const int64_t n_params = (int64_t)product_shape(tensor.shape);
  ggml_type data_qtype = tensor.dtype == "BF16" ? GGML_TYPE_BF16 : GGML_TYPE_F16;
  if (tensor.dtype == "F32" || tensor.dtype == "BF16") {
    if (tensor.shape.size() == 1 || n_params <= QUANTIZATION_THRESHOLD || contains_any(tensor.key, model.keys_hiprec)) {
      data_qtype = GGML_TYPE_F32;
    }
  }
  return data_qtype;
}

std::vector<int64_t> shape_fix_shape(const tensor_meta &tensor, const model_template &model, std::vector<kv_value> &kv) {
  std::vector<int64_t> shape = tensor.shape;
  const int64_t n_params = (int64_t)product_shape(shape);
  if (model.shape_fix && shape.size() > 1 && n_params >= REARRANGE_THRESHOLD && n_params % 256 == 0 && shape.back() % 256 != 0) {
    kv_value value;
    value.key = "comfy.gguf.orig_shape." + tensor.key;
    value.type = GGUF_TYPE_ARRAY;
    value.arr_i64 = shape;
    kv.push_back(std::move(value));
    return {n_params / 256, 256};
  }
  return shape;
}

std::string override_qtype(const std::string &key, const std::vector<std::pair<std::string, std::string>> &overrides) {
  for (const auto &item : overrides) {
    if (fnmatch_case(item.first.c_str(), key.c_str())) {
      return item.second;
    }
  }
  return "";
}

std::vector<tensor_plan> prepare_plans(
    const std::vector<tensor_meta> &tensors,
    const model_template &model,
    const std::string &file_type_name,
    const std::string &base_qtype_name,
    const cli_options &options,
    std::vector<kv_value> &kv,
    std::vector<tensor_info> &infos,
    conversion_result &result) {
  std::vector<tensor_plan> plans;
  std::map<std::string, int> counters = {{"attention_value", 0}, {"ffn_down", 0}};
  int max_dynamic_layer = -1;
  for (const tensor_meta &tensor : tensors) {
    max_dynamic_layer = std::max(max_dynamic_layer, dynamic_layer_index(tensor.key));
  }

  for (const tensor_meta &tensor : tensors) {
    if (contains_any(tensor.key, model.keys_ignore)) {
      continue;
    }
    if (tensor.key.size() > MAX_TENSOR_NAME_LENGTH) {
      fail("Can only handle tensor names up to 127 characters: " + tensor.key);
    }
    ggml_type storage_qtype = base_storage_type_for_meta(tensor, model);
    std::vector<int64_t> write_shape = shape_fix_shape(tensor, model, kv);
    bool quantize = policy_allows_quant_shape(tensor.key, write_shape, model, options.policy);
    if (matches_any(tensor.key, options.include) && write_shape.size() == 2) {
      quantize = true;
    }
    if (matches_any(tensor.key, options.exclude)) {
      quantize = false;
    }

    ggml_type target_qtype = storage_qtype;
    const std::string forced = override_qtype(tensor.key, options.tensor_overrides);
    if (!forced.empty()) {
      target_qtype = qtype_from_name(parse_tensor_qtype_name(forced));
      quantize = is_quantized_qtype(target_qtype);
    } else if (quantize) {
      std::string target_name = base_qtype_name;
      if (options.policy == "comfy" || options.policy == "dynamic") {
        target_name = mixed_policy_qtype(file_type_name, base_qtype_name, tensor.key, model, counters);
        if (options.policy == "dynamic") {
          target_name = dynamic_policy_qtype(file_type_name, target_name, tensor.key, max_dynamic_layer);
        }
      }
      target_qtype = qtype_from_name(target_name);
    }

    if (quantize && is_quantized_qtype(target_qtype)) {
      const uint64_t block_size = qtype_block_size(target_qtype);
      if (block_size == 0 || (uint64_t)write_shape.back() % block_size != 0) {
        result.fallback_counts[qtype_name(target_qtype)] += 1;
        target_qtype = GGML_TYPE_F16;
        quantize = false;
      }
    }

    tensor_plan plan;
    plan.key = tensor.key;
    plan.source_dtype = parse_source_dtype(tensor.dtype);
    plan.qtype = target_qtype;
    plan.source_shape = tensor.shape;
    plan.write_shape = write_shape;
    plan.data_begin = tensor.data_begin;
    plan.data_end = tensor.data_end;
    plan.n_values = product_shape(write_shape);
    plan.n_per_row = write_shape.empty() ? 1 : (uint64_t)write_shape.back();
    plan.n_rows = plan.n_values / plan.n_per_row;
    plan.expected_nbytes = qtype_nbytes(write_shape, target_qtype);

    tensor_info info;
    info.key = tensor.key;
    info.gguf_shape = write_shape;
    info.qtype = target_qtype;
    info.nbytes = plan.expected_nbytes;
    infos.push_back(info);
    plans.push_back(std::move(plan));
    result.tensor_type_counts[qtype_name(target_qtype)] += 1;
  }
  return plans;
}

void append_kv_value(std::vector<uint8_t> &out, const kv_value &kv) {
  append_string(out, kv.key);
  append_le<uint32_t>(out, (uint32_t)kv.type);
  if (kv.type == GGUF_TYPE_STRING) {
    append_string(out, kv.str);
  } else if (kv.type == GGUF_TYPE_UINT32) {
    append_le<uint32_t>(out, kv.u32);
  } else if (kv.type == GGUF_TYPE_ARRAY) {
    append_le<uint32_t>(out, (uint32_t)GGUF_TYPE_INT64);
    append_le<uint64_t>(out, (uint64_t)kv.arr_i64.size());
    for (int64_t value : kv.arr_i64) {
      append_le<int64_t>(out, value);
    }
  } else {
    fail("unsupported GGUF metadata value type");
  }
}

void write_gguf_metadata(FILE *out, const std::vector<kv_value> &kv, const std::vector<tensor_info> &infos) {
  std::vector<uint8_t> header;
  append_le<uint32_t>(header, GGUF_MAGIC);
  append_le<uint32_t>(header, GGUF_VERSION);
  append_le<uint64_t>(header, (uint64_t)infos.size());
  append_le<uint64_t>(header, (uint64_t)kv.size());
  write_all_file(out, header.data(), header.size());

  std::vector<uint8_t> kv_data;
  for (const kv_value &value : kv) {
    append_kv_value(kv_data, value);
  }
  write_all_file(out, kv_data.data(), kv_data.size());

  std::vector<uint8_t> ti_data;
  uint64_t offset = 0;
  for (const tensor_info &info : infos) {
    append_string(ti_data, info.key);
    append_le<uint32_t>(ti_data, (uint32_t)info.gguf_shape.size());
    for (size_t i = info.gguf_shape.size(); i > 0; --i) {
      append_le<uint64_t>(ti_data, (uint64_t)info.gguf_shape[i - 1]);
    }
    append_le<uint32_t>(ti_data, (uint32_t)info.qtype);
    append_le<uint64_t>(ti_data, offset);
    offset += gguf_pad(info.nbytes, GGUF_DEFAULT_ALIGNMENT);
  }
  write_all_file(out, ti_data.data(), ti_data.size());
}

unsigned int parse_positive_thread_count(const char *value) {
  if (value && value[0]) {
    errno = 0;
    char *end = nullptr;
    const long long parsed = std::strtoll(value, &end, 10);
    if (errno == 0 && end != value && *end == '\0' && parsed > 0) {
      return (unsigned int)std::min<long long>(parsed, (long long)std::numeric_limits<unsigned int>::max());
    }
  }
  return 0;
}

unsigned int native_thread_limit(unsigned int requested) {
  if (requested > 0) {
    return requested;
  }
  const unsigned int env_threads = parse_positive_thread_count(std::getenv("LIBGGUF_NUM_THREADS"));
  if (env_threads > 0) {
    return env_threads;
  }
  const unsigned int hardware = std::thread::hardware_concurrency();
  return hardware == 0 ? 1 : hardware;
}

unsigned int active_thread_count(uint64_t nrows, unsigned int max_threads) {
  if (nrows < 64 || max_threads <= 1) {
    return 1;
  }
  return (unsigned int)std::min<uint64_t>(nrows, (uint64_t)max_threads);
}

typedef size_t (*native_quantize_fn)(const float *RESTRICT, void *RESTRICT, int64_t, int64_t, const float *);

native_quantize_fn native_quantize_function(ggml_type qtype) {
  switch (qtype) {
  case GGML_TYPE_Q1_0: return quantize_q1_0;
  case GGML_TYPE_Q4_0: return quantize_q4_0;
  case GGML_TYPE_Q4_1: return quantize_q4_1;
  case GGML_TYPE_Q5_0: return quantize_q5_0;
  case GGML_TYPE_Q5_1: return quantize_q5_1;
  case GGML_TYPE_Q8_0: return quantize_q8_0;
  case GGML_TYPE_Q2_K: return quantize_q2_K;
  case GGML_TYPE_Q3_K: return quantize_q3_K;
  case GGML_TYPE_Q4_K: return quantize_q4_K;
  case GGML_TYPE_Q5_K: return quantize_q5_K;
  case GGML_TYPE_Q6_K: return quantize_q6_K;
  default: return nullptr;
  }
}

void decode_bf16_to_f32(const uint16_t *src, uint64_t count, float *dst) {
  uint64_t i = 0;
#if defined(LIBGGUF_NATIVE_SSE2)
  const __m128i zero = _mm_setzero_si128();
  for (; i + 8 <= count; i += 8) {
    const __m128i values = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + i));
    const __m128i lo = _mm_unpacklo_epi16(zero, values);
    const __m128i hi = _mm_unpackhi_epi16(zero, values);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + i), lo);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + i + 4), hi);
  }
#endif
  for (; i < count; ++i) {
    ggml_bf16_t value;
    value.bits = src[i];
    dst[i] = GGML_BF16_TO_FP32(value);
  }
}

void decode_f16_to_f32(const ggml_fp16_t *src, uint64_t count, float *dst) {
  if (libgguf_get_cpu_features().f16c) {
    libgguf_quantize_gguf_f16_to_f32_f16c(src, count, dst);
    return;
  }
  for (uint64_t i = 0; i < count; ++i) {
    dst[i] = GGML_FP16_TO_FP32(src[i]);
  }
}

void decode_source_bytes_to_f32(const tensor_plan &plan, const uint8_t *src, uint64_t value_count, float *dst) {
  switch (plan.source_dtype) {
  case NATIVE_DTYPE_F32:
    std::memcpy(dst, src, (size_t)(value_count * sizeof(float)));
    break;
  case NATIVE_DTYPE_F16: {
    decode_f16_to_f32((const ggml_fp16_t *)src, value_count, dst);
    break;
  }
  case NATIVE_DTYPE_BF16:
    decode_bf16_to_f32((const uint16_t *)src, value_count, dst);
    break;
  }
}

void read_source_chunk(const input_file &input, const tensor_plan &plan, uint64_t row_begin, uint64_t row_count, std::vector<uint8_t> &bytes) {
  const uint64_t value_begin = row_begin * plan.n_per_row;
  const uint64_t value_count = row_count * plan.n_per_row;
  const uint64_t byte_count = value_count * source_dtype_size(plan.source_dtype);
  bytes.resize((size_t)byte_count);
  std::string error;
  if (!input.read_at(plan.data_begin + value_begin * source_dtype_size(plan.source_dtype), bytes.data(), byte_count, &error)) {
    fail(error + " for " + plan.key);
  }
}

class tensor_quant_worker_pool {
public:
  explicit tensor_quant_worker_pool(unsigned int max_threads)
      : max_threads_(std::max(1u, max_threads)) {
    scratch_.resize(max_threads_);
    ok_.resize(max_threads_, true);
    if (max_threads_ > 1) {
      for (unsigned int i = 0; i < max_threads_; ++i) {
        threads_.emplace_back([this, i]() { worker_loop(i); });
      }
    }
  }

  tensor_quant_worker_pool(const tensor_quant_worker_pool &) = delete;
  tensor_quant_worker_pool &operator=(const tensor_quant_worker_pool &) = delete;

  ~tensor_quant_worker_pool() {
    if (threads_.empty()) {
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
      ++generation_;
    }
    task_cv_.notify_all();
    for (std::thread &thread : threads_) {
      thread.join();
    }
  }

  bool run(const tensor_plan &plan, const uint8_t *source_bytes, uint64_t row_count, void *dst, const float *imatrix, uint64_t scratch_bytes) {
    configure_task(plan, row_count, scratch_bytes);
    if (active_threads_ <= 1) {
      return quantize_range(0, source_bytes, 0, row_count, (uint8_t *)dst, imatrix);
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      plan_ = &plan;
      source_bytes_ = source_bytes;
      row_count_ = row_count;
      dst_ = (uint8_t *)dst;
      imatrix_ = imatrix;
      completed_ = 0;
      std::fill(ok_.begin(), ok_.begin() + active_threads_, true);
      ++generation_;
    }
    task_cv_.notify_all();

    std::unique_lock<std::mutex> lock(mutex_);
    done_cv_.wait(lock, [&]() { return completed_ == max_threads_; });
    for (unsigned int i = 0; i < active_threads_; ++i) {
      if (!ok_[i]) {
        return false;
      }
    }
    return true;
  }

private:
  void configure_task(const tensor_plan &plan, uint64_t row_count, uint64_t scratch_bytes) {
    plan_ = &plan;
    quantize_ = native_quantize_function(plan.qtype);
    row_size_ = libgguf_row_size(plan.qtype, (int64_t)plan.n_per_row);
    source_row_bytes_ = plan.n_per_row * source_dtype_size(plan.source_dtype);
    active_threads_ = active_thread_count(row_count, max_threads_);
    if (!quantize_ || row_size_ == 0) {
      fail("unsupported native quantizer for " + plan.key);
    }

    const uint64_t rows_per_thread = active_threads_ == 0 ? row_count : (row_count + (uint64_t)active_threads_ - 1) / (uint64_t)active_threads_;
    uint64_t tile_rows = plan.source_dtype == NATIVE_DTYPE_F32 ? rows_per_thread : std::max<uint64_t>(1, std::min<uint64_t>(256, rows_per_thread));
    if (plan.source_dtype != NATIVE_DTYPE_F32) {
      const uint64_t f32_scratch_budget = std::max<uint64_t>(
          plan.n_per_row * sizeof(float),
          scratch_bytes / std::max<uint64_t>(1, (uint64_t)active_threads_ * 4));
      tile_rows = std::min<uint64_t>(tile_rows, std::max<uint64_t>(1, f32_scratch_budget / (plan.n_per_row * sizeof(float))));
      for (unsigned int i = 0; i < active_threads_; ++i) {
        scratch_[i].resize((size_t)(tile_rows * plan.n_per_row));
      }
    }
    tile_rows_ = std::max<uint64_t>(1, tile_rows);
  }

  void worker_loop(unsigned int thread_id) {
    uint64_t seen_generation = 0;
    for (;;) {
      const tensor_plan *plan = nullptr;
      const uint8_t *source = nullptr;
      uint8_t *dst = nullptr;
      const float *imatrix = nullptr;
      uint64_t row_count = 0;
      unsigned int active_threads = 1;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        task_cv_.wait(lock, [&]() { return stop_ || generation_ != seen_generation; });
        if (stop_) {
          return;
        }
        seen_generation = generation_;
        plan = plan_;
        source = source_bytes_;
        dst = dst_;
        imatrix = imatrix_;
        row_count = row_count_;
        active_threads = active_threads_;
      }

      bool ok = true;
      if (thread_id < active_threads) {
        const uint64_t rows_per_thread = (row_count + (uint64_t)active_threads - 1) / (uint64_t)active_threads;
        const uint64_t begin = (uint64_t)thread_id * rows_per_thread;
        const uint64_t end = std::min<uint64_t>(row_count, begin + rows_per_thread);
        if (begin < end && plan != nullptr) {
          ok = quantize_range(thread_id, source, begin, end - begin, dst, imatrix);
        }
      }

      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (thread_id < active_threads_) {
          ok_[thread_id] = ok;
        }
        ++completed_;
      }
      done_cv_.notify_one();
    }
  }

  bool quantize_range(
      unsigned int thread_id,
      const uint8_t *source,
      uint64_t begin,
      uint64_t count,
      uint8_t *dst,
      const float *imatrix) {
    uint64_t processed = 0;
    while (processed < count) {
      const uint64_t tile_rows = std::min<uint64_t>(tile_rows_, count - processed);
      const uint64_t row = begin + processed;
      const uint8_t *tile_source = source + row * source_row_bytes_;
      uint8_t *tile_dst = dst + row * row_size_;
      const float *quant_source = nullptr;
      const tensor_plan &plan = *plan_;
      if (plan.source_dtype == NATIVE_DTYPE_F32) {
        quant_source = reinterpret_cast<const float *>(tile_source);
      } else {
        float *scratch = scratch_[thread_id].data();
        decode_source_bytes_to_f32(plan, tile_source, tile_rows * plan.n_per_row, scratch);
        quant_source = scratch;
      }
      const size_t written = quantize_(quant_source, tile_dst, (int64_t)tile_rows, (int64_t)plan.n_per_row, imatrix);
      if (written != tile_rows * (uint64_t)row_size_) {
        return false;
      }
      processed += tile_rows;
    }
    return true;
  }

  const unsigned int max_threads_ = 1;
  const tensor_plan *plan_ = nullptr;
  native_quantize_fn quantize_ = nullptr;
  size_t row_size_ = 0;
  unsigned int active_threads_ = 1;
  uint64_t source_row_bytes_ = 0;
  uint64_t tile_rows_ = 1;

  std::vector<std::thread> threads_;
  std::vector<std::vector<float>> scratch_;
  std::vector<bool> ok_;

  std::mutex mutex_;
  std::condition_variable task_cv_;
  std::condition_variable done_cv_;
  bool stop_ = false;
  uint64_t generation_ = 0;
  uint64_t completed_ = 0;
  const uint8_t *source_bytes_ = nullptr;
  uint64_t row_count_ = 0;
  uint8_t *dst_ = nullptr;
  const float *imatrix_ = nullptr;
};

template <typename Fn>
void measure_phase(timing_totals *timings, double timing_totals::*field, Fn &&fn) {
  if (!timings) {
    fn();
    return;
  }
  const auto begin = steady_clock::now();
  fn();
  const auto end = steady_clock::now();
  timings->*field += elapsed_seconds(begin, end);
}

bool converter_cuda_supported_qtype(ggml_type qtype) {
  switch (qtype) {
  case GGML_TYPE_Q4_0:
  case GGML_TYPE_Q8_0:
  case GGML_TYPE_Q2_K:
  case GGML_TYPE_Q3_K:
  case GGML_TYPE_Q4_K:
  case GGML_TYPE_Q5_K:
  case GGML_TYPE_Q6_K:
    return true;
  default:
    return false;
  }
}

bool converter_auto_prefers_cuda_qtype(ggml_type qtype) {
  switch (qtype) {
  case GGML_TYPE_Q2_K:
  case GGML_TYPE_Q3_K:
  case GGML_TYPE_Q4_K:
  case GGML_TYPE_Q5_K:
  case GGML_TYPE_Q6_K:
    return true;
  default:
    return false;
  }
}

bool conversion_plans_prefer_cuda(const std::vector<tensor_plan> &plans) {
  for (const tensor_plan &plan : plans) {
    if (is_quantized_qtype(plan.qtype) && converter_auto_prefers_cuda_qtype(plan.qtype)) {
      return true;
    }
  }
  return false;
}

class cuda_converter_context;

bool conversion_plan_requests_cuda(const tensor_plan &plan, const cli_options &options) {
  if (!is_quantized_qtype(plan.qtype)) {
    return false;
  }
  return options.backend == CONVERTER_BACKEND_CUDA ||
         (options.backend == CONVERTER_BACKEND_AUTO && converter_auto_prefers_cuda_qtype(plan.qtype));
}

#if defined(LIBGGUF_HAS_CUDA_NATIVE)
std::string cuda_status_message(libgguf_cuda_context *ctx, int status, const std::string &prefix) {
  std::string message = prefix;
  if (status != LIBGGUF_CUDA_STATUS_SUCCESS) {
    const char *last_error = libgguf_cuda_last_error(ctx);
    if (last_error && last_error[0]) {
      message += ": ";
      message += last_error;
    }
  }
  return message;
}

void require_cuda_success(libgguf_cuda_context *ctx, int status, const std::string &prefix) {
  if (status != LIBGGUF_CUDA_STATUS_SUCCESS) {
    fail(cuda_status_message(ctx, status, prefix));
  }
}

class cuda_converter_context {
public:
  cuda_converter_context() {
    libgguf_cuda_context *created = nullptr;
    const int status = libgguf_cuda_context_create(0, &created);
    if (status != LIBGGUF_CUDA_STATUS_SUCCESS) {
      std::string message = "failed to initialize CUDA backend";
      if (created) {
        const char *last_error = libgguf_cuda_last_error(created);
        if (last_error && last_error[0]) {
          message += ": ";
          message += last_error;
        }
        libgguf_cuda_context_destroy(created);
      }
      fail(message);
    }
    ctx_ = created;
    try {
      require_cuda_success(ctx_, libgguf_cuda_buffer_create(ctx_, 0, &input_), "failed to create CUDA input buffer");
      require_cuda_success(ctx_, libgguf_cuda_buffer_create(ctx_, 0, &output_), "failed to create CUDA output buffer");
      for (cuda_slot &slot : slots_) {
        require_cuda_success(ctx_, libgguf_cuda_host_buffer_create(ctx_, 0, &slot.host_input), "failed to create pinned CUDA input buffer");
        require_cuda_success(ctx_, libgguf_cuda_host_buffer_create(ctx_, 0, &slot.host_output), "failed to create pinned CUDA output buffer");
        require_cuda_success(ctx_, libgguf_cuda_event_create(ctx_, &slot.event_begin), "failed to create CUDA timing event");
        require_cuda_success(ctx_, libgguf_cuda_event_create(ctx_, &slot.event_after_h2d), "failed to create CUDA timing event");
        require_cuda_success(ctx_, libgguf_cuda_event_create(ctx_, &slot.event_after_quant), "failed to create CUDA timing event");
        require_cuda_success(ctx_, libgguf_cuda_event_create(ctx_, &slot.event_after_d2h), "failed to create CUDA timing event");
      }
    } catch (...) {
      cleanup();
      throw;
    }
  }

  cuda_converter_context(const cuda_converter_context &) = delete;
  cuda_converter_context &operator=(const cuda_converter_context &) = delete;

  ~cuda_converter_context() {
    cleanup();
  }

  bool supports(ggml_type qtype, uint64_t n_per_row) const {
    return converter_cuda_supported_qtype(qtype) &&
           libgguf_cuda_qtype_supported((int64_t)qtype) &&
           !libgguf_cuda_qtype_needs_imatrix((int64_t)qtype) &&
           libgguf_cuda_row_size((int64_t)qtype, (int64_t)n_per_row) > 0;
  }

  void resize_host_slot(size_t slot_index, size_t input_bytes, size_t output_bytes) {
    cuda_slot &slot = slot_at(slot_index);
    require_cuda_success(ctx_, libgguf_cuda_host_buffer_resize_discard(ctx_, slot.host_input, input_bytes), "failed to resize pinned CUDA input buffer");
    require_cuda_success(ctx_, libgguf_cuda_host_buffer_resize_discard(ctx_, slot.host_output, output_bytes), "failed to resize pinned CUDA output buffer");
  }

  float *host_input_data(size_t slot_index) {
    return (float *)libgguf_cuda_host_buffer_data(slot_at(slot_index).host_input);
  }

  uint8_t *host_output_data(size_t slot_index) {
    return (uint8_t *)libgguf_cuda_host_buffer_data(slot_at(slot_index).host_output);
  }

  void resize_device_buffers(size_t input_bytes, size_t output_bytes) {
    require_cuda_success(ctx_, libgguf_cuda_buffer_resize_discard(ctx_, input_, input_bytes), "failed to resize CUDA input buffer");
    require_cuda_success(ctx_, libgguf_cuda_buffer_resize_discard(ctx_, output_, output_bytes), "failed to resize CUDA output buffer");
  }

  void quantize_chunk(
      const tensor_plan &plan,
      const float *host_input,
      uint64_t row_count,
      uint8_t *host_output,
      size_t output_bytes,
      timing_totals *timings) {
    enqueue_quantize_chunk(0, plan, host_input, row_count, host_output, output_bytes, timings);
    finish_quantize_chunk(0, timings);
  }

  void enqueue_quantize_chunk(
      size_t slot_index,
      const tensor_plan &plan,
      const float *host_input,
      uint64_t row_count,
      uint8_t *host_output,
      size_t output_bytes,
      timing_totals *timings) {
    cuda_slot &slot = slot_at(slot_index);
    const size_t input_bytes = (size_t)(row_count * plan.n_per_row * sizeof(float));
    if (timings) {
      timings->cuda_chunks += 1;
      timings->cuda_max_input_bytes = std::max<uint64_t>(timings->cuda_max_input_bytes, (uint64_t)input_bytes);
      timings->cuda_max_output_bytes = std::max<uint64_t>(timings->cuda_max_output_bytes, (uint64_t)output_bytes);
    }
    if (timings) {
      require_cuda_success(ctx_, libgguf_cuda_event_record(ctx_, slot.event_begin), "failed to record CUDA timing event");
    }
    require_cuda_success(ctx_, libgguf_cuda_buffer_resize_discard(ctx_, input_, input_bytes), "failed to resize CUDA input buffer");
    require_cuda_success(ctx_, libgguf_cuda_h2d(ctx_, input_, 0, host_input, input_bytes), "failed to copy tensor chunk to CUDA");
    if (timings) {
      require_cuda_success(ctx_, libgguf_cuda_event_record(ctx_, slot.event_after_h2d), "failed to record CUDA timing event");
    }
    require_cuda_success(ctx_, libgguf_cuda_buffer_resize_discard(ctx_, output_, output_bytes), "failed to resize CUDA output buffer");
    require_cuda_success(
        ctx_,
        libgguf_cuda_quantize_f32_rows(
            ctx_,
            (const float *)libgguf_cuda_buffer_const_data(input_),
            nullptr,
            libgguf_cuda_buffer_data(output_),
            (int64_t)plan.qtype,
            (int64_t)row_count,
            (int64_t)plan.n_per_row),
        "failed to quantize tensor chunk on CUDA");
    if (timings) {
      require_cuda_success(ctx_, libgguf_cuda_event_record(ctx_, slot.event_after_quant), "failed to record CUDA timing event");
    }
    require_cuda_success(ctx_, libgguf_cuda_d2h(ctx_, host_output, output_, 0, output_bytes), "failed to copy tensor chunk from CUDA");
    require_cuda_success(ctx_, libgguf_cuda_event_record(ctx_, slot.event_after_d2h), "failed to record CUDA completion event");
  }

  void finish_quantize_chunk(size_t slot_index, timing_totals *timings) {
    cuda_slot &slot = slot_at(slot_index);
    require_cuda_success(ctx_, libgguf_cuda_event_synchronize(ctx_, slot.event_after_d2h), "failed to synchronize CUDA download");
    if (timings) {
      add_elapsed_ms(&timing_totals::h2d_s, slot.event_begin, slot.event_after_h2d, timings);
      add_elapsed_ms(&timing_totals::cuda_quant_s, slot.event_after_h2d, slot.event_after_quant, timings);
      add_elapsed_ms(&timing_totals::d2h_s, slot.event_after_quant, slot.event_after_d2h, timings);
    }
  }

private:
  struct cuda_slot {
    libgguf_cuda_host_buffer *host_input = nullptr;
    libgguf_cuda_host_buffer *host_output = nullptr;
    libgguf_cuda_event *event_begin = nullptr;
    libgguf_cuda_event *event_after_h2d = nullptr;
    libgguf_cuda_event *event_after_quant = nullptr;
    libgguf_cuda_event *event_after_d2h = nullptr;
  };

  void cleanup() {
    for (cuda_slot &slot : slots_) {
      if (slot.event_after_d2h) {
        libgguf_cuda_event_destroy(ctx_, slot.event_after_d2h);
        slot.event_after_d2h = nullptr;
      }
      if (slot.event_after_quant) {
        libgguf_cuda_event_destroy(ctx_, slot.event_after_quant);
        slot.event_after_quant = nullptr;
      }
      if (slot.event_after_h2d) {
        libgguf_cuda_event_destroy(ctx_, slot.event_after_h2d);
        slot.event_after_h2d = nullptr;
      }
      if (slot.event_begin) {
        libgguf_cuda_event_destroy(ctx_, slot.event_begin);
        slot.event_begin = nullptr;
      }
      if (slot.host_output) {
        libgguf_cuda_host_buffer_destroy(ctx_, slot.host_output);
        slot.host_output = nullptr;
      }
      if (slot.host_input) {
        libgguf_cuda_host_buffer_destroy(ctx_, slot.host_input);
        slot.host_input = nullptr;
      }
    }
    if (output_) {
      libgguf_cuda_buffer_destroy(ctx_, output_);
      output_ = nullptr;
    }
    if (input_) {
      libgguf_cuda_buffer_destroy(ctx_, input_);
      input_ = nullptr;
    }
    if (ctx_) {
      libgguf_cuda_context_destroy(ctx_);
      ctx_ = nullptr;
    }
  }

  cuda_slot &slot_at(size_t slot_index) {
    if (slot_index >= 2) {
      fail("invalid CUDA pipeline slot index");
    }
    return slots_[slot_index];
  }

  void add_elapsed_ms(double timing_totals::*field, libgguf_cuda_event *begin, libgguf_cuda_event *end, timing_totals *timings) {
    float elapsed_ms = 0.0f;
    require_cuda_success(ctx_, libgguf_cuda_event_elapsed_ms(ctx_, begin, end, &elapsed_ms), "failed to measure CUDA timing event");
    timings->*field += (double)elapsed_ms / 1000.0;
  }

  libgguf_cuda_context *ctx_ = nullptr;
  libgguf_cuda_buffer *input_ = nullptr;
  libgguf_cuda_buffer *output_ = nullptr;
  cuda_slot slots_[2];
};
#endif

bool conversion_plan_uses_cuda(const tensor_plan &plan, const cli_options &options, cuda_converter_context *cuda_ctx) {
  if (!conversion_plan_requests_cuda(plan, options)) {
    return false;
  }
#if defined(LIBGGUF_HAS_CUDA_NATIVE)
  return cuda_ctx && cuda_ctx->supports(plan.qtype, plan.n_per_row);
#else
  (void)cuda_ctx;
  return false;
#endif
}

std::set<std::string> select_cuda_verification_tensor_keys(
    const std::vector<tensor_plan> &plans,
    const cli_options &options,
    cuda_converter_context *cuda_ctx) {
  struct largest_candidate {
    uint64_t expected_nbytes = 0;
    std::string key;
    size_t index = 0;
  };
  std::set<std::string> selected;
  std::vector<largest_candidate> largest_candidates;
  std::vector<std::string> random_candidates;
  for (size_t i = 0; i < plans.size(); ++i) {
    if (conversion_plan_uses_cuda(plans[i], options, cuda_ctx)) {
      largest_candidates.push_back({plans[i].expected_nbytes, plans[i].key, i});
      random_candidates.push_back(plans[i].key);
    }
  }
  std::sort(largest_candidates.begin(), largest_candidates.end(), [](const largest_candidate &a, const largest_candidate &b) {
    if (a.expected_nbytes != b.expected_nbytes) {
      return a.expected_nbytes > b.expected_nbytes;
    }
    if (a.key != b.key) {
      return a.key < b.key;
    }
    return a.index < b.index;
  });
  const size_t selected_count =
      (size_t)std::min<uint64_t>(options.verify_cuda_large_tensors, (uint64_t)largest_candidates.size());
  for (size_t i = 0; i < selected_count; ++i) {
    selected.insert(largest_candidates[i].key);
  }
  if (options.verify_cuda_random_tensors > 0 && !random_candidates.empty()) {
    std::mt19937_64 rng(options.seed);
    std::shuffle(random_candidates.begin(), random_candidates.end(), rng);
    const size_t random_selected_count =
        (size_t)std::min<uint64_t>(options.verify_cuda_random_tensors, (uint64_t)random_candidates.size());
    for (size_t i = 0; i < random_selected_count; ++i) {
      selected.insert(random_candidates[i]);
    }
  }
  return selected;
}

void write_tensor_payload(
    FILE *out,
    const input_file &input,
    const tensor_plan &plan,
    const cli_options &options,
    tensor_quant_worker_pool &native_pool,
    cuda_converter_context *cuda_ctx,
    uint64_t *verify_cuda_remaining,
    const std::set<std::string> &verify_cuda_selected_keys,
    timing_totals *timings) {
  if (timings) {
    timings->tensors += 1;
  }
  const uint64_t scratch_bytes = options.scratch_bytes;
  write_pre_tensor_padding(out, GGUF_DEFAULT_ALIGNMENT);
  if (qtype_matches_source(plan.source_dtype, plan.qtype)) {
    const uint64_t buffer_bytes = std::max<uint64_t>(1, std::min<uint64_t>(scratch_bytes, plan.expected_nbytes));
    std::vector<uint8_t> buffer((size_t)buffer_bytes);
    uint64_t remaining = plan.expected_nbytes;
    uint64_t offset = plan.data_begin;
    while (remaining > 0) {
      const uint64_t chunk = std::min<uint64_t>(remaining, buffer.size());
      std::string error;
      bool read_ok = false;
      measure_phase(timings, &timing_totals::read_s, [&]() {
        read_ok = input.read_at(offset, buffer.data(), chunk, &error);
      });
      if (!read_ok) {
        fail(error + " for " + plan.key);
      }
      measure_phase(timings, &timing_totals::write_s, [&]() {
        write_all_file(out, buffer.data(), chunk);
      });
      offset += chunk;
      remaining -= chunk;
    }
    write_post_tensor_padding(out, plan.expected_nbytes, GGUF_DEFAULT_ALIGNMENT);
    return;
  }

  const size_t row_size = libgguf_row_size(plan.qtype, (int64_t)plan.n_per_row);
  if (row_size == 0) {
    fail("unsupported qtype or row width for " + plan.key);
  }
  const bool source_f32 = plan.source_dtype == NATIVE_DTYPE_F32;
  const bool native_quant = is_quantized_qtype(plan.qtype) && native_quantize_function(plan.qtype) != nullptr;
  const bool cuda_quant = conversion_plan_uses_cuda(plan, options, cuda_ctx);
  if (conversion_plan_requests_cuda(plan, options)) {
    if (options.backend == CONVERTER_BACKEND_CUDA && !cuda_quant && !options.cuda_fallback_cpu) {
      fail(
          "CUDA backend does not support " + qtype_name(plan.qtype) +
          " for tensor " + plan.key + "; pass --cuda-fallback cpu to use CPU for unsupported tensors");
    }
  }
  const bool verify_cuda_first =
      cuda_quant && verify_cuda_remaining && *verify_cuda_remaining > 0;
  const bool verify_cuda_selected =
      cuda_quant && verify_cuda_selected_keys.find(plan.key) != verify_cuda_selected_keys.end();
  const bool verify_cuda = verify_cuda_first || verify_cuda_selected;
  if (cuda_quant && timings) {
    timings->cuda_tensors += 1;
    if (verify_cuda) {
      timings->cuda_verified_tensors += 1;
    }
  }
  const uint64_t row_work_bytes =
      plan.n_per_row * source_dtype_size(plan.source_dtype) +
      (source_f32 ? 0 : plan.n_per_row * sizeof(float)) +
      (uint64_t)row_size;
  uint64_t rows_per_chunk = row_work_bytes == 0 ? 1 : scratch_bytes / row_work_bytes;
  if (rows_per_chunk == 0) {
    rows_per_chunk = 1;
  }
  if (cuda_quant && options.cuda_vram_bytes > 0) {
    const uint64_t device_row_bytes = plan.n_per_row * sizeof(float) + (uint64_t)row_size;
    rows_per_chunk = device_row_bytes == 0 ? 1 : options.cuda_vram_bytes / device_row_bytes;
    if (rows_per_chunk == 0) {
      rows_per_chunk = 1;
    }
  }
  rows_per_chunk = std::min<uint64_t>(rows_per_chunk, plan.n_rows);

#if defined(LIBGGUF_HAS_CUDA_NATIVE)
  if (cuda_quant && !verify_cuda && options.cuda_pipeline) {
    struct cuda_pipeline_slot_state {
      uint64_t row = 0;
      uint64_t row_count = 0;
      size_t expected = 0;
      std::vector<uint8_t> source_bytes;
    };

    const size_t max_input_bytes = (size_t)(rows_per_chunk * plan.n_per_row * sizeof(float));
    const size_t max_output_bytes = (size_t)(rows_per_chunk * (uint64_t)row_size);
    cuda_ctx->resize_device_buffers(max_input_bytes, max_output_bytes);
    cuda_ctx->resize_host_slot(0, max_input_bytes, max_output_bytes);
    cuda_ctx->resize_host_slot(1, max_input_bytes, max_output_bytes);

    cuda_pipeline_slot_state slots[2];
    auto prepare_and_enqueue = [&](size_t slot_index, uint64_t row) {
      cuda_pipeline_slot_state &slot = slots[slot_index];
      slot.row = row;
      slot.row_count = std::min<uint64_t>(rows_per_chunk, plan.n_rows - row);
      slot.expected = (size_t)(slot.row_count * (uint64_t)row_size);
      float *host_f32 = cuda_ctx->host_input_data(slot_index);
      const uint64_t value_begin = row * plan.n_per_row;
      const uint64_t value_count = slot.row_count * plan.n_per_row;
      if (source_f32) {
        const uint64_t byte_count = value_count * sizeof(float);
        std::string error;
        bool read_ok = false;
        measure_phase(timings, &timing_totals::read_s, [&]() {
          read_ok = input.read_at(plan.data_begin + value_begin * sizeof(float), host_f32, byte_count, &error);
        });
        if (!read_ok) {
          fail(error + " for " + plan.key);
        }
      } else {
        measure_phase(timings, &timing_totals::read_s, [&]() {
          read_source_chunk(input, plan, row, slot.row_count, slot.source_bytes);
        });
        measure_phase(timings, &timing_totals::cpu_convert_s, [&]() {
          decode_source_bytes_to_f32(plan, slot.source_bytes.data(), value_count, host_f32);
        });
      }
      cuda_ctx->enqueue_quantize_chunk(
          slot_index,
          plan,
          host_f32,
          slot.row_count,
          cuda_ctx->host_output_data(slot_index),
          slot.expected,
          timings);
    };
    auto finish_and_write = [&](size_t slot_index) {
      cuda_pipeline_slot_state &slot = slots[slot_index];
      cuda_ctx->finish_quantize_chunk(slot_index, timings);
      measure_phase(timings, &timing_totals::write_s, [&]() {
        write_all_file(out, cuda_ctx->host_output_data(slot_index), slot.expected);
      });
    };

    size_t active_slot = 0;
    uint64_t row = 0;
    prepare_and_enqueue(active_slot, row);
    row += slots[active_slot].row_count;
    while (row < plan.n_rows) {
      const size_t next_slot = 1 - active_slot;
      prepare_and_enqueue(next_slot, row);
      row += slots[next_slot].row_count;
      finish_and_write(active_slot);
      active_slot = next_slot;
    }
    finish_and_write(active_slot);
    write_post_tensor_padding(out, plan.expected_nbytes, GGUF_DEFAULT_ALIGNMENT);
    return;
  }
#endif

  std::vector<float> scratch;
  if (!source_f32) {
    scratch.resize((size_t)(rows_per_chunk * plan.n_per_row));
  }
  std::vector<uint8_t> source_bytes;
  std::vector<uint8_t> encoded((size_t)(rows_per_chunk * row_size));
  std::vector<uint8_t> verify_encoded;
  if (verify_cuda) {
    verify_encoded.resize(encoded.size());
  }
  for (uint64_t row = 0; row < plan.n_rows; row += rows_per_chunk) {
    const uint64_t row_count = std::min<uint64_t>(rows_per_chunk, plan.n_rows - row);
    measure_phase(timings, &timing_totals::read_s, [&]() {
      read_source_chunk(input, plan, row, row_count, source_bytes);
    });
    const float *chunk_src = nullptr;
    if (cuda_quant) {
#if defined(LIBGGUF_HAS_CUDA_NATIVE)
      measure_phase(timings, &timing_totals::cpu_convert_s, [&]() {
        if (source_f32) {
          chunk_src = reinterpret_cast<const float *>(source_bytes.data());
        } else {
          decode_source_bytes_to_f32(plan, source_bytes.data(), row_count * plan.n_per_row, scratch.data());
          chunk_src = scratch.data();
        }
      });
      const size_t expected = (size_t)(row_count * (uint64_t)row_size);
      cuda_ctx->quantize_chunk(plan, chunk_src, row_count, encoded.data(), expected, timings);
      if (verify_cuda) {
        const bool quant_ok = native_pool.run(plan, source_bytes.data(), row_count, verify_encoded.data(), nullptr, scratch_bytes);
        if (!quant_ok || std::memcmp(encoded.data(), verify_encoded.data(), expected) != 0) {
          fail("CUDA verification mismatch for " + plan.key + " at row " + std::to_string(row));
        }
      }
      measure_phase(timings, &timing_totals::write_s, [&]() {
        write_all_file(out, encoded.data(), expected);
      });
      continue;
#endif
    }
    if (native_quant) {
      bool quant_ok = false;
      measure_phase(timings, &timing_totals::cpu_convert_s, [&]() {
        quant_ok = native_pool.run(plan, source_bytes.data(), row_count, encoded.data(), nullptr, scratch_bytes);
      });
      if (!quant_ok) {
        fail("native quantization returned an unexpected byte count for " + plan.key);
      }
      measure_phase(timings, &timing_totals::write_s, [&]() {
        write_all_file(out, encoded.data(), row_count * (uint64_t)row_size);
      });
      continue;
    }
    size_t written = 0;
    const uint64_t expected = row_count * (uint64_t)row_size;
    measure_phase(timings, &timing_totals::cpu_convert_s, [&]() {
      if (source_f32) {
        chunk_src = reinterpret_cast<const float *>(source_bytes.data());
      } else {
        decode_source_bytes_to_f32(plan, source_bytes.data(), row_count * plan.n_per_row, scratch.data());
        chunk_src = scratch.data();
      }
      written = libgguf_quantize_chunk(plan.qtype, chunk_src, encoded.data(), 0, (int64_t)row_count, (int64_t)plan.n_per_row, nullptr);
    });
    if (written != expected) {
      fail("libgguf_quantize_chunk returned an unexpected byte count for " + plan.key);
    }
    measure_phase(timings, &timing_totals::write_s, [&]() {
      write_all_file(out, encoded.data(), expected);
    });
  }
  if (verify_cuda_first && verify_cuda_remaining && *verify_cuda_remaining != VERIFY_CUDA_ALL_TENSORS) {
    *verify_cuda_remaining -= 1;
  }
  write_post_tensor_padding(out, plan.expected_nbytes, GGUF_DEFAULT_ALIGNMENT);
}

void parse_tensor_type_arg(const std::string &value, std::vector<std::pair<std::string, std::string>> &out) {
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
  std::puts("  --backend cpu|cuda|auto    Conversion backend for quantized tensors (default: auto)");
  std::puts("  --cuda-fallback cpu        Use CPU for tensors unsupported by the CUDA converter");
  std::puts("  --verify-cuda-tensors N|all");
  std::puts("                           Compare CUDA tensors against CPU bytes");
  std::puts("  --verify-cuda-large-tensors N");
  std::puts("                           Compare the N largest CUDA tensors against CPU bytes");
  std::puts("  --verify-cuda-random-tensors N");
  std::puts("                           Compare N seeded random CUDA tensors against CPU bytes");
  std::puts("  --seed N                  Seed for random CUDA tensor verification (default: 0)");
  std::puts("  --cuda-vram-bytes N        CUDA device chunk budget in bytes (0 uses --scratch-bytes)");
  std::puts("  --cuda-batch-mb N          CUDA device chunk budget in MiB (alias for --cuda-vram-bytes)");
  std::puts("  --cuda-pipeline 0|1        Use two-slot CUDA host pipeline for unverified tensors (default: 1)");
  std::puts("  --timings                  Print conversion timing breakdown to stderr");
  std::puts("  --help                     Show this help");
}

cli_options parse_args(int argc, char **argv) {
  cli_options options;
  auto parse_positive_bytes = [](const std::string &value, const char *name) -> uint64_t {
    char *end = nullptr;
    errno = 0;
    unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
    if (errno != 0 || end == value.c_str() || *end != '\0' || parsed == 0) {
      fail(std::string(name) + " must be positive");
    }
    return (uint64_t)parsed;
  };
  auto parse_non_negative_integer = [](const std::string &value, const char *name) -> uint64_t {
    char *end = nullptr;
    errno = 0;
    unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
    if (errno != 0 || end == value.c_str() || *end != '\0' || value[0] == '-') {
      fail(std::string(name) + " must be a non-negative integer");
    }
    return (uint64_t)parsed;
  };
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto need_value = [&](const char *name) -> std::string {
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
      } else if (value == "cuda") {
        options.backend = CONVERTER_BACKEND_CUDA;
      } else {
        fail("--backend must be 'cpu', 'cuda', or 'auto'");
      }
    } else if (arg == "--cuda-fallback") {
      std::string value = need_value("--cuda-fallback");
      if (value != "cpu") {
        fail("--cuda-fallback must be 'cpu'");
      }
      options.cuda_fallback_cpu = true;
    } else if (arg == "--verify-cuda-tensors") {
      std::string value = need_value("--verify-cuda-tensors");
      if (value == "all") {
        options.verify_cuda_tensors = VERIFY_CUDA_ALL_TENSORS;
      } else {
        try {
          options.verify_cuda_tensors = parse_non_negative_integer(value, "--verify-cuda-tensors");
        } catch (...) {
          fail("--verify-cuda-tensors must be a non-negative integer or 'all'");
        }
      }
    } else if (arg == "--verify-cuda-large-tensors") {
      options.verify_cuda_large_tensors =
          parse_non_negative_integer(need_value("--verify-cuda-large-tensors"), "--verify-cuda-large-tensors");
    } else if (arg == "--verify-cuda-random-tensors") {
      options.verify_cuda_random_tensors =
          parse_non_negative_integer(need_value("--verify-cuda-random-tensors"), "--verify-cuda-random-tensors");
    } else if (arg == "--seed") {
      options.seed = parse_non_negative_integer(need_value("--seed"), "--seed");
    } else if (arg == "--cuda-vram-bytes") {
      options.cuda_vram_bytes = parse_non_negative_integer(need_value("--cuda-vram-bytes"), "--cuda-vram-bytes");
    } else if (arg == "--cuda-batch-mb") {
      const uint64_t mib = parse_non_negative_integer(need_value("--cuda-batch-mb"), "--cuda-batch-mb");
      if (mib > std::numeric_limits<uint64_t>::max() / (1024ULL * 1024ULL)) {
        fail("--cuda-batch-mb is too large");
      }
      options.cuda_vram_bytes = mib * 1024ULL * 1024ULL;
    } else if (arg == "--cuda-pipeline") {
      const std::string value = need_value("--cuda-pipeline");
      if (value == "0") {
        options.cuda_pipeline = false;
      } else if (value == "1") {
        options.cuda_pipeline = true;
      } else {
        fail("--cuda-pipeline must be 0 or 1");
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
  if (options.verify_cuda_tensors > 0 && options.backend == CONVERTER_BACKEND_CPU) {
    fail("--verify-cuda-tensors requires --backend cuda or auto");
  }
  if (options.verify_cuda_large_tensors > 0 && options.backend == CONVERTER_BACKEND_CPU) {
    fail("--verify-cuda-large-tensors requires --backend cuda or auto");
  }
  if (options.verify_cuda_random_tensors > 0 && options.backend == CONVERTER_BACKEND_CPU) {
    fail("--verify-cuda-random-tensors requires --backend cuda or auto");
  }
  if (options.cuda_vram_bytes > 0 && options.backend == CONVERTER_BACKEND_CPU) {
    fail("--cuda-vram-bytes requires --backend cuda or auto");
  }
  return options;
}

conversion_result convert(const cli_options &options) {
#if !defined(LIBGGUF_HAS_CUDA_NATIVE)
  if (options.backend == CONVERTER_BACKEND_CUDA) {
    fail("CUDA backend requested, but this executable was built without native CUDA support");
  }
#endif
  const auto total_begin = steady_clock::now();
  auto [file_type_name, base_qtype_name] = parse_qtype(options.qtype);
  std::string dst = options.dst.empty() ? default_output_path(options.src, file_type_name) : options.dst;
  if (file_exists(dst) && !options.overwrite) {
    fail("Output exists and overwriting is disabled: " + dst);
  }

  input_file input;
  std::string input_error;
  if (!input.open(options.src, &input_error)) {
    fail(input_error);
  }
  uint64_t data_start = 0;
  std::vector<tensor_meta> tensors = parse_safetensors_header(input, &data_start);
  (void)data_start;
  strip_prefix(tensors);
  model_template model = detect_arch(tensors);

  conversion_result result;
  result.output_path = dst;
  result.arch = model.arch;
  result.file_type_name = "MOSTLY_" + file_type_name;

  std::vector<kv_value> kv;
  kv.push_back({"general.architecture", GGUF_TYPE_STRING, 0, model.arch, {}});
  kv.push_back({"general.quantization_version", GGUF_TYPE_UINT32, GGML_QUANT_VERSION, {}, {}});
  kv.push_back({"general.file_type", GGUF_TYPE_UINT32, file_type_value(file_type_name), {}, {}});

  std::vector<tensor_info> infos;
  std::vector<tensor_plan> plans = prepare_plans(tensors, model, file_type_name, base_qtype_name, options, kv, infos, result);
  timing_totals timings;
  timing_totals *timings_ptr = options.timings ? &timings : nullptr;
  const unsigned int worker_threads = native_thread_limit(options.threads);
  tensor_quant_worker_pool native_pool(worker_threads);
#if defined(LIBGGUF_HAS_CUDA_NATIVE)
  std::unique_ptr<cuda_converter_context> cuda_ctx;
  if (options.backend == CONVERTER_BACKEND_CUDA) {
    cuda_ctx.reset(new cuda_converter_context());
  } else if (options.backend == CONVERTER_BACKEND_AUTO && conversion_plans_prefer_cuda(plans)) {
    try {
      cuda_ctx.reset(new cuda_converter_context());
    } catch (const std::runtime_error &) {
      cuda_ctx.reset();
    }
  }
#else
  cuda_converter_context *cuda_ctx = nullptr;
#endif
  const std::set<std::string> verify_cuda_selected_keys = select_cuda_verification_tensor_keys(
      plans,
      options,
#if defined(LIBGGUF_HAS_CUDA_NATIVE)
      cuda_ctx.get()
#else
      cuda_ctx
#endif
  );
  uint64_t verify_cuda_remaining = options.verify_cuda_tensors;

#if defined(_WIN32)
  FILE *out = nullptr;
  fopen_s(&out, dst.c_str(), "wb");
#else
  FILE *out = std::fopen(dst.c_str(), "wb");
#endif
  if (!out) {
    fail("failed to open output file: " + dst);
  }
  try {
    const auto metadata_begin = steady_clock::now();
    write_gguf_metadata(out, kv, infos);
    const auto metadata_end = steady_clock::now();
    if (timings_ptr) {
      timings.metadata_s += elapsed_seconds(metadata_begin, metadata_end);
    }
    for (const tensor_plan &plan : plans) {
      write_tensor_payload(
          out,
          input,
          plan,
          options,
          native_pool,
#if defined(LIBGGUF_HAS_CUDA_NATIVE)
          cuda_ctx.get(),
#else
          cuda_ctx,
#endif
          &verify_cuda_remaining,
          verify_cuda_selected_keys,
          timings_ptr);
    }
    if (std::fclose(out) != 0) {
      out = nullptr;
      fail("failed to close output file");
    }
    out = nullptr;
  } catch (...) {
    if (out) {
      std::fclose(out);
    }
    throw;
  }
  if (timings_ptr) {
    timings.total_s = elapsed_seconds(total_begin, steady_clock::now());
    std::fprintf(
        stderr,
        "Timings: total=%.3fs metadata=%.3fs read=%.3fs cpu_convert=%.3fs h2d=%.3fs cuda_quant=%.3fs d2h=%.3fs write=%.3fs tensors=%llu cuda_tensors=%llu cuda_verified=%llu cuda_chunks=%llu threads=%u scratch=%llu cuda_vram=%llu cuda_pipeline=%u cuda_max_input=%llu cuda_max_output=%llu\n",
        timings.total_s,
        timings.metadata_s,
        timings.read_s,
        timings.cpu_convert_s,
        timings.h2d_s,
        timings.cuda_quant_s,
        timings.d2h_s,
        timings.write_s,
        (unsigned long long)timings.tensors,
        (unsigned long long)timings.cuda_tensors,
        (unsigned long long)timings.cuda_verified_tensors,
        (unsigned long long)timings.cuda_chunks,
        worker_threads,
        (unsigned long long)options.scratch_bytes,
        (unsigned long long)options.cuda_vram_bytes,
        options.cuda_pipeline ? 1U : 0U,
        (unsigned long long)timings.cuda_max_input_bytes,
        (unsigned long long)timings.cuda_max_output_bytes);
  }
  return result;
}

} // namespace

int main(int argc, char **argv) {
  try {
    cli_options options = parse_args(argc, argv);
    conversion_result result = convert(options);
    std::printf("Wrote %s\n", result.output_path.c_str());
    std::printf("Architecture: %s\n", result.arch.c_str());
    std::printf("File type: %s\n", result.file_type_name.c_str());
    std::printf("Tensor types: ");
    bool first = true;
    for (const auto &item : result.tensor_type_counts) {
      if (!first) {
        std::printf(", ");
      }
      first = false;
      std::printf("%s=%d", item.first.c_str(), item.second);
    }
    std::printf("\n");
    if (!result.fallback_counts.empty()) {
      std::printf("Fallbacks: ");
      first = true;
      for (const auto &item : result.fallback_counts) {
        if (!first) {
          std::printf(", ");
        }
        first = false;
        std::printf("%s=%d", item.first.c_str(), item.second);
      }
      std::printf("\n");
    }
    return 0;
  } catch (const std::exception &exc) {
    std::fprintf(stderr, "error: %s\n", exc.what());
    return 1;
  }
}
