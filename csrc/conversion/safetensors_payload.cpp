#include "conversion/safetensors_payload.h"

#include "kernels/cpu/common/common.h"
#include "kernels/cpu/quantize/kernels.h"
#include <libgguf/libgguf.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <emmintrin.h>
#define LIBGGUF_PAYLOAD_SSE2 1
#endif

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

bool parse_payload_source_dtype(const char *name, payload_source_dtype *out)
{
  if (std::strcmp(name, "F32") == 0) { *out = PAYLOAD_DTYPE_F32; return true; }
  if (std::strcmp(name, "F16") == 0) { *out = PAYLOAD_DTYPE_F16; return true; }
  if (std::strcmp(name, "BF16") == 0) { *out = PAYLOAD_DTYPE_BF16; return true; }
  return false;
}

size_t payload_source_dtype_size(payload_source_dtype dtype)
{
  switch (dtype)
  {
  case PAYLOAD_DTYPE_F32: return sizeof(float);
  case PAYLOAD_DTYPE_F16:
  case PAYLOAD_DTYPE_BF16: return sizeof(uint16_t);
  }
  return 0;
}

static bool qtype_matches_source(payload_source_dtype dtype, ggml_type qtype)
{
  return (dtype == PAYLOAD_DTYPE_F32 && qtype == GGML_TYPE_F32) ||
         (dtype == PAYLOAD_DTYPE_F16 && qtype == GGML_TYPE_F16) ||
         (dtype == PAYLOAD_DTYPE_BF16 && qtype == GGML_TYPE_BF16);
}

static unsigned int payload_thread_count(int64_t nrows)
{
  if (nrows < 64) return 1;
  const char *env = std::getenv("LIBGGUF_NUM_THREADS");
  if (env != nullptr && env[0] != '\0')
  {
    errno = 0;
    char *end = nullptr;
    const long long parsed = std::strtoll(env, &end, 10);
    if (errno == 0 && end != env && parsed > 0)
      return (unsigned int)std::min<int64_t>((int64_t)parsed, nrows);
    return 1;
  }
  const unsigned int hardware = std::thread::hardware_concurrency();
  return hardware == 0 ? 1 : std::min<unsigned int>(hardware, (unsigned int)nrows);
}
struct mapped_file
{
  const uint8_t *data = nullptr;
  uint64_t size = 0;
#if defined(_WIN32)
  HANDLE file = INVALID_HANDLE_VALUE;
  HANDLE mapping = nullptr;
#else
  int fd = -1;
#endif

  bool map(const std::string &path, std::string *error)
  {
#if defined(_WIN32)
    int wide_len = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    if (wide_len <= 0)
    {
      *error = "failed to convert safetensors path to UTF-16";
      return false;
    }
    std::vector<wchar_t> wide_path((size_t)wide_len);
    if (MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wide_path.data(), wide_len) <= 0)
    {
      *error = "failed to convert safetensors path to UTF-16";
      return false;
    }

    file = CreateFileW(wide_path.data(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE)
    {
      *error = "failed to open safetensors file";
      return false;
    }
    LARGE_INTEGER file_size;
    if (!GetFileSizeEx(file, &file_size) || file_size.QuadPart < 0)
    {
      *error = "failed to stat safetensors file";
      return false;
    }
    size = (uint64_t)file_size.QuadPart;
    if (size == 0)
    {
      *error = "safetensors file is empty";
      return false;
    }
    mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mapping)
    {
      *error = "failed to create safetensors file mapping";
      return false;
    }
    data = (const uint8_t *)MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    if (!data)
    {
      *error = "failed to map safetensors file";
      return false;
    }
    return true;
#else
    fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0)
    {
      *error = std::string("failed to open safetensors file: ") + std::strerror(errno);
      return false;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 0)
    {
      *error = std::string("failed to stat safetensors file: ") + std::strerror(errno);
      return false;
    }
    size = (uint64_t)st.st_size;
    if (size == 0)
    {
      *error = "safetensors file is empty";
      return false;
    }
    void *mapped = mmap(nullptr, (size_t)size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED)
    {
      *error = std::string("failed to map safetensors file: ") + std::strerror(errno);
      return false;
    }
    data = (const uint8_t *)mapped;
    return true;
#endif
  }

  void close()
  {
#if defined(_WIN32)
    if (data)
    {
      UnmapViewOfFile(data);
      data = nullptr;
    }
    if (mapping)
    {
      CloseHandle(mapping);
      mapping = nullptr;
    }
    if (file != INVALID_HANDLE_VALUE)
    {
      CloseHandle(file);
      file = INVALID_HANDLE_VALUE;
    }
#else
    if (data)
    {
      munmap((void *)data, (size_t)size);
      data = nullptr;
    }
    if (fd >= 0)
    {
      ::close(fd);
      fd = -1;
    }
#endif
    size = 0;
  }

  ~mapped_file()
  {
    close();
  }
};

static bool fd_current_position(int fd, uint64_t *position, std::string *error)
{
#if defined(_WIN32)
  __int64 result = _lseeki64(fd, 0, SEEK_CUR);
  if (result < 0)
  {
    *error = "failed to query output file position";
    return false;
  }
#else
  off_t result = lseek(fd, 0, SEEK_CUR);
  if (result < 0)
  {
    *error = std::string("failed to query output file position: ") + std::strerror(errno);
    return false;
  }
#endif
  *position = (uint64_t)result;
  return true;
}

static bool write_all_fd(int fd, const void *data, uint64_t len, std::string *error)
{
  const uint8_t *cursor = (const uint8_t *)data;
  while (len > 0)
  {
#if defined(_WIN32)
    const unsigned int chunk = (unsigned int)std::min<uint64_t>(len, (uint64_t)UINT_MAX);
    int written = _write(fd, cursor, chunk);
    if (written < 0)
    {
      *error = "failed to write output file";
      return false;
    }
#else
    const size_t chunk = (size_t)std::min<uint64_t>(len, (uint64_t)std::numeric_limits<size_t>::max());
    ssize_t written = write(fd, cursor, chunk);
    if (written < 0)
    {
      if (errno == EINTR)
      {
        continue;
      }
      *error = std::string("failed to write output file: ") + std::strerror(errno);
      return false;
    }
#endif
    if (written == 0)
    {
      *error = "failed to write output file";
      return false;
    }
    cursor += written;
    len -= (uint64_t)written;
  }
  return true;
}

static bool write_zero_padding(int fd, uint64_t pad, std::string *error)
{
  static const uint8_t zeros[4096] = {};
  while (pad > 0)
  {
    const uint64_t chunk = std::min<uint64_t>(pad, sizeof(zeros));
    if (!write_all_fd(fd, zeros, chunk, error))
    {
      return false;
    }
    pad -= chunk;
  }
  return true;
}

static uint64_t gguf_pad(uint64_t n, uint64_t alignment)
{
  if (alignment == 0)
  {
    return n;
  }
  return ((n + alignment - 1) / alignment) * alignment;
}

static bool write_pre_tensor_padding(int fd, uint64_t alignment, std::string *error)
{
  uint64_t position = 0;
  if (!fd_current_position(fd, &position, error))
  {
    return false;
  }
  return write_zero_padding(fd, gguf_pad(position, alignment) - position, error);
}

static bool write_post_tensor_padding(int fd, uint64_t tensor_nbytes, uint64_t alignment, std::string *error)
{
  return write_zero_padding(fd, gguf_pad(tensor_nbytes, alignment) - tensor_nbytes, error);
}

static void decode_bf16_to_f32(const uint16_t *src, uint64_t count, float *dst)
{
  uint64_t i = 0;
#if defined(LIBGGUF_PAYLOAD_SSE2)
  const __m128i zero = _mm_setzero_si128();
  for (; i + 8 <= count; i += 8)
  {
    const __m128i values = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + i));
    const __m128i lo = _mm_unpacklo_epi16(zero, values);
    const __m128i hi = _mm_unpackhi_epi16(zero, values);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + i), lo);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + i + 4), hi);
  }
#endif
  for (; i < count; ++i)
  {
    ggml_bf16_t value;
    value.bits = src[i];
    dst[i] = LIBGGUF_BF16_TO_FP32(value);
  }
}

#if defined(LIBGGUF_PAYLOAD_SSE2)
static inline float q8_0_hmax_ps_native(__m128 v)
{
  __m128 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
  v = _mm_max_ps(v, shuf);
  shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 0, 3, 2));
  v = _mm_max_ps(v, shuf);
  return _mm_cvtss_f32(v);
}

static inline __m128 bf16_load4_to_ps_native(const uint16_t *src)
{
  const __m128i values = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(src));
  const __m128i expanded = _mm_unpacklo_epi16(_mm_setzero_si128(), values);
  return _mm_castsi128_ps(expanded);
}

static inline __m128i q8_0_round_away_from_zero_native(
    __m128 v,
    __m128 id,
    __m128 sign_mask,
    __m128 half,
    __m128 zero)
{
  const __m128 scaled = _mm_mul_ps(v, id);
  const __m128 abs_scaled = _mm_andnot_ps(sign_mask, scaled);
  const __m128 rounded_abs = _mm_add_ps(abs_scaled, half);
  const __m128i abs_i = _mm_cvttps_epi32(rounded_abs);
  const __m128 negative = _mm_cmplt_ps(scaled, zero);
  const __m128i sign_i = _mm_castps_si128(negative);
  return _mm_sub_epi32(_mm_xor_si128(abs_i, sign_i), sign_i);
}
#endif

static void quantize_bf16_row_q8_0_native(const uint16_t *src, block_q8_0 *dst, int64_t k)
{
  assert(k % QK8_0 == 0);
  const int64_t nb = k / QK8_0;

#if defined(LIBGGUF_PAYLOAD_SSE2)
  static_assert(QK8_0 == 32, "QK8_0 must be 32");
  const __m128 sign_mask = _mm_set1_ps(-0.0f);
  const __m128 half = _mm_set1_ps(0.5f);
  const __m128 zero = _mm_setzero_ps();
  for (int64_t i = 0; i < nb; ++i)
  {
    const uint16_t *xb = src + i * QK8_0;
    __m128 maxv = _mm_setzero_ps();
    for (int j = 0; j < QK8_0; j += 4)
    {
      const __m128 v = bf16_load4_to_ps_native(xb + j);
      maxv = _mm_max_ps(maxv, _mm_andnot_ps(sign_mask, v));
    }

    const float amax = q8_0_hmax_ps_native(maxv);
    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;
    const __m128 idv = _mm_set1_ps(id);

    dst[i].d = LIBGGUF_FP32_TO_FP16(d);

    for (int j = 0; j < QK8_0; j += 8)
    {
      const __m128i q0 = q8_0_round_away_from_zero_native(bf16_load4_to_ps_native(xb + j), idv, sign_mask, half, zero);
      const __m128i q1 = q8_0_round_away_from_zero_native(bf16_load4_to_ps_native(xb + j + 4), idv, sign_mask, half, zero);
      const __m128i packed_i16 = _mm_packs_epi32(q0, q1);
      const __m128i packed_i8 = _mm_packs_epi16(packed_i16, _mm_setzero_si128());
      _mm_storel_epi64(reinterpret_cast<__m128i *>(dst[i].qs + j), packed_i8);
    }
  }
#else
  float values[QK8_0];
  for (int64_t i = 0; i < nb; ++i)
  {
    const uint16_t *xb = src + i * QK8_0;
    float amax = 0.0f;
    for (int j = 0; j < QK8_0; ++j)
    {
      ggml_bf16_t value;
      value.bits = xb[j];
      values[j] = LIBGGUF_BF16_TO_FP32(value);
      amax = libgguf_max(amax, fabsf(values[j]));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;
    dst[i].d = LIBGGUF_FP32_TO_FP16(d);

    for (int j = 0; j < QK8_0; ++j)
    {
      dst[i].qs[j] = roundf(values[j] * id);
    }
  }
#endif
}

static size_t quantize_bf16_q8_0_chunk_native(const uint16_t *src, void *dst, int64_t nrows, int64_t n_per_row)
{
  const size_t row_size = libgguf_row_size(GGML_TYPE_Q8_0, n_per_row);
  const unsigned int nthreads = payload_thread_count(nrows);
  if (nthreads <= 1)
  {
    quantize_bf16_row_q8_0_native(src, reinterpret_cast<block_q8_0 *>(dst), nrows * n_per_row);
    return (size_t)nrows * row_size;
  }

  std::vector<std::thread> threads;
  std::vector<size_t> written(nthreads, 0);
  threads.reserve(nthreads);

  const int64_t rows_per_thread = (nrows + (int64_t)nthreads - 1) / (int64_t)nthreads;
  for (unsigned int thread_id = 0; thread_id < nthreads; ++thread_id)
  {
    const int64_t row_begin = (int64_t)thread_id * rows_per_thread;
    if (row_begin >= nrows)
    {
      written.resize(thread_id);
      break;
    }
    const int64_t row_count = std::min<int64_t>(rows_per_thread, nrows - row_begin);
    threads.emplace_back([=, &written]() {
      quantize_bf16_row_q8_0_native(
          src + row_begin * n_per_row,
          reinterpret_cast<block_q8_0 *>((uint8_t *)dst + (uint64_t)row_begin * row_size),
          row_count * n_per_row);
      written[thread_id] = (size_t)row_count * row_size;
    });
  }

  size_t total = 0;
  for (std::thread &thread : threads)
  {
    thread.join();
  }
  for (size_t count : written)
  {
    total += count;
  }
  return total;
}

static void decode_source_to_f32(const safetensors_payload_plan &plan, const uint8_t *source, uint64_t row_begin, uint64_t row_count, float *dst);

typedef size_t (*quantize_fn)(const float *LIBGGUF_RESTRICT, void *LIBGGUF_RESTRICT, int64_t, int64_t, const float *);

static quantize_fn quantize_function(ggml_type qtype)
{
  switch (qtype)
  {
  case GGML_TYPE_Q1_0:
    return quantize_q1_0;
  case GGML_TYPE_Q4_0:
    return quantize_q4_0;
  case GGML_TYPE_Q4_1:
    return quantize_q4_1;
  case GGML_TYPE_Q5_0:
    return quantize_q5_0;
  case GGML_TYPE_Q5_1:
    return quantize_q5_1;
  case GGML_TYPE_Q8_0:
    return quantize_q8_0;
  case GGML_TYPE_Q2_K:
    return quantize_q2_k;
  case GGML_TYPE_Q3_K:
    return quantize_q3_k;
  case GGML_TYPE_Q4_K:
    return quantize_q4_k;
  case GGML_TYPE_Q5_K:
    return quantize_q5_k;
  case GGML_TYPE_Q6_K:
    return quantize_q6_k;
  case GGML_TYPE_IQ2_XXS:
    return quantize_iq2_xxs;
  case GGML_TYPE_IQ2_XS:
    return quantize_iq2_xs;
  case GGML_TYPE_IQ2_S:
    return quantize_iq2_s;
  case GGML_TYPE_IQ3_XXS:
    return quantize_iq3_xxs;
  case GGML_TYPE_IQ3_S:
    return quantize_iq3_s;
  case GGML_TYPE_IQ1_S:
    return quantize_iq1_s;
  case GGML_TYPE_IQ1_M:
    return quantize_iq1_m;
  case GGML_TYPE_IQ4_NL:
    return quantize_iq4_nl;
  case GGML_TYPE_IQ4_XS:
    return quantize_iq4_xs;
  case GGML_TYPE_TQ1_0:
    return quantize_tq1_0;
  case GGML_TYPE_TQ2_0:
    return quantize_tq2_0;
  case GGML_TYPE_MXFP4:
    return quantize_mxfp4;
  case GGML_TYPE_NVFP4:
    return quantize_nvfp4;
  default:
    return nullptr;
  }
}

static bool quantize_decoded_source_chunk_native(
    const safetensors_payload_plan &plan,
    const uint8_t *source,
    uint64_t row_begin,
    uint64_t row_count,
    void *dst,
    const float *imatrix,
    std::string *error)
{
  quantize_fn quantize = quantize_function(plan.qtype);
  if (!quantize)
  {
    *error = "unsupported fused native quantization target for " + plan.key;
    return false;
  }

  const size_t row_size = libgguf_row_size(plan.qtype, (int64_t)plan.n_per_row);
  const unsigned int nthreads = payload_thread_count((int64_t)row_count);
  if (nthreads <= 1)
  {
    std::vector<float> scratch((size_t)(row_count * plan.n_per_row));
    decode_source_to_f32(plan, source, row_begin, row_count, scratch.data());
    const size_t written = quantize(scratch.data(), dst, (int64_t)row_count, (int64_t)plan.n_per_row, imatrix);
    if (written != row_count * (uint64_t)row_size)
    {
      *error = "fused native quantization returned an unexpected byte count for " + plan.key;
      return false;
    }
    return true;
  }

  std::vector<std::thread> threads;
  std::vector<size_t> written(nthreads, 0);
  threads.reserve(nthreads);
  const uint64_t rows_per_thread = (row_count + (uint64_t)nthreads - 1) / (uint64_t)nthreads;
  for (unsigned int thread_id = 0; thread_id < nthreads; ++thread_id)
  {
    const uint64_t local_row_begin = (uint64_t)thread_id * rows_per_thread;
    if (local_row_begin >= row_count)
    {
      written.resize(thread_id);
      break;
    }
    const uint64_t local_row_count = std::min<uint64_t>(rows_per_thread, row_count - local_row_begin);
    threads.emplace_back([=, &written, &plan]() {
      std::vector<float> scratch((size_t)(local_row_count * plan.n_per_row));
      decode_source_to_f32(plan, source, row_begin + local_row_begin, local_row_count, scratch.data());
      written[thread_id] = quantize(
          scratch.data(),
          (uint8_t *)dst + local_row_begin * row_size,
          (int64_t)local_row_count,
          (int64_t)plan.n_per_row,
          imatrix);
    });
  }

  for (std::thread &thread : threads)
  {
    thread.join();
  }
  for (size_t i = 0; i < written.size(); ++i)
  {
    const uint64_t local_row_begin = (uint64_t)i * rows_per_thread;
    const uint64_t local_row_count = std::min<uint64_t>(rows_per_thread, row_count - local_row_begin);
    const uint64_t expected = local_row_count * (uint64_t)row_size;
    if (written[i] != expected)
    {
      *error = "fused native quantization returned an unexpected byte count for " + plan.key;
      return false;
    }
  }
  return true;
}

static void decode_source_to_f32(const safetensors_payload_plan &plan, const uint8_t *source, uint64_t row_begin, uint64_t row_count, float *dst)
{
  const uint64_t value_begin = row_begin * plan.n_per_row;
  const uint64_t value_count = row_count * plan.n_per_row;
  const uint8_t *src = source + plan.data_begin + value_begin * payload_source_dtype_size(plan.source_dtype);

  switch (plan.source_dtype)
  {
  case PAYLOAD_DTYPE_F32:
    std::memcpy(dst, src, (size_t)(value_count * sizeof(float)));
    break;
  case PAYLOAD_DTYPE_F16:
  {
    const ggml_fp16_t *values = (const ggml_fp16_t *)src;
    for (uint64_t i = 0; i < value_count; ++i)
    {
      dst[i] = LIBGGUF_FP16_TO_FP32(values[i]);
    }
    break;
  }
  case PAYLOAD_DTYPE_BF16:
  {
    decode_bf16_to_f32((const uint16_t *)src, value_count, dst);
    break;
  }
  }
}

static const float *aligned_f32_source_rows(const safetensors_payload_plan &plan, const uint8_t *source, uint64_t row_begin)
{
  if (plan.source_dtype != PAYLOAD_DTYPE_F32)
  {
    return nullptr;
  }
  const uint64_t value_begin = row_begin * plan.n_per_row;
  const uint8_t *src = source + plan.data_begin + value_begin * sizeof(float);
  if ((reinterpret_cast<uintptr_t>(src) % alignof(float)) != 0)
  {
    return nullptr;
  }
  return reinterpret_cast<const float *>(src);
}

static void accumulate_imatrix_rows(const float *src_rows, uint64_t row_count, uint64_t n_per_row, std::vector<float> &imatrix)
{
  for (uint64_t r = 0; r < row_count; ++r)
  {
    const float *src_row = src_rows + r * n_per_row;
    for (uint64_t c = 0; c < n_per_row; ++c)
    {
      imatrix[(size_t)c] += src_row[c] * src_row[c];
    }
  }
}

static bool compute_auto_imatrix(safetensors_payload_plan &plan, const uint8_t *source, uint64_t rows_per_chunk, std::vector<float> &scratch, std::string *error)
{
  plan.auto_imatrix.assign((size_t)plan.n_per_row, 0.0f);
  for (uint64_t row = 0; row < plan.n_rows; row += rows_per_chunk)
  {
    const uint64_t row_count = std::min<uint64_t>(rows_per_chunk, plan.n_rows - row);
    const float *src_rows = aligned_f32_source_rows(plan, source, row);
    if (!src_rows)
    {
      decode_source_to_f32(plan, source, row, row_count, scratch.data());
      src_rows = scratch.data();
    }
    accumulate_imatrix_rows(src_rows, row_count, plan.n_per_row, plan.auto_imatrix);
  }
  if (plan.auto_imatrix.empty())
  {
    *error = "failed to compute imatrix";
    return false;
  }
  return true;
}

static bool write_payload_plan(
    int fd,
    const safetensors_payload_plan &plan,
    const uint8_t *source,
    uint64_t alignment,
    uint64_t scratch_bytes,
    const float *imatrix,
    std::string *error)
{
  if (!write_pre_tensor_padding(fd, alignment, error))
  {
    return false;
  }

  if (qtype_matches_source(plan.source_dtype, plan.qtype))
  {
    if (!write_all_fd(fd, source + plan.data_begin, plan.expected_nbytes, error))
    {
      return false;
    }
    return write_post_tensor_padding(fd, plan.expected_nbytes, alignment, error);
  }

  const size_t row_size = libgguf_row_size(plan.qtype, (int64_t)plan.n_per_row);
  if (plan.source_dtype == PAYLOAD_DTYPE_BF16 && plan.qtype == GGML_TYPE_Q8_0)
  {
    uint64_t rows_per_chunk = scratch_bytes / (uint64_t)row_size;
    if (rows_per_chunk == 0)
    {
      rows_per_chunk = 1;
    }
    rows_per_chunk = std::min<uint64_t>(rows_per_chunk, plan.n_rows);

    std::vector<uint8_t> encoded((size_t)(rows_per_chunk * row_size));
    for (uint64_t row = 0; row < plan.n_rows; row += rows_per_chunk)
    {
      const uint64_t row_count = std::min<uint64_t>(rows_per_chunk, plan.n_rows - row);
      const uint8_t *src = source + plan.data_begin + row * plan.n_per_row * sizeof(uint16_t);
      const size_t written = quantize_bf16_q8_0_chunk_native(
          reinterpret_cast<const uint16_t *>(src),
          encoded.data(),
          (int64_t)row_count,
          (int64_t)plan.n_per_row);
      const uint64_t expected = row_count * (uint64_t)row_size;
      if (written != expected)
      {
        *error = "native BF16 Q8_0 quantization returned an unexpected byte count for " + plan.key;
        return false;
      }
      if (!write_all_fd(fd, encoded.data(), expected, error))
      {
        return false;
      }
    }

    return write_post_tensor_padding(fd, plan.expected_nbytes, alignment, error);
  }

  if ((plan.source_dtype == PAYLOAD_DTYPE_BF16 || plan.source_dtype == PAYLOAD_DTYPE_F16) && quantize_function(plan.qtype) != nullptr)
  {
    const uint64_t row_work_bytes = plan.n_per_row * sizeof(float) + (uint64_t)row_size;
    uint64_t rows_per_chunk = row_work_bytes == 0 ? 1 : scratch_bytes / row_work_bytes;
    if (rows_per_chunk == 0)
    {
      rows_per_chunk = 1;
    }
    rows_per_chunk = std::min<uint64_t>(rows_per_chunk, plan.n_rows);

    std::vector<uint8_t> encoded((size_t)(rows_per_chunk * row_size));
    for (uint64_t row = 0; row < plan.n_rows; row += rows_per_chunk)
    {
      const uint64_t row_count = std::min<uint64_t>(rows_per_chunk, plan.n_rows - row);
      if (!quantize_decoded_source_chunk_native(plan, source, row, row_count, encoded.data(), imatrix, error))
      {
        return false;
      }
      const uint64_t expected = row_count * (uint64_t)row_size;
      if (!write_all_fd(fd, encoded.data(), expected, error))
      {
        return false;
      }
    }

    return write_post_tensor_padding(fd, plan.expected_nbytes, alignment, error);
  }

  const bool use_source_f32 = aligned_f32_source_rows(plan, source, 0) != nullptr;
  const uint64_t row_work_bytes = (use_source_f32 ? 0 : plan.n_per_row * sizeof(float)) + (uint64_t)row_size;
  uint64_t rows_per_chunk = row_work_bytes == 0 ? 1 : scratch_bytes / row_work_bytes;
  if (rows_per_chunk == 0)
  {
    rows_per_chunk = 1;
  }
  rows_per_chunk = std::min<uint64_t>(rows_per_chunk, plan.n_rows);

  std::vector<float> scratch;
  if (!use_source_f32)
  {
    scratch.resize((size_t)(rows_per_chunk * plan.n_per_row));
  }
  std::vector<uint8_t> encoded((size_t)(rows_per_chunk * row_size));

  for (uint64_t row = 0; row < plan.n_rows; row += rows_per_chunk)
  {
    const uint64_t row_count = std::min<uint64_t>(rows_per_chunk, plan.n_rows - row);
    const float *chunk_src = nullptr;
    if (use_source_f32)
    {
      chunk_src = aligned_f32_source_rows(plan, source, row);
      if (!chunk_src)
      {
        *error = "aligned F32 source became unavailable for " + plan.key;
        return false;
      }
    }
    else
    {
      decode_source_to_f32(plan, source, row, row_count, scratch.data());
      chunk_src = scratch.data();
    }
    const size_t written = libgguf_quantize_chunk(
        plan.qtype,
        chunk_src,
        encoded.data(),
        0,
        (int64_t)row_count,
        (int64_t)plan.n_per_row,
        imatrix);
    const uint64_t expected = row_count * (uint64_t)row_size;
    if (written != expected)
    {
      *error = "libgguf_quantize_chunk returned an unexpected byte count for " + plan.key;
      return false;
    }
    if (!write_all_fd(fd, encoded.data(), expected, error))
    {
      return false;
    }
  }

  return write_post_tensor_padding(fd, plan.expected_nbytes, alignment, error);
}

bool write_safetensors_payload(
    const std::string &path,
    int fd,
    std::vector<safetensors_payload_plan> &plans,
    uint64_t alignment,
    uint64_t scratch_bytes,
    uint64_t *data_bytes_written,
    std::string *error)
{
  mapped_file mapped;
  if (!mapped.map(path, error))
  {
    return false;
  }

  *data_bytes_written = 0;
  for (safetensors_payload_plan &plan : plans)
  {
    if (plan.data_end > mapped.size)
    {
      *error = "safetensors byte range exceeds file size for " + plan.key;
      return false;
    }

    const size_t row_size = libgguf_row_size(plan.qtype, (int64_t)plan.n_per_row);
    uint64_t row_work_bytes = plan.n_per_row * sizeof(float) + (uint64_t)row_size;
    uint64_t rows_per_chunk = row_work_bytes == 0 ? 1 : scratch_bytes / row_work_bytes;
    if (rows_per_chunk == 0)
    {
      rows_per_chunk = 1;
    }
    rows_per_chunk = std::min<uint64_t>(rows_per_chunk, plan.n_rows);

    const float *imatrix = plan.imatrix;
    if (libgguf_quantize_requires_imatrix(plan.qtype) && imatrix == nullptr)
    {
      const bool imatrix_uses_source_f32 = aligned_f32_source_rows(plan, mapped.data, 0) != nullptr;
      if (imatrix_uses_source_f32)
      {
        rows_per_chunk = plan.n_rows;
      }
      std::vector<float> decode_scratch;
      if (!imatrix_uses_source_f32)
      {
        decode_scratch.resize((size_t)(rows_per_chunk * plan.n_per_row));
      }
      if (!compute_auto_imatrix(plan, mapped.data, rows_per_chunk, decode_scratch, error))
      {
        return false;
      }
      imatrix = plan.auto_imatrix.data();
    }

    if (!write_payload_plan(fd, plan, mapped.data, alignment, scratch_bytes, imatrix, error))
    {
      return false;
    }
    *data_bytes_written += plan.expected_nbytes;
  }
  return true;
}
