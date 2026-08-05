#include "conversion/cpu_quantizer.h"

#include "conversion/common.h"
#include "cli/quantize/options.h"
#include "cli/quantize/policy.h"
#include "conversion/safetensors.h"

#include "kernels/cpu/common/common.h"
#include <libgguf/libgguf.h>

#include "kernels/cpu/quantize/kernels.h"

#include <algorithm>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <emmintrin.h>
#define LIBGGUF_NATIVE_SSE2 1
#endif
#if defined(__F16C__) || (defined(_MSC_VER) && defined(__AVX2__))
#include <immintrin.h>
#endif

typedef size_t (*native_quantize_fn)(const float *LIBGGUF_RESTRICT, void *LIBGGUF_RESTRICT, int64_t, int64_t, const float *);

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


native_quantize_fn native_quantize_function(ggml_type qtype) {
  switch (qtype) {
  case GGML_TYPE_Q1_0: return quantize_q1_0;
  case GGML_TYPE_Q4_0: return quantize_q4_0;
  case GGML_TYPE_Q4_1: return quantize_q4_1;
  case GGML_TYPE_Q5_0: return quantize_q5_0;
  case GGML_TYPE_Q5_1: return quantize_q5_1;
  case GGML_TYPE_Q8_0: return quantize_q8_0;
  case GGML_TYPE_Q2_K: return quantize_q2_k;
  case GGML_TYPE_Q3_K: return quantize_q3_k;
  case GGML_TYPE_Q4_K: return quantize_q4_k;
  case GGML_TYPE_Q5_K: return quantize_q5_k;
  case GGML_TYPE_Q6_K: return quantize_q6_k;
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
    dst[i] = LIBGGUF_BF16_TO_FP32(value);
  }
}

void decode_f16_to_f32(const ggml_fp16_t *src, uint64_t count, float *dst) {
#if defined(__F16C__) || (defined(_MSC_VER) && defined(__AVX2__))
  uint64_t i = 0;
  for (; i + 8 <= count; i += 8) {
    const __m128i values = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + i));
    _mm256_storeu_ps(dst + i, _mm256_cvtph_ps(values));
  }
  for (; i < count; ++i) {
    dst[i] = LIBGGUF_FP16_TO_FP32(src[i]);
  }
#else
  for (uint64_t i = 0; i < count; ++i) {
    dst[i] = LIBGGUF_FP16_TO_FP32(src[i]);
  }
#endif
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

struct cpu_quantizer::implementation {
public:
  explicit implementation(unsigned int max_threads)
      : max_threads_(std::max(1u, max_threads)) {
    scratch_.resize(max_threads_);
    ok_.resize(max_threads_, true);
    if (max_threads_ > 1) {
      for (unsigned int i = 0; i < max_threads_; ++i) {
        threads_.emplace_back([this, i]() { worker_loop(i); });
      }
    }
  }

  implementation(const implementation &) = delete;
  implementation &operator=(const implementation &) = delete;

  ~implementation() {
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

cpu_quantizer::cpu_quantizer(unsigned int max_threads)
    : implementation_(new implementation(max_threads)) {
}

bool cpu_quantizer_supports(ggml_type qtype) {
  return native_quantize_function(qtype) != nullptr;
}

cpu_quantizer::~cpu_quantizer() = default;

bool cpu_quantizer::run(
    const tensor_plan & plan,
    const uint8_t * source_bytes,
    uint64_t row_count,
    void * destination,
    const float * imatrix,
    uint64_t scratch_bytes
) {
    return implementation_->run(plan, source_bytes, row_count, destination, imatrix, scratch_bytes);
}
