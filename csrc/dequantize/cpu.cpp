#include "dequantize/cpu.h"

#include <algorithm>
#include <thread>
#include <vector>

#include <libgguf/libgguf.h>

#include "cpu/threading.h"
#include "kernels/cpu/dequantize/kernels.h"

size_t libgguf_dequantize_chunk_serial(
    enum ggml_type type,
    const void *src,
    float *dst,
    int64_t start,
    int64_t nrows,
    int64_t n_per_row)
{
  if (nrows <= 0)
  {
    return 0;
  }

  const size_t row_size = libgguf_row_size(type, n_per_row);
  if (row_size == 0 || n_per_row <= 0 || start < 0 || start % n_per_row != 0)
  {
    return 0;
  }

  const size_t start_row = (size_t)(start / n_per_row);
  const char *src_row = (const char *)src + start_row * row_size;
  float *dst_row = dst + start;
  const int64_t k = nrows * n_per_row;

  switch (type)
  {
  case GGML_TYPE_Q1_0:
    dequantize_q1_0((const block_q1_0 *)src_row, dst_row, k);
    break;
  case GGML_TYPE_Q4_0:
    dequantize_q4_0((const block_q4_0 *)src_row, dst_row, k);
    break;
  case GGML_TYPE_Q4_1:
    dequantize_q4_1((const block_q4_1 *)src_row, dst_row, k);
    break;
  case GGML_TYPE_Q5_0:
    dequantize_q5_0((const block_q5_0 *)src_row, dst_row, k);
    break;
  case GGML_TYPE_Q5_1:
    dequantize_q5_1((const block_q5_1 *)src_row, dst_row, k);
    break;
  case GGML_TYPE_Q8_0:
    dequantize_q8_0((const block_q8_0 *)src_row, dst_row, k);
    break;
  case GGML_TYPE_Q2_K:
    dequantize_q2_k((const block_q2_K *)src_row, dst_row, k);
    break;
  case GGML_TYPE_Q3_K:
    dequantize_q3_k((const block_q3_K *)src_row, dst_row, k);
    break;
  case GGML_TYPE_Q4_K:
    dequantize_q4_k((const block_q4_K *)src_row, dst_row, k);
    break;
  case GGML_TYPE_Q5_K:
    dequantize_q5_k((const block_q5_K *)src_row, dst_row, k);
    break;
  case GGML_TYPE_Q6_K:
    dequantize_q6_k((const block_q6_K *)src_row, dst_row, k);
    break;
  case GGML_TYPE_IQ2_XXS:
    dequantize_iq2_xxs((const block_iq2_xxs *)src_row, dst_row, k);
    break;
  case GGML_TYPE_IQ2_XS:
    dequantize_iq2_xs((const block_iq2_xs *)src_row, dst_row, k);
    break;
  case GGML_TYPE_IQ2_S:
    dequantize_iq2_s((const block_iq2_s *)src_row, dst_row, k);
    break;
  case GGML_TYPE_IQ3_XXS:
    dequantize_iq3_xxs((const block_iq3_xxs *)src_row, dst_row, k);
    break;
  case GGML_TYPE_IQ3_S:
    dequantize_iq3_s((const block_iq3_s *)src_row, dst_row, k);
    break;
  case GGML_TYPE_IQ1_S:
    dequantize_iq1_s((const block_iq1_s *)src_row, dst_row, k);
    break;
  case GGML_TYPE_IQ1_M:
    dequantize_iq1_m((const block_iq1_m *)src_row, dst_row, k);
    break;
  case GGML_TYPE_IQ4_NL:
    dequantize_iq4_nl((const block_iq4_nl *)src_row, dst_row, k);
    break;
  case GGML_TYPE_IQ4_XS:
    dequantize_iq4_xs((const block_iq4_xs *)src_row, dst_row, k);
    break;
  case GGML_TYPE_TQ1_0:
    dequantize_tq1_0((const block_tq1_0 *)src_row, dst_row, k);
    break;
  case GGML_TYPE_TQ2_0:
    dequantize_tq2_0((const block_tq2_0 *)src_row, dst_row, k);
    break;
  case GGML_TYPE_MXFP4:
    dequantize_mxfp4((const block_mxfp4 *)src_row, dst_row, k);
    break;
  case GGML_TYPE_NVFP4:
    dequantize_nvfp4((const block_nvfp4 *)src_row, dst_row, k);
    break;
  default:
    return 0;
  }

  return (size_t)nrows * (size_t)n_per_row * sizeof(float);
}

extern "C" LIBGGUF_API size_t libgguf_dequantize_chunk(
    enum ggml_type type,
    const void *src,
    float *dst,
    int64_t start,
    int64_t nrows,
    int64_t n_per_row)
{
  if (nrows <= 0)
  {
    return 0;
  }

  const size_t row_size = libgguf_row_size(type, n_per_row);
  if (row_size == 0 || n_per_row <= 0 || start < 0 || start % n_per_row != 0)
  {
    return 0;
  }

  const unsigned int nthreads = libgguf_cpu_thread_count(nrows);
  if (nthreads <= 1)
  {
    return libgguf_dequantize_chunk_serial(type, src, dst, start, nrows, n_per_row);
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
      written[thread_id] = libgguf_dequantize_chunk_serial(
          type,
          src,
          dst,
          start + row_begin * n_per_row,
          row_count,
          n_per_row);
    });
  }

  for (std::thread &thread : threads)
  {
    thread.join();
  }

  size_t total = 0;
  for (size_t n : written)
  {
    total += n;
  }
  return total;
}
