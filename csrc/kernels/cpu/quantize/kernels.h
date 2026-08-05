#pragma once

#include <cstddef>
#include <cstdint>

#include <libgguf/formats.h>

extern "C"
{
  size_t quantize_f32(const float *LIBGGUF_RESTRICT src, void *LIBGGUF_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_f16(const float *LIBGGUF_RESTRICT src, void *LIBGGUF_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_bf16(const float *LIBGGUF_RESTRICT src, void *LIBGGUF_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_iq2_xxs(const float *LIBGGUF_RESTRICT src, void *LIBGGUF_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_iq2_xs(const float *LIBGGUF_RESTRICT src, void *LIBGGUF_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_iq2_s(const float *LIBGGUF_RESTRICT src, void *LIBGGUF_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_iq3_xxs(const float *LIBGGUF_RESTRICT src, void *LIBGGUF_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_iq1_s(const float *LIBGGUF_RESTRICT src, void *LIBGGUF_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_iq1_m(const float *LIBGGUF_RESTRICT src, void *LIBGGUF_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_iq4_nl(const float *LIBGGUF_RESTRICT src, void *LIBGGUF_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_iq4_xs(const float *LIBGGUF_RESTRICT src, void *LIBGGUF_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_iq3_s(const float *LIBGGUF_RESTRICT src, void *LIBGGUF_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_tq1_0(const float *LIBGGUF_RESTRICT src, void *LIBGGUF_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_tq2_0(const float *LIBGGUF_RESTRICT src, void *LIBGGUF_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_q2_k(const float *LIBGGUF_RESTRICT src, void *LIBGGUF_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_q3_k(const float *LIBGGUF_RESTRICT src, void *LIBGGUF_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_q4_k(const float *LIBGGUF_RESTRICT src, void *LIBGGUF_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_q5_k(const float *LIBGGUF_RESTRICT src, void *LIBGGUF_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_q6_k(const float *LIBGGUF_RESTRICT src, void *LIBGGUF_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_q1_0(const float *LIBGGUF_RESTRICT src, void *LIBGGUF_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_q4_0(const float *LIBGGUF_RESTRICT src, void *LIBGGUF_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_q4_1(const float *LIBGGUF_RESTRICT src, void *LIBGGUF_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_q5_0(const float *LIBGGUF_RESTRICT src, void *LIBGGUF_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_q5_1(const float *LIBGGUF_RESTRICT src, void *LIBGGUF_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_q8_0(const float *LIBGGUF_RESTRICT src, void *LIBGGUF_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_mxfp4(const float *LIBGGUF_RESTRICT src, void *LIBGGUF_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_nvfp4(const float *LIBGGUF_RESTRICT src, void *LIBGGUF_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
}
