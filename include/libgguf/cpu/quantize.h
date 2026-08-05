#pragma once

#include <cstddef>
#include <cstdint>

#include <libgguf/formats.h>

extern "C"
{
  void quantize_row_q1_0(const float *RESTRICT x, block_q1_0 *RESTRICT y, int64_t k);
  void quantize_row_q4_0(const float *RESTRICT x, block_q4_0 *RESTRICT y, int64_t k);
  void quantize_row_q4_1(const float *RESTRICT x, block_q4_1 *RESTRICT y, int64_t k);
  void quantize_row_q5_0(const float *RESTRICT x, block_q5_0 *RESTRICT y, int64_t k);
  void quantize_row_q5_1(const float *RESTRICT x, block_q5_1 *RESTRICT y, int64_t k);
  void quantize_row_q8_0(const float *RESTRICT x, block_q8_0 *RESTRICT y, int64_t k);
  void quantize_row_mxfp4(const float *RESTRICT x, block_mxfp4 *RESTRICT y, int64_t k);
  void quantize_row_nvfp4(const float *RESTRICT x, block_nvfp4 *RESTRICT y, int64_t k);
  void quantize_row_q2_K(const float *RESTRICT x, block_q2_K *RESTRICT y, int64_t k);
  void quantize_row_q3_K(const float *RESTRICT x, block_q3_K *RESTRICT y, int64_t k);
  void quantize_row_q4_K(const float *RESTRICT x, block_q4_K *RESTRICT y, int64_t k);
  void quantize_row_q5_K(const float *RESTRICT x, block_q5_K *RESTRICT y, int64_t k);
  void quantize_row_q6_K(const float *RESTRICT x, block_q6_K *RESTRICT y, int64_t k);
  void quantize_row_tq1_0(const float *RESTRICT x, block_tq1_0 *RESTRICT y, int64_t k);
  void quantize_row_tq2_0(const float *RESTRICT x, block_tq2_0 *RESTRICT y, int64_t k);
  void quantize_row_iq3_xxs(const float *RESTRICT x, block_iq3_xxs *RESTRICT y, int64_t k);
  void quantize_row_iq4_nl(const float *RESTRICT x, block_iq4_nl *RESTRICT y, int64_t k);
  void quantize_row_iq4_xs(const float *RESTRICT x, block_iq4_xs *RESTRICT y, int64_t k);
  void quantize_row_iq3_s(const float *RESTRICT x, block_iq3_s *RESTRICT y, int64_t k);
  void quantize_row_iq2_s(const float *RESTRICT x, block_iq2_s *RESTRICT y, int64_t k);

  size_t quantize_iq2_xxs(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_iq2_xs(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_iq2_s(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_iq3_xxs(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_iq1_s(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_iq1_m(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_iq4_nl(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_iq4_xs(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_iq3_s(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_tq1_0(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_tq2_0(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_q2_K(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_q3_K(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_q4_K(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_q5_K(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_q6_K(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_q1_0(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_q4_0(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_q4_1(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_q5_0(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_q5_1(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_q8_0(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_mxfp4(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  size_t quantize_nvfp4(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
}
