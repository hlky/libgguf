#pragma once

#include <cstdint>

#include <libgguf/formats.h>

extern "C"
{
  void dequantize_row_q1_0(const block_q1_0 *RESTRICT x, float *RESTRICT y, int64_t k);
  void dequantize_row_q4_0(const block_q4_0 *RESTRICT x, float *RESTRICT y, int64_t k);
  void dequantize_row_q4_1(const block_q4_1 *RESTRICT x, float *RESTRICT y, int64_t k);
  void dequantize_row_q5_0(const block_q5_0 *RESTRICT x, float *RESTRICT y, int64_t k);
  void dequantize_row_q5_1(const block_q5_1 *RESTRICT x, float *RESTRICT y, int64_t k);
  void dequantize_row_q8_0(const block_q8_0 *RESTRICT x, float *RESTRICT y, int64_t k);
  void dequantize_row_mxfp4(const block_mxfp4 *RESTRICT x, float *RESTRICT y, int64_t k);
  void dequantize_row_nvfp4(const block_nvfp4 *RESTRICT x, float *RESTRICT y, int64_t k);
  void dequantize_row_q2_K(const block_q2_K *RESTRICT x, float *RESTRICT y, int64_t k);
  void dequantize_row_q3_K(const block_q3_K *RESTRICT x, float *RESTRICT y, int64_t k);
  void dequantize_row_q4_K(const block_q4_K *RESTRICT x, float *RESTRICT y, int64_t k);
  void dequantize_row_q5_K(const block_q5_K *RESTRICT x, float *RESTRICT y, int64_t k);
  void dequantize_row_q6_K(const block_q6_K *RESTRICT x, float *RESTRICT y, int64_t k);
  void dequantize_row_q8_K(const block_q8_K *RESTRICT x, float *RESTRICT y, int64_t k);
  void dequantize_row_tq1_0(const block_tq1_0 *RESTRICT x, float *RESTRICT y, int64_t k);
  void dequantize_row_tq2_0(const block_tq2_0 *RESTRICT x, float *RESTRICT y, int64_t k);
  void dequantize_row_iq2_xxs(const block_iq2_xxs *RESTRICT x, float *RESTRICT y, int64_t k);
  void dequantize_row_iq2_xs(const block_iq2_xs *RESTRICT x, float *RESTRICT y, int64_t k);
  void dequantize_row_iq2_s(const block_iq2_s *RESTRICT x, float *RESTRICT y, int64_t k);
  void dequantize_row_iq3_xxs(const block_iq3_xxs *RESTRICT x, float *RESTRICT y, int64_t k);
  void dequantize_row_iq1_s(const block_iq1_s *RESTRICT x, float *RESTRICT y, int64_t k);
  void dequantize_row_iq1_m(const block_iq1_m *RESTRICT x, float *RESTRICT y, int64_t k);
  void dequantize_row_iq4_nl(const block_iq4_nl *RESTRICT x, float *RESTRICT y, int64_t k);
  void dequantize_row_iq4_xs(const block_iq4_xs *RESTRICT x, float *RESTRICT y, int64_t k);
  void dequantize_row_iq3_s(const block_iq3_s *RESTRICT x, float *RESTRICT y, int64_t k);
}
