#pragma once

#include <cstdint>

#include <libgguf/formats.h>

extern "C"
{
  void dequantize_q1_0(const block_q1_0 *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k);
  void dequantize_q4_0(const block_q4_0 *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k);
  void dequantize_q4_1(const block_q4_1 *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k);
  void dequantize_q5_0(const block_q5_0 *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k);
  void dequantize_q5_1(const block_q5_1 *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k);
  void dequantize_q8_0(const block_q8_0 *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k);
  void dequantize_mxfp4(const block_mxfp4 *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k);
  void dequantize_nvfp4(const block_nvfp4 *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k);
  void dequantize_q2_k(const block_q2_K *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k);
  void dequantize_q3_k(const block_q3_K *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k);
  void dequantize_q4_k(const block_q4_K *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k);
  void dequantize_q5_k(const block_q5_K *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k);
  void dequantize_q6_k(const block_q6_K *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k);
  void dequantize_q8_K(const block_q8_K *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k);
  void dequantize_tq1_0(const block_tq1_0 *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k);
  void dequantize_tq2_0(const block_tq2_0 *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k);
  void dequantize_iq2_xxs(const block_iq2_xxs *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k);
  void dequantize_iq2_xs(const block_iq2_xs *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k);
  void dequantize_iq2_s(const block_iq2_s *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k);
  void dequantize_iq3_xxs(const block_iq3_xxs *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k);
  void dequantize_iq1_s(const block_iq1_s *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k);
  void dequantize_iq1_m(const block_iq1_m *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k);
  void dequantize_iq4_nl(const block_iq4_nl *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k);
  void dequantize_iq4_xs(const block_iq4_xs *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k);
  void dequantize_iq3_s(const block_iq3_s *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k);
}
