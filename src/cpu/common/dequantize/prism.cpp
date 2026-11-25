#include "cpu/common/common.h"
#include "cpu/common/dequantize/kernels.h"
#include <libgguf/cpu/dequantize/decoder.h>

// Byte layout and decoding from PrismML-Eng/llama.cpp at
// 668256445feafa00dfe5f194bac1ea37ae2c74e5, ggml/src/ggml-quants.c.
void dequantize_pq2_0(const block_pq2_0 *LIBGGUF_RESTRICT x,
                      float *LIBGGUF_RESTRICT y, int64_t k)
{
  assert(k % QK_PQ2_0 == 0);
  for (int64_t i = 0; i < k / QK_PQ2_0; ++i)
  {
    libgguf::cpu::BlockDecoder<GGML_TYPE_PQ2_0>::decode(
        x + i, y, [](uint16_t h) { return LIBGGUF_FP16_TO_FP32(h); });
    y += QK_PQ2_0;
  }
}

void dequantize_ptq1_0(const block_ptq1_0 *LIBGGUF_RESTRICT x,
                       float *LIBGGUF_RESTRICT y, int64_t k)
{
  assert(k % QK_PTQ1_0 == 0);
  for (int64_t i = 0; i < k / QK_PTQ1_0; ++i)
  {
    libgguf::cpu::BlockDecoder<GGML_TYPE_PTQ1_0>::decode(
        x + i, y, [](uint16_t h) { return LIBGGUF_FP16_TO_FP32(h); });
    y += QK_PTQ1_0;
  }
}
