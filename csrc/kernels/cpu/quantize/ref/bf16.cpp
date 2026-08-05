#include "kernels/cpu/common/common.h"
#include "kernels/cpu/quantize/kernels.h"

extern "C" size_t quantize_bf16(const float *LIBGGUF_RESTRICT src, void *LIBGGUF_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *)
{
  const size_t n = (size_t)nrows * (size_t)n_per_row;
  ggml_bf16_t *out = (ggml_bf16_t *)dst;
  for (size_t i = 0; i < n; ++i)
  {
    out[i] = LIBGGUF_FP32_TO_BF16(src[i]);
  }
  return n * sizeof(ggml_bf16_t);
}
