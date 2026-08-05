#include "kernels/cpu/quantize/kernels.h"

#include <cstring>

extern "C" size_t quantize_f32(const float *LIBGGUF_RESTRICT src, void *LIBGGUF_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *)
{
  const size_t n = (size_t)nrows * (size_t)n_per_row;
  std::memcpy(dst, src, n * sizeof(float));
  return n * sizeof(float);
}
