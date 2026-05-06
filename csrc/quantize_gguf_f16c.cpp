#include "common/libgguf_common.h"
#include "common/libgguf_internal.h"

#include <cstdint>

#if (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))) || defined(__F16C__)
#include <immintrin.h>
#define LIBGGUF_QUANTIZE_GGUF_BUILD_F16C 1
#endif

extern "C" void libgguf_quantize_gguf_f16_to_f32_f16c(const ggml_fp16_t *src, uint64_t count, float *dst)
{
#if defined(LIBGGUF_QUANTIZE_GGUF_BUILD_F16C)
  uint64_t i = 0;
  for (; i + 8 <= count; i += 8)
  {
    const __m128i values = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + i));
    const __m256 floats = _mm256_cvtph_ps(values);
    _mm256_storeu_ps(dst + i, floats);
  }
  for (; i < count; ++i)
  {
    dst[i] = GGML_FP16_TO_FP32(src[i]);
  }
#else
  for (uint64_t i = 0; i < count; ++i)
  {
    dst[i] = GGML_FP16_TO_FP32(src[i]);
  }
#endif
}
