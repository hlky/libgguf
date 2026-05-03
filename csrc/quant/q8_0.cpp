#include "libgguf_common.h"
#include "common/libgguf_cpu.h"

#include <cstring>

typedef void (*libgguf_q8_0_kernel_fn)(const float *RESTRICT, block_q8_0 *RESTRICT, int64_t);

struct libgguf_q8_0_selection
{
  const char *backend;
  libgguf_q8_0_kernel_fn kernel;
};

extern "C" void quantize_row_q8_0_sse2(const float *RESTRICT x, block_q8_0 *RESTRICT y, int64_t k);
extern "C" void quantize_row_q8_0_sse4_1(const float *RESTRICT x, block_q8_0 *RESTRICT y, int64_t k);
extern "C" void quantize_row_q8_0_avx2(const float *RESTRICT x, block_q8_0 *RESTRICT y, int64_t k);

void quantize_row_q8_0(const float *RESTRICT x, block_q8_0 *RESTRICT y, int64_t k)
{
  assert(k % QK8_0 == 0);
  const int nb = k / QK8_0;

  for (int i = 0; i < nb; i++)
  {
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++)
    {
      const float v = x[i * QK8_0 + j];
      amax = MAX(amax, fabsf(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = GGML_FP32_TO_FP16(d);

    for (int j = 0; j < QK8_0; ++j)
    {
      const float x0 = x[i * QK8_0 + j] * id;
      y[i].qs[j] = roundf(x0);
    }
  }
}

static libgguf_q8_0_selection libgguf_q8_0_select_kernel()
{
  const libgguf_cpu_features &features = libgguf_get_cpu_features();

  if (features.avx2)
  {
    return {"avx2", quantize_row_q8_0_avx2};
  }
  if (features.sse4_1)
  {
    return {"sse4_1", quantize_row_q8_0_sse4_1};
  }
  if (features.sse2)
  {
    return {"sse2", quantize_row_q8_0_sse2};
  }

  return {"ref", quantize_row_q8_0};
}

static const libgguf_q8_0_selection &libgguf_q8_0_selected()
{
  static const libgguf_q8_0_selection selected = libgguf_q8_0_select_kernel();
  return selected;
}

static libgguf_q8_0_kernel_fn libgguf_q8_0_kernel()
{
  return libgguf_q8_0_selected().kernel;
}

extern "C" const char *libgguf_q8_0_backend(void)
{
  return libgguf_q8_0_selected().backend;
}

extern "C" int libgguf_q8_0_cpu_supports_backend(const char *backend)
{
  const libgguf_cpu_features &features = libgguf_get_cpu_features();
  if (backend == nullptr)
  {
    return 0;
  }
  if (std::strcmp(backend, "ref") == 0)
  {
    return 1;
  }
  if (std::strcmp(backend, "sse2") == 0)
  {
    return features.sse2 ? 1 : 0;
  }
  if (std::strcmp(backend, "sse4_1") == 0)
  {
    return features.sse4_1 ? 1 : 0;
  }
  if (std::strcmp(backend, "avx2") == 0)
  {
    return features.avx2 ? 1 : 0;
  }
  return 0;
}

extern "C" size_t libgguf_quantize_q8_0_for_backend(
    const char *backend,
    const float *RESTRICT src,
    void *RESTRICT dst,
    int64_t nrows,
    int64_t n_per_row)
{
  libgguf_q8_0_kernel_fn kernel = nullptr;
  if (backend == nullptr || std::strcmp(backend, "ref") == 0)
  {
    kernel = quantize_row_q8_0;
  }
  else if (std::strcmp(backend, "sse2") == 0)
  {
    if (!libgguf_get_cpu_features().sse2)
    {
      return 0;
    }
    kernel = quantize_row_q8_0_sse2;
  }
  else if (std::strcmp(backend, "sse4_1") == 0)
  {
    if (!libgguf_get_cpu_features().sse4_1)
    {
      return 0;
    }
    kernel = quantize_row_q8_0_sse4_1;
  }
  else if (std::strcmp(backend, "avx2") == 0)
  {
    if (!libgguf_get_cpu_features().avx2)
    {
      return 0;
    }
    kernel = quantize_row_q8_0_avx2;
  }
  else
  {
    return 0;
  }

  kernel(src, (block_q8_0 *)dst, (int64_t)nrows * n_per_row);
  return nrows * libgguf_row_size(GGML_TYPE_Q8_0, n_per_row);
}

size_t quantize_q8_0(const float *RESTRICT src, void *RESTRICT dst, int64_t nrow, int64_t n_per_row, const float *quant_weights)
{
  (void)quant_weights; // not used
  libgguf_q8_0_kernel()(src, (block_q8_0 *)dst, (int64_t)nrow * n_per_row);
  return nrow * libgguf_row_size(GGML_TYPE_Q8_0, n_per_row);
}
