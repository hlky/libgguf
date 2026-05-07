#include "libgguf_common.h"
#include "common/libgguf_backend.h"

typedef void (*libgguf_nvfp4_kernel_fn)(const float *RESTRICT, block_nvfp4 *RESTRICT, int64_t);

struct libgguf_nvfp4_selection
{
  const char *backend;
  libgguf_nvfp4_kernel_fn kernel;
};

#if !LIBGGUF_CPU_BACKEND_REF
extern "C" void LIBGGUF_CPU_BACKEND_SYMBOL(quantize_row_nvfp4)(const float *RESTRICT x, block_nvfp4 *RESTRICT y, int64_t k);
#endif

void quantize_row_nvfp4(const float *RESTRICT x, block_nvfp4 *RESTRICT y, int64_t k)
{
  static const int qk = QK_NVFP4;
  static const int qk_sub = QK_NVFP4_SUB;
  static const int n_sub = QK_NVFP4 / QK_NVFP4_SUB;

  assert(k % qk == 0);

  const int nb = k / qk;

  for (int i = 0; i < nb; i++)
  {
    for (int s = 0; s < n_sub; s++)
    {
      const float *xb = x + i * qk + s * qk_sub;

      float amax = 0.0f;
      for (int j = 0; j < qk_sub; j++)
      {
        if (amax < fabsf(xb[j]))
        {
          amax = fabsf(xb[j]);
        }
      }

      // UE4M3 scale: amax / 6.0 maps the max E2M1 value (6.0) to amax
      const uint8_t ue = ggml_fp32_to_ue4m3(amax / 6.0f);
      y[i].d[s] = ue;
      const float d = ggml_ue4m3_to_fp32(ue);

      for (int j = 0; j < qk_sub / 2; ++j)
      {
        const uint8_t x0 = best_index_mxfp4(xb[0 + j], d);
        const uint8_t x1 = best_index_mxfp4(xb[qk_sub / 2 + j], d);

        y[i].qs[s * (qk_sub / 2) + j] = x0 | (x1 << 4);
      }
    }
  }
}

//
// 2-6 bit quantization in super-blocks
//

//
// ===================== Helper functions
//

static libgguf_nvfp4_selection libgguf_nvfp4_select_kernel()
{
#if LIBGGUF_CPU_BACKEND_REF
  return {LIBGGUF_CPU_BACKEND_NAME, quantize_row_nvfp4};
#else
  return {LIBGGUF_CPU_BACKEND_NAME, LIBGGUF_CPU_BACKEND_SYMBOL(quantize_row_nvfp4)};
#endif
}

static const libgguf_nvfp4_selection &libgguf_nvfp4_selected()
{
  static const libgguf_nvfp4_selection selected = libgguf_nvfp4_select_kernel();
  return selected;
}

extern "C" const char *libgguf_nvfp4_backend(void)
{
  return libgguf_nvfp4_selected().backend;
}

extern "C" int libgguf_nvfp4_cpu_supports_backend(const char *backend)
{
  return libgguf_cpu_backend_supports_request(backend) ? 1 : 0;
}

static libgguf_nvfp4_kernel_fn libgguf_nvfp4_kernel_for_backend(const char *backend)
{
  if (libgguf_cpu_backend_is_ref_request(backend))
    return quantize_row_nvfp4;
#if !LIBGGUF_CPU_BACKEND_REF
  if (libgguf_cpu_backend_is_compiled_request(backend))
    return LIBGGUF_CPU_BACKEND_SYMBOL(quantize_row_nvfp4);
#endif
  return nullptr;
}

extern "C" size_t libgguf_quantize_nvfp4_for_backend(
    const char *backend,
    const float *RESTRICT src,
    void *RESTRICT dst,
    int64_t nrows,
    int64_t n_per_row)
{
  libgguf_nvfp4_kernel_fn kernel = libgguf_nvfp4_kernel_for_backend(backend);
  if (!kernel)
    return 0;
  kernel(src, (block_nvfp4 *)dst, nrows * n_per_row);
  return nrows * libgguf_row_size(GGML_TYPE_NVFP4, n_per_row);
}

size_t quantize_nvfp4(const float *RESTRICT src, void *RESTRICT dst, int64_t nrow, int64_t n_per_row, const float *quant_weights)
{
  GGML_UNUSED(quant_weights);
  libgguf_nvfp4_selected().kernel(src, (block_nvfp4 *)dst, (int64_t)nrow * n_per_row);
  return nrow * libgguf_row_size(GGML_TYPE_NVFP4, n_per_row);
}

// ====================== Ternary (de)-quantization (BitNet b1.58 and TriLMs)
