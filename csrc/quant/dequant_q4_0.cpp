#include "common/libgguf_common.h"
#include "common/libgguf_cpu.h"

#include <cstdlib>
#include <cstring>

typedef void (*libgguf_dequant_q4_0_kernel_fn)(const block_q4_0 *RESTRICT, float *RESTRICT, int64_t);

struct libgguf_dequant_q4_0_selection
{
  const char *backend;
  libgguf_dequant_q4_0_kernel_fn kernel;
};

extern "C" void dequantize_row_q4_0_sse2(const block_q4_0 *RESTRICT x, float *RESTRICT y, int64_t k);
extern "C" void dequantize_row_q4_0_avx2(const block_q4_0 *RESTRICT x, float *RESTRICT y, int64_t k);
extern "C" void dequantize_row_q4_0_sse4_1(const block_q4_0 *RESTRICT x, float *RESTRICT y, int64_t k);
extern "C" void dequantize_row_q4_0_ref(const block_q4_0 *RESTRICT x, float *RESTRICT y, int64_t k);

static libgguf_dequant_q4_0_selection libgguf_dequant_q4_0_select_kernel()
{
  const char *forced = std::getenv("LIBGGUF_DEQUANT_Q4_0_BACKEND");
  const libgguf_cpu_features &features = libgguf_get_cpu_features();

  if (forced != nullptr && forced[0] != '\0')
  {
    if (std::strcmp(forced, "ref") == 0)
    {
      return {"ref", dequantize_row_q4_0_ref};
    }
    if (std::strcmp(forced, "sse4_1") == 0 && features.sse4_1)
    {
      return {"sse4_1", dequantize_row_q4_0_sse4_1};
    }
    if (std::strcmp(forced, "sse2") == 0 && features.sse2)
    {
      return {"sse2", dequantize_row_q4_0_sse2};
    }
    if (std::strcmp(forced, "avx2") == 0 && features.avx2)
    {
      return {"avx2", dequantize_row_q4_0_avx2};
    }
  }

  if (features.avx2)
  {
    return {"avx2", dequantize_row_q4_0_avx2};
  }
  if (features.sse4_1)
  {
    return {"sse4_1", dequantize_row_q4_0_sse4_1};
  }
  if (features.sse2)
  {
    return {"sse2", dequantize_row_q4_0_sse2};
  }

  return {"ref", dequantize_row_q4_0_ref};
}

static const libgguf_dequant_q4_0_selection &libgguf_dequant_q4_0_selected()
{
  static const libgguf_dequant_q4_0_selection selected = libgguf_dequant_q4_0_select_kernel();
  return selected;
}

extern "C" void dequantize_row_q4_0(const block_q4_0 *RESTRICT x, float *RESTRICT y, int64_t k)
{
  libgguf_dequant_q4_0_selected().kernel(x, y, k);
}

extern "C" const char *libgguf_dequant_q4_0_backend(void)
{
  return libgguf_dequant_q4_0_selected().backend;
}
