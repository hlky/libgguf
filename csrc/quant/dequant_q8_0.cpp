#include "common/libgguf_common.h"
#include "common/libgguf_cpu.h"

#include <cstring>

typedef void (*libgguf_dequant_q8_0_kernel_fn)(const block_q8_0 *RESTRICT, float *RESTRICT, int64_t);

struct libgguf_dequant_q8_0_selection
{
  const char *backend;
  libgguf_dequant_q8_0_kernel_fn kernel;
};

extern "C" void dequantize_row_q8_0_sse2(const block_q8_0 *RESTRICT x, float *RESTRICT y, int64_t k);
extern "C" void dequantize_row_q8_0_sse4_1(const block_q8_0 *RESTRICT x, float *RESTRICT y, int64_t k);
extern "C" void dequantize_row_q8_0_avx2(const block_q8_0 *RESTRICT x, float *RESTRICT y, int64_t k);
extern "C" void dequantize_row_q8_0_ref(const block_q8_0 *RESTRICT x, float *RESTRICT y, int64_t k);

static libgguf_dequant_q8_0_selection libgguf_dequant_q8_0_select_kernel()
{
  return {"ref", dequantize_row_q8_0_ref};
}

static const libgguf_dequant_q8_0_selection &libgguf_dequant_q8_0_selected()
{
  static const libgguf_dequant_q8_0_selection selected = libgguf_dequant_q8_0_select_kernel();
  return selected;
}

extern "C" void dequantize_row_q8_0(const block_q8_0 *RESTRICT x, float *RESTRICT y, int64_t k)
{
  libgguf_dequant_q8_0_selected().kernel(x, y, k);
}

extern "C" const char *libgguf_dequant_q8_0_backend(void)
{
  return libgguf_dequant_q8_0_selected().backend;
}
