#include "libgguf_common.h"

extern "C" void quantize_row_iq3_xxs_sse2(const float *RESTRICT x, block_iq3_xxs *RESTRICT y, int64_t k)
{
  // IQ3_XXS uses grid search and bit packing; keep this backend byte-identical.
  quantize_row_iq3_xxs_ref(x, y, k);
}
