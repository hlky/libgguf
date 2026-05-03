#include "libgguf_common.h"

extern "C" void quantize_row_iq2_xxs_sse2(const float *RESTRICT x, block_iq2_xxs *RESTRICT y, int64_t k)
{
  // IQ2_XXS uses weighted table-neighbour search; keep this backend byte-identical.
  quantize_iq2_xxs(x, y, 1, k, nullptr);
}
