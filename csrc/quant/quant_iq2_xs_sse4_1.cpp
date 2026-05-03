#include "libgguf_common.h"

extern "C" void quantize_row_iq2_xs_sse4_1(const float *RESTRICT x, block_iq2_xs *RESTRICT y, int64_t k)
{
  // IQ2_XS uses weighted table-neighbour search; keep this backend byte-identical.
  quantize_iq2_xs(x, y, 1, k, nullptr);
}
