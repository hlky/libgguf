#include "libgguf_common.h"

extern "C" void quantize_row_iq2_s_avx2(const float *RESTRICT x, block_iq2_s *RESTRICT y, int64_t k)
{
  // IQ2_S uses weighted table-neighbour search; keep this backend byte-identical.
  quantize_row_iq2_s_ref(x, y, k);
}
