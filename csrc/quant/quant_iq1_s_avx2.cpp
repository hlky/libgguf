#include "libgguf_common.h"

extern "C" void quantize_row_iq1_s_avx2(const float *RESTRICT x, block_iq1_s *RESTRICT y, int64_t k)
{
  // IQ1_S uses weighted table-neighbour search; keep this backend byte-identical.
  quantize_iq1_s(x, y, 1, k, nullptr);
}
