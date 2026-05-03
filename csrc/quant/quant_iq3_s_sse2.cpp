#include "libgguf_common.h"

extern "C" void quantize_row_iq3_s_sse2(const float *RESTRICT x, block_iq3_s *RESTRICT y, int64_t k)
{
  // IQ3_S uses weighted grid search and bit packing; keep this backend byte-identical.
  quantize_row_iq3_s_ref(x, y, k);
}
