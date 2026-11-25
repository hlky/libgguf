#include "cpu/common/common.h"
#include "cpu/common/dequantize/kernels.h"
#include "quantize/iq_tables.h"
#include <libgguf/cpu/dequantize/decoder.h>


// Share the scalar block implementation with bounded header consumers.

void dequantize_block_q8_0(const block_q8_0 * LIBGGUF_RESTRICT x, float * LIBGGUF_RESTRICT y, int64_t k) {
    static const int qk = QK8_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        libgguf::cpu::BlockDecoder<GGML_TYPE_Q8_0>::decode(
            x + i, y + i*qk, [](uint16_t h) { return LIBGGUF_FP16_TO_FP32(h); });
    }
}


void dequantize_q8_0(const block_q8_0 *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k)
{
  dequantize_block_q8_0(x, y, k);
}

