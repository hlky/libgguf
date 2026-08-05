#include "kernels/cpu/common/common.h"
#include "kernels/cpu/dequantize/kernels.h"
#include "quantize/iq_tables.h"


// Vendored from third_party/llama.cpp/ggml/src/ggml-quants.c.

void dequantize_block_q8_0(const block_q8_0 * LIBGGUF_RESTRICT x, float * LIBGGUF_RESTRICT y, int64_t k) {
    static const int qk = QK8_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = LIBGGUF_FP16_TO_FP32(x[i].d);

        for (int j = 0; j < qk; ++j) {
            y[i*qk + j] = x[i].qs[j]*d;
        }
    }
}


void dequantize_q8_0(const block_q8_0 *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k)
{
  dequantize_block_q8_0(x, y, k);
}

