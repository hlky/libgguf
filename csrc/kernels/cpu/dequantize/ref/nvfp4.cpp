#include "kernels/cpu/common/common.h"
#include "kernels/cpu/dequantize/kernels.h"
#include "quantize/iq_tables.h"


// Vendored from third_party/llama.cpp/ggml/src/ggml-quants.c.

void dequantize_block_nvfp4(const block_nvfp4 * LIBGGUF_RESTRICT x, float * LIBGGUF_RESTRICT y, int64_t k) {
    static const int qk = QK_NVFP4;
    static const int qk_sub = QK_NVFP4_SUB;
    static const int n_sub = QK_NVFP4 / QK_NVFP4_SUB;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        for (int s = 0; s < n_sub; s++) {
            const float d = libgguf_ue4m3_to_fp32(x[i].d[s]);
            float * yb = y + i*qk + s*qk_sub;

            for (int j = 0; j < qk_sub/2; ++j) {
                const int8_t v0 = kvalues_mxfp4[x[i].qs[s*(qk_sub/2) + j] & 0x0F];
                const int8_t v1 = kvalues_mxfp4[x[i].qs[s*(qk_sub/2) + j] >>   4];

                yb[j + 0       ] = v0*d;
                yb[j + qk_sub/2] = v1*d;
            }
        }
    }
}


void dequantize_nvfp4(const block_nvfp4 *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k)
{
  dequantize_block_nvfp4(x, y, k);
}

