#include "kernels/cpu/common/common.h"
#include "kernels/cpu/dequantize/kernels.h"
#include "quantize/iq_tables.h"


// Vendored from third_party/llama.cpp/ggml/src/ggml-quants.c.

void dequantize_block_q5_k(const block_q5_K * LIBGGUF_RESTRICT x, float * LIBGGUF_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const uint8_t * ql = x[i].qs;
        const uint8_t * qh = x[i].qh;

        const float d = LIBGGUF_FP16_TO_FP32(x[i].d);
        const float min = LIBGGUF_FP16_TO_FP32(x[i].dmin);

        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l]  >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32; is += 2;
            u1 <<= 2; u2 <<= 2;
        }
    }
}


void dequantize_q5_k(const block_q5_K *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k)
{
  dequantize_block_q5_k(x, y, k);
}

