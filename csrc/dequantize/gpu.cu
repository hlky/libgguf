#include "dequantize/gpu.h"

#include <libgguf/gpu/compat.h>

#include <libgguf/gpu/dequantize/kernels.h>

bool dequantize_type_supported(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_BF16:
        case GGML_TYPE_TQ1_0:
        case GGML_TYPE_TQ2_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q1_0:
            return true;
        default:
            return false;
    }
}

void dequantize_row(
    const void * x,
    void * y,
    enum ggml_type type,
    int k,
    libgguf_gpu_dtype dtype,
    StreamType stream
) {
    switch (type) {
        case GGML_TYPE_Q4_0:
            dequantize_q4_0(x, y, k, dtype, stream);
            return;
        case GGML_TYPE_Q4_1:
            dequantize_q4_1(x, y, k, dtype, stream);
            return;
        case GGML_TYPE_Q5_0:
            dequantize_q5_0(x, y, k, dtype, stream);
            return;
        case GGML_TYPE_Q5_1:
            dequantize_q5_1(x, y, k, dtype, stream);
            return;
        case GGML_TYPE_Q8_0:
            dequantize_q8_0(x, y, k, dtype, stream);
            return;
        case GGML_TYPE_Q2_K:
            dequantize_q2_k(x, y, k, dtype, stream);
            return;
        case GGML_TYPE_Q3_K:
            dequantize_q3_k(x, y, k, dtype, stream);
            return;
        case GGML_TYPE_Q4_K:
            dequantize_q4_k(x, y, k, dtype, stream);
            return;
        case GGML_TYPE_Q5_K:
            dequantize_q5_k(x, y, k, dtype, stream);
            return;
        case GGML_TYPE_Q6_K:
            dequantize_q6_k(x, y, k, dtype, stream);
            return;
        case GGML_TYPE_IQ2_XXS:
            dequantize_iq2_xxs(x, y, k, dtype, stream);
            return;
        case GGML_TYPE_IQ2_XS:
            dequantize_iq2_xs(x, y, k, dtype, stream);
            return;
        case GGML_TYPE_IQ3_XXS:
            dequantize_iq3_xxs(x, y, k, dtype, stream);
            return;
        case GGML_TYPE_IQ1_S:
            dequantize_iq1_s(x, y, k, dtype, stream);
            return;
        case GGML_TYPE_IQ4_NL:
            dequantize_iq4_nl(x, y, k, dtype, stream);
            return;
        case GGML_TYPE_IQ3_S:
            dequantize_iq3_s(x, y, k, dtype, stream);
            return;
        case GGML_TYPE_IQ2_S:
            dequantize_iq2_s(x, y, k, dtype, stream);
            return;
        case GGML_TYPE_IQ4_XS:
            dequantize_iq4_xs(x, y, k, dtype, stream);
            return;
        case GGML_TYPE_IQ1_M:
            dequantize_iq1_m(x, y, k, dtype, stream);
            return;
        case GGML_TYPE_BF16:
            dequantize_bf16(x, y, k, dtype, stream);
            return;
        case GGML_TYPE_TQ1_0:
            dequantize_tq1_0(x, y, k, dtype, stream);
            return;
        case GGML_TYPE_TQ2_0:
            dequantize_tq2_0(x, y, k, dtype, stream);
            return;
        case GGML_TYPE_MXFP4:
            dequantize_mxfp4(x, y, k, dtype, stream);
            return;
        case GGML_TYPE_NVFP4:
            dequantize_nvfp4(x, y, k, dtype, stream);
            return;
        case GGML_TYPE_Q1_0:
            dequantize_q1_0(x, y, k, dtype, stream);
            return;
        default:
            return;
    }
}
