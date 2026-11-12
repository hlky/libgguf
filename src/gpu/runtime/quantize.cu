#include "quantize/gpu.h"

#include <libgguf/gpu/compat.h>
#include <libgguf/gpu/quantize/kernels.h>
#include <libgguf/gpu/common.cuh>

bool quantize_type_supported(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_TQ1_0:
        case GGML_TYPE_TQ2_0:
            return true;
        default:
            return false;
    }
}

void quantize_row(
    const float * x,
    const float * quant_weights,
    void * y,
    enum ggml_type type,
    int64_t k,
    int64_t n_per_row,
    StreamType stream
) {
    switch (type) {
        case GGML_TYPE_IQ4_NL:
            quantize_iq4_nl(x, quant_weights, y, k, n_per_row, libgguf_gpu_dtype::f32, stream);
            return;
        case GGML_TYPE_IQ4_XS:
            quantize_iq4_xs(x, quant_weights, y, k, n_per_row, libgguf_gpu_dtype::f32, stream);
            return;
        case GGML_TYPE_IQ2_XXS:
            quantize_iq2_xxs(x, quant_weights, y, k, n_per_row, libgguf_gpu_dtype::f32, stream);
            return;
        case GGML_TYPE_IQ2_XS:
            quantize_iq2_xs(x, quant_weights, y, k, n_per_row, libgguf_gpu_dtype::f32, stream);
            return;
        case GGML_TYPE_IQ2_S:
            quantize_iq2_s(x, quant_weights, y, k, n_per_row, libgguf_gpu_dtype::f32, stream);
            return;
        case GGML_TYPE_IQ3_XXS:
            quantize_iq3_xxs(x, quant_weights, y, k, n_per_row, libgguf_gpu_dtype::f32, stream);
            return;
        case GGML_TYPE_IQ3_S:
            quantize_iq3_s(x, quant_weights, y, k, n_per_row, libgguf_gpu_dtype::f32, stream);
            return;
        case GGML_TYPE_IQ1_S:
            quantize_iq1_s(x, quant_weights, y, k, n_per_row, libgguf_gpu_dtype::f32, stream);
            return;
        case GGML_TYPE_IQ1_M:
            quantize_iq1_m(x, quant_weights, y, k, n_per_row, libgguf_gpu_dtype::f32, stream);
            return;
        case GGML_TYPE_MXFP4:
            quantize_mxfp4(x, quant_weights, y, k, n_per_row, libgguf_gpu_dtype::f32, stream);
            return;
        case GGML_TYPE_NVFP4:
            quantize_nvfp4(x, quant_weights, y, k, n_per_row, libgguf_gpu_dtype::f32, stream);
            return;
        case GGML_TYPE_Q1_0:
            quantize_q1_0(x, quant_weights, y, k, n_per_row, libgguf_gpu_dtype::f32, stream);
            return;
        case GGML_TYPE_Q2_K:
            quantize_q2_k(x, quant_weights, y, k, n_per_row, libgguf_gpu_dtype::f32, stream);
            return;
        case GGML_TYPE_Q3_K:
            quantize_q3_k(x, quant_weights, y, k, n_per_row, libgguf_gpu_dtype::f32, stream);
            return;
        case GGML_TYPE_Q4_K:
            quantize_q4_k(x, quant_weights, y, k, n_per_row, libgguf_gpu_dtype::f32, stream);
            return;
        case GGML_TYPE_Q4_0:
            quantize_q4_0(x, quant_weights, y, k, n_per_row, libgguf_gpu_dtype::f32, stream);
            return;
        case GGML_TYPE_Q4_1:
            quantize_q4_1(x, quant_weights, y, k, n_per_row, libgguf_gpu_dtype::f32, stream);
            return;
        case GGML_TYPE_Q5_0:
            quantize_q5_0(x, quant_weights, y, k, n_per_row, libgguf_gpu_dtype::f32, stream);
            return;
        case GGML_TYPE_Q5_1:
            quantize_q5_1(x, quant_weights, y, k, n_per_row, libgguf_gpu_dtype::f32, stream);
            return;
        case GGML_TYPE_Q5_K:
            quantize_q5_k(x, quant_weights, y, k, n_per_row, libgguf_gpu_dtype::f32, stream);
            return;
        case GGML_TYPE_Q6_K:
            quantize_q6_k(x, quant_weights, y, k, n_per_row, libgguf_gpu_dtype::f32, stream);
            return;
        case GGML_TYPE_Q8_0:
            quantize_q8_0(x, quant_weights, y, k, n_per_row, libgguf_gpu_dtype::f32, stream);
            return;
        case GGML_TYPE_TQ1_0:
            quantize_tq1_0(x, quant_weights, y, k, n_per_row, libgguf_gpu_dtype::f32, stream);
            return;
        case GGML_TYPE_TQ2_0:
            quantize_tq2_0(x, quant_weights, y, k, n_per_row, libgguf_gpu_dtype::f32, stream);
            return;
        default:
            return;
    }
}
