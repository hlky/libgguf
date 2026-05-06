#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_quantize_kernels.h"
#include "libgguf_cuda_common.h"

int64_t gguf_cuda_quantize_row_size_for_type(int64_t type, int64_t n) {
    switch (type) {

        case GGML_TYPE_IQ4_NL:
            return n * (int64_t)sizeof(block_iq4_nl) / QK4_NL;
        case GGML_TYPE_IQ4_XS:
            return n * (int64_t)sizeof(block_iq4_xs) / QK_K;
        case GGML_TYPE_IQ2_XXS:
            return n * (int64_t)sizeof(block_iq2_xxs) / QK_K;
        case GGML_TYPE_IQ2_XS:
            return n * (int64_t)sizeof(block_iq2_xs) / QK_K;
        case GGML_TYPE_IQ2_S:
            return n * (int64_t)sizeof(block_iq2_s) / QK_K;
        case GGML_TYPE_IQ3_XXS:
            return n * (int64_t)sizeof(block_iq3_xxs) / QK_K;
        case GGML_TYPE_IQ3_S:
            return n * (int64_t)sizeof(block_iq3_s) / QK_K;
        case GGML_TYPE_IQ1_S:
            return n * (int64_t)sizeof(block_iq1_s) / QK_K;
        case GGML_TYPE_IQ1_M:
            return n * (int64_t)sizeof(block_iq1_m) / QK_K;
        case GGML_TYPE_MXFP4:
            return n * (int64_t)sizeof(block_mxfp4) / QK_MXFP4;
        case GGML_TYPE_NVFP4:
            return n * (int64_t)sizeof(block_nvfp4) / QK_NVFP4;
        case GGML_TYPE_Q1_0:
            return n * (int64_t)sizeof(block_q1_0) / QK1_0;
        case GGML_TYPE_Q2_K:
            return n * (int64_t)sizeof(block_q2_K) / QK_K;
        case GGML_TYPE_Q3_K:
            return n * (int64_t)sizeof(block_q3_K) / QK_K;
        case GGML_TYPE_Q4_K:
            return n * (int64_t)sizeof(block_q4_K) / QK_K;
        case GGML_TYPE_Q4_0:
            return n * (int64_t)sizeof(block_q4_0) / QK4_0;
        case GGML_TYPE_Q4_1:
            return n * (int64_t)sizeof(block_q4_1) / QK4_1;
        case GGML_TYPE_Q5_0:
            return n * (int64_t)sizeof(block_q5_0) / QK5_0;
        case GGML_TYPE_Q5_1:
            return n * (int64_t)sizeof(block_q5_1) / QK5_1;
        case GGML_TYPE_Q5_K:
            return n * (int64_t)sizeof(block_q5_K) / QK_K;
        case GGML_TYPE_Q6_K:
            return n * (int64_t)sizeof(block_q6_K) / QK_K;
        case GGML_TYPE_Q8_0:
            return n * (int64_t)sizeof(block_q8_0) / QK8_0;
        case GGML_TYPE_TQ1_0:
            return n * (int64_t)sizeof(block_tq1_0) / QK_K;
        case GGML_TYPE_TQ2_0:
            return n * (int64_t)sizeof(block_tq2_0) / QK_K;
        default:
            return 0;
    }
}

int64_t gguf_cuda_quantize_block_size_for_type(int64_t type) {
    switch (type) {

        case GGML_TYPE_IQ4_NL:
            return QK4_NL;
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_TQ1_0:
        case GGML_TYPE_TQ2_0:
            return QK_K;
        case GGML_TYPE_MXFP4:
            return QK_MXFP4;
        case GGML_TYPE_NVFP4:
            return QK_NVFP4;
        case GGML_TYPE_Q1_0:
            return QK1_0;
        case GGML_TYPE_Q4_0:
            return QK4_0;
        case GGML_TYPE_Q4_1:
            return QK4_1;
        case GGML_TYPE_Q5_0:
            return QK5_0;
        case GGML_TYPE_Q5_1:
            return QK5_1;
        case GGML_TYPE_Q8_0:
            return QK8_0;
        default:
            return 0;
    }
}

bool gguf_cuda_quantize_type_needs_imatrix(int64_t type) {
    switch (type) {
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ1_S:
            return true;
        default:
            return false;
    }
}

void gguf_cuda_quantize_row(
    const float * x,
    const float * quant_weights,
    void * y,
    int64_t type,
    int64_t k,
    int64_t n_per_row,
    cudaStream_t stream
) {
    switch (type) {
        case GGML_TYPE_IQ4_NL:
            gguf_cuda_quantize_launch_iq4_nl(x, quant_weights, y, k, n_per_row, stream);
            return;
        case GGML_TYPE_IQ4_XS:
            gguf_cuda_quantize_launch_iq4_xs(x, quant_weights, y, k, n_per_row, stream);
            return;
        case GGML_TYPE_IQ2_XXS:
            gguf_cuda_quantize_launch_iq2_xxs(x, quant_weights, y, k, n_per_row, stream);
            return;
        case GGML_TYPE_IQ2_XS:
            gguf_cuda_quantize_launch_iq2_xs(x, quant_weights, y, k, n_per_row, stream);
            return;
        case GGML_TYPE_IQ2_S:
            gguf_cuda_quantize_launch_iq2_s(x, quant_weights, y, k, n_per_row, stream);
            return;
        case GGML_TYPE_IQ3_XXS:
            gguf_cuda_quantize_launch_iq3_xxs(x, quant_weights, y, k, n_per_row, stream);
            return;
        case GGML_TYPE_IQ3_S:
            gguf_cuda_quantize_launch_iq3_s(x, quant_weights, y, k, n_per_row, stream);
            return;
        case GGML_TYPE_IQ1_S:
            gguf_cuda_quantize_launch_iq1_s(x, quant_weights, y, k, n_per_row, stream);
            return;
        case GGML_TYPE_IQ1_M:
            gguf_cuda_quantize_launch_iq1_m(x, quant_weights, y, k, n_per_row, stream);
            return;
        case GGML_TYPE_MXFP4:
            gguf_cuda_quantize_launch_mxfp4(x, quant_weights, y, k, n_per_row, stream);
            return;
        case GGML_TYPE_NVFP4:
            gguf_cuda_quantize_launch_nvfp4(x, quant_weights, y, k, n_per_row, stream);
            return;
        case GGML_TYPE_Q1_0:
            gguf_cuda_quantize_launch_q1_0(x, quant_weights, y, k, n_per_row, stream);
            return;
        case GGML_TYPE_Q2_K:
            gguf_cuda_quantize_launch_q2_k(x, quant_weights, y, k, n_per_row, stream);
            return;
        case GGML_TYPE_Q3_K:
            gguf_cuda_quantize_launch_q3_k(x, quant_weights, y, k, n_per_row, stream);
            return;
        case GGML_TYPE_Q4_K:
            gguf_cuda_quantize_launch_q4_k(x, quant_weights, y, k, n_per_row, stream);
            return;
        case GGML_TYPE_Q4_0:
            gguf_cuda_quantize_launch_q4_0(x, quant_weights, y, k, n_per_row, stream);
            return;
        case GGML_TYPE_Q4_1:
            gguf_cuda_quantize_launch_q4_1(x, quant_weights, y, k, n_per_row, stream);
            return;
        case GGML_TYPE_Q5_0:
            gguf_cuda_quantize_launch_q5_0(x, quant_weights, y, k, n_per_row, stream);
            return;
        case GGML_TYPE_Q5_1:
            gguf_cuda_quantize_launch_q5_1(x, quant_weights, y, k, n_per_row, stream);
            return;
        case GGML_TYPE_Q5_K:
            gguf_cuda_quantize_launch_q5_k(x, quant_weights, y, k, n_per_row, stream);
            return;
        case GGML_TYPE_Q6_K:
            gguf_cuda_quantize_launch_q6_k(x, quant_weights, y, k, n_per_row, stream);
            return;
        case GGML_TYPE_Q8_0:
            gguf_cuda_quantize_launch_q8_0(x, quant_weights, y, k, n_per_row, stream);
            return;
        case GGML_TYPE_TQ1_0:
            gguf_cuda_quantize_launch_tq1_0(x, quant_weights, y, k, n_per_row, stream);
            return;
        case GGML_TYPE_TQ2_0:
            gguf_cuda_quantize_launch_tq2_0(x, quant_weights, y, k, n_per_row, stream);
            return;
        default:
            return;
    }
}
