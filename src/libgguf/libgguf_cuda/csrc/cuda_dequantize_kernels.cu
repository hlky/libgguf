#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifndef LIBGGUF_CUDA_NO_C10
#include <torch/all.h>
#endif

#include "cuda_dequantize_kernels.h"

void gguf_cuda_dequantize_row(
    const void * x,
    void * y,
    int64_t type,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
) {
    switch (type) {
        case 2:
            gguf_cuda_dequantize_launch_q4_0(x, y, k, dtype, stream);
            return;
        case 3:
            gguf_cuda_dequantize_launch_q4_1(x, y, k, dtype, stream);
            return;
        case 6:
            gguf_cuda_dequantize_launch_q5_0(x, y, k, dtype, stream);
            return;
        case 7:
            gguf_cuda_dequantize_launch_q5_1(x, y, k, dtype, stream);
            return;
        case 8:
            gguf_cuda_dequantize_launch_q8_0(x, y, k, dtype, stream);
            return;
        case 10:
            gguf_cuda_dequantize_launch_q2_k(x, y, k, dtype, stream);
            return;
        case 11:
            gguf_cuda_dequantize_launch_q3_k(x, y, k, dtype, stream);
            return;
        case 12:
            gguf_cuda_dequantize_launch_q4_k(x, y, k, dtype, stream);
            return;
        case 13:
            gguf_cuda_dequantize_launch_q5_k(x, y, k, dtype, stream);
            return;
        case 14:
            gguf_cuda_dequantize_launch_q6_k(x, y, k, dtype, stream);
            return;
        case 16:
            gguf_cuda_dequantize_launch_iq2_xxs(x, y, k, dtype, stream);
            return;
        case 17:
            gguf_cuda_dequantize_launch_iq2_xs(x, y, k, dtype, stream);
            return;
        case 18:
            gguf_cuda_dequantize_launch_iq3_xxs(x, y, k, dtype, stream);
            return;
        case 19:
            gguf_cuda_dequantize_launch_iq1_s(x, y, k, dtype, stream);
            return;
        case 20:
            gguf_cuda_dequantize_launch_iq4_nl(x, y, k, dtype, stream);
            return;
        case 21:
            gguf_cuda_dequantize_launch_iq3_s(x, y, k, dtype, stream);
            return;
        case 22:
            gguf_cuda_dequantize_launch_iq2_s(x, y, k, dtype, stream);
            return;
        case 23:
            gguf_cuda_dequantize_launch_iq4_xs(x, y, k, dtype, stream);
            return;
        case 29:
            gguf_cuda_dequantize_launch_iq1_m(x, y, k, dtype, stream);
            return;
        case 30:
            gguf_cuda_dequantize_launch_bf16(x, y, k, dtype, stream);
            return;
        case 34:
            gguf_cuda_dequantize_launch_tq1_0(x, y, k, dtype, stream);
            return;
        case 35:
            gguf_cuda_dequantize_launch_tq2_0(x, y, k, dtype, stream);
            return;
        case 39:
            gguf_cuda_dequantize_launch_mxfp4(x, y, k, dtype, stream);
            return;
        case 40:
            gguf_cuda_dequantize_launch_nvfp4(x, y, k, dtype, stream);
            return;
        case 41:
            gguf_cuda_dequantize_launch_q1_0(x, y, k, dtype, stream);
            return;
        default:
#ifndef LIBGGUF_CUDA_NO_C10
            TORCH_CHECK(false, "Unsupported GGML quantization type for CUDA dequantize: ", type);
#endif
            return;
    }
}
