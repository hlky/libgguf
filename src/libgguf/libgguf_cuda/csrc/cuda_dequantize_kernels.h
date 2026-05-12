#pragma once

#include <cstdint>

#ifdef LIBGGUF_CUDA_NO_C10
#include "dispatch_utils.h"
#else
#include <ATen/core/ScalarType.h>
#endif
#include <cuda_runtime.h>

void gguf_cuda_dequantize_row(
    const void * x,
    void * y,
    int64_t type,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
);

void gguf_cuda_dequantize_launch_q4_0(
    const void * x,
    void * y,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
);

void gguf_cuda_dequantize_launch_q4_1(
    const void * x,
    void * y,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
);

void gguf_cuda_dequantize_launch_q5_0(
    const void * x,
    void * y,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
);

void gguf_cuda_dequantize_launch_q5_1(
    const void * x,
    void * y,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
);

void gguf_cuda_dequantize_launch_q8_0(
    const void * x,
    void * y,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
);

void gguf_cuda_dequantize_launch_bf16(
    const void * x,
    void * y,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
);

void gguf_cuda_dequantize_launch_q1_0(
    const void * x,
    void * y,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
);

void gguf_cuda_dequantize_launch_tq1_0(
    const void * x,
    void * y,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
);

void gguf_cuda_dequantize_launch_tq2_0(
    const void * x,
    void * y,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
);

void gguf_cuda_dequantize_launch_mxfp4(
    const void * x,
    void * y,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
);

void gguf_cuda_dequantize_launch_nvfp4(
    const void * x,
    void * y,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
);

void gguf_cuda_dequantize_launch_q2_k(
    const void * x,
    void * y,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
);

void gguf_cuda_dequantize_launch_q3_k(
    const void * x,
    void * y,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
);

void gguf_cuda_dequantize_launch_q4_k(
    const void * x,
    void * y,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
);

void gguf_cuda_dequantize_launch_q5_k(
    const void * x,
    void * y,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
);

void gguf_cuda_dequantize_launch_q6_k(
    const void * x,
    void * y,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
);

void gguf_cuda_dequantize_launch_iq2_xxs(
    const void * x,
    void * y,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
);

void gguf_cuda_dequantize_launch_iq2_xs(
    const void * x,
    void * y,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
);

void gguf_cuda_dequantize_launch_iq2_s(
    const void * x,
    void * y,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
);

void gguf_cuda_dequantize_launch_iq3_xxs(
    const void * x,
    void * y,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
);

void gguf_cuda_dequantize_launch_iq3_s(
    const void * x,
    void * y,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
);

void gguf_cuda_dequantize_launch_iq1_s(
    const void * x,
    void * y,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
);

void gguf_cuda_dequantize_launch_iq1_m(
    const void * x,
    void * y,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
);

void gguf_cuda_dequantize_launch_iq4_nl(
    const void * x,
    void * y,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
);

void gguf_cuda_dequantize_launch_iq4_xs(
    const void * x,
    void * y,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
);
