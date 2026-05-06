#pragma once

#include <cstdint>

#include <cuda_runtime.h>

int64_t gguf_cuda_quantize_row_size_for_type(int64_t type, int64_t n);
int64_t gguf_cuda_quantize_block_size_for_type(int64_t type);
bool gguf_cuda_quantize_type_needs_imatrix(int64_t type);

void gguf_cuda_quantize_row(
    const float * x,
    const float * quant_weights,
    void * y,
    int64_t type,
    int64_t k,
    int64_t n_per_row,
    cudaStream_t stream
);

void gguf_cuda_quantize_launch_q8_0(
    const float * x,
    const float * quant_weights,
    void * y,
    int64_t k,
    int64_t n_per_row,
    cudaStream_t stream
);

void gguf_cuda_quantize_launch_q1_0(
    const float * x,
    const float * quant_weights,
    void * y,
    int64_t k,
    int64_t n_per_row,
    cudaStream_t stream
);

void gguf_cuda_quantize_launch_mxfp4(
    const float * x,
    const float * quant_weights,
    void * y,
    int64_t k,
    int64_t n_per_row,
    cudaStream_t stream
);

void gguf_cuda_quantize_launch_nvfp4(
    const float * x,
    const float * quant_weights,
    void * y,
    int64_t k,
    int64_t n_per_row,
    cudaStream_t stream
);

void gguf_cuda_quantize_launch_q4_0(
    const float * x,
    const float * quant_weights,
    void * y,
    int64_t k,
    int64_t n_per_row,
    cudaStream_t stream
);

void gguf_cuda_quantize_launch_q4_1(
    const float * x,
    const float * quant_weights,
    void * y,
    int64_t k,
    int64_t n_per_row,
    cudaStream_t stream
);

void gguf_cuda_quantize_launch_q5_0(
    const float * x,
    const float * quant_weights,
    void * y,
    int64_t k,
    int64_t n_per_row,
    cudaStream_t stream
);

void gguf_cuda_quantize_launch_q5_1(
    const float * x,
    const float * quant_weights,
    void * y,
    int64_t k,
    int64_t n_per_row,
    cudaStream_t stream
);

void gguf_cuda_quantize_launch_q2_k(
    const float * x,
    const float * quant_weights,
    void * y,
    int64_t k,
    int64_t n_per_row,
    cudaStream_t stream
);

void gguf_cuda_quantize_launch_q3_k(
    const float * x,
    const float * quant_weights,
    void * y,
    int64_t k,
    int64_t n_per_row,
    cudaStream_t stream
);

void gguf_cuda_quantize_launch_q4_k(
    const float * x,
    const float * quant_weights,
    void * y,
    int64_t k,
    int64_t n_per_row,
    cudaStream_t stream
);

void gguf_cuda_quantize_launch_q5_k(
    const float * x,
    const float * quant_weights,
    void * y,
    int64_t k,
    int64_t n_per_row,
    cudaStream_t stream
);

void gguf_cuda_quantize_launch_iq4_nl(
    const float * x,
    const float * quant_weights,
    void * y,
    int64_t k,
    int64_t n_per_row,
    cudaStream_t stream
);

void gguf_cuda_quantize_launch_iq4_xs(
    const float * x,
    const float * quant_weights,
    void * y,
    int64_t k,
    int64_t n_per_row,
    cudaStream_t stream
);

void gguf_cuda_quantize_launch_iq2_xxs(
    const float * x,
    const float * quant_weights,
    void * y,
    int64_t k,
    int64_t n_per_row,
    cudaStream_t stream
);

void gguf_cuda_quantize_launch_iq2_xs(
    const float * x,
    const float * quant_weights,
    void * y,
    int64_t k,
    int64_t n_per_row,
    cudaStream_t stream
);

void gguf_cuda_quantize_launch_iq2_s(
    const float * x,
    const float * quant_weights,
    void * y,
    int64_t k,
    int64_t n_per_row,
    cudaStream_t stream
);

void gguf_cuda_quantize_launch_iq3_xxs(
    const float * x,
    const float * quant_weights,
    void * y,
    int64_t k,
    int64_t n_per_row,
    cudaStream_t stream
);

void gguf_cuda_quantize_launch_iq3_s(
    const float * x,
    const float * quant_weights,
    void * y,
    int64_t k,
    int64_t n_per_row,
    cudaStream_t stream
);

void gguf_cuda_quantize_launch_iq1_s(
    const float * x,
    const float * quant_weights,
    void * y,
    int64_t k,
    int64_t n_per_row,
    cudaStream_t stream
);

void gguf_cuda_quantize_launch_iq1_m(
    const float * x,
    const float * quant_weights,
    void * y,
    int64_t k,
    int64_t n_per_row,
    cudaStream_t stream
);

void gguf_cuda_quantize_launch_tq1_0(
    const float * x,
    const float * quant_weights,
    void * y,
    int64_t k,
    int64_t n_per_row,
    cudaStream_t stream
);

void gguf_cuda_quantize_launch_tq2_0(
    const float * x,
    const float * quant_weights,
    void * y,
    int64_t k,
    int64_t n_per_row,
    cudaStream_t stream
);

void gguf_cuda_quantize_launch_q6_k(
    const float * x,
    const float * quant_weights,
    void * y,
    int64_t k,
    int64_t n_per_row,
    cudaStream_t stream
);
