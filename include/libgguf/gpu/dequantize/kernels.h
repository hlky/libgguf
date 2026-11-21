#pragma once

#include <cstdint>

#include <libgguf/gpu/dispatch.h>
#include <libgguf/gpu/compat.h>
#include <libgguf/gpu/types.h>

void dequantize_q4_0(
    const void * x,
    void * y,
    int k,
    libgguf_gpu_dtype dtype,
    StreamType stream
);

void dequantize_q4_1(
    const void * x,
    void * y,
    int k,
    libgguf_gpu_dtype dtype,
    StreamType stream
);

void dequantize_q5_0(
    const void * x,
    void * y,
    int k,
    libgguf_gpu_dtype dtype,
    StreamType stream
);

void dequantize_q5_1(
    const void * x,
    void * y,
    int k,
    libgguf_gpu_dtype dtype,
    StreamType stream
);

void dequantize_q8_0(
    const void * x,
    void * y,
    int k,
    libgguf_gpu_dtype dtype,
    StreamType stream
);

void dequantize_bf16(
    const void * x,
    void * y,
    int k,
    libgguf_gpu_dtype dtype,
    StreamType stream
);

void dequantize_q1_0(
    const void * x,
    void * y,
    int k,
    libgguf_gpu_dtype dtype,
    StreamType stream
);

void dequantize_tq1_0(
    const void * x,
    void * y,
    int k,
    libgguf_gpu_dtype dtype,
    StreamType stream
);

void dequantize_tq2_0(
    const void * x,
    void * y,
    int k,
    libgguf_gpu_dtype dtype,
    StreamType stream
);

void dequantize_mxfp4(
    const void * x,
    void * y,
    int k,
    libgguf_gpu_dtype dtype,
    StreamType stream
);

void dequantize_nvfp4(
    const void * x,
    void * y,
    int k,
    libgguf_gpu_dtype dtype,
    StreamType stream
);

void dequantize_q2_k(
    const void * x,
    void * y,
    int k,
    libgguf_gpu_dtype dtype,
    StreamType stream
);

void dequantize_q3_k(
    const void * x,
    void * y,
    int k,
    libgguf_gpu_dtype dtype,
    StreamType stream
);

void dequantize_q4_k(
    const void * x,
    void * y,
    int k,
    libgguf_gpu_dtype dtype,
    StreamType stream
);

void dequantize_q5_k(
    const void * x,
    void * y,
    int k,
    libgguf_gpu_dtype dtype,
    StreamType stream
);

void dequantize_q6_k(
    const void * x,
    void * y,
    int k,
    libgguf_gpu_dtype dtype,
    StreamType stream
);

void dequantize_iq2_xxs(
    const void * x,
    void * y,
    int k,
    libgguf_gpu_dtype dtype,
    StreamType stream
);

void dequantize_iq2_xs(
    const void * x,
    void * y,
    int k,
    libgguf_gpu_dtype dtype,
    StreamType stream
);

void dequantize_iq2_s(
    const void * x,
    void * y,
    int k,
    libgguf_gpu_dtype dtype,
    StreamType stream
);

void dequantize_iq3_xxs(
    const void * x,
    void * y,
    int k,
    libgguf_gpu_dtype dtype,
    StreamType stream
);

void dequantize_iq3_s(
    const void * x,
    void * y,
    int k,
    libgguf_gpu_dtype dtype,
    StreamType stream
);

void dequantize_iq1_s(
    const void * x,
    void * y,
    int k,
    libgguf_gpu_dtype dtype,
    StreamType stream
);

void dequantize_iq1_m(
    const void * x,
    void * y,
    int k,
    libgguf_gpu_dtype dtype,
    StreamType stream
);

void dequantize_iq4_nl(
    const void * x,
    void * y,
    int k,
    libgguf_gpu_dtype dtype,
    StreamType stream
);

void dequantize_iq4_xs(
    const void * x,
    void * y,
    int k,
    libgguf_gpu_dtype dtype,
    StreamType stream
);
