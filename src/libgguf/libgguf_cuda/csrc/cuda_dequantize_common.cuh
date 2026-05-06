#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "libgguf_cuda_common.h"

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static __global__ void dequantize_block(const void * __restrict__ vx, dst_t * __restrict__ y, const int k) {
    const int i = 2*(blockDim.x*blockIdx.x + threadIdx.x);

    if (i >= k) {
        return;
    }

    const int ib = i/qk; // block index
    const int iqs = (i%qk)/qr; // quant index
    const int iybs = i - i%qk; // y block start index
    const int y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    dfloat2 v;
    dequantize_kernel(vx, ib, iqs, v);

    y[iybs + iqs + 0]        = convert_from_float<dst_t>(v.x);
    y[iybs + iqs + y_offset] = convert_from_float<dst_t>(v.y);
}

static __device__ __forceinline__ float bitcast_u32_to_float(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static __device__ __forceinline__ float e8m0_to_float(uint8_t value) {
    const uint32_t bits = value < 2 ? (0x00200000u << value) : ((uint32_t)(value - 1) << 23);
    return bitcast_u32_to_float(bits);
}

static __device__ __forceinline__ float ue4m3_to_float(uint8_t value) {
    if (value == 0 || value == 0x7f) {
        return 0.0f;
    }
    const int exp = (value >> 3) & 0x0f;
    const int man = value & 0x07;
    const float raw = exp == 0 ? man * 0x1p-9f : ldexpf(1.0f + man * 0.125f, exp - 7);
    return raw * 0.5f;
}


static inline __device__ void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}


template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_cuda(const void * __restrict__ vx, dst_t * __restrict__ y, const int k, cudaStream_t stream) {
    const int num_blocks = (k + 2*CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / (2*CUDA_DEQUANTIZE_BLOCK_SIZE);
    dequantize_block<qk, qr, dequantize_kernel><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(vx, y, k);
}
