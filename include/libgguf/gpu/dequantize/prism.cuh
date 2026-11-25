#pragma once

#include <libgguf/gpu/dequantize/common.cuh>

namespace libgguf { namespace gpu {

template <typename dst_t>
static __device__ __forceinline__ void dequantize_pq2_0(
    const void *vx, dst_t *output, int ib, int j)
{
    const auto *x = static_cast<const block_pq2_0 *>(vx);
    const float d = __half2float(load_half(x[ib].d));
    const uint8_t packed = x[ib].qs[j];
#pragma unroll
    for (int slot = 0; slot < 4; ++slot)
        output[j * 4 + slot] = convert_from_float<dst_t>(
            d * (int((packed >> (2 * slot)) & 3) - 1));
}

template <typename dst_t>
static __device__ __forceinline__ void dequantize_ptq1_0(
    const void *vx, dst_t *output, int ib, int j)
{
    const auto *x = static_cast<const block_ptq1_0 *>(vx);
    const float d = __half2float(load_half(x[ib].d));
    const int width = j < 16 ? 16 : j < 24 ? 8 : 2;
    const int start = j < 16 ? 0 : j < 24 ? 16 : 24;
    const int base = j < 16 ? 0 : j < 24 ? 80 : 120;
    const int planes = j < 24 ? 5 : 4;
    const uint8_t packed = j < 24 ? x[ib].qs[j] : x[ib].qh[j - 24];
    int multiplier = 1;
#pragma unroll
    for (int plane = 0; plane < planes; ++plane)
    {
        const uint8_t code = uint8_t(packed * multiplier);
        output[base + j - start + plane * width] = convert_from_float<dst_t>(
            d * ((int(code) * 3 >> 8) - 1));
        multiplier *= 3;
    }
}

template <typename dst_t, bool DenseTrits>
static __global__ void dequantize_block_prism(
    const void *__restrict__ x, dst_t *__restrict__ y, const int k)
{
    constexpr int threads = DenseTrits ? 26 : 32;
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= (k / 128) * threads) return;
    const int ib = index / threads;
    const int j = index % threads;
    if constexpr (DenseTrits) dequantize_ptq1_0(x, y + ib * 128, ib, j);
    else dequantize_pq2_0(x, y + ib * 128, ib, j);
}

} }
