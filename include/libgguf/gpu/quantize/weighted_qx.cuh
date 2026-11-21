#pragma once

#include <libgguf/gpu/quantize/common.cuh>

namespace libgguf::gpu {

inline unsigned weighted_qx_groups(int64_t width) {
    const auto groups = (width / 32 + 63) / 64;
    return static_cast<unsigned>(groups > 32 ? 32 : groups);
}

// Importance weights are shared across rows. Reproduce the reference row sum
// in input order because its rounding can change the discrete codes.
template<int Bits, bool Affine = false, typename Input, typename Block>
static __global__ void quantize_weighted_qx(const Input *input, const float *importance,
                                           Block *output, int64_t width) {
    static_assert(Bits == 4 || Bits == 5);
    constexpr int block_size = 32;
    const auto *row = input + static_cast<int64_t>(blockIdx.x) * width;
    auto *blocks = output + static_cast<int64_t>(blockIdx.x) * (width / block_size);
    __shared__ float sigma2;
    if (width <= 2048) {
        if (threadIdx.x == 0) {
            float sum = 0;
            for (int64_t i = 0; i < width; ++i) {
                const float v = static_cast<float>(row[i]);
                sum = __fadd_rn(sum, __fmul_rn(v, v));
            }
            sigma2 = __fdiv_rn(sum, static_cast<float>(width));
        }
    } else {
        // Coalesce the row reads. The sum still follows the CPU's FP32 order.
        constexpr int chunk = 512;
        __shared__ float squares[chunk];
        float sum = 0;
        for (int64_t start = 0; start < width; start += chunk) {
            const int count = static_cast<int>(width - start < chunk ? width - start : chunk);
            for (int i = threadIdx.x; i < count; i += blockDim.x) {
                const float v = static_cast<float>(row[start + i]);
                squares[i] = __fmul_rn(v, v);
            }
            __syncthreads();
            if (threadIdx.x == 0)
                for (int i = 0; i < count; ++i) sum = __fadd_rn(sum, squares[i]);
            __syncthreads();
        }
        if (threadIdx.x == 0) sigma2 = __fdiv_rn(sum, static_cast<float>(width));
    }
    __syncthreads();
    for (int64_t block = static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
         block < width / block_size; block += static_cast<int64_t>(gridDim.y) * blockDim.x) {
        const auto offset = block * block_size;
        float weights[block_size];
        uint8_t codes[block_size];
        for (int j = 0; j < block_size; ++j) {
            const float value = static_cast<float>(row[offset + j]);
            weights[j] = __fmul_rn(importance[offset + j],
                                   sqrtf(__fadd_rn(sigma2, __fmul_rn(value, value))));
        }
        auto &dst = blocks[block];
        float scale;
        if constexpr (Affine) {
            uint8_t scratch[block_size];
            float negative_min;
            scale = make_qkx2_quants(block_size, (1 << Bits) - 1, row + offset, weights, codes, &negative_min,
                                     scratch, -0.9f, 0.05f, 36, false);
            dst.m = compute_fp32_to_fp16(-negative_min);
        } else {
            scale = make_qx_quants_rmse1(block_size, 1 << (Bits - 1), row + offset,
                                         reinterpret_cast<int8_t *>(codes), weights);
        }
        dst.d = compute_fp32_to_fp16(scale);
        uint32_t high = 0;
        for (int j = 0; j < block_size / 2; ++j) {
            const auto lo = static_cast<uint8_t>(codes[j]);
            const auto hi = static_cast<uint8_t>(codes[j + 16]);
            dst.qs[j] = (lo & 15) | ((hi & 15) << 4);
            if constexpr (Bits == 5) {
                high |= static_cast<uint32_t>(lo >> 4) << j;
                high |= static_cast<uint32_t>(hi >> 4) << (j + 16);
            }
        }
        if constexpr (Bits == 5)
            store_u32_le(dst.qh, high);
    }
}
}
