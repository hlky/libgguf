#pragma once

#include <libgguf/gpu/quantize/common.cuh>

namespace libgguf::gpu {

// The CPU reference fits a scale and minimum per group, then quantizes those
// fits across the 256-value block. Keep intermediate fits and codes in LDS.
template<int Bits, class Input, class Block>
static __global__ void quantize_weighted_k(const Input* src, const float* importance,
                                          Block* dst, int64_t blocks_per_row) {
    static_assert(Bits == 2 || Bits == 4 || Bits == 5);
    constexpr int width = Bits == 2 ? 16 : 32;
    constexpr int groups = QK_K / width;
    constexpr int max_code = (1 << Bits) - 1;
    constexpr int max_scale = Bits == 2 ? 15 : 63;
    const int lane = threadIdx.x;
    const int64_t block = blockIdx.x;
    const Input* x = src + block * QK_K;
    const float* qw = importance + (block % blocks_per_row) * QK_K;
    auto& y = dst[block];
    __shared__ float sigma2, rounded_scale, rounded_min;
    __shared__ float scales[groups], mins[groups], sums[groups];
    __shared__ uint8_t scale_codes[groups], min_codes[groups], codes[QK_K];

    if (lane == 0) {
        float sum = 0;
        for (int j = 0; j < QK_K; ++j) {
            const float v = static_cast<float>(x[j]);
            sum = __fadd_rn(sum, __fmul_rn(v,v));
        }
        sigma2 = sum * ((Bits == 2 ? 1.0f : 2.0f) / QK_K);
    }
    __syncthreads();
    if (lane < groups) {
        float weights[width], total = 0;
        uint8_t scratch[width];
        const int start = lane * width;
        for (int i = 0; i < width; ++i) {
            const float v = static_cast<float>(x[start+i]);
            weights[i] = qw[start+i] * sqrtf(sigma2 + v*v);
            total += weights[i];
        }
        sums[lane] = total;
        scales[lane] = make_qkx2_quants(width, max_code, x+start, weights, codes+start,
                                        mins+lane, scratch, -0.9f, 0.05f, 36, false);
    }
    __syncthreads();
    if (lane == 0) {
        const auto d = compute_fp32_to_fp16(make_qp_quants(groups,max_scale,scales,scale_codes,sums));
        const auto m = compute_fp32_to_fp16(make_qp_quants(groups,max_scale,mins,min_codes,sums));
        y.d = d;
        y.dmin = m;
        rounded_scale = __half2float(load_half(d));
        rounded_min = __half2float(load_half(m));
    }
    __syncthreads();
    if constexpr (Bits == 2) {
        if (lane < groups) y.scales[lane] = scale_codes[lane] | (min_codes[lane] << 4);
    } else if (lane < 4) {
        const int j = lane+4;
        y.scales[lane] = scale_codes[lane] | ((scale_codes[j] >> 4) << 6);
        y.scales[j] = min_codes[lane] | ((min_codes[j] >> 4) << 6);
        y.scales[j+4] = (scale_codes[j] & 15) | ((min_codes[j] & 15) << 4);
    }
    for (int i=lane; i<QK_K; i+=blockDim.x) {
        const int group = i / width;
        const float d = rounded_scale * scale_codes[group];
        if (d != 0) {
            const float m = rounded_min * min_codes[group];
            codes[i] = clamp_int(nearest_int(__fdiv_rn(static_cast<float>(x[i])+m,d)),0,max_code);
        }
    }
    __syncthreads();
    if (lane < 32) {
        if constexpr (Bits == 2) {
            for (int i=0; i<QK_K; i+=128)
                y.qs[i/4+lane] = codes[i+lane] | (codes[i+lane+32]<<2) |
                                 (codes[i+lane+64]<<4) | (codes[i+lane+96]<<6);
        } else {
            uint8_t high = 0;
            for (int i=0; i<QK_K; i+=64) {
                const unsigned lo=codes[i+lane], hi=codes[i+lane+32];
                y.qs[i/2+lane]=(lo&15) | ((hi&15)<<4);
                if constexpr (Bits == 5)
                    high |= ((lo>>4) | ((hi>>4)<<1)) << (i/32);
            }
            if constexpr (Bits == 5) y.qh[lane]=high;
        }
    }
}
}
