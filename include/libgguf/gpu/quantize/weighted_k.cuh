#pragma once

#include <libgguf/gpu/quantize/common.cuh>

namespace libgguf::gpu {

// The CPU reference fits a scale and minimum per group, then quantizes those
// fits across the 256-value block. Keep intermediate fits and codes in LDS.
template<int Bits, int BlocksPerWorkgroup = 1, class Input, class Block>
static __global__ void quantize_weighted_k(const Input* src, const float* importance,
                                          Block* dst, int64_t blocks_per_row) {
    static_assert(Bits == 2 || Bits == 4 || Bits == 5);
    constexpr int width = Bits == 2 ? 16 : 32;
    constexpr int groups = QK_K / width;
    constexpr int max_code = (1 << Bits) - 1;
    constexpr int max_scale = Bits == 2 ? 15 : 63;
    static_assert(BlocksPerWorkgroup == 1 || (BlocksPerWorkgroup == 8 && Bits == 4));
    constexpr int team_width = 64 / BlocksPerWorkgroup;
    const int workers = BlocksPerWorkgroup == 1 ? int(blockDim.x) : team_width;
    const int team = BlocksPerWorkgroup == 1 ? 0 : int(threadIdx.x) / team_width;
    const int lane = BlocksPerWorkgroup == 1 ? int(threadIdx.x) : int(threadIdx.x) % team_width;
    const int64_t block = int64_t(blockIdx.x) * BlocksPerWorkgroup + team;
    const Input* x = src + block * QK_K;
    const float* qw = importance + (block % blocks_per_row) * QK_K;
    auto& y = dst[block];
    struct BlockState {
        float sigma2, rounded_scale, rounded_min;
        float scales[groups], mins[groups], sums[groups];
        uint8_t scale_codes[groups], min_codes[groups], codes[QK_K];
    };
    __shared__ BlockState states[BlocksPerWorkgroup];
    auto& state = states[team];

    if (lane == 0) {
        float sum = 0;
        for (int j = 0; j < QK_K; ++j) {
            const float v = static_cast<float>(x[j]);
            sum = __fadd_rn(sum, __fmul_rn(v,v));
        }
        state.sigma2 = sum * ((Bits == 2 ? 1.0f : 2.0f) / QK_K);
    }
    __syncthreads();
    if (lane < groups) {
        float weights[width], total = 0;
        uint8_t scratch[width];
        const int start = lane * width;
        for (int i = 0; i < width; ++i) {
            const float v = static_cast<float>(x[start+i]);
            weights[i] = qw[start+i] * sqrtf(state.sigma2 + v*v);
            total += weights[i];
        }
        state.sums[lane] = total;
        state.scales[lane] = make_qkx2_quants(width, max_code, x+start, weights, state.codes+start,
                                        state.mins+lane, scratch, -0.9f, 0.05f, 36, false);
    }
    __syncthreads();
    if (lane == 0) {
        const auto d = compute_fp32_to_fp16(make_qp_quants(groups,max_scale,state.scales,state.scale_codes,state.sums));
        const auto m = compute_fp32_to_fp16(make_qp_quants(groups,max_scale,state.mins,state.min_codes,state.sums));
        y.d = d;
        y.dmin = m;
        state.rounded_scale = __half2float(load_half(d));
        state.rounded_min = __half2float(load_half(m));
    }
    __syncthreads();
    if constexpr (Bits == 2) {
        if (lane < groups) y.scales[lane] = state.scale_codes[lane] | (state.min_codes[lane] << 4);
    } else if (lane < 4) {
        const int j = lane+4;
        y.scales[lane] = state.scale_codes[lane] | ((state.scale_codes[j] >> 4) << 6);
        y.scales[j] = state.min_codes[lane] | ((state.min_codes[j] >> 4) << 6);
        y.scales[j+4] = (state.scale_codes[j] & 15) | ((state.min_codes[j] & 15) << 4);
    }
    for (int i=lane; i<QK_K; i+=workers) {
        const int group = i / width;
        const float d = state.rounded_scale * state.scale_codes[group];
        if (d != 0) {
            const float m = state.rounded_min * state.min_codes[group];
            state.codes[i] = clamp_int(nearest_int(__fdiv_rn(static_cast<float>(x[i])+m,d)),0,max_code);
        }
    }
    __syncthreads();
    for (int byte = lane; byte < 32; byte += workers) {
        if constexpr (Bits == 2) {
            for (int i=0; i<QK_K; i+=128)
                y.qs[i/4+byte] = state.codes[i+byte] | (state.codes[i+byte+32]<<2) |
                                 (state.codes[i+byte+64]<<4) | (state.codes[i+byte+96]<<6);
        } else {
            uint8_t high = 0;
            for (int i=0; i<QK_K; i+=64) {
                const unsigned lo=state.codes[i+byte], hi=state.codes[i+byte+32];
                y.qs[i/2+byte]=(lo&15) | ((hi&15)<<4);
                if constexpr (Bits == 5)
                    high |= ((lo>>4) | ((hi>>4)<<1)) << (i/32);
            }
            if constexpr (Bits == 5) y.qh[byte]=high;
        }
    }
}
}
