#include <libgguf/gpu/compat.h>
#include <libgguf/gpu/quantize/common.cuh>
#include <libgguf/gpu/quantize/kernels.h>

static __device__ __forceinline__ int tq_round_away(float v) {
    const int abs_i = (int)(fabsf(v) + 0.5f);
    return v < 0.0f ? -abs_i : abs_i;
}

static __global__ void quantize_block_tq1_0_warp(const float * __restrict__ x, block_tq1_0 * __restrict__ y, int64_t n_blocks) {
    constexpr unsigned mask = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps_per_cta = blockDim.x >> 5;
    const int64_t ib = (int64_t)blockIdx.x * warps_per_cta + warp;
    if (ib >= n_blocks) {
        return;
    }

    const float * xb = x + ib * QK_K;
    float amax = 0.0f;
    for (int j = lane; j < QK_K; j += 32) {
        amax = fmaxf(amax, fabsf(xb[j]));
    }
    amax = __shfl_sync(mask, warp_reduce_max(amax), 0);

    const float d = amax;
    const float id = d != 0.0f ? __fdiv_rn(1.0f, d) : 0.0f;
    if (lane == 0) {
        y[ib].d = compute_fp32_to_fp16(d);
    }

    uint8_t q = 0;
    for (int n = 0; n < 5; ++n) {
        const int xi = tq_round_away(__fmul_rn(xb[lane + n * 32], id)) + 1;
        q *= 3;
        q += xi;
    }
    y[ib].qs[lane] = pack_trits_5(q);

    if (lane < 16) {
        q = 0;
        constexpr int offset = 5 * 32;
        for (int n = 0; n < 5; ++n) {
            const int xi = tq_round_away(__fmul_rn(xb[offset + lane + n * 16], id)) + 1;
            q *= 3;
            q += xi;
        }
        y[ib].qs[32 + lane] = pack_trits_5(q);
    }

    if (lane < (int)sizeof(block_tq1_0::qh)) {
        q = 0;
        constexpr int offset = 5 * 32 + 5 * 16;
        for (int m = 0; m < 4; ++m) {
            const int xi = tq_round_away(__fmul_rn(xb[offset + lane + m * (int)sizeof(block_tq1_0::qh)], id)) + 1;
            q *= 3;
            q += xi;
        }
        q *= 3;
        y[ib].qh[lane] = pack_trits_5(q);
    }
}


void quantize_launch_tq1_0(
    const float * x,
    const float * quant_weights,
    void * y,
    int64_t k,
    int64_t n_per_row,
    StreamType stream
) {
    (void)quant_weights;
    (void)n_per_row;
    constexpr int tq1_threads = 128;
    constexpr int tq1_blocks_per_cta = tq1_threads / 32;
    const int64_t n_blocks = k / QK_K;
    const int blocks = (int)((n_blocks + tq1_blocks_per_cta - 1) / tq1_blocks_per_cta);
    quantize_block_tq1_0_warp<<<blocks, tq1_threads, 0, stream>>>(x, (block_tq1_0 *)y, n_blocks);
}
