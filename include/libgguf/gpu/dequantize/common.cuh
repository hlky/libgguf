#pragma once
#include <libgguf/gpu/compat.h>
#include <libgguf/gpu/common.cuh>

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

