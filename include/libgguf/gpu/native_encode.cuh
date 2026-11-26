#pragma once

#include <libgguf/gpu/native_fragment.cuh>
#include <limits>

namespace libgguf::native_detail {

LIBGGUF_NATIVE_HD std::uint64_t double_bits(double value) {
#if (defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return __builtin_bit_cast(std::uint64_t, value);
#else
    std::uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
#endif
}

// Positive finite scale conversion directly from binary64, avoiding double rounding.
LIBGGUF_NATIVE_HD std::uint16_t scale_half(double value) {
    if (value <= 0) return 0;
    if (value < 0x1p-24) value = 0x1p-24;
    if (value > 65504.0) value = 65504.0;
    const auto bits = double_bits(value);
    const int exponent = static_cast<int>((bits >> 52) & 2047u) - 1023;
    const auto significand = (bits & 0x000fffffffffffffull) | (1ull << 52);
    const int shift = exponent < -14 ? 28 - exponent : 42;
    auto rounded = significand >> shift;
    const auto remainder = significand & ((1ull << shift) - 1);
    const auto midpoint = 1ull << (shift - 1);
    if (remainder > midpoint || (remainder == midpoint && (rounded & 1))) ++rounded;
    return static_cast<std::uint16_t>(exponent < -14 ? rounded
        : static_cast<std::uint64_t>((exponent + 14) * 1024) + rounded);
}

LIBGGUF_NATIVE_HD int rounded_code(double value, int lo, int hi) {
    if (value <= lo) return lo;
    if (value >= hi) return hi;
    int result = static_cast<int>(value);
    const double fraction = value - result;
    if (fraction > 0.5 || (fraction == 0.5 && result % 2 != 0)) ++result;
    if (fraction < -0.5 || (fraction == -0.5 && result % 2 != 0)) --result;
    return result;
}

LIBGGUF_NATIVE_HD void store_half(std::uint8_t* output, std::uint16_t bits) {
    output[0] = static_cast<std::uint8_t>(bits);
    output[1] = static_cast<std::uint8_t>(bits >> 8);
}

LIBGGUF_NATIVE_HD bool finite(float value) {
    return value <= std::numeric_limits<float>::max()
        && value >= -std::numeric_limits<float>::max();
}

LIBGGUF_NATIVE_HD bool has_importance(const float* weights, int count) {
    if (weights) for (int k = 0; k < count; ++k) if (weights[k] > 0) return true;
    return false;
}

LIBGGUF_NATIVE_HD double score_codes(
    const float* input, const float* weights, int count, float scale, const int* q) {
    const bool weighted = has_importance(weights, count);
    double score = 0;
    for (int k = 0; k < count; ++k) {
        const float decoded = scale * static_cast<float>(q[k]);
        const double error = static_cast<double>(input[k]) - static_cast<double>(decoded);
        const double square = error * error;
        const double term = square * (weighted ? static_cast<double>(weights[k]) : 1.0);
        score = score + term;
    }
    return score;
}

LIBGGUF_NATIVE_HD std::uint16_t fit_group(
    const float* input, const float* weights, int count, int lo, int hi, int* best_q) {
    double initial = 0;
    for (int k = 0; k < count; ++k) {
        const double candidate = input[k] >= 0
            ? static_cast<double>(input[k]) / hi : static_cast<double>(input[k]) / lo;
        if (candidate > initial) initial = candidate;
    }
    if (initial == 0) {
        for (int k = 0; k < count; ++k) best_q[k] = 0;
        return 0;
    }
    double best_score = std::numeric_limits<double>::infinity();
    std::uint16_t best_half = 0;
    int candidate_q[64]{};
    for (int j = 0; j <= 16; ++j) {
        const auto half = scale_half(initial * (16 + j) / 32.0);
        const float scale = gpu::fragment_detail::fp16_to_float(half);
        for (int k = 0; k < count; ++k)
            candidate_q[k] = rounded_code(static_cast<double>(input[k]) / scale, lo, hi);
        const double error = score_codes(input, weights, count, scale, candidate_q);
        if (error < best_score) {
            best_score = error;
            best_half = half;
            for (int k = 0; k < count; ++k) best_q[k] = candidate_q[k];
        }
    }
    return best_half;
}

LIBGGUF_NATIVE_HD void write_quants(
    NativeFormat format, std::uint8_t* output, const int* q, int count) {
    const int start = format == NativeFormat::s4h256 ? 10 : 2;
    if (format == NativeFormat::s8g32) {
        for (int k = 0; k < count; ++k) output[start + k] = static_cast<std::uint8_t>(q[k]);
    } else {
        for (int k = 0; k < count; k += 2)
            output[start + k / 2] = static_cast<std::uint8_t>((q[k] & 15) | ((q[k + 1] & 15) << 4));
    }
}

// One block per invocation; no runtime, allocation, subgroup or instruction dependency.
LIBGGUF_NATIVE_HD void encode_block(
    NativeFormat format, const float* input, const float* weights, std::uint8_t* output) {
    int best_q[256]{};
    if (format != NativeFormat::s4h256) {
        const auto info = native_format_info(format);
        const bool eight = format == NativeFormat::s8g32;
        const auto half = fit_group(input, weights, info.values_per_block,
            eight ? -127 : -8, eight ? 127 : 7, best_q);
        store_half(output, half);
        write_quants(format, output, best_q, info.values_per_block);
        return;
    }
    float local[8]{};
    double maximum = 0;
    for (int g = 0; g < 8; ++g) {
        const auto half = fit_group(input + g * 32, weights ? weights + g * 32 : nullptr,
            32, -8, 7, best_q + g * 32);
        local[g] = gpu::fragment_detail::fp16_to_float(half);
        if (local[g] > maximum) maximum = local[g];
    }
    for (int i = 0; i < 138; ++i) output[i] = 0;
    if (maximum == 0) return;
    double best_score = std::numeric_limits<double>::infinity();
    int candidate_q[256]{};
    std::uint8_t codes[8]{};
    const bool weighted = has_importance(weights, 256);
    for (int j = 0; j <= 16; ++j) {
        const auto half = scale_half((maximum / 255.0) * (16 + j) / 32.0);
        const float base = gpu::fragment_detail::fp16_to_float(half);
        double score = 0;
        for (int g = 0; g < 8; ++g) {
            codes[g] = local[g] == 0 ? 0 : static_cast<std::uint8_t>(
                rounded_code(static_cast<double>(local[g]) / base, 1, 255));
            const float scale = base * static_cast<float>(codes[g]);
            for (int k = g * 32; k < (g + 1) * 32; ++k) {
                candidate_q[k] = scale == 0 ? 0 : rounded_code(static_cast<double>(input[k]) / scale, -8, 7);
                const float decoded = scale * static_cast<float>(candidate_q[k]);
                const double error = static_cast<double>(input[k]) - decoded;
                const double square = error * error;
                const double term = square * (weighted ? static_cast<double>(weights[k]) : 1.0);
                score = score + term;
            }
        }
        if (score < best_score) {
            best_score = score;
            store_half(output, half);
            for (int g = 0; g < 8; ++g) output[2 + g] = codes[g];
            for (int k = 0; k < 256; ++k) best_q[k] = candidate_q[k];
        }
    }
    write_quants(format, output, best_q, 256);
}

} // namespace libgguf::native_detail
