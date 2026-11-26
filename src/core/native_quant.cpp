#include <libgguf/native_quant.h>
#if defined(LIBGGUF_NATIVE_SIMD_DECODE)
#include "cpu/common/native_decode.h"
#endif
#include <libgguf/gpu/native_encode.cuh>
#include <limits>

namespace {
using libgguf::NativeFormat;
bool multiply(size_t a, size_t b, size_t& result) {
    if (b && a > std::numeric_limits<size_t>::max() / b) return false;
    result = a * b;
    return true;
}
bool span(const void* data, size_t length) {
    const auto base = reinterpret_cast<uintptr_t>(data);
    return data && length <= std::numeric_limits<uintptr_t>::max() - base;
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}
libgguf_native_status sizes(libgguf_native_format format, size_t rows, size_t columns,
    size_t& packed, size_t& dense) {
    size_t row = 0;
    const auto status = libgguf_native_row_size(format, columns, &row);
    if (status != LIBGGUF_NATIVE_OK) return status;
    if (!rows) return LIBGGUF_NATIVE_INVALID_SHAPE;
    size_t values = 0;
    if (!multiply(row, rows, packed) || !multiply(columns, rows, values)
        || !multiply(values, sizeof(float), dense)) return LIBGGUF_NATIVE_OVERFLOW;
    return LIBGGUF_NATIVE_OK;
}
}

extern "C" LIBGGUF_NATIVE_API libgguf_native_status libgguf_native_row_size(
    libgguf_native_format format, size_t columns, size_t* bytes) {
    if (!bytes) return LIBGGUF_NATIVE_INVALID_BUFFER;
    const auto info = libgguf::native_format_info(static_cast<NativeFormat>(format));
    if (!info.name) return LIBGGUF_NATIVE_INVALID_FORMAT;
    if (!columns || columns % info.values_per_block) return LIBGGUF_NATIVE_INVALID_SHAPE;
    if (!multiply(columns / info.values_per_block, info.bytes_per_block, *bytes))
        return LIBGGUF_NATIVE_OVERFLOW;
    return LIBGGUF_NATIVE_OK;
}

extern "C" LIBGGUF_NATIVE_API libgguf_native_status libgguf_native_validate_rows(
    libgguf_native_format format, const void* source, size_t source_bytes,
    size_t rows, size_t columns) {
    size_t packed = 0, dense = 0;
    const auto status = sizes(format, rows, columns, packed, dense);
    if (status != LIBGGUF_NATIVE_OK) return status;
    if (source_bytes != packed || !span(source, source_bytes)) return LIBGGUF_NATIVE_INVALID_BUFFER;
    const auto kind = static_cast<NativeFormat>(format);
    const auto info = libgguf::native_format_info(kind);
    const auto* bytes = static_cast<const std::uint8_t*>(source);
    for (size_t offset = 0; offset < packed; offset += info.bytes_per_block)
        if (!libgguf::native_detail::valid_block(kind, bytes + offset)) return LIBGGUF_NATIVE_INVALID_VALUE;
    return LIBGGUF_NATIVE_OK;
}

extern "C" LIBGGUF_NATIVE_API libgguf_native_status libgguf_native_encode_rows(
    libgguf_native_format format, const float* source, size_t source_bytes,
    size_t rows, size_t columns, const float* importance, size_t importance_bytes,
    void* destination, size_t destination_bytes) {
    size_t packed = 0, dense = 0;
    const auto status = sizes(format, rows, columns, packed, dense);
    if (status != LIBGGUF_NATIVE_OK) return status;
    if (source_bytes != dense || destination_bytes != packed || !span(source, dense)
        || !span(destination, packed) || reinterpret_cast<uintptr_t>(source) % alignof(float)
        || overlap(source, dense, destination, packed)) return LIBGGUF_NATIVE_INVALID_BUFFER;
    if (importance ? (importance_bytes != columns * sizeof(float)
        || !span(importance, importance_bytes) || reinterpret_cast<uintptr_t>(importance) % alignof(float)
        || overlap(importance, importance_bytes, destination, packed)) : importance_bytes != 0)
        return LIBGGUF_NATIVE_INVALID_BUFFER;
#if defined(LIBGGUF_NATIVE_SIMD_DECODE)
    if (!libgguf::native_cpu::valid_f32(source, dense / sizeof(float), false)
        || (importance && !libgguf::native_cpu::valid_f32(importance, columns, true)))
        return LIBGGUF_NATIVE_INVALID_VALUE;
#else
    for (size_t i = 0; i < dense / sizeof(float); ++i)
        if (!libgguf::native_detail::finite(source[i])) return LIBGGUF_NATIVE_INVALID_VALUE;
    if (importance) for (size_t i = 0; i < columns; ++i)
        if (!libgguf::native_detail::finite(importance[i]) || importance[i] < 0)
            return LIBGGUF_NATIVE_INVALID_VALUE;
#endif
    const auto kind = static_cast<NativeFormat>(format);
    const auto info = libgguf::native_format_info(kind);
    auto* output = static_cast<std::uint8_t*>(destination);
    for (size_t row = 0; row < rows; ++row)
        for (size_t k = 0; k < columns; k += info.values_per_block) {
            libgguf::native_detail::encode_block(kind, source + row * columns + k,
                importance ? importance + k : nullptr, output);
            output += info.bytes_per_block;
        }
    return LIBGGUF_NATIVE_OK;
}

extern "C" LIBGGUF_NATIVE_API libgguf_native_status libgguf_native_decode_rows(
    libgguf_native_format format, const void* source, size_t source_bytes,
    size_t rows, size_t columns, float* destination, size_t destination_bytes) {
    size_t packed = 0, dense = 0;
    auto status = sizes(format, rows, columns, packed, dense);
    if (status != LIBGGUF_NATIVE_OK) return status;
    if (destination_bytes != dense || !span(destination, dense)
        || reinterpret_cast<uintptr_t>(destination) % alignof(float)
        || !span(source, source_bytes) || overlap(source, source_bytes, destination, dense))
        return LIBGGUF_NATIVE_INVALID_BUFFER;
    status = libgguf_native_validate_rows(format, source, source_bytes, rows, columns);
    if (status != LIBGGUF_NATIVE_OK) return status;
    const auto kind = static_cast<NativeFormat>(format);
    const auto info = libgguf::native_format_info(kind);
    const auto* input = static_cast<const std::uint8_t*>(source);
#if defined(LIBGGUF_NATIVE_SIMD_DECODE)
    libgguf::native_cpu::decode(kind, input, rows * columns / info.values_per_block, destination);
#else
    for (size_t i = 0; i < rows * columns; i += info.values_per_block) {
        for (unsigned k = 0; k < info.values_per_block; ++k)
            destination[i + k] = libgguf::native_detail::scale(kind, input, k)
                * static_cast<float>(libgguf::native_detail::quant(kind, input, k));
        input += info.bytes_per_block;
    }
#endif
    return LIBGGUF_NATIVE_OK;
}
