#pragma once

#include <stddef.h>
#include <stdint.h>

#if (defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
#define LIBGGUF_NATIVE_API
#elif defined(_WIN32)
#define LIBGGUF_NATIVE_API __declspec(dllexport)
#else
#define LIBGGUF_NATIVE_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Separate from ggml_type. These numbers are not GGUF tensor type codes.
typedef enum libgguf_native_format {
    LIBGGUF_NATIVE_S4G32 = 0,
    LIBGGUF_NATIVE_S4G64 = 1,
    LIBGGUF_NATIVE_S8G32 = 2,
    LIBGGUF_NATIVE_S4H256 = 3
} libgguf_native_format;

typedef enum libgguf_native_status {
    LIBGGUF_NATIVE_OK = 0,
    LIBGGUF_NATIVE_INVALID_FORMAT = 1,
    LIBGGUF_NATIVE_INVALID_SHAPE = 2,
    LIBGGUF_NATIVE_INVALID_BUFFER = 3,
    LIBGGUF_NATIVE_INVALID_VALUE = 4,
    LIBGGUF_NATIVE_OVERFLOW = 5,
    LIBGGUF_NATIVE_GPU_ERROR = 6
} libgguf_native_status;

// Metadata-only query. Rows must contain whole canonical blocks.
LIBGGUF_NATIVE_API libgguf_native_status libgguf_native_row_size(
    libgguf_native_format format, size_t columns, size_t* bytes);

// Validate the complete input before writing. Input/output spans must not overlap.
// importance is optional, with exactly columns F32 values shared by all rows.
LIBGGUF_NATIVE_API libgguf_native_status libgguf_native_encode_rows(
    libgguf_native_format format, const float* source, size_t source_bytes,
    size_t rows, size_t columns, const float* importance, size_t importance_bytes,
    void* destination, size_t destination_bytes);

LIBGGUF_NATIVE_API libgguf_native_status libgguf_native_decode_rows(
    libgguf_native_format format, const void* source, size_t source_bytes,
    size_t rows, size_t columns, float* destination, size_t destination_bytes);

LIBGGUF_NATIVE_API libgguf_native_status libgguf_native_validate_rows(
    libgguf_native_format format, const void* source, size_t source_bytes,
    size_t rows, size_t columns);

#ifdef __cplusplus
}
#endif
