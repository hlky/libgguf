#pragma once

#include <libgguf/native_quant.h>

#ifdef __cplusplus
extern "C" {
#endif

// Asynchronous, allocation-free out variants. All data pointers and device_status
// refer to device memory on the stream's device. device_status is a caller-owned
// aligned uint32_t, is overwritten on every invocation, and must not alias inputs
// or output. Keep every span alive until stream completion. No synchronization or
// default-stream substitution occurs. Null stream denotes the runtime default.
//
// Return value checks host-visible metadata/enqueue errors only. On completion,
// device_status contains a libgguf_native_status. Invalid device values leave the
// entire destination untouched. Runtime enqueue failures return GPU_ERROR; work
// already enqueued still owns its buffers until the caller completes the stream.
LIBGGUF_NATIVE_API libgguf_native_status libgguf_native_encode_gpu_into(
    libgguf_native_format format, const float* source, size_t source_bytes,
    size_t rows, size_t columns, const float* importance, size_t importance_bytes,
    void* destination, size_t destination_bytes, uint32_t* device_status, void* stream);

LIBGGUF_NATIVE_API libgguf_native_status libgguf_native_decode_gpu_into(
    libgguf_native_format format, const void* source, size_t source_bytes,
    size_t rows, size_t columns, float* destination, size_t destination_bytes,
    uint32_t* device_status, void* stream);

#ifdef __cplusplus
}
#endif
