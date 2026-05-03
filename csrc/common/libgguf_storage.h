#pragma once

#include <cstddef>

#include "libgguf_common.h"

extern "C" const char *libgguf_storage_backend(void);
extern "C" int libgguf_storage_cpu_supports_backend(const char *backend);
extern "C" int libgguf_storage_set_backend(const char *backend);

extern "C" size_t libgguf_store_f32(const float *src, void *dst, size_t n);
extern "C" size_t libgguf_store_f16(const float *src, void *dst, size_t n);
extern "C" size_t libgguf_store_bf16(const float *src, void *dst, size_t n);
