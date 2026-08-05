#pragma once

#include <cstring>

#ifndef LIBGGUF_CPU_BACKEND_REF
#define LIBGGUF_CPU_BACKEND_REF 0
#endif

#ifndef LIBGGUF_CPU_BACKEND_SSE2
#define LIBGGUF_CPU_BACKEND_SSE2 0
#endif

#ifndef LIBGGUF_CPU_BACKEND_SSE4_1
#define LIBGGUF_CPU_BACKEND_SSE4_1 0
#endif

#ifndef LIBGGUF_CPU_BACKEND_AVX2
#define LIBGGUF_CPU_BACKEND_AVX2 0
#endif

#if !LIBGGUF_CPU_BACKEND_REF && !LIBGGUF_CPU_BACKEND_SSE2 && !LIBGGUF_CPU_BACKEND_SSE4_1 && !LIBGGUF_CPU_BACKEND_AVX2
#undef LIBGGUF_CPU_BACKEND_REF
#define LIBGGUF_CPU_BACKEND_REF 1
#endif

#if (LIBGGUF_CPU_BACKEND_REF + LIBGGUF_CPU_BACKEND_SSE2 + LIBGGUF_CPU_BACKEND_SSE4_1 + LIBGGUF_CPU_BACKEND_AVX2) != 1
#error "Exactly one LIBGGUF_CPU_BACKEND_* macro must be enabled"
#endif

#if LIBGGUF_CPU_BACKEND_REF
#define LIBGGUF_CPU_BACKEND_NAME "ref"
#elif LIBGGUF_CPU_BACKEND_SSE2
#define LIBGGUF_CPU_BACKEND_NAME "sse2"
#define LIBGGUF_CPU_BACKEND_SUFFIX sse2
#elif LIBGGUF_CPU_BACKEND_SSE4_1
#define LIBGGUF_CPU_BACKEND_NAME "sse4_1"
#define LIBGGUF_CPU_BACKEND_SUFFIX sse4_1
#elif LIBGGUF_CPU_BACKEND_AVX2
#define LIBGGUF_CPU_BACKEND_NAME "avx2"
#define LIBGGUF_CPU_BACKEND_SUFFIX avx2
#endif

#define LIBGGUF_CPU_BACKEND_APPEND2(base, suffix) base##_##suffix
#define LIBGGUF_CPU_BACKEND_APPEND(base, suffix) LIBGGUF_CPU_BACKEND_APPEND2(base, suffix)
#define LIBGGUF_CPU_BACKEND_SYMBOL(base) LIBGGUF_CPU_BACKEND_APPEND(base, LIBGGUF_CPU_BACKEND_SUFFIX)

static inline bool libgguf_cpu_backend_is_ref_request(const char *backend)
{
  return backend == nullptr || std::strcmp(backend, "ref") == 0;
}

static inline bool libgguf_cpu_backend_is_compiled_request(const char *backend)
{
  return backend != nullptr && std::strcmp(backend, LIBGGUF_CPU_BACKEND_NAME) == 0;
}

static inline bool libgguf_cpu_backend_supports_request(const char *backend)
{
  if (backend == nullptr)
  {
    return false;
  }
  return std::strcmp(backend, "ref") == 0 || libgguf_cpu_backend_is_compiled_request(backend);
}
