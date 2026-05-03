#include "libgguf_storage.h"

#include <cstring>

#include "libgguf_cpu.h"

typedef void (*libgguf_store_bf16_fn)(const float *src, ggml_bf16_t *dst, size_t n);

struct libgguf_storage_backend_fns
{
  const char *name;
  libgguf_store_bf16_fn store_bf16;
};

static void libgguf_store_bf16_ref_impl(const float *src, ggml_bf16_t *dst, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    dst[i] = GGML_FP32_TO_BF16(src[i]);
  }
}

extern "C" void libgguf_store_bf16_sse2(const float *src, ggml_bf16_t *dst, size_t n);
extern "C" void libgguf_store_bf16_sse4_1(const float *src, ggml_bf16_t *dst, size_t n);
extern "C" void libgguf_store_bf16_avx2(const float *src, ggml_bf16_t *dst, size_t n);

static const libgguf_storage_backend_fns REF_BACKEND = {
    "ref",
    libgguf_store_bf16_ref_impl,
};

static const libgguf_storage_backend_fns SSE2_BACKEND = {
    "sse2",
    libgguf_store_bf16_sse2,
};

static const libgguf_storage_backend_fns SSE4_1_BACKEND = {
    "sse4_1",
    libgguf_store_bf16_sse4_1,
};

static const libgguf_storage_backend_fns AVX2_BACKEND = {
    "avx2",
    libgguf_store_bf16_avx2,
};

static const libgguf_storage_backend_fns *libgguf_storage_backend_for_name(const char *backend)
{
  if (backend == nullptr || backend[0] == '\0')
  {
    return nullptr;
  }

  if (std::strcmp(backend, "ref") == 0)
  {
    return &REF_BACKEND;
  }

  const libgguf_cpu_features &features = libgguf_get_cpu_features();
  if (std::strcmp(backend, "sse2") == 0 && features.sse2)
  {
    return &SSE2_BACKEND;
  }
  if (std::strcmp(backend, "sse4_1") == 0 && features.sse4_1)
  {
    return &SSE4_1_BACKEND;
  }
  if (std::strcmp(backend, "avx2") == 0 && features.avx2)
  {
    return &AVX2_BACKEND;
  }
  return nullptr;
}

static const libgguf_storage_backend_fns *&libgguf_storage_override_slot()
{
  static const libgguf_storage_backend_fns *override_selected = nullptr;
  return override_selected;
}

static const libgguf_storage_backend_fns &libgguf_storage_selected()
{
  static const libgguf_storage_backend_fns *auto_selected = []() {
    const libgguf_cpu_features &features = libgguf_get_cpu_features();
    if (features.avx2)
    {
      return &AVX2_BACKEND;
    }
    if (features.sse4_1)
    {
      return &SSE4_1_BACKEND;
    }
    if (features.sse2)
    {
      return &SSE2_BACKEND;
    }
    return &REF_BACKEND;
  }();
  const libgguf_storage_backend_fns *override_selected = libgguf_storage_override_slot();
  return *(override_selected ? override_selected : auto_selected);
}

extern "C" const char *libgguf_storage_backend(void)
{
  return libgguf_storage_selected().name;
}

extern "C" int libgguf_storage_cpu_supports_backend(const char *backend)
{
  return libgguf_storage_backend_for_name(backend) ? 1 : 0;
}

extern "C" int libgguf_storage_set_backend(const char *backend)
{
  if (backend != nullptr && std::strcmp(backend, "auto") == 0)
  {
    libgguf_storage_override_slot() = nullptr;
    return 1;
  }
  const libgguf_storage_backend_fns *selected = libgguf_storage_backend_for_name(backend);
  if (!selected)
  {
    return 0;
  }
  libgguf_storage_override_slot() = selected;
  return 1;
}

extern "C" size_t libgguf_store_f32(const float *src, void *dst, size_t n)
{
  std::memcpy(dst, src, n * sizeof(float));
  return n * sizeof(float);
}

extern "C" size_t libgguf_store_f16(const float *src, void *dst, size_t n)
{
  ggml_fp16_t *out = (ggml_fp16_t *)dst;
  for (size_t i = 0; i < n; ++i)
  {
    out[i] = GGML_FP32_TO_FP16(src[i]);
  }
  return n * sizeof(ggml_fp16_t);
}

extern "C" size_t libgguf_store_bf16(const float *src, void *dst, size_t n)
{
  libgguf_storage_selected().store_bf16(src, (ggml_bf16_t *)dst, n);
  return n * sizeof(ggml_bf16_t);
}
