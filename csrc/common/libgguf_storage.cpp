#include "libgguf_storage.h"
#include "common/libgguf_backend.h"

#include <cstring>

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

#if !LIBGGUF_CPU_BACKEND_REF
extern "C" void LIBGGUF_CPU_BACKEND_SYMBOL(libgguf_store_bf16)(const float *src, ggml_bf16_t *dst, size_t n);
#endif

static const libgguf_storage_backend_fns REF_BACKEND = {
    "ref",
    libgguf_store_bf16_ref_impl,
};

#if !LIBGGUF_CPU_BACKEND_REF
static const libgguf_storage_backend_fns COMPILED_BACKEND = {
    LIBGGUF_CPU_BACKEND_NAME,
    LIBGGUF_CPU_BACKEND_SYMBOL(libgguf_store_bf16),
};
#endif

static const libgguf_storage_backend_fns &libgguf_storage_default_backend()
{
#if LIBGGUF_CPU_BACKEND_REF
  return REF_BACKEND;
#else
  return COMPILED_BACKEND;
#endif
}

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

#if !LIBGGUF_CPU_BACKEND_REF
  if (libgguf_cpu_backend_is_compiled_request(backend))
  {
    return &COMPILED_BACKEND;
  }
#endif
  return nullptr;
}

static const libgguf_storage_backend_fns *&libgguf_storage_selected_slot()
{
  static const libgguf_storage_backend_fns *selected = &libgguf_storage_default_backend();
  return selected;
}

static const libgguf_storage_backend_fns &libgguf_storage_selected()
{
  return *libgguf_storage_selected_slot();
}

extern "C" const char *libgguf_storage_backend(void)
{
  return libgguf_storage_selected().name;
}

extern "C" int libgguf_storage_cpu_supports_backend(const char *backend)
{
  return libgguf_cpu_backend_supports_request(backend) ? 1 : 0;
}

extern "C" int libgguf_storage_set_backend(const char *backend)
{
  if (backend != nullptr && std::strcmp(backend, "auto") == 0)
  {
    libgguf_storage_selected_slot() = &libgguf_storage_default_backend();
    return 1;
  }
  const libgguf_storage_backend_fns *selected = libgguf_storage_backend_for_name(backend);
  if (!selected)
  {
    return 0;
  }
  libgguf_storage_selected_slot() = selected;
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
