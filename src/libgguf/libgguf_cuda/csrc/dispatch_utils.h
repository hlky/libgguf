#pragma once

#ifdef LIBGGUF_CUDA_NO_C10

#include <cuda_fp16.h>

namespace at {
enum class ScalarType {
    Float,
    Half,
};
}  // namespace at

#define VLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  do {                                                \
    switch (TYPE) {                                   \
      case at::ScalarType::Float: {                   \
        using scalar_t = float;                       \
        __VA_ARGS__();                                \
        break;                                        \
      }                                               \
      case at::ScalarType::Half: {                    \
        using scalar_t = half;                        \
        __VA_ARGS__();                                \
        break;                                        \
      }                                               \
    }                                                 \
  } while (false)

#else

#include <torch/all.h>

#define VLLM_DISPATCH_CASE_FLOATING_TYPES(...)         \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define VLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

#endif
