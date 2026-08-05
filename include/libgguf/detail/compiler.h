#pragma once

#include <cstdlib>

#define LIBGGUF_UNUSED(x) (void)(x)

#if defined(_MSC_VER)
#define LIBGGUF_UNREACHABLE() __assume(0)
#define LIBGGUF_EXTENSION
#define LIBGGUF_RESTRICT __restrict
#elif defined(__clang__)
#define LIBGGUF_UNREACHABLE() __builtin_unreachable()
#define LIBGGUF_EXTENSION __extension__
#define LIBGGUF_RESTRICT __restrict
#elif defined(__GNUC__)
#define LIBGGUF_UNREACHABLE() __builtin_unreachable()
#define LIBGGUF_EXTENSION __extension__
#define LIBGGUF_RESTRICT __restrict__
#else
#define LIBGGUF_UNREACHABLE() abort()
#define LIBGGUF_EXTENSION
#define LIBGGUF_RESTRICT
#endif
