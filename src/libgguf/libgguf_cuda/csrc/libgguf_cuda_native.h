#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum libgguf_cuda_status {
    LIBGGUF_CUDA_STATUS_SUCCESS = 0,
    LIBGGUF_CUDA_STATUS_INVALID_ARGUMENT = 1,
    LIBGGUF_CUDA_STATUS_UNSUPPORTED_TYPE = 2,
    LIBGGUF_CUDA_STATUS_CUDA_ERROR = 3,
    LIBGGUF_CUDA_STATUS_ALLOCATION_FAILED = 4,
};

typedef struct libgguf_cuda_context libgguf_cuda_context;
typedef struct libgguf_cuda_buffer libgguf_cuda_buffer;

int libgguf_cuda_context_create(int device, libgguf_cuda_context ** out);
void libgguf_cuda_context_destroy(libgguf_cuda_context * ctx);
const char * libgguf_cuda_last_error(const libgguf_cuda_context * ctx);
int libgguf_cuda_synchronize(libgguf_cuda_context * ctx);

int libgguf_cuda_qtype_supported(int64_t qtype);
int libgguf_cuda_qtype_needs_imatrix(int64_t qtype);
int64_t libgguf_cuda_block_size(int64_t qtype);
int64_t libgguf_cuda_row_size(int64_t qtype, int64_t n_per_row);

int libgguf_cuda_buffer_create(libgguf_cuda_context * ctx, size_t size, libgguf_cuda_buffer ** out);
void libgguf_cuda_buffer_destroy(libgguf_cuda_context * ctx, libgguf_cuda_buffer * buffer);
int libgguf_cuda_buffer_resize(libgguf_cuda_context * ctx, libgguf_cuda_buffer * buffer, size_t size);
int libgguf_cuda_buffer_reserve(libgguf_cuda_context * ctx, libgguf_cuda_buffer * buffer, size_t size);
void * libgguf_cuda_buffer_data(libgguf_cuda_buffer * buffer);
const void * libgguf_cuda_buffer_const_data(const libgguf_cuda_buffer * buffer);
size_t libgguf_cuda_buffer_size(const libgguf_cuda_buffer * buffer);

int libgguf_cuda_h2d(
    libgguf_cuda_context * ctx,
    libgguf_cuda_buffer * dst,
    size_t dst_offset,
    const void * src,
    size_t size
);
int libgguf_cuda_d2h(
    libgguf_cuda_context * ctx,
    void * dst,
    const libgguf_cuda_buffer * src,
    size_t src_offset,
    size_t size
);

int libgguf_cuda_quantize_f32_rows(
    libgguf_cuda_context * ctx,
    const float * device_input,
    const float * device_imatrix,
    void * device_output,
    int64_t qtype,
    int64_t rows,
    int64_t n_per_row
);

#ifdef __cplusplus
}
#endif
