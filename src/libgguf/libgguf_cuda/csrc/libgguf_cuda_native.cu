#include "libgguf_cuda_native.h"

#include <cuda_runtime.h>

#include <new>
#include <string>

#include "cuda_quantize_kernels.h"

struct libgguf_cuda_context {
    int device = 0;
    cudaStream_t stream = nullptr;
    std::string last_error;
};

struct libgguf_cuda_buffer {
    void * data = nullptr;
    size_t size = 0;
    size_t capacity = 0;
};

struct libgguf_cuda_event {
    cudaEvent_t event = nullptr;
};

struct libgguf_cuda_host_buffer {
    void * data = nullptr;
    size_t size = 0;
    size_t capacity = 0;
};

static int set_error(libgguf_cuda_context * ctx, int status, const char * message) {
    if (ctx) {
        ctx->last_error = message ? message : "";
    }
    return status;
}

static int set_cuda_error(libgguf_cuda_context * ctx, cudaError_t error) {
    if (ctx) {
        ctx->last_error = cudaGetErrorString(error);
    }
    return LIBGGUF_CUDA_STATUS_CUDA_ERROR;
}

static int set_device(libgguf_cuda_context * ctx) {
    if (!ctx) {
        return LIBGGUF_CUDA_STATUS_INVALID_ARGUMENT;
    }
    const cudaError_t error = cudaSetDevice(ctx->device);
    if (error != cudaSuccess) {
        return set_cuda_error(ctx, error);
    }
    return LIBGGUF_CUDA_STATUS_SUCCESS;
}

extern "C" {

int libgguf_cuda_context_create(int device, libgguf_cuda_context ** out) {
    if (!out) {
        return LIBGGUF_CUDA_STATUS_INVALID_ARGUMENT;
    }
    *out = nullptr;
    libgguf_cuda_context * ctx = new (std::nothrow) libgguf_cuda_context();
    if (!ctx) {
        return LIBGGUF_CUDA_STATUS_ALLOCATION_FAILED;
    }
    ctx->device = device;
    cudaError_t error = cudaSetDevice(device);
    if (error != cudaSuccess) {
        const int status = set_cuda_error(ctx, error);
        delete ctx;
        return status;
    }
    error = cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking);
    if (error != cudaSuccess) {
        const int status = set_cuda_error(ctx, error);
        delete ctx;
        return status;
    }
    *out = ctx;
    return LIBGGUF_CUDA_STATUS_SUCCESS;
}

void libgguf_cuda_context_destroy(libgguf_cuda_context * ctx) {
    if (!ctx) {
        return;
    }
    cudaSetDevice(ctx->device);
    if (ctx->stream) {
        cudaStreamDestroy(ctx->stream);
    }
    delete ctx;
}

const char * libgguf_cuda_last_error(const libgguf_cuda_context * ctx) {
    if (!ctx) {
        return "libgguf CUDA context is null";
    }
    return ctx->last_error.c_str();
}

int libgguf_cuda_event_create(libgguf_cuda_context * ctx, libgguf_cuda_event ** out) {
    if (!ctx || !out) {
        return set_error(ctx, LIBGGUF_CUDA_STATUS_INVALID_ARGUMENT, "event create requires a context and output pointer");
    }
    *out = nullptr;
    const int status = set_device(ctx);
    if (status != LIBGGUF_CUDA_STATUS_SUCCESS) {
        return status;
    }
    libgguf_cuda_event * event = new (std::nothrow) libgguf_cuda_event();
    if (!event) {
        return set_error(ctx, LIBGGUF_CUDA_STATUS_ALLOCATION_FAILED, "failed to allocate CUDA event handle");
    }
    const cudaError_t error = cudaEventCreate(&event->event);
    if (error != cudaSuccess) {
        delete event;
        return set_cuda_error(ctx, error);
    }
    *out = event;
    return set_error(ctx, LIBGGUF_CUDA_STATUS_SUCCESS, "");
}

void libgguf_cuda_event_destroy(libgguf_cuda_context * ctx, libgguf_cuda_event * event) {
    if (!event) {
        return;
    }
    if (ctx) {
        cudaSetDevice(ctx->device);
    }
    if (event->event) {
        cudaEventDestroy(event->event);
    }
    delete event;
}

int libgguf_cuda_event_record(libgguf_cuda_context * ctx, libgguf_cuda_event * event) {
    if (!ctx || !event || !event->event) {
        return set_error(ctx, LIBGGUF_CUDA_STATUS_INVALID_ARGUMENT, "event record requires a context and event");
    }
    const int status = set_device(ctx);
    if (status != LIBGGUF_CUDA_STATUS_SUCCESS) {
        return status;
    }
    const cudaError_t error = cudaEventRecord(event->event, ctx->stream);
    if (error != cudaSuccess) {
        return set_cuda_error(ctx, error);
    }
    return set_error(ctx, LIBGGUF_CUDA_STATUS_SUCCESS, "");
}

int libgguf_cuda_event_synchronize(libgguf_cuda_context * ctx, libgguf_cuda_event * event) {
    if (!ctx || !event || !event->event) {
        return set_error(ctx, LIBGGUF_CUDA_STATUS_INVALID_ARGUMENT, "event synchronize requires a context and event");
    }
    const int status = set_device(ctx);
    if (status != LIBGGUF_CUDA_STATUS_SUCCESS) {
        return status;
    }
    const cudaError_t error = cudaEventSynchronize(event->event);
    if (error != cudaSuccess) {
        return set_cuda_error(ctx, error);
    }
    return set_error(ctx, LIBGGUF_CUDA_STATUS_SUCCESS, "");
}

int libgguf_cuda_event_elapsed_ms(
    libgguf_cuda_context * ctx,
    const libgguf_cuda_event * start,
    const libgguf_cuda_event * end,
    float * out
) {
    if (!ctx || !start || !start->event || !end || !end->event || !out) {
        return set_error(ctx, LIBGGUF_CUDA_STATUS_INVALID_ARGUMENT, "event elapsed requires a context, events, and output pointer");
    }
    float elapsed = 0.0f;
    const cudaError_t error = cudaEventElapsedTime(&elapsed, start->event, end->event);
    if (error != cudaSuccess) {
        return set_cuda_error(ctx, error);
    }
    *out = elapsed;
    return set_error(ctx, LIBGGUF_CUDA_STATUS_SUCCESS, "");
}

int libgguf_cuda_qtype_supported(int64_t qtype) {
    return gguf_cuda_quantize_block_size_for_type(qtype) > 0;
}

int libgguf_cuda_qtype_needs_imatrix(int64_t qtype) {
    return gguf_cuda_quantize_type_needs_imatrix(qtype) ? 1 : 0;
}

int64_t libgguf_cuda_block_size(int64_t qtype) {
    return gguf_cuda_quantize_block_size_for_type(qtype);
}

int64_t libgguf_cuda_row_size(int64_t qtype, int64_t n_per_row) {
    if (n_per_row < 0) {
        return 0;
    }
    return gguf_cuda_quantize_row_size_for_type(qtype, n_per_row);
}

int libgguf_cuda_buffer_create(libgguf_cuda_context * ctx, size_t size, libgguf_cuda_buffer ** out) {
    if (!ctx || !out) {
        return set_error(ctx, LIBGGUF_CUDA_STATUS_INVALID_ARGUMENT, "buffer create requires a context and output pointer");
    }
    *out = nullptr;
    libgguf_cuda_buffer * buffer = new (std::nothrow) libgguf_cuda_buffer();
    if (!buffer) {
        return set_error(ctx, LIBGGUF_CUDA_STATUS_ALLOCATION_FAILED, "failed to allocate CUDA buffer handle");
    }
    const int status = libgguf_cuda_buffer_resize_discard(ctx, buffer, size);
    if (status != LIBGGUF_CUDA_STATUS_SUCCESS) {
        delete buffer;
        return status;
    }
    *out = buffer;
    return LIBGGUF_CUDA_STATUS_SUCCESS;
}

void libgguf_cuda_buffer_destroy(libgguf_cuda_context * ctx, libgguf_cuda_buffer * buffer) {
    if (!buffer) {
        return;
    }
    if (ctx) {
        cudaSetDevice(ctx->device);
    }
    if (buffer->data) {
        cudaFree(buffer->data);
    }
    delete buffer;
}

int libgguf_cuda_buffer_resize_discard(libgguf_cuda_context * ctx, libgguf_cuda_buffer * buffer, size_t size) {
    if (!ctx || !buffer) {
        return set_error(ctx, LIBGGUF_CUDA_STATUS_INVALID_ARGUMENT, "buffer discard resize requires a context and buffer");
    }
    if (size <= buffer->capacity) {
        buffer->size = size;
        return set_error(ctx, LIBGGUF_CUDA_STATUS_SUCCESS, "");
    }
    int status = set_device(ctx);
    if (status != LIBGGUF_CUDA_STATUS_SUCCESS) {
        return status;
    }
    if (buffer->data) {
        const cudaError_t free_error = cudaFree(buffer->data);
        buffer->data = nullptr;
        buffer->size = 0;
        buffer->capacity = 0;
        if (free_error != cudaSuccess) {
            return set_cuda_error(ctx, free_error);
        }
    }
    void * data = nullptr;
    const cudaError_t alloc_error = cudaMalloc(&data, size);
    if (alloc_error != cudaSuccess) {
        return set_cuda_error(ctx, alloc_error);
    }
    buffer->data = data;
    buffer->size = size;
    buffer->capacity = size;
    return set_error(ctx, LIBGGUF_CUDA_STATUS_SUCCESS, "");
}

void * libgguf_cuda_buffer_data(libgguf_cuda_buffer * buffer) {
    return buffer ? buffer->data : nullptr;
}

const void * libgguf_cuda_buffer_const_data(const libgguf_cuda_buffer * buffer) {
    return buffer ? buffer->data : nullptr;
}

size_t libgguf_cuda_buffer_size(const libgguf_cuda_buffer * buffer) {
    return buffer ? buffer->size : 0;
}

int libgguf_cuda_host_buffer_create(libgguf_cuda_context * ctx, size_t size, libgguf_cuda_host_buffer ** out) {
    if (!ctx || !out) {
        return set_error(ctx, LIBGGUF_CUDA_STATUS_INVALID_ARGUMENT, "host buffer create requires a context and output pointer");
    }
    *out = nullptr;
    libgguf_cuda_host_buffer * buffer = new (std::nothrow) libgguf_cuda_host_buffer();
    if (!buffer) {
        return set_error(ctx, LIBGGUF_CUDA_STATUS_ALLOCATION_FAILED, "failed to allocate CUDA host buffer handle");
    }
    const int status = libgguf_cuda_host_buffer_resize_discard(ctx, buffer, size);
    if (status != LIBGGUF_CUDA_STATUS_SUCCESS) {
        delete buffer;
        return status;
    }
    *out = buffer;
    return LIBGGUF_CUDA_STATUS_SUCCESS;
}

void libgguf_cuda_host_buffer_destroy(libgguf_cuda_context * ctx, libgguf_cuda_host_buffer * buffer) {
    if (!buffer) {
        return;
    }
    if (ctx) {
        cudaSetDevice(ctx->device);
    }
    if (buffer->data) {
        cudaFreeHost(buffer->data);
    }
    delete buffer;
}

int libgguf_cuda_host_buffer_resize_discard(libgguf_cuda_context * ctx, libgguf_cuda_host_buffer * buffer, size_t size) {
    if (!ctx || !buffer) {
        return set_error(ctx, LIBGGUF_CUDA_STATUS_INVALID_ARGUMENT, "host buffer discard resize requires a context and buffer");
    }
    if (size <= buffer->capacity) {
        buffer->size = size;
        return set_error(ctx, LIBGGUF_CUDA_STATUS_SUCCESS, "");
    }
    int status = set_device(ctx);
    if (status != LIBGGUF_CUDA_STATUS_SUCCESS) {
        return status;
    }
    if (buffer->data) {
        const cudaError_t free_error = cudaFreeHost(buffer->data);
        buffer->data = nullptr;
        buffer->size = 0;
        buffer->capacity = 0;
        if (free_error != cudaSuccess) {
            return set_cuda_error(ctx, free_error);
        }
    }
    void * data = nullptr;
    const cudaError_t alloc_error = cudaMallocHost(&data, size);
    if (alloc_error != cudaSuccess) {
        return set_cuda_error(ctx, alloc_error);
    }
    buffer->data = data;
    buffer->size = size;
    buffer->capacity = size;
    return set_error(ctx, LIBGGUF_CUDA_STATUS_SUCCESS, "");
}

void * libgguf_cuda_host_buffer_data(libgguf_cuda_host_buffer * buffer) {
    return buffer ? buffer->data : nullptr;
}

const void * libgguf_cuda_host_buffer_const_data(const libgguf_cuda_host_buffer * buffer) {
    return buffer ? buffer->data : nullptr;
}

size_t libgguf_cuda_host_buffer_size(const libgguf_cuda_host_buffer * buffer) {
    return buffer ? buffer->size : 0;
}

int libgguf_cuda_h2d(
    libgguf_cuda_context * ctx,
    libgguf_cuda_buffer * dst,
    size_t dst_offset,
    const void * src,
    size_t size
) {
    if (!ctx || !dst || (!src && size > 0) || dst_offset > dst->size || size > dst->size - dst_offset) {
        return set_error(ctx, LIBGGUF_CUDA_STATUS_INVALID_ARGUMENT, "invalid host-to-device copy arguments");
    }
    const int status = set_device(ctx);
    if (status != LIBGGUF_CUDA_STATUS_SUCCESS || size == 0) {
        return status;
    }
    const cudaError_t error = cudaMemcpyAsync(
        static_cast<char *>(dst->data) + dst_offset,
        src,
        size,
        cudaMemcpyHostToDevice,
        ctx->stream
    );
    if (error != cudaSuccess) {
        return set_cuda_error(ctx, error);
    }
    return set_error(ctx, LIBGGUF_CUDA_STATUS_SUCCESS, "");
}

int libgguf_cuda_d2h(
    libgguf_cuda_context * ctx,
    void * dst,
    const libgguf_cuda_buffer * src,
    size_t src_offset,
    size_t size
) {
    if (!ctx || (!dst && size > 0) || !src || src_offset > src->size || size > src->size - src_offset) {
        return set_error(ctx, LIBGGUF_CUDA_STATUS_INVALID_ARGUMENT, "invalid device-to-host copy arguments");
    }
    const int status = set_device(ctx);
    if (status != LIBGGUF_CUDA_STATUS_SUCCESS || size == 0) {
        return status;
    }
    const cudaError_t error = cudaMemcpyAsync(
        dst,
        static_cast<const char *>(src->data) + src_offset,
        size,
        cudaMemcpyDeviceToHost,
        ctx->stream
    );
    if (error != cudaSuccess) {
        return set_cuda_error(ctx, error);
    }
    return set_error(ctx, LIBGGUF_CUDA_STATUS_SUCCESS, "");
}

int libgguf_cuda_quantize_f32_rows(
    libgguf_cuda_context * ctx,
    const float * device_input,
    const float * device_imatrix,
    void * device_output,
    int64_t qtype,
    int64_t rows,
    int64_t n_per_row
) {
    if (!ctx || !device_input || !device_output || rows < 0 || n_per_row <= 0) {
        return set_error(ctx, LIBGGUF_CUDA_STATUS_INVALID_ARGUMENT, "invalid quantize arguments");
    }
    const int64_t block_size = gguf_cuda_quantize_block_size_for_type(qtype);
    if (block_size <= 0 || gguf_cuda_quantize_row_size_for_type(qtype, n_per_row) <= 0) {
        return set_error(ctx, LIBGGUF_CUDA_STATUS_UNSUPPORTED_TYPE, "unsupported CUDA quantization type");
    }
    if (n_per_row % block_size != 0) {
        return set_error(ctx, LIBGGUF_CUDA_STATUS_INVALID_ARGUMENT, "row width is not divisible by quantization block size");
    }
    if (gguf_cuda_quantize_type_needs_imatrix(qtype) && !device_imatrix) {
        return set_error(ctx, LIBGGUF_CUDA_STATUS_INVALID_ARGUMENT, "quantization type requires an imatrix");
    }
    const int status = set_device(ctx);
    if (status != LIBGGUF_CUDA_STATUS_SUCCESS || rows == 0) {
        return status;
    }
    gguf_cuda_quantize_row(device_input, device_imatrix, device_output, qtype, rows * n_per_row, n_per_row, ctx->stream);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        return set_cuda_error(ctx, error);
    }
    return set_error(ctx, LIBGGUF_CUDA_STATUS_SUCCESS, "");
}

}
