#include "conversion/gpu_quantizer.h"

#include <libgguf/gpu/compat.h>

#include <algorithm>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include "quantize/gpu.h"
#include <libgguf/formats.h>

namespace {

enum libgguf_gpu_status {
    LIBGGUF_GPU_STATUS_SUCCESS = 0,
    LIBGGUF_GPU_STATUS_INVALID_ARGUMENT = 1,
    LIBGGUF_GPU_STATUS_UNSUPPORTED_TYPE = 2,
    LIBGGUF_GPU_STATUS_GPU_ERROR = 3,
    LIBGGUF_GPU_STATUS_ALLOCATION_FAILED = 4,
};

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
using gpu_error_t = hipError_t;
using gpu_event_t = hipEvent_t;
#define gpu_event_create hipEventCreate
#define gpu_event_destroy hipEventDestroy
#define gpu_event_elapsed_time hipEventElapsedTime
#define gpu_event_record hipEventRecord
#define gpu_event_synchronize hipEventSynchronize
#define gpu_free hipFree
#define gpu_free_host hipHostFree
#define gpu_get_error_string hipGetErrorString
#define gpu_get_last_error hipGetLastError
#define gpu_malloc hipMalloc
#define gpu_malloc_host hipHostMalloc
#define gpu_memcpy_async hipMemcpyAsync
#define gpu_memcpy_device_to_host hipMemcpyDeviceToHost
#define gpu_memcpy_host_to_device hipMemcpyHostToDevice
#define gpu_set_device hipSetDevice
#define gpu_stream_create_with_flags hipStreamCreateWithFlags
#define gpu_stream_destroy hipStreamDestroy
#define gpu_stream_non_blocking hipStreamNonBlocking
#define gpu_success hipSuccess
#else
using gpu_error_t = cudaError_t;
using gpu_event_t = cudaEvent_t;
#define gpu_event_create cudaEventCreate
#define gpu_event_destroy cudaEventDestroy
#define gpu_event_elapsed_time cudaEventElapsedTime
#define gpu_event_record cudaEventRecord
#define gpu_event_synchronize cudaEventSynchronize
#define gpu_free cudaFree
#define gpu_free_host cudaFreeHost
#define gpu_get_error_string cudaGetErrorString
#define gpu_get_last_error cudaGetLastError
#define gpu_malloc cudaMalloc
#define gpu_malloc_host cudaMallocHost
#define gpu_memcpy_async cudaMemcpyAsync
#define gpu_memcpy_device_to_host cudaMemcpyDeviceToHost
#define gpu_memcpy_host_to_device cudaMemcpyHostToDevice
#define gpu_set_device cudaSetDevice
#define gpu_stream_create_with_flags cudaStreamCreateWithFlags
#define gpu_stream_destroy cudaStreamDestroy
#define gpu_stream_non_blocking cudaStreamNonBlocking
#define gpu_success cudaSuccess
#endif

struct libgguf_gpu_context {
    int device = 0;
    StreamType stream = nullptr;
    std::string last_error;
};

struct libgguf_gpu_buffer {
    void * data = nullptr;
    size_t size = 0;
    size_t capacity = 0;
};

struct libgguf_gpu_event {
    gpu_event_t event = nullptr;
};

struct libgguf_gpu_host_buffer {
    void * data = nullptr;
    size_t size = 0;
    size_t capacity = 0;
};


int libgguf_gpu_buffer_resize_discard(libgguf_gpu_context * ctx, libgguf_gpu_buffer * buffer, size_t size);
int libgguf_gpu_host_buffer_resize_discard(libgguf_gpu_context * ctx, libgguf_gpu_host_buffer * buffer, size_t size);
static int set_error(libgguf_gpu_context * ctx, int status, const char * message) {
    if (ctx) {
        ctx->last_error = message ? message : "";
    }
    return status;
}

static int set_gpu_error(libgguf_gpu_context * ctx, gpu_error_t error) {
    if (ctx) {
        ctx->last_error = gpu_get_error_string(error);
    }
    return LIBGGUF_GPU_STATUS_GPU_ERROR;
}

static int set_device(libgguf_gpu_context * ctx) {
    if (!ctx) {
        return LIBGGUF_GPU_STATUS_INVALID_ARGUMENT;
    }
    const gpu_error_t error = gpu_set_device(ctx->device);
    if (error != gpu_success) {
        return set_gpu_error(ctx, error);
    }
    return LIBGGUF_GPU_STATUS_SUCCESS;
}

int libgguf_gpu_context_create(int device, libgguf_gpu_context ** out) {
    if (!out) {
        return LIBGGUF_GPU_STATUS_INVALID_ARGUMENT;
    }
    *out = nullptr;
    libgguf_gpu_context * ctx = new (std::nothrow) libgguf_gpu_context();
    if (!ctx) {
        return LIBGGUF_GPU_STATUS_ALLOCATION_FAILED;
    }
    ctx->device = device;
    gpu_error_t error = gpu_set_device(device);
    if (error != gpu_success) {
        const int status = set_gpu_error(ctx, error);
        delete ctx;
        return status;
    }
    error = gpu_stream_create_with_flags(&ctx->stream, gpu_stream_non_blocking);
    if (error != gpu_success) {
        const int status = set_gpu_error(ctx, error);
        delete ctx;
        return status;
    }
    *out = ctx;
    return LIBGGUF_GPU_STATUS_SUCCESS;
}

void libgguf_gpu_context_destroy(libgguf_gpu_context * ctx) {
    if (!ctx) {
        return;
    }
    gpu_set_device(ctx->device);
    if (ctx->stream) {
        gpu_stream_destroy(ctx->stream);
    }
    delete ctx;
}

const char * libgguf_gpu_last_error(const libgguf_gpu_context * ctx) {
    if (!ctx) {
        return "libgguf GPU context is null";
    }
    return ctx->last_error.c_str();
}

int libgguf_gpu_event_create(libgguf_gpu_context * ctx, libgguf_gpu_event ** out) {
    if (!ctx || !out) {
        return set_error(ctx, LIBGGUF_GPU_STATUS_INVALID_ARGUMENT, "event create requires a context and output pointer");
    }
    *out = nullptr;
    const int status = set_device(ctx);
    if (status != LIBGGUF_GPU_STATUS_SUCCESS) {
        return status;
    }
    libgguf_gpu_event * event = new (std::nothrow) libgguf_gpu_event();
    if (!event) {
        return set_error(ctx, LIBGGUF_GPU_STATUS_ALLOCATION_FAILED, "failed to allocate GPU event handle");
    }
    const gpu_error_t error = gpu_event_create(&event->event);
    if (error != gpu_success) {
        delete event;
        return set_gpu_error(ctx, error);
    }
    *out = event;
    return set_error(ctx, LIBGGUF_GPU_STATUS_SUCCESS, "");
}

void libgguf_gpu_event_destroy(libgguf_gpu_context * ctx, libgguf_gpu_event * event) {
    if (!event) {
        return;
    }
    if (ctx) {
        gpu_set_device(ctx->device);
    }
    if (event->event) {
        gpu_event_destroy(event->event);
    }
    delete event;
}

int libgguf_gpu_event_record(libgguf_gpu_context * ctx, libgguf_gpu_event * event) {
    if (!ctx || !event || !event->event) {
        return set_error(ctx, LIBGGUF_GPU_STATUS_INVALID_ARGUMENT, "event record requires a context and event");
    }
    const int status = set_device(ctx);
    if (status != LIBGGUF_GPU_STATUS_SUCCESS) {
        return status;
    }
    const gpu_error_t error = gpu_event_record(event->event, ctx->stream);
    if (error != gpu_success) {
        return set_gpu_error(ctx, error);
    }
    return set_error(ctx, LIBGGUF_GPU_STATUS_SUCCESS, "");
}

int libgguf_gpu_event_synchronize(libgguf_gpu_context * ctx, libgguf_gpu_event * event) {
    if (!ctx || !event || !event->event) {
        return set_error(ctx, LIBGGUF_GPU_STATUS_INVALID_ARGUMENT, "event synchronize requires a context and event");
    }
    const int status = set_device(ctx);
    if (status != LIBGGUF_GPU_STATUS_SUCCESS) {
        return status;
    }
    const gpu_error_t error = gpu_event_synchronize(event->event);
    if (error != gpu_success) {
        return set_gpu_error(ctx, error);
    }
    return set_error(ctx, LIBGGUF_GPU_STATUS_SUCCESS, "");
}

int libgguf_gpu_event_elapsed_ms(
    libgguf_gpu_context * ctx,
    const libgguf_gpu_event * start,
    const libgguf_gpu_event * end,
    float * out
) {
    if (!ctx || !start || !start->event || !end || !end->event || !out) {
        return set_error(ctx, LIBGGUF_GPU_STATUS_INVALID_ARGUMENT, "event elapsed requires a context, events, and output pointer");
    }
    float elapsed = 0.0f;
    const gpu_error_t error = gpu_event_elapsed_time(&elapsed, start->event, end->event);
    if (error != gpu_success) {
        return set_gpu_error(ctx, error);
    }
    *out = elapsed;
    return set_error(ctx, LIBGGUF_GPU_STATUS_SUCCESS, "");
}

int libgguf_gpu_qtype_supported(int64_t qtype) {
    return quantize_type_supported((ggml_type)qtype) ? 1 : 0;
}

int libgguf_gpu_qtype_needs_imatrix(int64_t qtype) {
    return libgguf_quantize_requires_imatrix((ggml_type)qtype) ? 1 : 0;
}

int64_t libgguf_gpu_block_size(int64_t qtype) {
    return libgguf_block_size((ggml_type)qtype);
}

int64_t libgguf_gpu_row_size(int64_t qtype, int64_t n_per_row) {
    if (n_per_row < 0) {
        return 0;
    }
    return (int64_t)libgguf_row_size((ggml_type)qtype, n_per_row);
}

int libgguf_gpu_buffer_create(libgguf_gpu_context * ctx, size_t size, libgguf_gpu_buffer ** out) {
    if (!ctx || !out) {
        return set_error(ctx, LIBGGUF_GPU_STATUS_INVALID_ARGUMENT, "buffer create requires a context and output pointer");
    }
    *out = nullptr;
    libgguf_gpu_buffer * buffer = new (std::nothrow) libgguf_gpu_buffer();
    if (!buffer) {
        return set_error(ctx, LIBGGUF_GPU_STATUS_ALLOCATION_FAILED, "failed to allocate GPU buffer handle");
    }
    const int status = libgguf_gpu_buffer_resize_discard(ctx, buffer, size);
    if (status != LIBGGUF_GPU_STATUS_SUCCESS) {
        delete buffer;
        return status;
    }
    *out = buffer;
    return LIBGGUF_GPU_STATUS_SUCCESS;
}

void libgguf_gpu_buffer_destroy(libgguf_gpu_context * ctx, libgguf_gpu_buffer * buffer) {
    if (!buffer) {
        return;
    }
    if (ctx) {
        gpu_set_device(ctx->device);
    }
    if (buffer->data) {
        gpu_free(buffer->data);
    }
    delete buffer;
}

int libgguf_gpu_buffer_resize_discard(libgguf_gpu_context * ctx, libgguf_gpu_buffer * buffer, size_t size) {
    if (!ctx || !buffer) {
        return set_error(ctx, LIBGGUF_GPU_STATUS_INVALID_ARGUMENT, "buffer discard resize requires a context and buffer");
    }
    if (size <= buffer->capacity) {
        buffer->size = size;
        return set_error(ctx, LIBGGUF_GPU_STATUS_SUCCESS, "");
    }
    int status = set_device(ctx);
    if (status != LIBGGUF_GPU_STATUS_SUCCESS) {
        return status;
    }
    if (buffer->data) {
        const gpu_error_t free_error = gpu_free(buffer->data);
        buffer->data = nullptr;
        buffer->size = 0;
        buffer->capacity = 0;
        if (free_error != gpu_success) {
            return set_gpu_error(ctx, free_error);
        }
    }
    void * data = nullptr;
    const gpu_error_t alloc_error = gpu_malloc(&data, size);
    if (alloc_error != gpu_success) {
        return set_gpu_error(ctx, alloc_error);
    }
    buffer->data = data;
    buffer->size = size;
    buffer->capacity = size;
    return set_error(ctx, LIBGGUF_GPU_STATUS_SUCCESS, "");
}

void * libgguf_gpu_buffer_data(libgguf_gpu_buffer * buffer) {
    return buffer ? buffer->data : nullptr;
}

const void * libgguf_gpu_buffer_const_data(const libgguf_gpu_buffer * buffer) {
    return buffer ? buffer->data : nullptr;
}

size_t libgguf_gpu_buffer_size(const libgguf_gpu_buffer * buffer) {
    return buffer ? buffer->size : 0;
}

int libgguf_gpu_host_buffer_create(libgguf_gpu_context * ctx, size_t size, libgguf_gpu_host_buffer ** out) {
    if (!ctx || !out) {
        return set_error(ctx, LIBGGUF_GPU_STATUS_INVALID_ARGUMENT, "host buffer create requires a context and output pointer");
    }
    *out = nullptr;
    libgguf_gpu_host_buffer * buffer = new (std::nothrow) libgguf_gpu_host_buffer();
    if (!buffer) {
        return set_error(ctx, LIBGGUF_GPU_STATUS_ALLOCATION_FAILED, "failed to allocate GPU host buffer handle");
    }
    const int status = libgguf_gpu_host_buffer_resize_discard(ctx, buffer, size);
    if (status != LIBGGUF_GPU_STATUS_SUCCESS) {
        delete buffer;
        return status;
    }
    *out = buffer;
    return LIBGGUF_GPU_STATUS_SUCCESS;
}

void libgguf_gpu_host_buffer_destroy(libgguf_gpu_context * ctx, libgguf_gpu_host_buffer * buffer) {
    if (!buffer) {
        return;
    }
    if (ctx) {
        gpu_set_device(ctx->device);
    }
    if (buffer->data) {
        gpu_free_host(buffer->data);
    }
    delete buffer;
}

int libgguf_gpu_host_buffer_resize_discard(libgguf_gpu_context * ctx, libgguf_gpu_host_buffer * buffer, size_t size) {
    if (!ctx || !buffer) {
        return set_error(ctx, LIBGGUF_GPU_STATUS_INVALID_ARGUMENT, "host buffer discard resize requires a context and buffer");
    }
    if (size <= buffer->capacity) {
        buffer->size = size;
        return set_error(ctx, LIBGGUF_GPU_STATUS_SUCCESS, "");
    }
    int status = set_device(ctx);
    if (status != LIBGGUF_GPU_STATUS_SUCCESS) {
        return status;
    }
    if (buffer->data) {
        const gpu_error_t free_error = gpu_free_host(buffer->data);
        buffer->data = nullptr;
        buffer->size = 0;
        buffer->capacity = 0;
        if (free_error != gpu_success) {
            return set_gpu_error(ctx, free_error);
        }
    }
    void * data = nullptr;
    const gpu_error_t alloc_error = gpu_malloc_host(&data, size);
    if (alloc_error != gpu_success) {
        return set_gpu_error(ctx, alloc_error);
    }
    buffer->data = data;
    buffer->size = size;
    buffer->capacity = size;
    return set_error(ctx, LIBGGUF_GPU_STATUS_SUCCESS, "");
}

void * libgguf_gpu_host_buffer_data(libgguf_gpu_host_buffer * buffer) {
    return buffer ? buffer->data : nullptr;
}

const void * libgguf_gpu_host_buffer_const_data(const libgguf_gpu_host_buffer * buffer) {
    return buffer ? buffer->data : nullptr;
}

size_t libgguf_gpu_host_buffer_size(const libgguf_gpu_host_buffer * buffer) {
    return buffer ? buffer->size : 0;
}

int libgguf_gpu_h2d(
    libgguf_gpu_context * ctx,
    libgguf_gpu_buffer * dst,
    size_t dst_offset,
    const void * src,
    size_t size
) {
    if (!ctx || !dst || (!src && size > 0) || dst_offset > dst->size || size > dst->size - dst_offset) {
        return set_error(ctx, LIBGGUF_GPU_STATUS_INVALID_ARGUMENT, "invalid host-to-device copy arguments");
    }
    const int status = set_device(ctx);
    if (status != LIBGGUF_GPU_STATUS_SUCCESS || size == 0) {
        return status;
    }
    const gpu_error_t error = gpu_memcpy_async(
        static_cast<char *>(dst->data) + dst_offset,
        src,
        size,
        gpu_memcpy_host_to_device,
        ctx->stream
    );
    if (error != gpu_success) {
        return set_gpu_error(ctx, error);
    }
    return set_error(ctx, LIBGGUF_GPU_STATUS_SUCCESS, "");
}

int libgguf_gpu_d2h(
    libgguf_gpu_context * ctx,
    void * dst,
    const libgguf_gpu_buffer * src,
    size_t src_offset,
    size_t size
) {
    if (!ctx || (!dst && size > 0) || !src || src_offset > src->size || size > src->size - src_offset) {
        return set_error(ctx, LIBGGUF_GPU_STATUS_INVALID_ARGUMENT, "invalid device-to-host copy arguments");
    }
    const int status = set_device(ctx);
    if (status != LIBGGUF_GPU_STATUS_SUCCESS || size == 0) {
        return status;
    }
    const gpu_error_t error = gpu_memcpy_async(
        dst,
        static_cast<const char *>(src->data) + src_offset,
        size,
        gpu_memcpy_device_to_host,
        ctx->stream
    );
    if (error != gpu_success) {
        return set_gpu_error(ctx, error);
    }
    return set_error(ctx, LIBGGUF_GPU_STATUS_SUCCESS, "");
}

int libgguf_gpu_quantize_f32_rows(
    libgguf_gpu_context * ctx,
    const float * device_input,
    const float * device_imatrix,
    void * device_output,
    int64_t qtype,
    int64_t rows,
    int64_t n_per_row
) {
    if (!ctx || !device_input || !device_output || rows < 0 || n_per_row <= 0) {
        return set_error(ctx, LIBGGUF_GPU_STATUS_INVALID_ARGUMENT, "invalid quantize arguments");
    }
    const ggml_type type = (ggml_type)qtype;
    const int64_t block_size = libgguf_block_size(type);
    if (!quantize_type_supported(type) || block_size <= 0 || libgguf_row_size(type, n_per_row) == 0) {
        return set_error(ctx, LIBGGUF_GPU_STATUS_UNSUPPORTED_TYPE, "unsupported GPU quantization type");
    }
    if (n_per_row % block_size != 0) {
        return set_error(ctx, LIBGGUF_GPU_STATUS_INVALID_ARGUMENT, "row width is not divisible by quantization block size");
    }
    if (libgguf_quantize_requires_imatrix(type) && !device_imatrix) {
        return set_error(ctx, LIBGGUF_GPU_STATUS_INVALID_ARGUMENT, "quantization type requires an imatrix");
    }
    const int status = set_device(ctx);
    if (status != LIBGGUF_GPU_STATUS_SUCCESS || rows == 0) {
        return status;
    }
    quantize_row(device_input, device_imatrix, device_output, type, rows * n_per_row, n_per_row, ctx->stream);
    const gpu_error_t error = gpu_get_last_error();
    if (error != gpu_success) {
        return set_gpu_error(ctx, error);
    }
    return set_error(ctx, LIBGGUF_GPU_STATUS_SUCCESS, "");
}

std::string gpu_status_message(libgguf_gpu_context * ctx, int status, const std::string & prefix) {
    std::string message = prefix;
    if (status != LIBGGUF_GPU_STATUS_SUCCESS) {
        const char * last_error = libgguf_gpu_last_error(ctx);
        if (last_error && last_error[0]) {
            message += ": ";
            message += last_error;
        }
    }
    return message;
}

void require_gpu_success(libgguf_gpu_context * ctx, int status, const std::string & prefix) {
    if (status != LIBGGUF_GPU_STATUS_SUCCESS) {
        throw std::runtime_error(gpu_status_message(ctx, status, prefix));
    }
}

bool supported_qtype(ggml_type qtype) {
    switch (qtype) {
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_Q8_0:
    case GGML_TYPE_Q2_K:
    case GGML_TYPE_Q3_K:
    case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q5_K:
    case GGML_TYPE_Q6_K:
        return true;
    default:
        return false;
    }
}

} // namespace

struct gpu_quantizer::implementation {
    struct slot {
        libgguf_gpu_host_buffer * host_input = nullptr;
        libgguf_gpu_host_buffer * host_output = nullptr;
        libgguf_gpu_event * event_begin = nullptr;
        libgguf_gpu_event * event_after_h2d = nullptr;
        libgguf_gpu_event * event_after_quant = nullptr;
        libgguf_gpu_event * event_after_d2h = nullptr;
    };

    libgguf_gpu_context * context = nullptr;
    libgguf_gpu_buffer * input = nullptr;
    libgguf_gpu_buffer * output = nullptr;
    slot slots[2];

    slot & slot_at(size_t slot_index) {
        if (slot_index >= 2) {
            throw std::runtime_error("invalid GPU pipeline slot index");
        }
        return slots[slot_index];
    }

    void add_elapsed_ms(
        double * destination,
        libgguf_gpu_event * begin,
        libgguf_gpu_event * end
    ) {
        float elapsed_ms = 0.0f;
        require_gpu_success(
            context,
            libgguf_gpu_event_elapsed_ms(context, begin, end, &elapsed_ms),
            "failed to measure GPU timing event"
        );
        *destination += (double)elapsed_ms / 1000.0;
    }

    void cleanup() {
        for (slot & current : slots) {
            if (current.event_after_d2h) {
                libgguf_gpu_event_destroy(context, current.event_after_d2h);
                current.event_after_d2h = nullptr;
            }
            if (current.event_after_quant) {
                libgguf_gpu_event_destroy(context, current.event_after_quant);
                current.event_after_quant = nullptr;
            }
            if (current.event_after_h2d) {
                libgguf_gpu_event_destroy(context, current.event_after_h2d);
                current.event_after_h2d = nullptr;
            }
            if (current.event_begin) {
                libgguf_gpu_event_destroy(context, current.event_begin);
                current.event_begin = nullptr;
            }
            if (current.host_output) {
                libgguf_gpu_host_buffer_destroy(context, current.host_output);
                current.host_output = nullptr;
            }
            if (current.host_input) {
                libgguf_gpu_host_buffer_destroy(context, current.host_input);
                current.host_input = nullptr;
            }
        }
        if (output) {
            libgguf_gpu_buffer_destroy(context, output);
            output = nullptr;
        }
        if (input) {
            libgguf_gpu_buffer_destroy(context, input);
            input = nullptr;
        }
        if (context) {
            libgguf_gpu_context_destroy(context);
            context = nullptr;
        }
    }
};

gpu_quantizer::gpu_quantizer() : implementation_(new implementation()) {
    libgguf_gpu_context * created = nullptr;
    const int status = libgguf_gpu_context_create(0, &created);
    if (status != LIBGGUF_GPU_STATUS_SUCCESS) {
        std::string message = "failed to initialize GPU backend";
        if (created) {
            message = gpu_status_message(created, status, message);
            libgguf_gpu_context_destroy(created);
        }
        throw std::runtime_error(message);
    }
    implementation_->context = created;
    try {
        require_gpu_success(created, libgguf_gpu_buffer_create(created, 0, &implementation_->input), "failed to create GPU input buffer");
        require_gpu_success(created, libgguf_gpu_buffer_create(created, 0, &implementation_->output), "failed to create GPU output buffer");
        for (implementation::slot & current : implementation_->slots) {
            require_gpu_success(created, libgguf_gpu_host_buffer_create(created, 0, &current.host_input), "failed to create pinned GPU input buffer");
            require_gpu_success(created, libgguf_gpu_host_buffer_create(created, 0, &current.host_output), "failed to create pinned GPU output buffer");
            require_gpu_success(created, libgguf_gpu_event_create(created, &current.event_begin), "failed to create GPU timing event");
            require_gpu_success(created, libgguf_gpu_event_create(created, &current.event_after_h2d), "failed to create GPU timing event");
            require_gpu_success(created, libgguf_gpu_event_create(created, &current.event_after_quant), "failed to create GPU timing event");
            require_gpu_success(created, libgguf_gpu_event_create(created, &current.event_after_d2h), "failed to create GPU timing event");
        }
    } catch (...) {
        implementation_->cleanup();
        throw;
    }
}

gpu_quantizer::~gpu_quantizer() {
    implementation_->cleanup();
}

bool gpu_quantizer::supports(ggml_type qtype, uint64_t n_per_row) const {
    return supported_qtype(qtype) &&
           libgguf_gpu_qtype_supported((int64_t)qtype) &&
           !libgguf_gpu_qtype_needs_imatrix((int64_t)qtype) &&
           libgguf_gpu_row_size((int64_t)qtype, (int64_t)n_per_row) > 0;
}

void gpu_quantizer::resize_host_slot(size_t slot_index, size_t input_bytes, size_t output_bytes) {
    implementation::slot & current = implementation_->slot_at(slot_index);
    require_gpu_success(implementation_->context, libgguf_gpu_host_buffer_resize_discard(implementation_->context, current.host_input, input_bytes), "failed to resize pinned GPU input buffer");
    require_gpu_success(implementation_->context, libgguf_gpu_host_buffer_resize_discard(implementation_->context, current.host_output, output_bytes), "failed to resize pinned GPU output buffer");
}

float * gpu_quantizer::host_input_data(size_t slot_index) {
    return (float *)libgguf_gpu_host_buffer_data(implementation_->slot_at(slot_index).host_input);
}

uint8_t * gpu_quantizer::host_output_data(size_t slot_index) {
    return (uint8_t *)libgguf_gpu_host_buffer_data(implementation_->slot_at(slot_index).host_output);
}

void gpu_quantizer::resize_device_buffers(size_t input_bytes, size_t output_bytes) {
    require_gpu_success(implementation_->context, libgguf_gpu_buffer_resize_discard(implementation_->context, implementation_->input, input_bytes), "failed to resize GPU input buffer");
    require_gpu_success(implementation_->context, libgguf_gpu_buffer_resize_discard(implementation_->context, implementation_->output, output_bytes), "failed to resize GPU output buffer");
}

void gpu_quantizer::quantize_chunk(
    ggml_type qtype,
    uint64_t n_per_row,
    const float * host_input,
    uint64_t row_count,
    uint8_t * host_output,
    size_t output_bytes,
    gpu_quantizer_timings * timings
) {
    enqueue_quantize_chunk(0, qtype, n_per_row, host_input, row_count, host_output, output_bytes, timings);
    finish_quantize_chunk(0, timings);
}

void gpu_quantizer::enqueue_quantize_chunk(
    size_t slot_index,
    ggml_type qtype,
    uint64_t n_per_row,
    const float * host_input,
    uint64_t row_count,
    uint8_t * host_output,
    size_t output_bytes,
    gpu_quantizer_timings * timings
) {
    implementation::slot & current = implementation_->slot_at(slot_index);
    const size_t input_bytes = (size_t)(row_count * n_per_row * sizeof(float));
    if (timings) {
        timings->chunks += 1;
        timings->max_input_bytes = std::max<uint64_t>(timings->max_input_bytes, (uint64_t)input_bytes);
        timings->max_output_bytes = std::max<uint64_t>(timings->max_output_bytes, (uint64_t)output_bytes);
        require_gpu_success(implementation_->context, libgguf_gpu_event_record(implementation_->context, current.event_begin), "failed to record GPU timing event");
    }
    require_gpu_success(implementation_->context, libgguf_gpu_buffer_resize_discard(implementation_->context, implementation_->input, input_bytes), "failed to resize GPU input buffer");
    require_gpu_success(implementation_->context, libgguf_gpu_h2d(implementation_->context, implementation_->input, 0, host_input, input_bytes), "failed to copy tensor chunk to GPU");
    if (timings) {
        require_gpu_success(implementation_->context, libgguf_gpu_event_record(implementation_->context, current.event_after_h2d), "failed to record GPU timing event");
    }
    require_gpu_success(implementation_->context, libgguf_gpu_buffer_resize_discard(implementation_->context, implementation_->output, output_bytes), "failed to resize GPU output buffer");
    require_gpu_success(
        implementation_->context,
        libgguf_gpu_quantize_f32_rows(
            implementation_->context,
            (const float *)libgguf_gpu_buffer_const_data(implementation_->input),
            nullptr,
            libgguf_gpu_buffer_data(implementation_->output),
            (int64_t)qtype,
            (int64_t)row_count,
            (int64_t)n_per_row
        ),
        "failed to quantize tensor chunk on GPU"
    );
    if (timings) {
        require_gpu_success(implementation_->context, libgguf_gpu_event_record(implementation_->context, current.event_after_quant), "failed to record GPU timing event");
    }
    require_gpu_success(implementation_->context, libgguf_gpu_d2h(implementation_->context, host_output, implementation_->output, 0, output_bytes), "failed to copy tensor chunk from GPU");
    require_gpu_success(implementation_->context, libgguf_gpu_event_record(implementation_->context, current.event_after_d2h), "failed to record GPU completion event");
}

void gpu_quantizer::finish_quantize_chunk(size_t slot_index, gpu_quantizer_timings * timings) {
    implementation::slot & current = implementation_->slot_at(slot_index);
    require_gpu_success(implementation_->context, libgguf_gpu_event_synchronize(implementation_->context, current.event_after_d2h), "failed to synchronize GPU download");
    if (timings) {
        implementation_->add_elapsed_ms(&timings->h2d_s, current.event_begin, current.event_after_h2d);
        implementation_->add_elapsed_ms(&timings->quantize_s, current.event_after_h2d, current.event_after_quant);
        implementation_->add_elapsed_ms(&timings->d2h_s, current.event_after_quant, current.event_after_d2h);
    }
}
