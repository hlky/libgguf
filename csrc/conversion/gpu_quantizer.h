#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include <libgguf/types.h>

struct gpu_quantizer_timings {
    double h2d_s = 0.0;
    double quantize_s = 0.0;
    double d2h_s = 0.0;
    uint64_t chunks = 0;
    uint64_t max_input_bytes = 0;
    uint64_t max_output_bytes = 0;
};

class gpu_quantizer {
public:
    gpu_quantizer();
    ~gpu_quantizer();

    gpu_quantizer(const gpu_quantizer &) = delete;
    gpu_quantizer &operator=(const gpu_quantizer &) = delete;

    bool supports(ggml_type qtype, uint64_t n_per_row) const;

    void resize_host_slot(size_t slot_index, size_t input_bytes, size_t output_bytes);
    float * host_input_data(size_t slot_index);
    uint8_t * host_output_data(size_t slot_index);
    void resize_device_buffers(size_t input_bytes, size_t output_bytes);

    void quantize_chunk(
        ggml_type qtype,
        uint64_t n_per_row,
        const float * host_input,
        uint64_t row_count,
        uint8_t * host_output,
        size_t output_bytes,
        gpu_quantizer_timings * timings
    );

    void enqueue_quantize_chunk(
        size_t slot_index,
        ggml_type qtype,
        uint64_t n_per_row,
        const float * host_input,
        uint64_t row_count,
        uint8_t * host_output,
        size_t output_bytes,
        gpu_quantizer_timings * timings
    );

    void finish_quantize_chunk(size_t slot_index, gpu_quantizer_timings * timings);

private:
    struct implementation;
    std::unique_ptr<implementation> implementation_;
};
