#pragma once

#include "conversion/types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

struct input_file;

unsigned int native_thread_limit(unsigned int requested);
bool cpu_quantizer_supports(ggml_type qtype);
void decode_source_bytes_to_f32(
    const tensor_plan & plan,
    const uint8_t * source,
    uint64_t value_count,
    float * destination
);
void read_source_chunk(
    const input_file & input,
    const tensor_plan & plan,
    uint64_t row_begin,
    uint64_t row_count,
    std::vector<uint8_t> & bytes
);

class cpu_quantizer {
public:
    explicit cpu_quantizer(unsigned int max_threads);
    ~cpu_quantizer();

    cpu_quantizer(const cpu_quantizer &) = delete;
    cpu_quantizer & operator=(const cpu_quantizer &) = delete;

    bool run(
        const tensor_plan & plan,
        const uint8_t * source_bytes,
        uint64_t row_count,
        void * destination,
        const float * imatrix,
        uint64_t scratch_bytes
    );

private:
    struct implementation;
    std::unique_ptr<implementation> implementation_;
};
