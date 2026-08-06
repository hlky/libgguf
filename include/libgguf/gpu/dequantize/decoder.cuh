#pragma once

#include <cstdint>

#include <libgguf/gpu/dequantize/bf16.cuh>
#include <libgguf/gpu/dequantize/iq1_m.cuh>
#include <libgguf/gpu/dequantize/iq1_s.cuh>
#include <libgguf/gpu/dequantize/iq2_s.cuh>
#include <libgguf/gpu/dequantize/iq2_xs.cuh>
#include <libgguf/gpu/dequantize/iq2_xxs.cuh>
#include <libgguf/gpu/dequantize/iq3_s.cuh>
#include <libgguf/gpu/dequantize/iq3_xxs.cuh>
#include <libgguf/gpu/dequantize/iq4_nl.cuh>
#include <libgguf/gpu/dequantize/iq4_xs.cuh>
#include <libgguf/gpu/dequantize/mxfp4.cuh>
#include <libgguf/gpu/dequantize/nvfp4.cuh>
#include <libgguf/gpu/dequantize/q1_0.cuh>
#include <libgguf/gpu/dequantize/q2_k.cuh>
#include <libgguf/gpu/dequantize/q3_k.cuh>
#include <libgguf/gpu/dequantize/q4_0.cuh>
#include <libgguf/gpu/dequantize/q4_1.cuh>
#include <libgguf/gpu/dequantize/q4_k.cuh>
#include <libgguf/gpu/dequantize/q5_0.cuh>
#include <libgguf/gpu/dequantize/q5_1.cuh>
#include <libgguf/gpu/dequantize/q5_k.cuh>
#include <libgguf/gpu/dequantize/q6_k.cuh>
#include <libgguf/gpu/dequantize/q8_0.cuh>
#include <libgguf/gpu/dequantize/tq1_0.cuh>
#include <libgguf/gpu/dequantize/tq2_0.cuh>
#include <libgguf/types.h>

namespace libgguf
{
namespace gpu
{

template <ggml_type Format>
struct BlockDecoder;

template <>
struct BlockDecoder<GGML_TYPE_BF16>
{
    static constexpr int64_t values_per_block = 1;
    static constexpr int64_t values_per_decode_group = 1;
    static constexpr int64_t threads_per_decode_group = 1;

    template <typename dst_t>
    static __device__ __forceinline__ void decode_group(
        const void * table, dst_t * output, int64_t group_index, int thread_index)
    {
        if (thread_index == 0)
        {
            output[0] = convert_from_float<dst_t>(dequantize_bf16(table, (int)group_index));
        }
    }
};

#define LIBGGUF_DEFINE_BLOCK_DECODER(FORMAT, VALUES_PER_BLOCK, VALUES_PER_GROUP, THREADS, FUNCTION) \
    template <>                                                                                       \
    struct BlockDecoder<FORMAT>                                                                       \
    {                                                                                                  \
        static constexpr int64_t values_per_block = VALUES_PER_BLOCK;                                 \
        static constexpr int64_t values_per_decode_group = VALUES_PER_GROUP;                           \
        static constexpr int64_t threads_per_decode_group = THREADS;                                   \
                                                                                                       \
        template <typename dst_t>                                                                      \
        static __device__ __forceinline__ void decode_group(                                           \
            const void * table, dst_t * output, int64_t group_index, int thread_index)                 \
        {                                                                                              \
            FUNCTION(table, output, (int)group_index, thread_index);                                   \
        }                                                                                              \
    }

LIBGGUF_DEFINE_BLOCK_DECODER(GGML_TYPE_Q2_K,    QK_K,    QK_K, 64, dequantize_q2_k);
LIBGGUF_DEFINE_BLOCK_DECODER(GGML_TYPE_Q3_K,    QK_K,    QK_K, 64, dequantize_q3_k);
LIBGGUF_DEFINE_BLOCK_DECODER(GGML_TYPE_Q4_K,    QK_K,    QK_K, 32, dequantize_q4_k);
LIBGGUF_DEFINE_BLOCK_DECODER(GGML_TYPE_Q5_K,    QK_K,    QK_K, 64, dequantize_q5_k);
LIBGGUF_DEFINE_BLOCK_DECODER(GGML_TYPE_Q6_K,    QK_K,    QK_K, 64, dequantize_q6_k);
LIBGGUF_DEFINE_BLOCK_DECODER(GGML_TYPE_IQ1_M,   QK_K,    QK_K, 32, dequantize_iq1_m);
LIBGGUF_DEFINE_BLOCK_DECODER(GGML_TYPE_IQ1_S,   QK_K,    QK_K, 32, dequantize_iq1_s);
LIBGGUF_DEFINE_BLOCK_DECODER(GGML_TYPE_IQ2_S,   QK_K,    QK_K, 32, dequantize_iq2_s);
LIBGGUF_DEFINE_BLOCK_DECODER(GGML_TYPE_IQ2_XS,  QK_K,    QK_K, 32, dequantize_iq2_xs);
LIBGGUF_DEFINE_BLOCK_DECODER(GGML_TYPE_IQ2_XXS, QK_K,    QK_K, 32, dequantize_iq2_xxs);
LIBGGUF_DEFINE_BLOCK_DECODER(GGML_TYPE_IQ3_S,   QK_K,    QK_K, 32, dequantize_iq3_s);
LIBGGUF_DEFINE_BLOCK_DECODER(GGML_TYPE_IQ3_XXS, QK_K,    QK_K, 32, dequantize_iq3_xxs);
LIBGGUF_DEFINE_BLOCK_DECODER(GGML_TYPE_IQ4_NL,  QK4_NL,  QK_K, 32, dequantize_iq4_nl);
LIBGGUF_DEFINE_BLOCK_DECODER(GGML_TYPE_IQ4_XS,  QK_K,    QK_K, 32, dequantize_iq4_xs);

#undef LIBGGUF_DEFINE_BLOCK_DECODER

template <ggml_type Format, int64_t ValuesPerBlock>
struct PairBlockDecoder;

template <ggml_type Format, int64_t ValuesPerBlock>
struct PairBlockDecoder
{
    static constexpr int64_t values_per_block = ValuesPerBlock;
    static constexpr int64_t values_per_decode_group = ValuesPerBlock;
    static constexpr int64_t threads_per_decode_group = ValuesPerBlock / 2;
    static constexpr int64_t values_per_decode = 2;
};

template <>
struct BlockDecoder<GGML_TYPE_Q1_0> : PairBlockDecoder<GGML_TYPE_Q1_0, QK1_0>
{
    template <typename dst_t>
    static __device__ __forceinline__ void decode_group(
        const void * table, dst_t * output, int64_t group_index, int thread_index)
    {
        dfloat2 value;
        dequantize_q1_0(table, (int)(group_index * threads_per_decode_group + thread_index), value);
        output[2 * thread_index] = convert_from_float<dst_t>(value.x);
        output[2 * thread_index + 1] = convert_from_float<dst_t>(value.y);
    }
};

#define LIBGGUF_DEFINE_SPLIT_PAIR_DECODER(FORMAT, QK, FUNCTION)                           \
    template <>                                                                           \
    struct BlockDecoder<FORMAT> : PairBlockDecoder<FORMAT, QK>                           \
    {                                                                                     \
        static __device__ __forceinline__ dfloat2 decode(                                 \
            const void * table, int64_t block_index, int64_t value_offset)               \
        {                                                                                 \
            dfloat2 value;                                                                \
            FUNCTION(table, (int)block_index, (int)(value_offset / 2), value);            \
            return value;                                                                 \
        }                                                                                 \
                                                                                          \
        template <typename dst_t>                                                         \
        static __device__ __forceinline__ void decode_group(                              \
            const void * table, dst_t * output, int64_t group_index, int thread_index)    \
        {                                                                                 \
            const dfloat2 value = decode(table, group_index, 2 * thread_index);           \
            output[thread_index] = convert_from_float<dst_t>(value.x);                    \
            output[thread_index + QK / 2] = convert_from_float<dst_t>(value.y);            \
        }                                                                                 \
    }

LIBGGUF_DEFINE_SPLIT_PAIR_DECODER(GGML_TYPE_Q4_0, QK4_0, dequantize_q4_0);
LIBGGUF_DEFINE_SPLIT_PAIR_DECODER(GGML_TYPE_Q4_1, QK4_1, dequantize_q4_1);
LIBGGUF_DEFINE_SPLIT_PAIR_DECODER(GGML_TYPE_Q5_0, QK5_0, dequantize_q5_0);
LIBGGUF_DEFINE_SPLIT_PAIR_DECODER(GGML_TYPE_Q5_1, QK5_1, dequantize_q5_1);

#undef LIBGGUF_DEFINE_SPLIT_PAIR_DECODER

template <>
struct BlockDecoder<GGML_TYPE_Q8_0> : PairBlockDecoder<GGML_TYPE_Q8_0, QK8_0>
{
    static __device__ __forceinline__ dfloat2 decode(
        const void * table, int64_t block_index, int64_t value_offset)
    {
        dfloat2 value;
        dequantize_q8_0(table, (int)block_index, (int)value_offset, value);
        return value;
    }

    template <typename dst_t>
    static __device__ __forceinline__ void decode_group(
        const void * table, dst_t * output, int64_t group_index, int thread_index)
    {
        const dfloat2 value = decode(table, group_index, 2 * thread_index);
        output[2 * thread_index] = convert_from_float<dst_t>(value.x);
        output[2 * thread_index + 1] = convert_from_float<dst_t>(value.y);
    }
};

template <>
struct BlockDecoder<GGML_TYPE_MXFP4> : PairBlockDecoder<GGML_TYPE_MXFP4, QK_MXFP4>
{
    template <typename dst_t>
    static __device__ __forceinline__ void decode_group(
        const void * table, dst_t * output, int64_t group_index, int thread_index)
    {
        dfloat2 value;
        dequantize_mxfp4(table, (int)(group_index * threads_per_decode_group + thread_index), value);
        output[thread_index] = convert_from_float<dst_t>(value.x);
        output[thread_index + QK_MXFP4 / 2] = convert_from_float<dst_t>(value.y);
    }
};

template <>
struct BlockDecoder<GGML_TYPE_NVFP4> : PairBlockDecoder<GGML_TYPE_NVFP4, QK_NVFP4>
{
    template <typename dst_t>
    static __device__ __forceinline__ void decode_group(
        const void * table, dst_t * output, int64_t group_index, int thread_index)
    {
        dfloat2 value;
        dequantize_nvfp4(table, (int)(group_index * threads_per_decode_group + thread_index), value);
        const int group = thread_index / 8;
        const int local = thread_index - group * 8;
        output[group * 16 + local] = convert_from_float<dst_t>(value.x);
        output[group * 16 + local + 8] = convert_from_float<dst_t>(value.y);
    }
};

template <>
struct BlockDecoder<GGML_TYPE_TQ1_0>
{
    static constexpr int64_t values_per_block = QK_K;
    static constexpr int64_t values_per_decode_group = QK_K;
    static constexpr int64_t threads_per_decode_group =
        sizeof(block_tq1_0::qs) + sizeof(block_tq1_0::qh);

    template <typename dst_t>
    static __device__ __forceinline__ void decode_group(
        const void * table, dst_t * output, int64_t group_index, int thread_index)
    {
        dequantize_tq1_0(table, output, (int)group_index, thread_index);
    }
};

template <>
struct BlockDecoder<GGML_TYPE_TQ2_0>
{
    static constexpr int64_t values_per_block = QK_K;
    static constexpr int64_t values_per_decode_group = QK_K;
    static constexpr int64_t threads_per_decode_group = sizeof(block_tq2_0::qs);

    template <typename dst_t>
    static __device__ __forceinline__ void decode_group(
        const void * table, dst_t * output, int64_t group_index, int thread_index)
    {
        dequantize_tq2_0(table, output, (int)group_index, thread_index);
    }
};

}
}
