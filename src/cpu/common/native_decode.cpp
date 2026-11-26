#include "cpu/common/native_decode.h"
#include <libgguf/gpu/native_encode.cuh>
#include <immintrin.h>

namespace libgguf::native_cpu {
namespace {
void decode8(__m128i codes,float scale,float* output) {
#if defined(LIBGGUF_NATIVE_DECODE_AVX2)
    const auto q=_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(codes));
    _mm256_storeu_ps(output,_mm256_mul_ps(q,_mm256_set1_ps(scale)));
#else
    const auto s8=_mm_cmpgt_epi8(_mm_setzero_si128(),codes);
    const auto words=_mm_unpacklo_epi8(codes,s8);
    const auto s16=_mm_cmpgt_epi16(_mm_setzero_si128(),words);
    const auto first=_mm_cvtepi32_ps(_mm_unpacklo_epi16(words,s16));
    const auto second=_mm_cvtepi32_ps(_mm_unpackhi_epi16(words,s16));
    const auto factor=_mm_set1_ps(scale);
    _mm_storeu_ps(output,_mm_mul_ps(first,factor));
    _mm_storeu_ps(output+4,_mm_mul_ps(second,factor));
#endif
}
void decode32_i4(const std::uint8_t* source,float scale,float* output) {
    const auto bytes=_mm_loadu_si128(reinterpret_cast<const __m128i*>(source));
    const auto mask=_mm_set1_epi8(15),sign=_mm_set1_epi8(8);
    const auto low=_mm_and_si128(bytes,mask),high=_mm_and_si128(_mm_srli_epi16(bytes,4),mask);
    const auto first=_mm_sub_epi8(_mm_xor_si128(_mm_unpacklo_epi8(low,high),sign),sign);
    const auto second=_mm_sub_epi8(_mm_xor_si128(_mm_unpackhi_epi8(low,high),sign),sign);
    decode8(first,scale,output);decode8(_mm_srli_si128(first,8),scale,output+8);
    decode8(second,scale,output+16);decode8(_mm_srli_si128(second,8),scale,output+24);
}
}
void decode(NativeFormat format,const std::uint8_t* source,std::size_t blocks,float* output) {
    const auto info=native_format_info(format);
    for(std::size_t block=0;block<blocks;++block) {
        const float base=native_detail::read_half(source);
        if(format==NativeFormat::s8g32) {
            for(int k=0;k<32;k+=8)decode8(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(source+2+k)),base,output+k);
        } else {
            for(unsigned k=0;k<info.values_per_block;k+=32) {
                const float scale=format==NativeFormat::s4h256 ? base*float(source[2+k/32]) : base;
                decode32_i4(source+(format==NativeFormat::s4h256?10:2)+k/2,scale,output+k);
            }
        }
        source+=info.bytes_per_block;output+=info.values_per_block;
    }
}
bool valid_f32(const float* source,std::size_t values,bool nonnegative) {
    std::size_t i=0;
#if defined(LIBGGUF_NATIVE_DECODE_AVX2)
    for(;values-i>=8;i+=8) {
        const auto data=_mm256_loadu_ps(source+i);
        const auto exponent=_mm256_and_si256(_mm256_castps_si256(data),_mm256_set1_epi32(0x7f800000));
        if(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(exponent,_mm256_set1_epi32(0x7f800000)))))return false;
        if(nonnegative && _mm256_movemask_ps(_mm256_cmp_ps(data,_mm256_setzero_ps(),_CMP_LT_OQ)))return false;
    }
#else
    for(;values-i>=4;i+=4) {
        const auto data=_mm_loadu_ps(source+i);
        const auto exponent=_mm_and_si128(_mm_castps_si128(data),_mm_set1_epi32(0x7f800000));
        if(_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(exponent,_mm_set1_epi32(0x7f800000)))))return false;
        if(nonnegative && _mm_movemask_ps(_mm_cmplt_ps(data,_mm_setzero_ps())))return false;
    }
#endif
    for(;i<values;++i)if(!native_detail::finite(source[i]) || (nonnegative && source[i]<0))return false;
    return true;
}
} // namespace libgguf::native_cpu
