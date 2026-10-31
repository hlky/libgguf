#pragma once
#include <libgguf/gpu/compat.h>
#include <libgguf/gpu/dequantize/decoder.cuh>
namespace libgguf::gpu::gfx942 {
// Decode in the format-defined lanes, then write contiguous FP32 vectors.
// Packed GGUF bytes and the canonical decoder remain unchanged.
template<ggml_type F> struct Decoder : BlockDecoder<F> {};
template<> struct Decoder<GGML_TYPE_IQ4_NL> {
  static constexpr int values_per_decode_group=32,threads_per_decode_group=4;
  __device__ static void decode_group(const void* source,float* out,int64_t group,int lane) {
    dequantize_iq4_nl(static_cast<const char*>(source)+group*sizeof(block_iq4_nl),out,0,lane*8);
  }
};
template<ggml_type Format,int Threads,int GroupsPerWave>
__global__ __launch_bounds__(Threads) void decode_coalesced(const void* input,float* output,int64_t count) {
#if defined(__HIP_DEVICE_COMPILE__)
  static_assert(Threads==128 || Threads==256);
#if !defined(__gfx942__)
#error "gfx942 decoder pipeline requires physical wave64"
#endif
  using D=Decoder<Format>;constexpr int G=D::values_per_decode_group,T=D::threads_per_decode_group;
  static_assert(T<=64 && G%4==0);
  const int lane=threadIdx.x&63,wave=threadIdx.x/64;
  __shared__ float tile[Threads/64][G];
  for(int step=0;step<GroupsPerWave;++step) {
    const auto group=(int64_t(blockIdx.x)*(Threads/64)+wave)*GroupsPerWave+step;
    // Canonical decoder reads complete encoded groups only. The checked host
    // contract requires count to be a multiple of G.
    if(group<count/G && lane<T) D::decode_group(input,tile[wave],group,lane);
    __syncthreads();
    if(group<count/G) {
      using V=float __attribute__((ext_vector_type(4)));
      for(int i=lane*4;i<G;i+=64*4) {
        V value;__builtin_memcpy(&value,tile[wave]+i,sizeof(value));
        __builtin_memcpy(output+group*G+i,&value,sizeof(value));
      }
    }
    __syncthreads();
  }
#endif
}
template<int Threads,int GroupsPerWave,bool Checked=true> inline hipError_t launch_decode(ggml_type type,const void* input,float* output,int64_t count,hipStream_t stream) {
  if(count<0 || (count && (!input || !output)))return hipErrorInvalidValue;
  if(!count)return hipSuccess;
  if constexpr(Checked) {
  hipDeviceProp_t p{};int device=0;
  auto status=hipGetDevice(&device);if(status!=hipSuccess)return status;
  status=hipGetDeviceProperties(&p,device);if(status!=hipSuccess)return status;
  if(p.warpSize!=64 || __builtin_strncmp(p.gcnArchName,"gfx942",6))return hipErrorInvalidDevice;
  }
#define LIBGGUF_GFX942_DECODE_CASE(F) case F: {using D=Decoder<F>;if(count%D::values_per_decode_group)return hipErrorInvalidValue;const auto groups=count/D::values_per_decode_group;const auto blocks=(groups+(Threads/64)*GroupsPerWave-1)/((Threads/64)*GroupsPerWave);if(blocks>2147483647)return hipErrorInvalidValue;decode_coalesced<F,Threads,GroupsPerWave><<<dim3(blocks),dim3(Threads),0,stream>>>(input,output,count);break;}
  switch(type) {
    LIBGGUF_GFX942_DECODE_CASE(GGML_TYPE_Q1_0)
    LIBGGUF_GFX942_DECODE_CASE(GGML_TYPE_Q4_0)
    LIBGGUF_GFX942_DECODE_CASE(GGML_TYPE_Q4_1)
    LIBGGUF_GFX942_DECODE_CASE(GGML_TYPE_Q5_0)
    LIBGGUF_GFX942_DECODE_CASE(GGML_TYPE_Q5_1)
    LIBGGUF_GFX942_DECODE_CASE(GGML_TYPE_Q8_0)
    LIBGGUF_GFX942_DECODE_CASE(GGML_TYPE_Q2_K)
    LIBGGUF_GFX942_DECODE_CASE(GGML_TYPE_Q3_K)
    LIBGGUF_GFX942_DECODE_CASE(GGML_TYPE_Q4_K)
    LIBGGUF_GFX942_DECODE_CASE(GGML_TYPE_Q5_K)
    LIBGGUF_GFX942_DECODE_CASE(GGML_TYPE_Q6_K)
    LIBGGUF_GFX942_DECODE_CASE(GGML_TYPE_IQ1_S)
    LIBGGUF_GFX942_DECODE_CASE(GGML_TYPE_IQ1_M)
    LIBGGUF_GFX942_DECODE_CASE(GGML_TYPE_IQ2_XXS)
    LIBGGUF_GFX942_DECODE_CASE(GGML_TYPE_IQ2_XS)
    LIBGGUF_GFX942_DECODE_CASE(GGML_TYPE_IQ2_S)
    LIBGGUF_GFX942_DECODE_CASE(GGML_TYPE_IQ3_XXS)
    LIBGGUF_GFX942_DECODE_CASE(GGML_TYPE_IQ3_S)
    LIBGGUF_GFX942_DECODE_CASE(GGML_TYPE_IQ4_NL)
    LIBGGUF_GFX942_DECODE_CASE(GGML_TYPE_IQ4_XS)
    LIBGGUF_GFX942_DECODE_CASE(GGML_TYPE_TQ1_0)
    LIBGGUF_GFX942_DECODE_CASE(GGML_TYPE_TQ2_0)
    LIBGGUF_GFX942_DECODE_CASE(GGML_TYPE_MXFP4)
    LIBGGUF_GFX942_DECODE_CASE(GGML_TYPE_NVFP4)
    default:return hipErrorInvalidValue;
  }
#undef LIBGGUF_GFX942_DECODE_CASE
  return hipPeekAtLastError();
}
}
