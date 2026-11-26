#include <libgguf/native_gpu.h>
#include <libgguf/gpu/compat.h>
#include <libgguf/gpu/native_encode.cuh>
#include <limits>

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
#define NATIVE_STREAM hipStream_t
#define NATIVE_PEEK hipPeekAtLastError
#define NATIVE_SUCCESS hipSuccess
#else
#define NATIVE_STREAM cudaStream_t
#define NATIVE_PEEK cudaPeekAtLastError
#define NATIVE_SUCCESS cudaSuccess
#endif

namespace {
using libgguf::NativeFormat;
constexpr unsigned threads = 128;

bool valid_span(const void* p,size_t bytes) {
    return p && bytes && bytes <= std::numeric_limits<uintptr_t>::max()-reinterpret_cast<uintptr_t>(p);
}
bool overlap(const void* a,size_t na,const void* b,size_t nb) {
    return reinterpret_cast<uintptr_t>(a)<reinterpret_cast<uintptr_t>(b)+nb && reinterpret_cast<uintptr_t>(b)<reinterpret_cast<uintptr_t>(a)+na;
}
libgguf_native_status geometry(libgguf_native_format f,size_t rows,size_t columns,size_t& packed,size_t& dense) {
    const auto info=libgguf::native_format_info(static_cast<NativeFormat>(f));
    if(!info.name)return LIBGGUF_NATIVE_INVALID_FORMAT;
    if(!rows || !columns || columns%info.values_per_block)return LIBGGUF_NATIVE_INVALID_SHAPE;
    if(columns>SIZE_MAX/rows || rows*columns>SIZE_MAX/sizeof(float))return LIBGGUF_NATIVE_OVERFLOW;
    dense=rows*columns*sizeof(float);packed=rows*(columns/info.values_per_block)*info.bytes_per_block;
    // Bound the launch independently of device-specific grid maxima.
    if((dense/sizeof(float)+threads-1)/threads>2147483647u)return LIBGGUF_NATIVE_OVERFLOW;
    return LIBGGUF_NATIVE_OK;
}
bool valid_buffers(const void* source,size_t source_bytes,void* dest,size_t dest_bytes,uint32_t* status) {
    return valid_span(source,source_bytes) && valid_span(dest,dest_bytes) && valid_span(status,4)
        && reinterpret_cast<uintptr_t>(status)%4==0 && !overlap(source,source_bytes,dest,dest_bytes)
        && !overlap(source,source_bytes,status,4) && !overlap(dest,dest_bytes,status,4);
}
// Validation completes before any writes on the supplied stream. Its only
// reduction is the identical invalid-value status, so scheduling cannot change it.
__global__ void initialize_status(uint32_t* status) { *status=LIBGGUF_NATIVE_OK; }
__global__ void validate_encode(const float* source,size_t count,const float* importance,size_t columns,uint32_t* status) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count && !libgguf::native_detail::finite(source[i]))atomicExch(status,uint32_t(LIBGGUF_NATIVE_INVALID_VALUE));
    if(importance && i<columns && (!libgguf::native_detail::finite(importance[i]) || importance[i]<0))atomicExch(status,uint32_t(LIBGGUF_NATIVE_INVALID_VALUE));
}
__global__ void validate_decode(NativeFormat format,const unsigned char* source,size_t blocks,uint32_t* status) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const auto info=libgguf::native_format_info(format);
    if(i<blocks && !libgguf::native_detail::valid_block(format,source+i*info.bytes_per_block))atomicExch(status,uint32_t(LIBGGUF_NATIVE_INVALID_VALUE));
}
__global__ void encode(NativeFormat format,const float* source,const float* importance,size_t columns,unsigned char* dest,size_t blocks,const uint32_t* status) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=blocks || *status!=LIBGGUF_NATIVE_OK)return;
    const auto info=libgguf::native_format_info(format);
    libgguf::native_detail::encode_block(format,source+i*info.values_per_block,
        importance?importance+(i*info.values_per_block)%columns:nullptr,dest+i*info.bytes_per_block);
}
__global__ void decode(NativeFormat format,const unsigned char* source,float* dest,size_t count,const uint32_t* status) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=count || *status!=LIBGGUF_NATIVE_OK)return;
    const auto info=libgguf::native_format_info(format);
    const auto* block=source+i/info.values_per_block*info.bytes_per_block;
    dest[i]=libgguf::native_detail::scale(format,block,i%info.values_per_block)*float(libgguf::native_detail::quant(format,block,i%info.values_per_block));
}
}
extern "C" LIBGGUF_NATIVE_API libgguf_native_status libgguf_native_encode_gpu_into(
    libgguf_native_format f,const float* source,size_t source_bytes,size_t rows,size_t columns,
    const float* importance,size_t importance_bytes,void* dest,size_t dest_bytes,uint32_t* status,void* stream) {
    size_t packed=0,dense=0;auto result=geometry(f,rows,columns,packed,dense);if(result)return result;
    if(source_bytes!=dense || dest_bytes!=packed || !valid_buffers(source,source_bytes,dest,dest_bytes,status)
        || reinterpret_cast<uintptr_t>(source)%4)return LIBGGUF_NATIVE_INVALID_BUFFER;
    if(importance?(importance_bytes!=columns*4 || !valid_span(importance,importance_bytes)
        || reinterpret_cast<uintptr_t>(importance)%4 || overlap(importance,importance_bytes,dest,dest_bytes)
        || overlap(importance,importance_bytes,status,4)):importance_bytes!=0)return LIBGGUF_NATIVE_INVALID_BUFFER;
    auto s=reinterpret_cast<NATIVE_STREAM>(stream);const auto format=static_cast<NativeFormat>(f);
    const size_t blocks=rows*columns/libgguf::native_format_info(format).values_per_block;
    initialize_status<<<1,1,0,s>>>(status);
    if(NATIVE_PEEK()!=NATIVE_SUCCESS)return LIBGGUF_NATIVE_GPU_ERROR;
    validate_encode<<<unsigned((rows*columns+threads-1)/threads),threads,0,s>>>(source,rows*columns,importance,columns,status);
    if(NATIVE_PEEK()!=NATIVE_SUCCESS)return LIBGGUF_NATIVE_GPU_ERROR;
    encode<<<unsigned((blocks+threads-1)/threads),threads,0,s>>>(format,source,importance,columns,static_cast<unsigned char*>(dest),blocks,status);
    return NATIVE_PEEK()==NATIVE_SUCCESS?LIBGGUF_NATIVE_OK:LIBGGUF_NATIVE_GPU_ERROR;
}
extern "C" LIBGGUF_NATIVE_API libgguf_native_status libgguf_native_decode_gpu_into(
    libgguf_native_format f,const void* source,size_t source_bytes,size_t rows,size_t columns,
    float* dest,size_t dest_bytes,uint32_t* status,void* stream) {
    size_t packed=0,dense=0;auto result=geometry(f,rows,columns,packed,dense);if(result)return result;
    if(source_bytes!=packed || dest_bytes!=dense || !valid_buffers(source,source_bytes,dest,dest_bytes,status)
        || reinterpret_cast<uintptr_t>(dest)%4)return LIBGGUF_NATIVE_INVALID_BUFFER;
    auto s=reinterpret_cast<NATIVE_STREAM>(stream);const auto format=static_cast<NativeFormat>(f);
    const size_t blocks=rows*columns/libgguf::native_format_info(format).values_per_block;
    initialize_status<<<1,1,0,s>>>(status);
    if(NATIVE_PEEK()!=NATIVE_SUCCESS)return LIBGGUF_NATIVE_GPU_ERROR;
    validate_decode<<<unsigned((blocks+threads-1)/threads),threads,0,s>>>(format,static_cast<const unsigned char*>(source),blocks,status);
    if(NATIVE_PEEK()!=NATIVE_SUCCESS)return LIBGGUF_NATIVE_GPU_ERROR;
    decode<<<unsigned((rows*columns+threads-1)/threads),threads,0,s>>>(format,static_cast<const unsigned char*>(source),dest,rows*columns,status);
    return NATIVE_PEEK()==NATIVE_SUCCESS?LIBGGUF_NATIVE_OK:LIBGGUF_NATIVE_GPU_ERROR;
}
