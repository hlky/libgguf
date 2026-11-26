#include <libgguf/gpu/compat.h>
#include <libgguf/native_gpu.h>
#include <torch/all.h>
#if defined(USE_ROCM)
#include <c10/hip/HIPGuard.h>
#include <c10/hip/HIPStream.h>
#else
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif
#include <limits>

namespace {
void check(const torch::Tensor& tensor,const torch::Device& device,torch::ScalarType dtype) {
    TORCH_CHECK(tensor.is_cuda() && tensor.device()==device,"native GPU tensors must share a CUDA device");
    TORCH_CHECK(tensor.scalar_type()==dtype && tensor.is_contiguous(),"native GPU tensor dtype/contiguity differs");
}
size_t row_bytes(int64_t format,int64_t columns) {
    TORCH_CHECK(format>=0 && format<=3,"unknown native quantization format");
    TORCH_CHECK(columns>0,"native GPU width must be positive");
    size_t bytes=0;
    TORCH_CHECK(libgguf_native_row_size(static_cast<libgguf_native_format>(format),static_cast<size_t>(columns),&bytes)==LIBGGUF_NATIVE_OK,
        "native GPU width must contain whole canonical blocks");
    TORCH_CHECK(bytes<=static_cast<size_t>(std::numeric_limits<int64_t>::max()),"native GPU row bytes exceed i64");
    return bytes;
}
void status_tensor(const torch::Tensor& status,const torch::Device& device) {
    check(status,device,torch::kInt32);
    TORCH_CHECK(status.dim()==1 && status.numel()==1,"native GPU status must be a one-element I32 tensor");
}
}
void native_quantize_into(torch::Tensor source,int64_t format,torch::Tensor output,torch::Tensor status,
    std::optional<torch::Tensor> const& importance) {
    TORCH_CHECK(source.is_cuda() && source.dim()==2 && source.size(0)>0,"native GPU encoder requires nonempty rank-two input");
    check(source,source.device(),torch::kFloat32);check(output,source.device(),torch::kUInt8);status_tensor(status,source.device());
    const auto bytes=row_bytes(format,source.size(1));
    TORCH_CHECK(output.dim()==2 && output.size(0)==source.size(0) && output.size(1)==static_cast<int64_t>(bytes),"native GPU output shape differs");
    const at::cuda::OptionalCUDAGuard guard(device_of(source));
    const auto stream=c10::cuda::getCurrentCUDAStream();
    const float* weights=nullptr;size_t weights_bytes=0;
    if(importance) {
        check(*importance,source.device(),torch::kFloat32);
        TORCH_CHECK(importance->dim()==1 && importance->numel()==source.size(1),"native GPU importance must be a K vector");
        weights=importance->data_ptr<float>();weights_bytes=importance->nbytes();importance->record_stream(stream.unwrap());
    }
    source.record_stream(stream.unwrap());output.record_stream(stream.unwrap());status.record_stream(stream.unwrap());
    const auto result=libgguf_native_encode_gpu_into(static_cast<libgguf_native_format>(format),source.data_ptr<float>(),source.nbytes(),
        source.size(0),source.size(1),weights,weights_bytes,output.data_ptr(),output.nbytes(),
        reinterpret_cast<uint32_t*>(status.data_ptr<int32_t>()),stream.stream());
    TORCH_CHECK(result==LIBGGUF_NATIVE_OK,"native GPU encoder metadata/enqueue error: ",int(result));
}
void native_dequantize_into(torch::Tensor encoded,int64_t format,int64_t columns,torch::Tensor output,torch::Tensor status) {
    TORCH_CHECK(encoded.is_cuda() && encoded.dim()==2 && encoded.size(0)>0,"native GPU decoder requires nonempty rank-two input");
    check(encoded,encoded.device(),torch::kUInt8);check(output,encoded.device(),torch::kFloat32);status_tensor(status,encoded.device());
    const auto bytes=row_bytes(format,columns);
    TORCH_CHECK(encoded.size(1)==static_cast<int64_t>(bytes) && output.dim()==2 && output.size(0)==encoded.size(0) && output.size(1)==columns,"native GPU decode shape differs");
    const at::cuda::OptionalCUDAGuard guard(device_of(encoded));
    const auto stream=c10::cuda::getCurrentCUDAStream();
    encoded.record_stream(stream.unwrap());output.record_stream(stream.unwrap());status.record_stream(stream.unwrap());
    const auto result=libgguf_native_decode_gpu_into(static_cast<libgguf_native_format>(format),encoded.data_ptr(),encoded.nbytes(),
        encoded.size(0),columns,output.data_ptr<float>(),output.nbytes(),reinterpret_cast<uint32_t*>(status.data_ptr<int32_t>()),stream.stream());
    TORCH_CHECK(result==LIBGGUF_NATIVE_OK,"native GPU decoder metadata/enqueue error: ",int(result));
}
