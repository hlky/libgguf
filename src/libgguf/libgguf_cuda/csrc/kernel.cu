#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>

#include "cuda_compat.h"
#include "dispatch_utils.h"

#include "common.h"
#include "dequantize.h"

torch::Tensor ggml_dequantize(torch::Tensor W,  // quant weight
                              int64_t type, int64_t m, int64_t n,
                              std::optional<at::ScalarType> const& dtype) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(W));
  auto dtype_ = dtype.value_or(torch::kFloat16);
  auto options = torch::TensorOptions().dtype(dtype_).device(W.device());
  at::Tensor DW = torch::empty({m, n}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  VLLM_DISPATCH_FLOATING_TYPES(DW.scalar_type(), "ggml_dequantize", [&] {
    auto to_cuda = ggml_get_to_cuda<scalar_t>(type);
    TORCH_CHECK(to_cuda != nullptr, "Unsupported GGML quantization type for CUDA dequantize: ", type);
    to_cuda((void*)W.data_ptr(), (scalar_t*)DW.data_ptr(), m * n, stream);
  });

  return DW;
}
