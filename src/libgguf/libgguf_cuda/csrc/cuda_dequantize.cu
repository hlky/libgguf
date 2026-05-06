#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>

#include "cuda_dequantize_kernels.h"

torch::Tensor dequantize(torch::Tensor W, int64_t type, int64_t m, int64_t n,
                         std::optional<at::ScalarType> const &dtype) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(W));
  auto dtype_ = dtype.value_or(torch::kFloat16);
  auto options = torch::TensorOptions().dtype(dtype_).device(W.device());
  at::Tensor DW = torch::empty({m, n}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  gguf_cuda_dequantize_row((void *)W.data_ptr(), (void *)DW.data_ptr(), type, m * n, DW.scalar_type(), stream);

  return DW;
}
