#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <vector>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>

#include "cuda_compat.h"
#include "dispatch_utils.h"

#include "libgguf_cuda_common.h"
#include "dequantize.h"
#include "quantize.h"

torch::Tensor dequantize(torch::Tensor W,  // quant weight
                              int64_t type, int64_t m, int64_t n,
                              std::optional<at::ScalarType> const& dtype) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(W));
  auto dtype_ = dtype.value_or(torch::kFloat16);
  auto options = torch::TensorOptions().dtype(dtype_).device(W.device());
  at::Tensor DW = torch::empty({m, n}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  VLLM_DISPATCH_FLOATING_TYPES(DW.scalar_type(), "dequantize", [&] {
    auto to_cuda = ggml_get_to_cuda<scalar_t>(type);
    TORCH_CHECK(to_cuda != nullptr, "Unsupported GGML quantization type for CUDA dequantize: ", type);
    to_cuda((void*)W.data_ptr(), (scalar_t*)DW.data_ptr(), m * n, stream);
  });

  return DW;
}

torch::Tensor quantize(torch::Tensor W, int64_t type, std::optional<torch::Tensor> const &imatrix) {
  TORCH_CHECK(W.is_cuda(), "CUDA quantize expects a CUDA tensor");
  TORCH_CHECK(W.scalar_type() == torch::kFloat32, "CUDA quantize expects float32 input");
  TORCH_CHECK(W.dim() >= 1, "CUDA quantize expects at least one dimension");
  TORCH_CHECK(W.is_contiguous(), "CUDA quantize expects contiguous input");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(W));
  const int64_t n = W.size(-1);
  const int64_t qk = gguf_cuda_quantize_block_size(type);
  const int64_t row_size = gguf_cuda_quantize_row_size(type, n);
  TORCH_CHECK(qk != 0 && row_size != 0, "Unsupported GGML quantization type for CUDA quantize: ", type);
  TORCH_CHECK(n % qk == 0, "CUDA quantize input width must be divisible by the quantization block size");

  int64_t m = 1;
  std::vector<int64_t> out_shape;
  out_shape.reserve(W.dim());
  for (int64_t dim = 0; dim < W.dim() - 1; ++dim) {
    const int64_t size = W.size(dim);
    m *= size;
    out_shape.push_back(size);
  }
  out_shape.push_back(row_size);

  auto options = torch::TensorOptions().dtype(torch::kUInt8).device(W.device());
  at::Tensor QW = torch::empty(out_shape, options);
  const float *quant_weights_ptr = nullptr;
  if (gguf_cuda_quantize_needs_imatrix(type)) {
    TORCH_CHECK(imatrix.has_value(), "CUDA quantize requires imatrix for GGML quantization type: ", type);
    const torch::Tensor &quant_weights = imatrix.value();
    TORCH_CHECK(quant_weights.is_cuda(), "CUDA quantize imatrix must be a CUDA tensor");
    TORCH_CHECK(quant_weights.device() == W.device(), "CUDA quantize imatrix must be on the same device as input");
    TORCH_CHECK(quant_weights.scalar_type() == torch::kFloat32, "CUDA quantize imatrix must be float32");
    TORCH_CHECK(quant_weights.dim() == 1, "CUDA quantize imatrix must be one-dimensional");
    TORCH_CHECK(quant_weights.numel() >= n, "CUDA quantize imatrix must have at least input width elements");
    TORCH_CHECK(quant_weights.is_contiguous(), "CUDA quantize imatrix must be contiguous");
    quant_weights_ptr = (const float *)quant_weights.data_ptr();
  }
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  quantize_row_cuda((const float *)W.data_ptr(), quant_weights_ptr, (void *)QW.data_ptr(), type, m * n, n, stream);
  return QW;
}
