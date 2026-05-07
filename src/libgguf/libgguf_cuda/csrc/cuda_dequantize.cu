#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <limits>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>

#include "cuda_dequantize_kernels.h"
#include "cuda_quantize_kernels.h"
#include "libgguf.h"

namespace {

bool gguf_cuda_dequantize_dtype_supported(at::ScalarType dtype) {
  return dtype == torch::kFloat16 || dtype == torch::kBFloat16 || dtype == torch::kFloat32;
}

int64_t gguf_cuda_dequantize_block_size_for_type(int64_t type) {
  if (type == GGML_TYPE_BF16) {
    return 1;
  }
  return gguf_cuda_quantize_block_size_for_type(type);
}

int64_t gguf_cuda_dequantize_row_size_for_type(int64_t type, int64_t n) {
  if (type == GGML_TYPE_BF16) {
    TORCH_CHECK(n <= std::numeric_limits<int64_t>::max() / (int64_t)sizeof(uint16_t),
                "CUDA dequantize BF16 row size is too large");
    return n * (int64_t)sizeof(uint16_t);
  }
  return gguf_cuda_quantize_row_size_for_type(type, n);
}

}  // namespace

torch::Tensor dequantize(torch::Tensor W, int64_t type, int64_t m, int64_t n,
                         std::optional<at::ScalarType> const &dtype) {
  TORCH_CHECK(W.is_cuda(), "CUDA dequantize expects a CUDA tensor");
  TORCH_CHECK(W.scalar_type() == torch::kUInt8, "CUDA dequantize expects uint8 input");
  TORCH_CHECK(W.is_contiguous(), "CUDA dequantize expects contiguous input");
  TORCH_CHECK(m > 0, "CUDA dequantize expects positive row count");
  TORCH_CHECK(n > 0, "CUDA dequantize expects positive row width");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(W));
  auto dtype_ = dtype.value_or(torch::kFloat16);
  TORCH_CHECK(gguf_cuda_dequantize_dtype_supported(dtype_),
              "CUDA dequantize output dtype must be float16, bfloat16, or float32");
  const int64_t qk = gguf_cuda_dequantize_block_size_for_type(type);
  TORCH_CHECK(qk != 0, "Unsupported GGML quantization type for CUDA dequantize: ", type);
  TORCH_CHECK(n % qk == 0, "CUDA dequantize output width must be divisible by the quantization block size");
  TORCH_CHECK(m <= std::numeric_limits<int>::max() / n,
              "CUDA dequantize output element count exceeds int32 range");
  const int64_t row_size = gguf_cuda_dequantize_row_size_for_type(type, n);
  TORCH_CHECK(row_size != 0, "Unsupported GGML quantization type for CUDA dequantize: ", type);
  TORCH_CHECK(m <= std::numeric_limits<int64_t>::max() / row_size,
              "CUDA dequantize encoded input size is too large");
  const int64_t expected_numel = m * row_size;
  TORCH_CHECK(W.numel() == expected_numel,
              "CUDA dequantize input has ", W.numel(), " bytes, expected ", expected_numel);

  auto options = torch::TensorOptions().dtype(dtype_).device(W.device());
  at::Tensor DW = torch::empty({m, n}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  gguf_cuda_dequantize_row((void *)W.data_ptr(), (void *)DW.data_ptr(), type, (int)(m * n), DW.scalar_type(), stream);

  return DW;
}
