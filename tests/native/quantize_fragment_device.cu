#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>

#include <libgguf/gpu/integer/fragment.cuh>
#include <libgguf/gpu/quantize/q8_0.cuh>

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#define LIBGGUF_GPU_ERROR_T hipError_t
#define LIBGGUF_GPU_SUCCESS hipSuccess
#define LIBGGUF_GPU_MALLOC hipMalloc
#define LIBGGUF_GPU_FREE hipFree
#define LIBGGUF_GPU_MEMCPY hipMemcpy
#define LIBGGUF_GPU_D2H hipMemcpyDeviceToHost
#define LIBGGUF_GPU_SYNC hipDeviceSynchronize
#define LIBGGUF_GPU_LAUNCH(kernel, ...) \
  hipLaunchKernelGGL(kernel, dim3(1), dim3(32), 0, 0, __VA_ARGS__)
#else
#include <cuda_runtime.h>
#define LIBGGUF_GPU_ERROR_T cudaError_t
#define LIBGGUF_GPU_SUCCESS cudaSuccess
#define LIBGGUF_GPU_MALLOC cudaMalloc
#define LIBGGUF_GPU_FREE cudaFree
#define LIBGGUF_GPU_MEMCPY cudaMemcpy
#define LIBGGUF_GPU_D2H cudaMemcpyDeviceToHost
#define LIBGGUF_GPU_SYNC cudaDeviceSynchronize
#define LIBGGUF_GPU_LAUNCH(kernel, ...) kernel<<<1, 32>>>(__VA_ARGS__)
#endif

struct DeviceResult {
  float scale;
  float legacy_scale;
  std::int8_t quant;
  std::int8_t legacy_quant;
  std::int8_t q4_quant;
  float q4_scale;
};

template <typename Src>
__device__ __forceinline__ float legacy_quantize_q8_0(
    Src value, std::int8_t & quant) {
  const float input = static_cast<float>(value);
  const auto mask = __activemask();
  const float absolute_maximum = __shfl_sync(
      mask, warp_reduce_max(fabsf(input)), 0);
  const float scale = __fdiv_rn(absolute_maximum, 127.0f);
  const float inverse_scale = scale != 0.0f
      ? __fdiv_rn(1.0f, scale)
      : 0.0f;
  quant = static_cast<std::int8_t>(
      roundf(__fmul_rn(input, inverse_scale)));
  return scale;
}

__global__ void quantize_fragments(block_q4_0 q4, DeviceResult * output) {
  const int lane = static_cast<int>(threadIdx.x);
  std::int8_t quant = 0;
  std::int8_t legacy_quant = 0;
  const float value = static_cast<float>(lane - 16);
  output[lane].scale = libgguf::gpu::quantize_q8_0(value, quant);
  output[lane].legacy_scale = legacy_quantize_q8_0(value, legacy_quant);
  output[lane].quant = quant;
  output[lane].legacy_quant = legacy_quant;

  using Q4Integer =
      libgguf::gpu::BlockIntegerFragment<GGML_TYPE_Q4_0>;
  const auto * encoded = reinterpret_cast<const std::uint8_t *>(&q4);
  output[lane].q4_quant = Q4Integer::decode_raw_quant(encoded, lane);
  output[lane].q4_scale = Q4Integer::decode_raw_scale(encoded);
}

int main() {
  block_q4_0 q4{};
  q4.d = 0x3800u;
  for (int index = 0; index < 16; ++index)
    q4.qs[index] = static_cast<std::uint8_t>(
        index | ((15 - index) << 4));

  DeviceResult * device_output = nullptr;
  bool ok = LIBGGUF_GPU_MALLOC(
      reinterpret_cast<void **>(&device_output),
      32 * sizeof(DeviceResult)) == LIBGGUF_GPU_SUCCESS;
  if (!ok) return 1;
  LIBGGUF_GPU_LAUNCH(quantize_fragments, q4, device_output);
  ok &= LIBGGUF_GPU_SYNC() == LIBGGUF_GPU_SUCCESS;
  std::array<DeviceResult, 32> output{};
  ok &= LIBGGUF_GPU_MEMCPY(output.data(), device_output,
                           output.size() * sizeof(DeviceResult),
                           LIBGGUF_GPU_D2H) == LIBGGUF_GPU_SUCCESS;
  ok &= LIBGGUF_GPU_FREE(device_output) == LIBGGUF_GPU_SUCCESS;

  const float expected_scale = 16.0f / 127.0f;
  for (int lane = 0; lane < 32; ++lane) {
    const auto expected_quant = static_cast<std::int8_t>(
        std::round(static_cast<float>(lane - 16) / expected_scale));
    const int packed_index = lane & 15;
    const auto expected_q4 = static_cast<std::int8_t>(
        lane < 16 ? packed_index - 8 : 7 - packed_index);
    ok &= output[lane].scale == expected_scale;
    ok &= output[lane].scale == output[lane].legacy_scale;
    ok &= output[lane].quant == expected_quant;
    ok &= output[lane].quant == output[lane].legacy_quant;
    ok &= output[lane].q4_quant == expected_q4;
    ok &= output[lane].q4_scale == 0.5f;
  }
  if (!ok) std::cerr << "device quantize/integer fragment parity failed\n";
  return ok ? 0 : 1;
}
