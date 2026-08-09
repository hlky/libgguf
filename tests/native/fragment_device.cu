#include <array>
#include <cstdint>
#include <iostream>

#include <libgguf/gpu/dequantize/fragment.cuh>

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

using Q4 = libgguf::gpu::BlockFragmentDecoder<GGML_TYPE_Q4_0>;
using Q8 = libgguf::gpu::BlockFragmentDecoder<GGML_TYPE_Q8_0>;

__global__ void decode_fragments(Q4::Block q4, Q8::Block q8, float* output) {
  const int index = static_cast<int>(threadIdx.x);
  output[index] = Q4::decode_value(q4, index);
  output[32 + index] = Q8::decode_value(q8, index);
  if (index == 0) {
    Q4::decode_8(q4, 8,
                 [=] __device__(int slot, float value) {
                   output[64 + slot] = value;
                 });
    Q8::decode_8(q8, 8,
                 [=] __device__(int slot, float value) {
                   output[72 + slot] = value;
                 });
  }
}

int main() {
  Q4::Block q4{};
  Q8::Block q8{};
  q4.d = 0x3800u;
  q8.d = 0x3400u;
  for (int i = 0; i < 16; ++i)
    q4.qs[i] = static_cast<std::uint8_t>(i | ((15 - i) << 4));
  for (int i = 0; i < 32; ++i)
    q8.qs[i] = static_cast<std::int8_t>((i * 3) % 17 - 8);
  float* device_output = nullptr;
  bool ok = LIBGGUF_GPU_MALLOC(reinterpret_cast<void**>(&device_output),
                               80 * sizeof(float)) == LIBGGUF_GPU_SUCCESS;
  if (!ok) return 1;
  LIBGGUF_GPU_LAUNCH(decode_fragments, q4, q8, device_output);
  ok &= LIBGGUF_GPU_SYNC() == LIBGGUF_GPU_SUCCESS;
  std::array<float, 80> output{};
  ok &= LIBGGUF_GPU_MEMCPY(output.data(), device_output,
                           output.size() * sizeof(float), LIBGGUF_GPU_D2H) ==
        LIBGGUF_GPU_SUCCESS;
  ok &= LIBGGUF_GPU_FREE(device_output) == LIBGGUF_GPU_SUCCESS;
  for (int i = 0; i < 8; ++i) {
    ok &= output[64 + i] == output[8 + i];
    ok &= output[72 + i] == output[40 + i];
  }
  if (!ok) std::cerr << "device block fragment parity failed\n";
  return ok ? 0 : 1;
}
