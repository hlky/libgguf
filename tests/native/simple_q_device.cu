#include <array>
#include <cstdint>
#include <iostream>

#include <libgguf/device/simple_q.h>

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#define LIBGGUF_GPU_ERROR_T hipError_t
#define LIBGGUF_GPU_SUCCESS hipSuccess
#define LIBGGUF_GPU_MALLOC hipMalloc
#define LIBGGUF_GPU_FREE hipFree
#define LIBGGUF_GPU_MEMCPY hipMemcpy
#define LIBGGUF_GPU_D2H hipMemcpyDeviceToHost
#define LIBGGUF_GPU_SYNC hipDeviceSynchronize
#define LIBGGUF_GPU_LAST_ERROR hipGetLastError
#define LIBGGUF_GPU_ERROR_STRING hipGetErrorString
#define LIBGGUF_GPU_LAUNCH(kernel, blocks, threads, ...) \
  hipLaunchKernelGGL(kernel, dim3(blocks), dim3(threads), 0, 0, __VA_ARGS__)
#else
#include <cuda_runtime.h>
#define LIBGGUF_GPU_ERROR_T cudaError_t
#define LIBGGUF_GPU_SUCCESS cudaSuccess
#define LIBGGUF_GPU_MALLOC cudaMalloc
#define LIBGGUF_GPU_FREE cudaFree
#define LIBGGUF_GPU_MEMCPY cudaMemcpy
#define LIBGGUF_GPU_D2H cudaMemcpyDeviceToHost
#define LIBGGUF_GPU_SYNC cudaDeviceSynchronize
#define LIBGGUF_GPU_LAST_ERROR cudaGetLastError
#define LIBGGUF_GPU_ERROR_STRING cudaGetErrorString
#define LIBGGUF_GPU_LAUNCH(kernel, blocks, threads, ...) \
  kernel<<<blocks, threads>>>(__VA_ARGS__)
#endif

namespace {

using libgguf::device::q4_0_block;
using libgguf::device::q8_0_block;

__global__ void decode_simple_q(q4_0_block q4, q8_0_block q8, float* output) {
  const std::uint32_t index = threadIdx.x;
  if (index < 32) {
    output[index] = libgguf::device::decode_q4_0(q4, index);
    output[32 + index] = libgguf::device::decode_q8_0(q8, index);
  }
  if (index == 0) {
    libgguf::device::decode_q4_0_8(q4, 12, output + 64);
    libgguf::device::decode_q8_0_8(q8, 12, output + 72);
  }
}

bool check(LIBGGUF_GPU_ERROR_T status, const char* operation) {
  if (status == LIBGGUF_GPU_SUCCESS) return true;
  std::cerr << operation << ": " << LIBGGUF_GPU_ERROR_STRING(status) << '\n';
  return false;
}

}  // namespace

int main() {
  q4_0_block q4{};
  q4.scale_bits = 0x3800u;
  for (std::uint32_t index = 0; index < 16; ++index) {
    q4.quants[index] = static_cast<std::uint8_t>(
        index | ((15u - index) << 4));
  }
  q8_0_block q8{};
  q8.scale_bits = 0x3400u;
  constexpr std::array<std::int8_t, 32> q8_quants{
      -128, -64, -32, -16, -8, -4, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7,
         8,   9,  10,  11, 12, 13, 14, 15, 16, 24, 32, 48, 63, 64, 100, 127,
  };
  for (std::uint32_t index = 0; index < q8_quants.size(); ++index) {
    q8.quants[index] = q8_quants[index];
  }

  float* device_output = nullptr;
  if (!check(LIBGGUF_GPU_MALLOC(reinterpret_cast<void**>(&device_output),
                                80 * sizeof(float)),
             "allocate device output")) {
    return 1;
  }
  LIBGGUF_GPU_LAUNCH(decode_simple_q, 1, 32, q4, q8, device_output);
  bool ok = check(LIBGGUF_GPU_LAST_ERROR(), "launch decode kernel") &&
            check(LIBGGUF_GPU_SYNC(), "synchronize decode kernel");
  std::array<float, 80> output{};
  ok &= check(LIBGGUF_GPU_MEMCPY(output.data(), device_output,
                                output.size() * sizeof(float), LIBGGUF_GPU_D2H),
              "copy decode output");
  ok &= check(LIBGGUF_GPU_FREE(device_output), "free device output");
  for (std::uint32_t index = 0; index < 32; ++index) {
    const int q4_quant = index < 16 ? static_cast<int>(index) - 8
                                    : 15 - static_cast<int>(index - 16) - 8;
    const float expected_q4 = 0.5f * static_cast<float>(q4_quant);
    const float expected_q8 = 0.25f * static_cast<float>(q8_quants[index]);
    if (output[index] != expected_q4 || output[32 + index] != expected_q8) {
      std::cerr << "device decode mismatch at index " << index << '\n';
      ok = false;
    }
  }
  for (std::uint32_t index = 0; index < 8; ++index) {
    if (output[64 + index] != output[12 + index] ||
        output[72 + index] != output[32 + 12 + index]) {
      std::cerr << "device fragment decode mismatch at index " << index
                << '\n';
      ok = false;
    }
  }
  return ok ? 0 : 1;
}

#undef LIBGGUF_GPU_ERROR_T
#undef LIBGGUF_GPU_SUCCESS
#undef LIBGGUF_GPU_MALLOC
#undef LIBGGUF_GPU_FREE
#undef LIBGGUF_GPU_MEMCPY
#undef LIBGGUF_GPU_D2H
#undef LIBGGUF_GPU_SYNC
#undef LIBGGUF_GPU_LAST_ERROR
#undef LIBGGUF_GPU_ERROR_STRING
#undef LIBGGUF_GPU_LAUNCH
