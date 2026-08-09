#include <libgguf/gpu/dequantize/fragment.cuh>

#if defined(HIP_INCLUDE_HIP_HIP_RUNTIME_H) || \
    defined(HIP_INCLUDE_HIP_HIP_RUNTIME_API_H)
#error "fragment.cuh must not include the HIP runtime"
#endif

__attribute__((global)) void fragment_noruntime_probe(
    const block_q4_0* q4, const block_q8_0* q8, float* output) {
  using Q4 = libgguf::gpu::BlockFragmentDecoder<GGML_TYPE_Q4_0>;
  using Q8 = libgguf::gpu::BlockFragmentDecoder<GGML_TYPE_Q8_0>;
  Q4::decode_8(*q4, 8, [=] __attribute__((device))(int slot, float value) {
    output[slot] = value;
  });
  Q8::decode_8(*q8, 8, [=] __attribute__((device))(int slot, float value) {
    output[8 + slot] = value;
  });
}
