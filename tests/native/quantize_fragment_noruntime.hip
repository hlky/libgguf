#include <libgguf/gpu/integer/fragment.cuh>
#include <libgguf/gpu/quantize/fragment.cuh>

#if defined(HIP_INCLUDE_HIP_HIP_RUNTIME_H) || \
    defined(HIP_INCLUDE_HIP_HIP_RUNTIME_API_H)
#error "quantize/integer fragment headers must not include the HIP runtime"
#endif

struct ProbeOperations {
  __attribute__((device)) float reduce_absmax(float value) const {
    return value < 0.0f ? -value : value;
  }
  __attribute__((device)) float divide(float lhs, float rhs) const {
    return lhs / rhs;
  }
  __attribute__((device)) float multiply(float lhs, float rhs) const {
    return lhs * rhs;
  }
  __attribute__((device)) std::int8_t round_to_int8(float value) const {
    return static_cast<std::int8_t>(value);
  }
};

__attribute__((global)) void quantize_fragment_noruntime_probe(
    float value, const std::uint8_t * q4, std::int8_t * output,
    float * scale) {
  libgguf::gpu::BlockFragmentQuantizer<GGML_TYPE_Q8_0>::quantize_lane(
      value, output[0], ProbeOperations{});
  using Q4Integer =
      libgguf::gpu::BlockIntegerFragment<GGML_TYPE_Q4_0>;
  output[1] = Q4Integer::decode_raw_quant(q4, 0);
  scale[0] = Q4Integer::decode_raw_scale(q4);
}
