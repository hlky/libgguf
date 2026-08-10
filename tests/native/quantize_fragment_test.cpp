#include <cmath>
#include <cstdint>
#include <iostream>

#include <libgguf/gpu/integer/fragment.cuh>
#include <libgguf/gpu/quantize/fragment.cuh>

namespace {

struct HostOperations {
  float absolute_maximum;

  float reduce_absmax(float) const { return absolute_maximum; }
  float divide(float numerator, float denominator) const {
    return numerator / denominator;
  }
  float multiply(float lhs, float rhs) const { return lhs * rhs; }
  std::int8_t round_to_int8(float value) const {
    return static_cast<std::int8_t>(std::round(value));
  }
};

}  // namespace

int main() {
  using Quantizer =
      libgguf::gpu::BlockFragmentQuantizer<GGML_TYPE_Q8_0>;
  using Q4Integer =
      libgguf::gpu::BlockIntegerFragment<GGML_TYPE_Q4_0>;
  using Q8Integer =
      libgguf::gpu::BlockIntegerFragment<GGML_TYPE_Q8_0>;

  static_assert(libgguf::gpu::block_fragment_quantizer_api_revision == 1);
  static_assert(libgguf::gpu::block_integer_fragment_api_revision == 1);
  static_assert(Quantizer::values_per_block == 32);
  static_assert(Quantizer::encoded_bytes == 34);
  static_assert(Q4Integer::encoded_bytes == 18);
  static_assert(Q8Integer::encoded_bytes == 34);

  std::int8_t quant = 1;
  const float zero_scale = Quantizer::quantize_lane(
      0.0f, quant, HostOperations{0.0f});
  bool ok = zero_scale == 0.0f && quant == 0;

  const float scale = Quantizer::quantize_lane(
      2.0f, quant, HostOperations{4.0f});
  ok &= scale == 4.0f / 127.0f && quant == 64;

  block_q8_0 q8{};
  q8.d = 0x3800u;
  q8.qs[0] = -127;
  q8.qs[31] = 126;
  const auto * q8_raw = reinterpret_cast<const std::uint8_t *>(&q8);
  ok &= Q8Integer::decode_raw_scale(q8_raw) == 0.5f;
  ok &= Q8Integer::decode_raw_quant(q8_raw, 0) == -127;
  ok &= Q8Integer::decode_raw_quant(q8_raw, 31) == 126;

  block_q4_0 q4{};
  q4.d = 0x3400u;
  q4.qs[0] = 0xf0u;
  const auto * q4_raw = reinterpret_cast<const std::uint8_t *>(&q4);
  ok &= Q4Integer::decode_raw_scale(q4_raw) == 0.25f;
  ok &= Q4Integer::decode_raw_quant(q4_raw, 0) == -8;
  ok &= Q4Integer::decode_raw_quant(q4_raw, 16) == 7;

  if (!ok) std::cerr << "quantize/integer fragment contract failed\n";
  return ok ? 0 : 1;
}
