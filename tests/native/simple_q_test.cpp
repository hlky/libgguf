#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <type_traits>

#include <libgguf/device/simple_q.h>
#include <libgguf/formats.h>
#include <libgguf/libgguf.h>

namespace {

using libgguf::device::q4_0_block;
using libgguf::device::q8_0_block;

constexpr std::array<float, 32> q4_expected{
    -4.0f, -3.5f, -3.0f, -2.5f, -2.0f, -1.5f, -1.0f, -0.5f,
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  2.5f,  3.0f,  3.5f,
     3.5f,  3.0f,  2.5f,  2.0f,  1.5f,  1.0f,  0.5f,  0.0f,
    -0.5f, -1.0f, -1.5f, -2.0f, -2.5f, -3.0f, -3.5f, -4.0f,
};

constexpr std::array<std::int8_t, 32> q8_quants{
    -128, -64, -32, -16, -8, -4, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7,
       8,   9,  10,  11, 12, 13, 14, 15, 16, 24, 32, 48, 63, 64, 100, 127,
};

constexpr std::array<float, 32> q8_expected{
    -32.0f, -16.0f, -8.0f, -4.0f, -2.0f, -1.0f, -0.5f, -0.25f,
      0.0f,   0.25f,  0.5f,  0.75f, 1.0f,  1.25f, 1.5f,  1.75f,
      2.0f,   2.25f,  2.5f,  2.75f, 3.0f,  3.25f, 3.5f,  3.75f,
      4.0f,   6.0f,   8.0f, 12.0f, 15.75f, 16.0f, 25.0f, 31.75f,
};

q4_0_block make_q4() {
  q4_0_block block{};
  block.scale_bits = 0x3800u;  // 0.5
  for (std::uint32_t index = 0; index < 16; ++index) {
    block.quants[index] = static_cast<std::uint8_t>(
        index | ((15u - index) << 4));
  }
  return block;
}

q8_0_block make_q8() {
  q8_0_block block{};
  block.scale_bits = 0x3400u;  // 0.25
  for (std::uint32_t index = 0; index < q8_quants.size(); ++index) {
    block.quants[index] = q8_quants[index];
  }
  return block;
}

std::uint32_t float_bits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

bool test_fp16_edges() {
  using libgguf::device::detail::fp16_bits_to_float;
  bool ok = float_bits(fp16_bits_to_float(0x0000u)) == 0x00000000u;
  ok &= float_bits(fp16_bits_to_float(0x8000u)) == 0x80000000u;
  ok &= float_bits(fp16_bits_to_float(0x0001u)) == 0x33800000u;
  ok &= float_bits(fp16_bits_to_float(0x03ffu)) == 0x387fc000u;
  ok &= float_bits(fp16_bits_to_float(0x0400u)) == 0x38800000u;
  ok &= float_bits(fp16_bits_to_float(0x7bffu)) == 0x477fe000u;
  ok &= std::isinf(fp16_bits_to_float(0x7c00u));
  ok &= std::isinf(fp16_bits_to_float(0xfc00u));
  ok &= std::isnan(fp16_bits_to_float(0x7e01u));
  return ok;
}

template <class Block, class Expected>
bool compare_reference(ggml_type type, const Block& block,
                       float (*decode)(const Block&, std::uint32_t),
                       const Expected& expected, const char* label) {
  std::array<float, libgguf::device::simple_q_block_values> reference{};
  block_q4_0 q4_reference{};
  block_q8_0 q8_reference{};
  const void* reference_block = nullptr;
  if constexpr (std::is_same_v<Block, q4_0_block>) {
    static_assert(sizeof(q4_reference) == sizeof(block));
    std::memcpy(&q4_reference, &block, sizeof(block));
    reference_block = &q4_reference;
  } else {
    static_assert(sizeof(q8_reference) == sizeof(block));
    std::memcpy(&q8_reference, &block, sizeof(block));
    reference_block = &q8_reference;
  }
  const std::size_t bytes = libgguf_dequantize_chunk(
      type, reference_block, reference.data(), 0, 1, reference.size());
  if (bytes != reference.size() * sizeof(float)) {
    std::cerr << label << ": CPU reference returned " << bytes
              << " bytes, expected " << reference.size() * sizeof(float)
              << '\n';
    return false;
  }
  for (std::uint32_t index = 0; index < reference.size(); ++index) {
    const float actual = decode(block, index);
    if (std::memcmp(&actual, &reference[index], sizeof(float)) != 0 ||
        std::memcmp(&actual, &expected[index], sizeof(float)) != 0) {
      std::cerr << label << ": mismatch at index " << index << ": "
                << actual << " != " << reference[index] << " != "
                << expected[index] << '\n';
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  static_assert(sizeof(q4_0_block) == 18);
  static_assert(sizeof(q8_0_block) == 34);
  static_assert(libgguf::device::simple_q_block_values == 32);
  static_assert(libgguf::device::simple_q_api_revision == 1);

  const q4_0_block q4 = make_q4();
  const q8_0_block q8 = make_q8();
  bool ok = test_fp16_edges();
  ok &= compare_reference(GGML_TYPE_Q4_0, q4,
                              libgguf::device::decode_q4_0, q4_expected,
                              "Q4_0");
  ok &= compare_reference(GGML_TYPE_Q8_0, q8,
                          libgguf::device::decode_q8_0, q8_expected, "Q8_0");

  std::array<float, 8> fragment{};
  for (std::uint32_t base = 0; base <= 24; ++base) {
    libgguf::device::decode_q4_0_8(q4, base, fragment.data());
    for (std::uint32_t index = 0; index < fragment.size(); ++index) {
      ok &= fragment[index] == q4_expected[base + index];
    }
    libgguf::device::decode_q8_0_8(q8, base, fragment.data());
    for (std::uint32_t index = 0; index < fragment.size(); ++index) {
      ok &= fragment[index] == q8_expected[base + index];
    }
  }

  std::array<float, 32> source{};
  for (std::uint32_t index = 0; index < source.size(); ++index) {
    source[index] = (static_cast<int>(index) - 15) * 0.375f;
  }
  block_q4_0 cpu_quantized_q4{};
  block_q8_0 cpu_quantized_q8{};
  ok &= libgguf_quantize_chunk(GGML_TYPE_Q4_0, source.data(),
                               &cpu_quantized_q4, 0, 1, source.size(), nullptr) ==
        sizeof(cpu_quantized_q4);
  ok &= libgguf_quantize_chunk(GGML_TYPE_Q8_0, source.data(),
                               &cpu_quantized_q8, 0, 1, source.size(), nullptr) ==
        sizeof(cpu_quantized_q8);
  q4_0_block simple_quantized_q4{};
  q8_0_block simple_quantized_q8{};
  std::memcpy(&simple_quantized_q4, &cpu_quantized_q4,
              sizeof(simple_quantized_q4));
  std::memcpy(&simple_quantized_q8, &cpu_quantized_q8,
              sizeof(simple_quantized_q8));
  std::array<float, 32> cpu_dequantized{};
  libgguf_dequantize_chunk(GGML_TYPE_Q4_0, &cpu_quantized_q4,
                           cpu_dequantized.data(), 0, 1, source.size());
  for (std::uint32_t index = 0; index < source.size(); ++index) {
    ok &= libgguf::device::decode_q4_0(simple_quantized_q4, index) ==
          cpu_dequantized[index];
  }
  libgguf_dequantize_chunk(GGML_TYPE_Q8_0, &cpu_quantized_q8,
                           cpu_dequantized.data(), 0, 1, source.size());
  for (std::uint32_t index = 0; index < source.size(); ++index) {
    ok &= libgguf::device::decode_q8_0(simple_quantized_q8, index) ==
          cpu_dequantized[index];
  }
  if (!ok) std::cerr << "simple Q host fragment parity failed\n";
  return ok ? 0 : 1;
}
