#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>

#include <libgguf/gpu/dequantize/fragment.cuh>
#include <libgguf/libgguf.h>

namespace {

using Q4 = libgguf::gpu::BlockFragmentDecoder<GGML_TYPE_Q4_0>;
using Q8 = libgguf::gpu::BlockFragmentDecoder<GGML_TYPE_Q8_0>;

Q4::Block make_q4() {
  Q4::Block block{};
  block.d = 0x3800u;
  for (std::uint32_t index = 0; index < 16; ++index)
    block.qs[index] = static_cast<std::uint8_t>(
        index | ((15u - index) << 4));
  return block;
}

Q8::Block make_q8() {
  Q8::Block block{};
  block.d = 0x3400u;
  for (std::uint32_t index = 0; index < 32; ++index)
    block.qs[index] = static_cast<std::int8_t>(
        (static_cast<int>(index) * 3) % 17 - 8);
  return block;
}

template <typename Decoder>
bool compare(const typename Decoder::Block& block, ggml_type format) {
  std::array<float, Decoder::values_per_block> reference{};
  if (libgguf_dequantize_chunk(format, &block, reference.data(), 0, 1,
                               reference.size()) !=
      reference.size() * sizeof(float))
    return false;
  bool ok = true;
  for (int index = 0; index < Decoder::values_per_block; ++index)
    ok &= Decoder::decode_value(block, index) == reference[index];
  for (int first = 0; first < Decoder::values_per_block; first += 8) {
    Decoder::decode_8(block, first, [&](int slot, float value) {
      ok &= value == reference[first + slot];
    });
  }
  return ok;
}

}  // namespace

int main() {
  static_assert(libgguf::gpu::block_fragment_decoder_api_revision == 1);
  static_assert(sizeof(Q4::Block) == 18);
  static_assert(sizeof(Q8::Block) == 34);
  const bool ok = compare<Q4>(make_q4(), GGML_TYPE_Q4_0) &&
                  compare<Q8>(make_q8(), GGML_TYPE_Q8_0);
  if (!ok) std::cerr << "block fragment decoder parity failed\n";
  return ok ? 0 : 1;
}
