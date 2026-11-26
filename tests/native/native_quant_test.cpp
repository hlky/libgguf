#include <libgguf/native_quant.h>
#include <libgguf/gpu/native_encode.cuh>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <cmath>

static void check(bool value, const char* message) {
    if (!value) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}

int main() {
    using namespace libgguf;
    // Exhaustive half round-trip, independent expected representation.
    for (unsigned bits = 1; bits < 0x7c00; ++bits) {
        const float value = gpu::fragment_detail::fp16_to_float(bits);
        check(native_detail::scale_half(static_cast<double>(value)) == bits, "half round-trip");
    }
    check(native_detail::rounded_code(-2.5, -8, 7) == -2, "negative even tie");
    check(native_detail::rounded_code(-3.5, -8, 7) == -4, "negative odd tie");
    check(native_detail::rounded_code(2.5, -8, 7) == 2, "positive even tie");
    check(native_detail::rounded_code(3.5, -8, 7) == 4, "positive odd tie");
    for (unsigned kind = 0; kind < 4; ++kind) {
        const auto format = static_cast<libgguf_native_format>(kind);
        const auto info = native_format_info(static_cast<NativeFormat>(kind));
        std::array<float, 512> input{}, decoded{};
        std::array<std::uint8_t, 1024> encoded{};
        for (unsigned k = 0; k < input.size(); ++k) input[k] = static_cast<float>(static_cast<int>(k % 16) - 8);
        size_t bytes = 0;
        check(libgguf_native_row_size(format, 256, &bytes) == LIBGGUF_NATIVE_OK, "row query");
        check(libgguf_native_encode_rows(format, input.data(), sizeof(input), 2, 256,
            nullptr, 0, encoded.data(), bytes * 2) == LIBGGUF_NATIVE_OK, "encode");
        check(libgguf_native_decode_rows(format, encoded.data(), bytes * 2, 2, 256,
            decoded.data(), sizeof(decoded)) == LIBGGUF_NATIVE_OK, "decode");
        check(libgguf_native_validate_rows(format, encoded.data(), bytes * 2 - 1, 2, 256)
            == LIBGGUF_NATIVE_INVALID_BUFFER, "truncated source");
        check(libgguf_native_row_size(format, info.values_per_block - 1, &bytes)
            == LIBGGUF_NATIVE_INVALID_SHAPE, "block divisibility");
        // Stored representations exercise every finite positive FP16 scale,
        // including subnormals, independently of the encoder's fitted choices.
        for (unsigned bits=0;bits<0x7c00;++bits) {
            encoded.fill(0);
            encoded[0]=static_cast<std::uint8_t>(bits);encoded[1]=static_cast<std::uint8_t>(bits>>8);
            if (kind==3 && bits) for(int g=0;g<8;++g) encoded[2+g]=static_cast<std::uint8_t>(g==0?0:g*36);
            const int start=kind==3?10:2;
            for(unsigned k=0;k<info.values_per_block;++k) {
                const bool active=bits && (kind!=3 || k/32!=0);
                const int q=active ? (kind==2?int((bits+k)%255)-127:int((bits+k)%16)-8) : 0;
                if(kind==2)encoded[start+k]=static_cast<std::uint8_t>(q);
                else encoded[start+k/2]|=static_cast<std::uint8_t>((q&15)<<((k&1)*4));
            }
            check(libgguf_native_decode_rows(format,encoded.data(),info.bytes_per_block,1,info.values_per_block,
                decoded.data(),info.values_per_block*4)==LIBGGUF_NATIVE_OK,"stored scale SIMD decode");
            const int exponent=(bits>>10)&31,mantissa=bits&1023;
            const float base=exponent?std::ldexp(float(1024+mantissa),exponent-25):std::ldexp(float(mantissa),-24);
            for(unsigned k=0;k<info.values_per_block;++k) {
                const bool active=bits && (kind!=3 || k/32!=0);
                const int q=active ? (kind==2?int((bits+k)%255)-127:int((bits+k)%16)-8) : 0;
                const float scale=kind==3?base*float(k/32==0?0:(k/32)*36):base;
                check(decoded[k]==scale*float(q),"stored scale/reference decode mismatch");
            }
        }
    }
    size_t bytes = 0;
    check(libgguf_native_row_size(static_cast<libgguf_native_format>(99), 32, &bytes)
        == LIBGGUF_NATIVE_INVALID_FORMAT, "separate format namespace");
    std::puts("native format, rounding, encoding and validation contracts passed");
}
