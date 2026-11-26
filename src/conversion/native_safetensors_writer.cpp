#include "conversion/native_safetensors_writer.h"
#include "conversion/common.h"
#include "conversion/safetensors.h"
#include <libgguf/gpu/native_fragment.cuh>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <vector>

namespace {
std::string quote(const std::string& value) {
    std::string result = "\"";
    const char* hex = "0123456789abcdef";
    for (const unsigned char c : value) {
        if (c == '"' || c == '\\') { result += '\\'; result += static_cast<char>(c); }
        else if (c < 32) { result += "\\u00"; result += hex[c >> 4]; result += hex[c & 15]; }
        else result += static_cast<char>(c);
    }
    return result + '"';
}
void write(std::ofstream& file, const void* data, std::size_t bytes) {
    file.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!file) fail("failed to write native Safetensors output");
}
struct TempOutput {
    std::filesystem::path path;
    bool published = false;
    ~TempOutput() { if (!published) { std::error_code ignored; std::filesystem::remove(path, ignored); } }
};
}

void convert_native_safetensors(const std::string& source, const std::string& destination,
    libgguf_native_format format, std::size_t budget) {
    const auto info = libgguf::native_format_info(static_cast<libgguf::NativeFormat>(format));
    if (!info.name) fail("unknown native format");
    if (std::filesystem::exists(std::filesystem::u8path(destination))) fail("native output already exists");
    input_file input;
    std::string error;
    if (!input.open(source, &error)) fail(error);
    unsigned char length_bytes[8]{};
    if (!input.read_at(0, length_bytes, 8, &error)) fail(error);
    uint64_t header_bytes = 0;
    for (int i = 0; i < 8; ++i) header_bytes |= uint64_t(length_bytes[i]) << (i * 8);
    if (!header_bytes || header_bytes > 8u * 1024u * 1024u) fail("native source header exceeds 8 MiB");
    uint64_t data_start = 0;
    std::map<std::string,std::string> source_metadata;
    auto tensors = parse_safetensors_header(input, &data_start, &source_metadata);
    if (source_metadata.count("libgguf.quantization.v1")) fail("native source is already quantized");
    std::sort(tensors.begin(), tensors.end(), [](const auto& a, const auto& b) { return a.key < b.key; });
    std::set<std::string> names;
    uint64_t offset = 0;
    std::ostringstream fields, descriptors;
    bool first = true, first_quant = true;
    for (const auto& tensor : tensors) {
        if (!names.insert(tensor.key).second) fail("duplicate native source tensor name");
        uint64_t bytes = tensor.data_end - tensor.data_begin;
        const bool quantized = tensor.shape.size() == 2;
        size_t row_bytes = 0;
        if (quantized) {
            if (libgguf_native_row_size(format, tensor.shape[1], &row_bytes) != 0)
                fail("native format width is incompatible for " + tensor.key);
            if (!checked_mul_u64(tensor.shape[0], row_bytes, &bytes)) fail("native output extent overflows");
            if (!first_quant) descriptors << ',';
            first_quant = false;
            descriptors << quote(tensor.key) << ":{\"format\":" << quote(info.name)
                << ",\"format_revision\":1,\"encoding\":\"canonical_rows\",\"encoding_revision\":1,\"logical_shape\":["
                << tensor.shape[0] << ',' << tensor.shape[1] << "],\"payload\":" << quote(tensor.key)
                << ",\"encoder\":{\"name\":\"clip_grid\",\"revision\":1,\"calibration_digest\":null}}";
        }
        if (bytes > std::numeric_limits<uint64_t>::max() - offset) fail("native file extent overflows");
        if (!first) fields << ',';
        first = false;
        fields << quote(tensor.key) << ":{\"dtype\":" << quote(quantized ? "U8" : tensor.dtype) << ",\"shape\":[";
        if (quantized) fields << tensor.shape[0] << ',' << row_bytes;
        else for (size_t i = 0; i < tensor.shape.size(); ++i) { if (i) fields << ','; fields << tensor.shape[i]; }
        fields << "],\"data_offsets\":[" << offset << ',' << offset + bytes << "]}";
        offset += bytes;
    }
    std::string metadata = "{\"schema_version\":1,\"tensors\":{" + descriptors.str() + "}}";
    std::string header = "{\"__metadata__\":{\"libgguf.quantization.v1\":" + quote(metadata);
    for (const auto& field : source_metadata) header += ',' + quote(field.first) + ':' + quote(field.second);
    header += '}';
    if (!first) header += ',' + fields.str();
    header += '}';
    if (header.size() > 8u * 1024u * 1024u) fail("native output metadata exceeds 8 MiB");
    while (header.size() % 8) header += ' ';
    const auto dest = std::filesystem::u8path(destination);
    auto temporary_path = dest;
    temporary_path += ".tmp-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    TempOutput temporary{temporary_path};
    std::ofstream output(temporary.path, std::ios::binary | std::ios::trunc);
    if (!output) fail("cannot create native output temporary file");
    uint64_t size = header.size();
    unsigned char prefix[8]{};
    for (int i = 0; i < 8; ++i) prefix[i] = static_cast<unsigned char>(size >> (i * 8));
    write(output, prefix, 8); write(output, header.data(), header.size());
    for (const auto& tensor : tensors) {
        if (tensor.shape.size() != 2) {
            if (!budget) fail("native conversion scratch budget is zero");
            std::vector<unsigned char> buffer(static_cast<size_t>((std::min)(uint64_t(budget), tensor.data_end - tensor.data_begin)));
            for (auto pos = tensor.data_begin; pos < tensor.data_end;) {
                const auto count = (std::min)(uint64_t(buffer.size()), tensor.data_end - pos);
                if (!input.read_at(pos, buffer.data(), count, &error)) fail(error);
                write(output, buffer.data(), static_cast<size_t>(count)); pos += count;
            }
            continue;
        }
        const auto columns = static_cast<size_t>(tensor.shape[1]);
        const auto rows = static_cast<size_t>(tensor.shape[0]);
        const auto dtype = parse_source_dtype(tensor.dtype);
        const size_t element_bytes = source_dtype_size(dtype);
        size_t encoded_row = 0;
        if (libgguf_native_row_size(format, columns, &encoded_row) != 0) fail("invalid row width");
        uint64_t per_row = 0;
        if (!checked_mul_u64(columns, element_bytes + sizeof(float), &per_row)
            || encoded_row > std::numeric_limits<uint64_t>::max() - per_row) fail("scratch extent overflows");
        per_row += encoded_row;
        if (per_row > budget) fail("native conversion needs a larger scratch budget for " + tensor.key);
        const size_t batch = (std::min)(rows, budget / static_cast<size_t>(per_row));
        std::vector<unsigned char> raw(batch * columns * element_bytes), encoded(batch * encoded_row);
        std::vector<float> decoded(batch * columns);
        for (size_t start = 0; start < rows; start += batch) {
            const auto count = (std::min)(batch, rows - start);
            if (!input.read_at(tensor.data_begin + start * columns * element_bytes,
                raw.data(), count * columns * element_bytes, &error)) fail(error);
            for (size_t k = 0; k < count * columns; ++k) {
                if (dtype == NATIVE_DTYPE_F32) {
                    const auto* p = raw.data() + k * 4;
                    const uint32_t bits = uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
                    decoded[k] = libgguf::gpu::fragment_detail::float_from_bits(bits);
                } else {
                    const auto bits = libgguf::gpu::fragment_detail::load_u16_le(raw.data() + k * 2);
                    decoded[k] = dtype == NATIVE_DTYPE_F16
                        ? libgguf::gpu::fragment_detail::fp16_to_float(bits)
                        : libgguf::gpu::fragment_detail::float_from_bits(uint32_t(bits) << 16);
                }
            }
            const auto status = libgguf_native_encode_rows(format, decoded.data(), count * columns * 4,
                count, columns, nullptr, 0, encoded.data(), count * encoded_row);
            if (status != LIBGGUF_NATIVE_OK) fail("native encoder rejected source tensor " + tensor.key);
            write(output, encoded.data(), count * encoded_row);
        }
    }
    output.close();
    if (!output) fail("failed to close native output");
    // Link publication is atomic and cannot replace a destination created by a
    // competing writer. The completed temporary is in the same directory.
    std::filesystem::create_hard_link(temporary.path, dest);
    temporary.published = true;
    std::error_code ignored;
    std::filesystem::remove(temporary.path, ignored);
}
