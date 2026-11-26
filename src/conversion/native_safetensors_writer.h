#pragma once

#include <libgguf/native_quant.h>
#include <cstddef>
#include <string>

// Preserve names and plain tensors; encode rank-two matrices in bounded row batches.
// Throws on incompatible widths, invalid source bytes, or an existing destination.
void convert_native_safetensors(const std::string& source, const std::string& destination,
    libgguf_native_format format, std::size_t scratch_bytes = 32u * 1024u * 1024u);
