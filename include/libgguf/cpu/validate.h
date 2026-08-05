#pragma once

#include <cstddef>

#include <libgguf/types.h>

bool ggml_validate_row_data(enum ggml_type type, const void *data, size_t nbytes);
