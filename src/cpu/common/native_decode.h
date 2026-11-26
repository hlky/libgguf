#pragma once
#include <libgguf/common/native_formats.h>
#include <cstddef>
#include <cstdint>
namespace libgguf::native_cpu {
// Private validated-buffer implementations selected by the library CPU target.
void decode(NativeFormat,const std::uint8_t*,std::size_t blocks,float*);
bool valid_f32(const float*,std::size_t values,bool nonnegative);
}
