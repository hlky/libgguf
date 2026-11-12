#pragma once
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1201__)
#error "libgguf gfx1201 pack compiled for another architecture"
#endif
namespace libgguf::rocm::gfx1201 {
struct Architecture { static constexpr int wave_size = 32; };
}
