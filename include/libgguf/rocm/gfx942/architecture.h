#pragma once
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx942__)
#error "libgguf gfx942 pack compiled for another architecture"
#endif
namespace libgguf::rocm::gfx942 {
struct Architecture { static constexpr int wave_size = 64; };
}
