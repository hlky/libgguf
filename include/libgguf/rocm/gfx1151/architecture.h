#pragma once
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
#error "libgguf gfx1151 pack compiled for another architecture"
#endif
namespace libgguf::rocm::gfx1151 {
struct Architecture { static constexpr int wave_size = 32; };
}
