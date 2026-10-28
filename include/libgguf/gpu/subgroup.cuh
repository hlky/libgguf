#pragma once

#include <stdint.h>
#include <libgguf/gpu/compat.h>

namespace libgguf::gpu::subgroup32 {

// GGUF quantizers use logical groups of 32 values on both wave32 and wave64.
// These helpers assume the one-dimensional thread blocks used by the quantizers.
inline constexpr int width = 32;

static __device__ __forceinline__ uint32_t ballot(bool predicate) {
    const uint64_t physical = static_cast<uint64_t>(__ballot_sync(__activemask(), predicate));
    const unsigned base = (static_cast<unsigned>(threadIdx.x) % warpSize) & ~31u;
    return static_cast<uint32_t>(physical >> base);
}

} // namespace libgguf::gpu::subgroup32
