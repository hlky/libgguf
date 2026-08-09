#include <libgguf/device/simple_q.h>

#if defined(HIP_INCLUDE_HIP_HIP_RUNTIME_H) || \
    defined(HIP_INCLUDE_HIP_HIP_RUNTIME_API_H) || \
    defined(__HIP_RUNTIME_H__) || defined(HIP_RUNTIME_H) || \
    defined(__HIP_RUNTIME_API_H__)
#error "simple_q.h must not include the HIP runtime"
#endif
#if defined(C10_UTIL_HALF_H_) || defined(ATEN_CORE_SCALARTYPE_H)
#error "simple_q.h must not include Torch or ATen"
#endif

__attribute__((global)) void simple_q_noruntime_probe(
    const libgguf::device::q4_0_block* q4,
    const libgguf::device::q8_0_block* q8, float* output) {
  libgguf::device::decode_q4_0_8(*q4, 12, output);
  libgguf::device::decode_q8_0_8(*q8, 12, output + 8);
}
