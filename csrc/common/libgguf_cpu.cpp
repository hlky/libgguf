#include "libgguf_cpu.h"

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
#include <cpuid.h>
#endif

static void libgguf_cpuid(int leaf, int subleaf, int regs[4])
{
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
  __cpuidex(regs, leaf, subleaf);
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
  unsigned int eax = 0;
  unsigned int ebx = 0;
  unsigned int ecx = 0;
  unsigned int edx = 0;
  __cpuid_count((unsigned int)leaf, (unsigned int)subleaf, eax, ebx, ecx, edx);
  regs[0] = (int)eax;
  regs[1] = (int)ebx;
  regs[2] = (int)ecx;
  regs[3] = (int)edx;
#else
  (void)leaf;
  (void)subleaf;
  regs[0] = regs[1] = regs[2] = regs[3] = 0;
#endif
}

static unsigned long long libgguf_xgetbv(unsigned int index)
{
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
  return _xgetbv(index);
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
  unsigned int eax = 0;
  unsigned int edx = 0;
  __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
  return ((unsigned long long)edx << 32) | eax;
#else
  (void)index;
  return 0;
#endif
}

static libgguf_cpu_features libgguf_detect_cpu_features()
{
  libgguf_cpu_features features = {false, false};

#if defined(_M_X64) || defined(__x86_64__)
  features.sse2 = true;
#endif

#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
  int regs[4] = {0, 0, 0, 0};
  libgguf_cpuid(0, 0, regs);
  const int max_leaf = regs[0];
  if (max_leaf < 1)
  {
    return features;
  }

  libgguf_cpuid(1, 0, regs);
  features.sse2 = features.sse2 || ((regs[3] & (1 << 26)) != 0);

  const bool osxsave = (regs[2] & (1 << 27)) != 0;
  const bool avx = (regs[2] & (1 << 28)) != 0;
  const bool os_supports_ymm = osxsave && ((libgguf_xgetbv(0) & 0x6) == 0x6);
  if (max_leaf >= 7 && avx && os_supports_ymm)
  {
    libgguf_cpuid(7, 0, regs);
    features.avx2 = (regs[1] & (1 << 5)) != 0;
  }
#endif

  return features;
}

const libgguf_cpu_features &libgguf_get_cpu_features()
{
  static const libgguf_cpu_features features = libgguf_detect_cpu_features();
  return features;
}
