#include "cpu/threading.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <thread>

unsigned int libgguf_cpu_thread_count(int64_t row_count)
{
  if (row_count < 64)
  {
    return 1;
  }

  const char *env = std::getenv("LIBGGUF_NUM_THREADS");
  if (env != nullptr && env[0] != '\0')
  {
    errno = 0;
    char *end = nullptr;
    const long long parsed = std::strtoll(env, &end, 10);
    if (errno == 0 && end != env && parsed > 0)
    {
      return (unsigned int)std::min<int64_t>((int64_t)parsed, row_count);
    }
    return 1;
  }

  const unsigned int hardware = std::thread::hardware_concurrency();
  if (hardware == 0)
  {
    return 1;
  }
  return std::min<unsigned int>(hardware, (unsigned int)row_count);
}
