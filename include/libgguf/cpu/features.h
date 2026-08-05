#pragma once

struct libgguf_cpu_features
{
  bool sse2;
  bool sse4_1;
  bool f16c;
  bool avx2;
};

const libgguf_cpu_features &libgguf_get_cpu_features();
