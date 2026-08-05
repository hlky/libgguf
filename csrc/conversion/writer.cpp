#include "conversion/writer.h"

#include "conversion/common.h"

#include <algorithm>
#include <limits>

constexpr uint32_t GGUF_MAGIC = 0x46554747u;
constexpr uint32_t GGUF_VERSION = 3;
uint64_t gguf_pad(uint64_t n, uint64_t alignment) {
  return ((n + alignment - 1) / alignment) * alignment;
}

template <typename T>
void append_le(std::vector<uint8_t> &out, T value) {
  uint8_t *ptr = reinterpret_cast<uint8_t *>(&value);
  out.insert(out.end(), ptr, ptr + sizeof(T));
}

void append_string(std::vector<uint8_t> &out, const std::string &value) {
  append_le<uint64_t>(out, (uint64_t)value.size());
  out.insert(out.end(), value.begin(), value.end());
}

void write_all_file(FILE *f, const void *data, uint64_t len) {
  const uint8_t *cursor = (const uint8_t *)data;
  while (len > 0) {
    const size_t chunk = (size_t)std::min<uint64_t>(len, (uint64_t)std::numeric_limits<size_t>::max());
    const size_t written = std::fwrite(cursor, 1, chunk, f);
    if (written == 0) {
      fail("failed to write output file");
    }
    cursor += written;
    len -= written;
  }
}

void write_zeros_file(FILE *f, uint64_t len) {
  static const uint8_t zeros[4096] = {};
  while (len > 0) {
    const uint64_t chunk = std::min<uint64_t>(len, sizeof(zeros));
    write_all_file(f, zeros, chunk);
    len -= chunk;
  }
}

uint64_t tell_file(FILE *f) {
#if defined(_WIN32)
  __int64 pos = _ftelli64(f);
#else
  long pos = std::ftell(f);
#endif
  if (pos < 0) {
    fail("failed to query output file position");
  }
  return (uint64_t)pos;
}

void write_pre_tensor_padding(FILE *f, uint64_t alignment) {
  const uint64_t pos = tell_file(f);
  write_zeros_file(f, gguf_pad(pos, alignment) - pos);
}

void write_post_tensor_padding(FILE *f, uint64_t tensor_nbytes, uint64_t alignment) {
  write_zeros_file(f, gguf_pad(tensor_nbytes, alignment) - tensor_nbytes);
}


void append_kv_value(std::vector<uint8_t> &out, const kv_value &kv) {
  append_string(out, kv.key);
  append_le<uint32_t>(out, (uint32_t)kv.type);
  if (kv.type == GGUF_TYPE_STRING) {
    append_string(out, kv.str);
  } else if (kv.type == GGUF_TYPE_UINT32) {
    append_le<uint32_t>(out, kv.u32);
  } else if (kv.type == GGUF_TYPE_ARRAY) {
    append_le<uint32_t>(out, (uint32_t)GGUF_TYPE_INT64);
    append_le<uint64_t>(out, (uint64_t)kv.arr_i64.size());
    for (int64_t value : kv.arr_i64) {
      append_le<int64_t>(out, value);
    }
  } else {
    fail("unsupported GGUF metadata value type");
  }
}

void write_gguf_metadata(FILE *out, const std::vector<kv_value> &kv, const std::vector<tensor_info> &infos) {
  std::vector<uint8_t> header;
  append_le<uint32_t>(header, GGUF_MAGIC);
  append_le<uint32_t>(header, GGUF_VERSION);
  append_le<uint64_t>(header, (uint64_t)infos.size());
  append_le<uint64_t>(header, (uint64_t)kv.size());
  write_all_file(out, header.data(), header.size());

  std::vector<uint8_t> kv_data;
  for (const kv_value &value : kv) {
    append_kv_value(kv_data, value);
  }
  write_all_file(out, kv_data.data(), kv_data.size());

  std::vector<uint8_t> ti_data;
  uint64_t offset = 0;
  for (const tensor_info &info : infos) {
    append_string(ti_data, info.key);
    append_le<uint32_t>(ti_data, (uint32_t)info.gguf_shape.size());
    for (size_t i = info.gguf_shape.size(); i > 0; --i) {
      append_le<uint64_t>(ti_data, (uint64_t)info.gguf_shape[i - 1]);
    }
    append_le<uint32_t>(ti_data, (uint32_t)info.qtype);
    append_le<uint64_t>(ti_data, offset);
    offset += gguf_pad(info.nbytes, GGUF_DEFAULT_ALIGNMENT);
  }
  write_all_file(out, ti_data.data(), ti_data.size());
}


