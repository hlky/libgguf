#pragma once

#include "conversion/types.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
struct input_file {
#if defined(_WIN32)
  HANDLE file = INVALID_HANDLE_VALUE;
#else
  int fd = -1;
#endif
  uint64_t size = 0;

  bool open(const std::string &path, std::string *error) {
#if defined(_WIN32)
    int wide_len = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    if (wide_len <= 0) {
      *error = "failed to convert path to UTF-16";
      return false;
    }
    std::vector<wchar_t> wide_path((size_t)wide_len);
    if (MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wide_path.data(), wide_len) <= 0) {
      *error = "failed to convert path to UTF-16";
      return false;
    }
    file = CreateFileW(wide_path.data(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
      *error = "failed to open input file";
      return false;
    }
    LARGE_INTEGER file_size;
    if (!GetFileSizeEx(file, &file_size) || file_size.QuadPart < 0) {
      *error = "failed to stat input file";
      return false;
    }
    size = (uint64_t)file_size.QuadPart;
    return true;
#else
    fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
      *error = std::string("failed to open input file: ") + std::strerror(errno);
      return false;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 0) {
      *error = std::string("failed to stat input file: ") + std::strerror(errno);
      return false;
    }
    size = (uint64_t)st.st_size;
    return true;
#endif
  }

  bool read_at(uint64_t offset, void *dst, uint64_t len, std::string *error) const {
    if (offset > size || len > size - offset) {
      *error = "read range exceeds input file size";
      return false;
    }
    uint8_t *cursor = (uint8_t *)dst;
    uint64_t remaining = len;
#if defined(_WIN32)
    LARGE_INTEGER pos;
    while (remaining > 0) {
      pos.QuadPart = (LONGLONG)offset;
      OVERLAPPED ov{};
      ov.Offset = pos.LowPart;
      ov.OffsetHigh = pos.HighPart;
      DWORD chunk = (DWORD)std::min<uint64_t>(remaining, (uint64_t)std::numeric_limits<DWORD>::max());
      DWORD got = 0;
      if (!ReadFile(file, cursor, chunk, &got, &ov)) {
        *error = "failed to read input file";
        return false;
      }
      if (got == 0) {
        *error = "unexpected end of input file";
        return false;
      }
      cursor += got;
      offset += got;
      remaining -= got;
    }
#else
    while (remaining > 0) {
      const size_t chunk = (size_t)std::min<uint64_t>(remaining, (uint64_t)std::numeric_limits<size_t>::max());
      const ssize_t got = pread(fd, cursor, chunk, (off_t)offset);
      if (got < 0) {
        if (errno == EINTR) {
          continue;
        }
        *error = std::string("failed to read input file: ") + std::strerror(errno);
        return false;
      }
      if (got == 0) {
        *error = "unexpected end of input file";
        return false;
      }
      cursor += got;
      offset += (uint64_t)got;
      remaining -= (uint64_t)got;
    }
#endif
    return true;
  }

  void close() {
#if defined(_WIN32)
    if (file != INVALID_HANDLE_VALUE) {
      CloseHandle(file);
      file = INVALID_HANDLE_VALUE;
    }
#else
    if (fd >= 0) {
      ::close(fd);
      fd = -1;
    }
#endif
    size = 0;
  }

  ~input_file() {
    close();
  }
};

std::vector<tensor_meta> parse_safetensors_header(const input_file & input, uint64_t * data_start);
void strip_prefix(std::vector<tensor_meta> & tensors);
size_t source_dtype_size(native_source_dtype dtype);
native_source_dtype parse_source_dtype(const std::string & dtype);
