#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <emmintrin.h>
#define LIBGGUF_NATIVE_SSE2 1
#endif

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "common/libgguf_common.h"
#include "common/libgguf_internal.h"
#include "libgguf.h"

extern "C" const char *libgguf_q4_0_backend(void);
extern "C" int libgguf_q4_0_cpu_supports_backend(const char *backend);
extern "C" size_t libgguf_quantize_q4_0_for_backend(
    const char *backend,
    const float *src,
    void *dst,
    int64_t nrows,
    int64_t n_per_row);
extern "C" const char *libgguf_q8_0_backend(void);
extern "C" int libgguf_q8_0_cpu_supports_backend(const char *backend);
extern "C" size_t libgguf_quantize_q8_0_for_backend(
    const char *backend,
    const float *src,
    void *dst,
    int64_t nrows,
    int64_t n_per_row);
extern "C" const char *libgguf_q1_0_backend(void);
extern "C" int libgguf_q1_0_cpu_supports_backend(const char *backend);
extern "C" size_t libgguf_quantize_q1_0_for_backend(
    const char *backend,
    const float *src,
    void *dst,
    int64_t nrows,
    int64_t n_per_row);
extern "C" const char *libgguf_q4_1_backend(void);
extern "C" int libgguf_q4_1_cpu_supports_backend(const char *backend);
extern "C" size_t libgguf_quantize_q4_1_for_backend(
    const char *backend,
    const float *src,
    void *dst,
    int64_t nrows,
    int64_t n_per_row);
extern "C" const char *libgguf_q5_0_backend(void);
extern "C" int libgguf_q5_0_cpu_supports_backend(const char *backend);
extern "C" size_t libgguf_quantize_q5_0_for_backend(
    const char *backend,
    const float *src,
    void *dst,
    int64_t nrows,
    int64_t n_per_row);
extern "C" const char *libgguf_q5_1_backend(void);
extern "C" int libgguf_q5_1_cpu_supports_backend(const char *backend);
extern "C" size_t libgguf_quantize_q5_1_for_backend(
    const char *backend,
    const float *src,
    void *dst,
    int64_t nrows,
    int64_t n_per_row);
extern "C" const char *libgguf_mxfp4_backend(void);
extern "C" int libgguf_mxfp4_cpu_supports_backend(const char *backend);
extern "C" size_t libgguf_quantize_mxfp4_for_backend(
    const char *backend,
    const float *src,
    void *dst,
    int64_t nrows,
    int64_t n_per_row);
extern "C" const char *libgguf_nvfp4_backend(void);
extern "C" int libgguf_nvfp4_cpu_supports_backend(const char *backend);
extern "C" size_t libgguf_quantize_nvfp4_for_backend(
    const char *backend,
    const float *src,
    void *dst,
    int64_t nrows,
    int64_t n_per_row);
extern "C" const char *libgguf_q2_k_backend(void);
extern "C" int libgguf_q2_k_cpu_supports_backend(const char *backend);
extern "C" size_t libgguf_quantize_q2_k_for_backend(
    const char *backend,
    const float *src,
    void *dst,
    int64_t nrows,
    int64_t n_per_row);
extern "C" const char *libgguf_q3_k_backend(void);
extern "C" int libgguf_q3_k_cpu_supports_backend(const char *backend);
extern "C" size_t libgguf_quantize_q3_k_for_backend(
    const char *backend,
    const float *src,
    void *dst,
    int64_t nrows,
    int64_t n_per_row);
extern "C" const char *libgguf_q4_k_backend(void);
extern "C" int libgguf_q4_k_cpu_supports_backend(const char *backend);
extern "C" size_t libgguf_quantize_q4_k_for_backend(
    const char *backend,
    const float *src,
    void *dst,
    int64_t nrows,
    int64_t n_per_row);
extern "C" const char *libgguf_q5_k_backend(void);
extern "C" int libgguf_q5_k_cpu_supports_backend(const char *backend);
extern "C" size_t libgguf_quantize_q5_k_for_backend(
    const char *backend,
    const float *src,
    void *dst,
    int64_t nrows,
    int64_t n_per_row);
extern "C" const char *libgguf_q6_k_backend(void);
extern "C" int libgguf_q6_k_cpu_supports_backend(const char *backend);
extern "C" size_t libgguf_quantize_q6_k_for_backend(
    const char *backend,
    const float *src,
    void *dst,
    int64_t nrows,
    int64_t n_per_row);
extern "C" int libgguf_iq4_nl_cpu_supports_backend(const char *backend);
extern "C" size_t libgguf_quantize_iq4_nl_for_backend(
    const char *backend,
    const float *src,
    void *dst,
    int64_t nrows,
    int64_t n_per_row);
extern "C" int libgguf_iq4_xs_cpu_supports_backend(const char *backend);
extern "C" size_t libgguf_quantize_iq4_xs_for_backend(
    const char *backend,
    const float *src,
    void *dst,
    int64_t nrows,
    int64_t n_per_row);
extern "C" const char *libgguf_tq1_0_backend(void);
extern "C" int libgguf_tq1_0_cpu_supports_backend(const char *backend);
extern "C" size_t libgguf_quantize_tq1_0_for_backend(
    const char *backend,
    const float *src,
    void *dst,
    int64_t nrows,
    int64_t n_per_row);
extern "C" const char *libgguf_tq2_0_backend(void);
extern "C" int libgguf_tq2_0_cpu_supports_backend(const char *backend);
extern "C" size_t libgguf_quantize_tq2_0_for_backend(
    const char *backend,
    const float *src,
    void *dst,
    int64_t nrows,
    int64_t n_per_row);
extern "C" const char *libgguf_common_quant_backend(void);
extern "C" int libgguf_common_quant_cpu_supports_backend(const char *backend);
extern "C" int libgguf_common_quant_set_backend(const char *backend);
extern "C" uint64_t libgguf_common_quant_probe_for_backend(const char *backend);
extern "C" const char *libgguf_storage_backend(void);
extern "C" int libgguf_storage_cpu_supports_backend(const char *backend);
extern "C" int libgguf_storage_set_backend(const char *backend);
extern "C" const char *libgguf_dequant_backend(int type);
extern "C" int libgguf_dequant_cpu_supports_backend(const char *backend);
extern "C" size_t libgguf_dequantize_for_backend(
    int type,
    const char *backend,
    const void *src,
    float *dst,
    int64_t nrows,
    int64_t n_per_row);

static bool parse_int64(PyObject *value, int64_t *out);

static PyObject *py_quantize_for_backend(
    PyObject *args,
    ggml_type type,
    const char *error_message,
    size_t (*quantize_for_backend)(const char *, const float *, void *, int64_t, int64_t))
{
  const char *backend = nullptr;
  PyObject *src_obj;
  PyObject *n_rows_obj;
  PyObject *n_per_row_obj;

  if (!PyArg_ParseTuple(args, "sOOO", &backend, &src_obj, &n_rows_obj, &n_per_row_obj))
  {
    return nullptr;
  }

  int64_t n_rows = 0;
  int64_t n_per_row = 0;
  if (!parse_int64(n_rows_obj, &n_rows) || !parse_int64(n_per_row_obj, &n_per_row))
  {
    return nullptr;
  }
  if (n_rows < 0 || n_per_row <= 0)
  {
    PyErr_SetString(PyExc_ValueError, "n_rows must be non-negative and n_per_row must be positive");
    return nullptr;
  }

  Py_buffer src_view;
  if (PyObject_GetBuffer(src_obj, &src_view, PyBUF_CONTIG_RO) < 0)
  {
    return nullptr;
  }

  const size_t row_size = libgguf_row_size(type, n_per_row);
  const uint64_t src_required = (uint64_t)n_rows * (uint64_t)n_per_row * sizeof(float);
  if ((uint64_t)src_view.len < src_required)
  {
    PyBuffer_Release(&src_view);
    PyErr_SetString(PyExc_ValueError, "src buffer is smaller than n_rows * n_per_row float32 values");
    return nullptr;
  }

  const uint64_t out_size_u64 = (uint64_t)n_rows * (uint64_t)row_size;
  if (out_size_u64 > (uint64_t)std::numeric_limits<Py_ssize_t>::max())
  {
    PyBuffer_Release(&src_view);
    PyErr_SetString(PyExc_OverflowError, "quantized output is too large");
    return nullptr;
  }

  PyObject *out = PyBytes_FromStringAndSize(nullptr, (Py_ssize_t)out_size_u64);
  if (!out)
  {
    PyBuffer_Release(&src_view);
    return nullptr;
  }

  size_t written = 0;
  Py_BEGIN_ALLOW_THREADS
  written = quantize_for_backend(
      backend,
      (const float *)src_view.buf,
      PyBytes_AS_STRING(out),
      n_rows,
      n_per_row);
  Py_END_ALLOW_THREADS
  PyBuffer_Release(&src_view);

  if (written != (size_t)out_size_u64)
  {
    Py_DECREF(out);
    PyErr_SetString(PyExc_ValueError, error_message);
    return nullptr;
  }

  return out;
}

static bool parse_int64(PyObject *value, int64_t *out)
{
  long long parsed = PyLong_AsLongLong(value);
  if (parsed == -1 && PyErr_Occurred())
  {
    return false;
  }
  *out = (int64_t)parsed;
  return true;
}

enum native_source_dtype
{
  NATIVE_DTYPE_F32,
  NATIVE_DTYPE_F16,
  NATIVE_DTYPE_BF16,
};

static constexpr uint64_t NATIVE_DEFAULT_SCRATCH_BYTES = 32ull * 1024ull * 1024ull;

struct native_tensor_plan
{
  std::string key;
  native_source_dtype source_dtype = NATIVE_DTYPE_F32;
  ggml_type qtype = GGML_TYPE_F32;
  uint64_t data_begin = 0;
  uint64_t data_end = 0;
  uint64_t n_values = 0;
  uint64_t n_rows = 0;
  uint64_t n_per_row = 0;
  uint64_t expected_nbytes = 0;
  Py_buffer imatrix_view{};
  bool has_imatrix_view = false;
  std::vector<float> auto_imatrix;
};

static void release_native_plan_buffers(std::vector<native_tensor_plan> &plans)
{
  for (native_tensor_plan &plan : plans)
  {
    if (plan.has_imatrix_view)
    {
      PyBuffer_Release(&plan.imatrix_view);
      plan.has_imatrix_view = false;
    }
  }
}

static bool checked_mul_u64(uint64_t a, uint64_t b, uint64_t *out)
{
  if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a)
  {
    return false;
  }
  *out = a * b;
  return true;
}

static bool parse_shape_product(PyObject *shape_obj, const char *field_name, std::vector<int64_t> *shape, uint64_t *product)
{
  PyObject *seq = PySequence_Fast(shape_obj, field_name);
  if (!seq)
  {
    return false;
  }

  const Py_ssize_t len = PySequence_Fast_GET_SIZE(seq);

  shape->clear();
  shape->reserve((size_t)len);
  uint64_t total = 1;
  PyObject **items = PySequence_Fast_ITEMS(seq);
  for (Py_ssize_t i = 0; i < len; ++i)
  {
    long long dim = PyLong_AsLongLong(items[i]);
    if (dim == -1 && PyErr_Occurred())
    {
      Py_DECREF(seq);
      return false;
    }
    if (dim <= 0)
    {
      Py_DECREF(seq);
      PyErr_Format(PyExc_ValueError, "%s dimensions must be positive", field_name);
      return false;
    }
    uint64_t next_total = 0;
    if (!checked_mul_u64(total, (uint64_t)dim, &next_total))
    {
      Py_DECREF(seq);
      PyErr_Format(PyExc_OverflowError, "%s element count is too large", field_name);
      return false;
    }
    total = next_total;
    shape->push_back((int64_t)dim);
  }

  Py_DECREF(seq);
  *product = total;
  return true;
}

static PyObject *get_required_mapping_item(PyObject *mapping, const char *key)
{
  PyObject *value = PyMapping_GetItemString(mapping, key);
  if (!value)
  {
    PyErr_Clear();
    PyErr_Format(PyExc_ValueError, "native payload plan is missing %s", key);
    return nullptr;
  }
  return value;
}

static bool parse_u64_item(PyObject *mapping, const char *key, uint64_t *out)
{
  PyObject *value = get_required_mapping_item(mapping, key);
  if (!value)
  {
    return false;
  }
  unsigned long long parsed = PyLong_AsUnsignedLongLong(value);
  Py_DECREF(value);
  if (parsed == (unsigned long long)-1 && PyErr_Occurred())
  {
    return false;
  }
  *out = (uint64_t)parsed;
  return true;
}

static bool parse_int_item(PyObject *mapping, const char *key, int *out)
{
  PyObject *value = get_required_mapping_item(mapping, key);
  if (!value)
  {
    return false;
  }
  long parsed = PyLong_AsLong(value);
  Py_DECREF(value);
  if (parsed == -1 && PyErr_Occurred())
  {
    return false;
  }
  if (parsed < std::numeric_limits<int>::min() || parsed > std::numeric_limits<int>::max())
  {
    PyErr_Format(PyExc_OverflowError, "%s is outside int range", key);
    return false;
  }
  *out = (int)parsed;
  return true;
}

static bool parse_source_dtype(const char *dtype, native_source_dtype *out)
{
  if (std::strcmp(dtype, "F32") == 0)
  {
    *out = NATIVE_DTYPE_F32;
    return true;
  }
  if (std::strcmp(dtype, "F16") == 0)
  {
    *out = NATIVE_DTYPE_F16;
    return true;
  }
  if (std::strcmp(dtype, "BF16") == 0)
  {
    *out = NATIVE_DTYPE_BF16;
    return true;
  }
  return false;
}

static size_t source_dtype_size(native_source_dtype dtype)
{
  switch (dtype)
  {
  case NATIVE_DTYPE_F32:
    return sizeof(float);
  case NATIVE_DTYPE_F16:
  case NATIVE_DTYPE_BF16:
    return sizeof(uint16_t);
  }
  return 0;
}

static bool qtype_matches_source(native_source_dtype dtype, ggml_type qtype)
{
  return (dtype == NATIVE_DTYPE_F32 && qtype == GGML_TYPE_F32) ||
         (dtype == NATIVE_DTYPE_F16 && qtype == GGML_TYPE_F16) ||
         (dtype == NATIVE_DTYPE_BF16 && qtype == GGML_TYPE_BF16);
}

static unsigned int native_quantize_thread_count(int64_t nrows)
{
  if (nrows < 64)
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
      return (unsigned int)std::min<int64_t>((int64_t)parsed, nrows);
    }
    return 1;
  }

  const unsigned int hardware = std::thread::hardware_concurrency();
  if (hardware == 0)
  {
    return 1;
  }
  return std::min<unsigned int>(hardware, (unsigned int)nrows);
}

static bool parse_native_plan(PyObject *item, native_tensor_plan *plan)
{
  PyObject *key_obj = get_required_mapping_item(item, "key");
  if (!key_obj)
  {
    return false;
  }
  const char *key = PyUnicode_AsUTF8(key_obj);
  if (!key)
  {
    Py_DECREF(key_obj);
    return false;
  }
  plan->key = key;
  Py_DECREF(key_obj);

  PyObject *dtype_obj = get_required_mapping_item(item, "source_dtype");
  if (!dtype_obj)
  {
    return false;
  }
  const char *dtype = PyUnicode_AsUTF8(dtype_obj);
  if (!dtype)
  {
    Py_DECREF(dtype_obj);
    return false;
  }
  if (!parse_source_dtype(dtype, &plan->source_dtype))
  {
    PyErr_Format(PyExc_ValueError, "unsupported source dtype for %s: %s", plan->key.c_str(), dtype);
    Py_DECREF(dtype_obj);
    return false;
  }
  Py_DECREF(dtype_obj);

  int qtype = 0;
  if (!parse_int_item(item, "qtype", &qtype))
  {
    return false;
  }
  if (qtype < 0 || qtype >= GGML_TYPE_COUNT)
  {
    PyErr_Format(PyExc_ValueError, "unsupported qtype for %s: %d", plan->key.c_str(), qtype);
    return false;
  }
  plan->qtype = (ggml_type)qtype;

  if (!parse_u64_item(item, "data_begin", &plan->data_begin) ||
      !parse_u64_item(item, "data_end", &plan->data_end) ||
      !parse_u64_item(item, "nbytes", &plan->expected_nbytes))
  {
    return false;
  }
  if (plan->data_end < plan->data_begin)
  {
    PyErr_Format(PyExc_ValueError, "invalid safetensors byte range for %s", plan->key.c_str());
    return false;
  }

  PyObject *source_shape_obj = get_required_mapping_item(item, "source_shape");
  if (!source_shape_obj)
  {
    return false;
  }
  std::vector<int64_t> source_shape;
  uint64_t source_values = 0;
  bool parsed_source_shape = parse_shape_product(source_shape_obj, "source_shape", &source_shape, &source_values);
  Py_DECREF(source_shape_obj);
  if (!parsed_source_shape)
  {
    return false;
  }

  PyObject *write_shape_obj = get_required_mapping_item(item, "write_shape");
  if (!write_shape_obj)
  {
    return false;
  }
  std::vector<int64_t> write_shape;
  uint64_t write_values = 0;
  bool parsed_write_shape = parse_shape_product(write_shape_obj, "write_shape", &write_shape, &write_values);
  Py_DECREF(write_shape_obj);
  if (!parsed_write_shape)
  {
    return false;
  }
  if (source_values != write_values)
  {
    PyErr_Format(PyExc_ValueError, "source/write element counts differ for %s", plan->key.c_str());
    return false;
  }
  plan->n_values = write_values;
  plan->n_per_row = write_shape.empty() ? 1 : (uint64_t)write_shape.back();
  plan->n_rows = write_values / plan->n_per_row;

  uint64_t expected_source_bytes = 0;
  if (!checked_mul_u64(plan->n_values, (uint64_t)source_dtype_size(plan->source_dtype), &expected_source_bytes))
  {
    PyErr_Format(PyExc_OverflowError, "source byte count is too large for %s", plan->key.c_str());
    return false;
  }
  if (plan->data_end - plan->data_begin != expected_source_bytes)
  {
    PyErr_Format(PyExc_ValueError, "safetensors byte length does not match shape/dtype for %s", plan->key.c_str());
    return false;
  }

  const size_t row_size = libgguf_row_size(plan->qtype, (int64_t)plan->n_per_row);
  if (row_size == 0)
  {
    PyErr_Format(PyExc_ValueError, "unsupported qtype or row width for %s", plan->key.c_str());
    return false;
  }
  uint64_t expected_output_bytes = 0;
  if (!checked_mul_u64(plan->n_rows, (uint64_t)row_size, &expected_output_bytes))
  {
    PyErr_Format(PyExc_OverflowError, "output byte count is too large for %s", plan->key.c_str());
    return false;
  }
  if (plan->expected_nbytes != expected_output_bytes)
  {
    PyErr_Format(PyExc_ValueError, "expected nbytes does not match qtype/shape for %s", plan->key.c_str());
    return false;
  }

  PyObject *imatrix_obj = PyMapping_GetItemString(item, "imatrix");
  if (!imatrix_obj)
  {
    PyErr_Clear();
  }
  else
  {
    if (imatrix_obj != Py_None)
    {
      if (PyObject_GetBuffer(imatrix_obj, &plan->imatrix_view, PyBUF_CONTIG_RO) < 0)
      {
        Py_DECREF(imatrix_obj);
        return false;
      }
      plan->has_imatrix_view = true;
      const uint64_t imatrix_required = plan->n_per_row * sizeof(float);
      if ((uint64_t)plan->imatrix_view.len < imatrix_required)
      {
        Py_DECREF(imatrix_obj);
        PyErr_Format(PyExc_ValueError, "imatrix is smaller than n_per_row for %s", plan->key.c_str());
        return false;
      }
    }
    Py_DECREF(imatrix_obj);
  }

  return true;
}

struct native_mapped_file
{
  const uint8_t *data = nullptr;
  uint64_t size = 0;
#if defined(_WIN32)
  HANDLE file = INVALID_HANDLE_VALUE;
  HANDLE mapping = nullptr;
#else
  int fd = -1;
#endif

  bool map(const std::string &path, std::string *error)
  {
#if defined(_WIN32)
    int wide_len = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    if (wide_len <= 0)
    {
      *error = "failed to convert safetensors path to UTF-16";
      return false;
    }
    std::vector<wchar_t> wide_path((size_t)wide_len);
    if (MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wide_path.data(), wide_len) <= 0)
    {
      *error = "failed to convert safetensors path to UTF-16";
      return false;
    }

    file = CreateFileW(wide_path.data(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE)
    {
      *error = "failed to open safetensors file";
      return false;
    }
    LARGE_INTEGER file_size;
    if (!GetFileSizeEx(file, &file_size) || file_size.QuadPart < 0)
    {
      *error = "failed to stat safetensors file";
      return false;
    }
    size = (uint64_t)file_size.QuadPart;
    if (size == 0)
    {
      *error = "safetensors file is empty";
      return false;
    }
    mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mapping)
    {
      *error = "failed to create safetensors file mapping";
      return false;
    }
    data = (const uint8_t *)MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    if (!data)
    {
      *error = "failed to map safetensors file";
      return false;
    }
    return true;
#else
    fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0)
    {
      *error = std::string("failed to open safetensors file: ") + std::strerror(errno);
      return false;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 0)
    {
      *error = std::string("failed to stat safetensors file: ") + std::strerror(errno);
      return false;
    }
    size = (uint64_t)st.st_size;
    if (size == 0)
    {
      *error = "safetensors file is empty";
      return false;
    }
    void *mapped = mmap(nullptr, (size_t)size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED)
    {
      *error = std::string("failed to map safetensors file: ") + std::strerror(errno);
      return false;
    }
    data = (const uint8_t *)mapped;
    return true;
#endif
  }

  void close()
  {
#if defined(_WIN32)
    if (data)
    {
      UnmapViewOfFile(data);
      data = nullptr;
    }
    if (mapping)
    {
      CloseHandle(mapping);
      mapping = nullptr;
    }
    if (file != INVALID_HANDLE_VALUE)
    {
      CloseHandle(file);
      file = INVALID_HANDLE_VALUE;
    }
#else
    if (data)
    {
      munmap((void *)data, (size_t)size);
      data = nullptr;
    }
    if (fd >= 0)
    {
      ::close(fd);
      fd = -1;
    }
#endif
    size = 0;
  }

  ~native_mapped_file()
  {
    close();
  }
};

static bool fd_current_position(int fd, uint64_t *position, std::string *error)
{
#if defined(_WIN32)
  __int64 result = _lseeki64(fd, 0, SEEK_CUR);
  if (result < 0)
  {
    *error = "failed to query output file position";
    return false;
  }
#else
  off_t result = lseek(fd, 0, SEEK_CUR);
  if (result < 0)
  {
    *error = std::string("failed to query output file position: ") + std::strerror(errno);
    return false;
  }
#endif
  *position = (uint64_t)result;
  return true;
}

static bool write_all_fd(int fd, const void *data, uint64_t len, std::string *error)
{
  const uint8_t *cursor = (const uint8_t *)data;
  while (len > 0)
  {
#if defined(_WIN32)
    const unsigned int chunk = (unsigned int)std::min<uint64_t>(len, (uint64_t)UINT_MAX);
    int written = _write(fd, cursor, chunk);
    if (written < 0)
    {
      *error = "failed to write output file";
      return false;
    }
#else
    const size_t chunk = (size_t)std::min<uint64_t>(len, (uint64_t)std::numeric_limits<size_t>::max());
    ssize_t written = write(fd, cursor, chunk);
    if (written < 0)
    {
      if (errno == EINTR)
      {
        continue;
      }
      *error = std::string("failed to write output file: ") + std::strerror(errno);
      return false;
    }
#endif
    if (written == 0)
    {
      *error = "failed to write output file";
      return false;
    }
    cursor += written;
    len -= (uint64_t)written;
  }
  return true;
}

static bool write_zero_padding(int fd, uint64_t pad, std::string *error)
{
  static const uint8_t zeros[4096] = {};
  while (pad > 0)
  {
    const uint64_t chunk = std::min<uint64_t>(pad, sizeof(zeros));
    if (!write_all_fd(fd, zeros, chunk, error))
    {
      return false;
    }
    pad -= chunk;
  }
  return true;
}

static uint64_t gguf_pad(uint64_t n, uint64_t alignment)
{
  if (alignment == 0)
  {
    return n;
  }
  return ((n + alignment - 1) / alignment) * alignment;
}

static bool write_pre_tensor_padding(int fd, uint64_t alignment, std::string *error)
{
  uint64_t position = 0;
  if (!fd_current_position(fd, &position, error))
  {
    return false;
  }
  return write_zero_padding(fd, gguf_pad(position, alignment) - position, error);
}

static bool write_post_tensor_padding(int fd, uint64_t tensor_nbytes, uint64_t alignment, std::string *error)
{
  return write_zero_padding(fd, gguf_pad(tensor_nbytes, alignment) - tensor_nbytes, error);
}

static void decode_bf16_to_f32(const uint16_t *src, uint64_t count, float *dst)
{
  uint64_t i = 0;
#if defined(LIBGGUF_NATIVE_SSE2)
  const __m128i zero = _mm_setzero_si128();
  for (; i + 8 <= count; i += 8)
  {
    const __m128i values = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + i));
    const __m128i lo = _mm_unpacklo_epi16(zero, values);
    const __m128i hi = _mm_unpackhi_epi16(zero, values);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + i), lo);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + i + 4), hi);
  }
#endif
  for (; i < count; ++i)
  {
    ggml_bf16_t value;
    value.bits = src[i];
    dst[i] = GGML_BF16_TO_FP32(value);
  }
}

#if defined(LIBGGUF_NATIVE_SSE2)
static inline float q8_0_hmax_ps_native(__m128 v)
{
  __m128 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
  v = _mm_max_ps(v, shuf);
  shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 0, 3, 2));
  v = _mm_max_ps(v, shuf);
  return _mm_cvtss_f32(v);
}

static inline __m128 bf16_load4_to_ps_native(const uint16_t *src)
{
  const __m128i values = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(src));
  const __m128i expanded = _mm_unpacklo_epi16(_mm_setzero_si128(), values);
  return _mm_castsi128_ps(expanded);
}

static inline __m128i q8_0_round_away_from_zero_native(
    __m128 v,
    __m128 id,
    __m128 sign_mask,
    __m128 half,
    __m128 zero)
{
  const __m128 scaled = _mm_mul_ps(v, id);
  const __m128 abs_scaled = _mm_andnot_ps(sign_mask, scaled);
  const __m128 rounded_abs = _mm_add_ps(abs_scaled, half);
  const __m128i abs_i = _mm_cvttps_epi32(rounded_abs);
  const __m128 negative = _mm_cmplt_ps(scaled, zero);
  const __m128i sign_i = _mm_castps_si128(negative);
  return _mm_sub_epi32(_mm_xor_si128(abs_i, sign_i), sign_i);
}
#endif

static void quantize_bf16_row_q8_0_native(const uint16_t *src, block_q8_0 *dst, int64_t k)
{
  assert(k % QK8_0 == 0);
  const int64_t nb = k / QK8_0;

#if defined(LIBGGUF_NATIVE_SSE2)
  static_assert(QK8_0 == 32, "QK8_0 must be 32");
  const __m128 sign_mask = _mm_set1_ps(-0.0f);
  const __m128 half = _mm_set1_ps(0.5f);
  const __m128 zero = _mm_setzero_ps();
  for (int64_t i = 0; i < nb; ++i)
  {
    const uint16_t *xb = src + i * QK8_0;
    __m128 maxv = _mm_setzero_ps();
    for (int j = 0; j < QK8_0; j += 4)
    {
      const __m128 v = bf16_load4_to_ps_native(xb + j);
      maxv = _mm_max_ps(maxv, _mm_andnot_ps(sign_mask, v));
    }

    const float amax = q8_0_hmax_ps_native(maxv);
    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;
    const __m128 idv = _mm_set1_ps(id);

    dst[i].d = GGML_FP32_TO_FP16(d);

    for (int j = 0; j < QK8_0; j += 8)
    {
      const __m128i q0 = q8_0_round_away_from_zero_native(bf16_load4_to_ps_native(xb + j), idv, sign_mask, half, zero);
      const __m128i q1 = q8_0_round_away_from_zero_native(bf16_load4_to_ps_native(xb + j + 4), idv, sign_mask, half, zero);
      const __m128i packed_i16 = _mm_packs_epi32(q0, q1);
      const __m128i packed_i8 = _mm_packs_epi16(packed_i16, _mm_setzero_si128());
      _mm_storel_epi64(reinterpret_cast<__m128i *>(dst[i].qs + j), packed_i8);
    }
  }
#else
  float values[QK8_0];
  for (int64_t i = 0; i < nb; ++i)
  {
    const uint16_t *xb = src + i * QK8_0;
    float amax = 0.0f;
    for (int j = 0; j < QK8_0; ++j)
    {
      ggml_bf16_t value;
      value.bits = xb[j];
      values[j] = GGML_BF16_TO_FP32(value);
      amax = MAX(amax, fabsf(values[j]));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;
    dst[i].d = GGML_FP32_TO_FP16(d);

    for (int j = 0; j < QK8_0; ++j)
    {
      dst[i].qs[j] = roundf(values[j] * id);
    }
  }
#endif
}

static size_t quantize_bf16_q8_0_chunk_native(const uint16_t *src, void *dst, int64_t nrows, int64_t n_per_row)
{
  const size_t row_size = libgguf_row_size(GGML_TYPE_Q8_0, n_per_row);
  const unsigned int nthreads = native_quantize_thread_count(nrows);
  if (nthreads <= 1)
  {
    quantize_bf16_row_q8_0_native(src, reinterpret_cast<block_q8_0 *>(dst), nrows * n_per_row);
    return (size_t)nrows * row_size;
  }

  std::vector<std::thread> threads;
  std::vector<size_t> written(nthreads, 0);
  threads.reserve(nthreads);

  const int64_t rows_per_thread = (nrows + (int64_t)nthreads - 1) / (int64_t)nthreads;
  for (unsigned int thread_id = 0; thread_id < nthreads; ++thread_id)
  {
    const int64_t row_begin = (int64_t)thread_id * rows_per_thread;
    if (row_begin >= nrows)
    {
      written.resize(thread_id);
      break;
    }
    const int64_t row_count = std::min<int64_t>(rows_per_thread, nrows - row_begin);
    threads.emplace_back([=, &written]() {
      quantize_bf16_row_q8_0_native(
          src + row_begin * n_per_row,
          reinterpret_cast<block_q8_0 *>((uint8_t *)dst + (uint64_t)row_begin * row_size),
          row_count * n_per_row);
      written[thread_id] = (size_t)row_count * row_size;
    });
  }

  size_t total = 0;
  for (std::thread &thread : threads)
  {
    thread.join();
  }
  for (size_t count : written)
  {
    total += count;
  }
  return total;
}

static void decode_source_to_f32(const native_tensor_plan &plan, const uint8_t *source, uint64_t row_begin, uint64_t row_count, float *dst);

typedef size_t (*native_quantize_fn)(const float *RESTRICT, void *RESTRICT, int64_t, int64_t, const float *);

static native_quantize_fn native_quantize_function(ggml_type qtype)
{
  switch (qtype)
  {
  case GGML_TYPE_Q1_0:
    return quantize_q1_0;
  case GGML_TYPE_Q4_0:
    return quantize_q4_0;
  case GGML_TYPE_Q4_1:
    return quantize_q4_1;
  case GGML_TYPE_Q5_0:
    return quantize_q5_0;
  case GGML_TYPE_Q5_1:
    return quantize_q5_1;
  case GGML_TYPE_Q8_0:
    return quantize_q8_0;
  case GGML_TYPE_Q2_K:
    return quantize_q2_K;
  case GGML_TYPE_Q3_K:
    return quantize_q3_K;
  case GGML_TYPE_Q4_K:
    return quantize_q4_K;
  case GGML_TYPE_Q5_K:
    return quantize_q5_K;
  case GGML_TYPE_Q6_K:
    return quantize_q6_K;
  case GGML_TYPE_IQ2_XXS:
    return quantize_iq2_xxs;
  case GGML_TYPE_IQ2_XS:
    return quantize_iq2_xs;
  case GGML_TYPE_IQ2_S:
    return quantize_iq2_s;
  case GGML_TYPE_IQ3_XXS:
    return quantize_iq3_xxs;
  case GGML_TYPE_IQ3_S:
    return quantize_iq3_s;
  case GGML_TYPE_IQ1_S:
    return quantize_iq1_s;
  case GGML_TYPE_IQ1_M:
    return quantize_iq1_m;
  case GGML_TYPE_IQ4_NL:
    return quantize_iq4_nl;
  case GGML_TYPE_IQ4_XS:
    return quantize_iq4_xs;
  case GGML_TYPE_TQ1_0:
    return quantize_tq1_0;
  case GGML_TYPE_TQ2_0:
    return quantize_tq2_0;
  case GGML_TYPE_MXFP4:
    return quantize_mxfp4;
  case GGML_TYPE_NVFP4:
    return quantize_nvfp4;
  default:
    return nullptr;
  }
}

static bool quantize_decoded_source_chunk_native(
    const native_tensor_plan &plan,
    const uint8_t *source,
    uint64_t row_begin,
    uint64_t row_count,
    void *dst,
    const float *imatrix,
    std::string *error)
{
  native_quantize_fn quantize = native_quantize_function(plan.qtype);
  if (!quantize)
  {
    *error = "unsupported fused native quantization target for " + plan.key;
    return false;
  }

  const size_t row_size = libgguf_row_size(plan.qtype, (int64_t)plan.n_per_row);
  const unsigned int nthreads = native_quantize_thread_count((int64_t)row_count);
  if (nthreads <= 1)
  {
    std::vector<float> scratch((size_t)(row_count * plan.n_per_row));
    decode_source_to_f32(plan, source, row_begin, row_count, scratch.data());
    const size_t written = quantize(scratch.data(), dst, (int64_t)row_count, (int64_t)plan.n_per_row, imatrix);
    if (written != row_count * (uint64_t)row_size)
    {
      *error = "fused native quantization returned an unexpected byte count for " + plan.key;
      return false;
    }
    return true;
  }

  std::vector<std::thread> threads;
  std::vector<size_t> written(nthreads, 0);
  threads.reserve(nthreads);
  const uint64_t rows_per_thread = (row_count + (uint64_t)nthreads - 1) / (uint64_t)nthreads;
  for (unsigned int thread_id = 0; thread_id < nthreads; ++thread_id)
  {
    const uint64_t local_row_begin = (uint64_t)thread_id * rows_per_thread;
    if (local_row_begin >= row_count)
    {
      written.resize(thread_id);
      break;
    }
    const uint64_t local_row_count = std::min<uint64_t>(rows_per_thread, row_count - local_row_begin);
    threads.emplace_back([=, &written, &plan]() {
      std::vector<float> scratch((size_t)(local_row_count * plan.n_per_row));
      decode_source_to_f32(plan, source, row_begin + local_row_begin, local_row_count, scratch.data());
      written[thread_id] = quantize(
          scratch.data(),
          (uint8_t *)dst + local_row_begin * row_size,
          (int64_t)local_row_count,
          (int64_t)plan.n_per_row,
          imatrix);
    });
  }

  for (std::thread &thread : threads)
  {
    thread.join();
  }
  for (size_t i = 0; i < written.size(); ++i)
  {
    const uint64_t local_row_begin = (uint64_t)i * rows_per_thread;
    const uint64_t local_row_count = std::min<uint64_t>(rows_per_thread, row_count - local_row_begin);
    const uint64_t expected = local_row_count * (uint64_t)row_size;
    if (written[i] != expected)
    {
      *error = "fused native quantization returned an unexpected byte count for " + plan.key;
      return false;
    }
  }
  return true;
}

static void decode_source_to_f32(const native_tensor_plan &plan, const uint8_t *source, uint64_t row_begin, uint64_t row_count, float *dst)
{
  const uint64_t value_begin = row_begin * plan.n_per_row;
  const uint64_t value_count = row_count * plan.n_per_row;
  const uint8_t *src = source + plan.data_begin + value_begin * source_dtype_size(plan.source_dtype);

  switch (plan.source_dtype)
  {
  case NATIVE_DTYPE_F32:
    std::memcpy(dst, src, (size_t)(value_count * sizeof(float)));
    break;
  case NATIVE_DTYPE_F16:
  {
    const ggml_fp16_t *values = (const ggml_fp16_t *)src;
    for (uint64_t i = 0; i < value_count; ++i)
    {
      dst[i] = GGML_FP16_TO_FP32(values[i]);
    }
    break;
  }
  case NATIVE_DTYPE_BF16:
  {
    decode_bf16_to_f32((const uint16_t *)src, value_count, dst);
    break;
  }
  }
}

static const float *aligned_f32_source_rows(const native_tensor_plan &plan, const uint8_t *source, uint64_t row_begin)
{
  if (plan.source_dtype != NATIVE_DTYPE_F32)
  {
    return nullptr;
  }
  const uint64_t value_begin = row_begin * plan.n_per_row;
  const uint8_t *src = source + plan.data_begin + value_begin * sizeof(float);
  if ((reinterpret_cast<uintptr_t>(src) % alignof(float)) != 0)
  {
    return nullptr;
  }
  return reinterpret_cast<const float *>(src);
}

static void accumulate_imatrix_rows(const float *src_rows, uint64_t row_count, uint64_t n_per_row, std::vector<float> &imatrix)
{
  for (uint64_t r = 0; r < row_count; ++r)
  {
    const float *src_row = src_rows + r * n_per_row;
    for (uint64_t c = 0; c < n_per_row; ++c)
    {
      imatrix[(size_t)c] += src_row[c] * src_row[c];
    }
  }
}

static bool compute_auto_imatrix(native_tensor_plan &plan, const uint8_t *source, uint64_t rows_per_chunk, std::vector<float> &scratch, std::string *error)
{
  plan.auto_imatrix.assign((size_t)plan.n_per_row, 0.0f);
  for (uint64_t row = 0; row < plan.n_rows; row += rows_per_chunk)
  {
    const uint64_t row_count = std::min<uint64_t>(rows_per_chunk, plan.n_rows - row);
    const float *src_rows = aligned_f32_source_rows(plan, source, row);
    if (!src_rows)
    {
      decode_source_to_f32(plan, source, row, row_count, scratch.data());
      src_rows = scratch.data();
    }
    accumulate_imatrix_rows(src_rows, row_count, plan.n_per_row, plan.auto_imatrix);
  }
  if (plan.auto_imatrix.empty())
  {
    *error = "failed to compute imatrix";
    return false;
  }
  return true;
}

static bool write_native_plan(
    int fd,
    const native_tensor_plan &plan,
    const uint8_t *source,
    uint64_t alignment,
    uint64_t scratch_bytes,
    const float *imatrix,
    std::string *error)
{
  if (!write_pre_tensor_padding(fd, alignment, error))
  {
    return false;
  }

  if (qtype_matches_source(plan.source_dtype, plan.qtype))
  {
    if (!write_all_fd(fd, source + plan.data_begin, plan.expected_nbytes, error))
    {
      return false;
    }
    return write_post_tensor_padding(fd, plan.expected_nbytes, alignment, error);
  }

  const size_t row_size = libgguf_row_size(plan.qtype, (int64_t)plan.n_per_row);
  if (plan.source_dtype == NATIVE_DTYPE_BF16 && plan.qtype == GGML_TYPE_Q8_0)
  {
    uint64_t rows_per_chunk = scratch_bytes / (uint64_t)row_size;
    if (rows_per_chunk == 0)
    {
      rows_per_chunk = 1;
    }
    rows_per_chunk = std::min<uint64_t>(rows_per_chunk, plan.n_rows);

    std::vector<uint8_t> encoded((size_t)(rows_per_chunk * row_size));
    for (uint64_t row = 0; row < plan.n_rows; row += rows_per_chunk)
    {
      const uint64_t row_count = std::min<uint64_t>(rows_per_chunk, plan.n_rows - row);
      const uint8_t *src = source + plan.data_begin + row * plan.n_per_row * sizeof(uint16_t);
      const size_t written = quantize_bf16_q8_0_chunk_native(
          reinterpret_cast<const uint16_t *>(src),
          encoded.data(),
          (int64_t)row_count,
          (int64_t)plan.n_per_row);
      const uint64_t expected = row_count * (uint64_t)row_size;
      if (written != expected)
      {
        *error = "native BF16 Q8_0 quantization returned an unexpected byte count for " + plan.key;
        return false;
      }
      if (!write_all_fd(fd, encoded.data(), expected, error))
      {
        return false;
      }
    }

    return write_post_tensor_padding(fd, plan.expected_nbytes, alignment, error);
  }

  if ((plan.source_dtype == NATIVE_DTYPE_BF16 || plan.source_dtype == NATIVE_DTYPE_F16) && native_quantize_function(plan.qtype) != nullptr)
  {
    const uint64_t row_work_bytes = plan.n_per_row * sizeof(float) + (uint64_t)row_size;
    uint64_t rows_per_chunk = row_work_bytes == 0 ? 1 : scratch_bytes / row_work_bytes;
    if (rows_per_chunk == 0)
    {
      rows_per_chunk = 1;
    }
    rows_per_chunk = std::min<uint64_t>(rows_per_chunk, plan.n_rows);

    std::vector<uint8_t> encoded((size_t)(rows_per_chunk * row_size));
    for (uint64_t row = 0; row < plan.n_rows; row += rows_per_chunk)
    {
      const uint64_t row_count = std::min<uint64_t>(rows_per_chunk, plan.n_rows - row);
      if (!quantize_decoded_source_chunk_native(plan, source, row, row_count, encoded.data(), imatrix, error))
      {
        return false;
      }
      const uint64_t expected = row_count * (uint64_t)row_size;
      if (!write_all_fd(fd, encoded.data(), expected, error))
      {
        return false;
      }
    }

    return write_post_tensor_padding(fd, plan.expected_nbytes, alignment, error);
  }

  const bool use_source_f32 = aligned_f32_source_rows(plan, source, 0) != nullptr;
  const uint64_t row_work_bytes = (use_source_f32 ? 0 : plan.n_per_row * sizeof(float)) + (uint64_t)row_size;
  uint64_t rows_per_chunk = row_work_bytes == 0 ? 1 : scratch_bytes / row_work_bytes;
  if (rows_per_chunk == 0)
  {
    rows_per_chunk = 1;
  }
  rows_per_chunk = std::min<uint64_t>(rows_per_chunk, plan.n_rows);

  std::vector<float> scratch;
  if (!use_source_f32)
  {
    scratch.resize((size_t)(rows_per_chunk * plan.n_per_row));
  }
  std::vector<uint8_t> encoded((size_t)(rows_per_chunk * row_size));

  for (uint64_t row = 0; row < plan.n_rows; row += rows_per_chunk)
  {
    const uint64_t row_count = std::min<uint64_t>(rows_per_chunk, plan.n_rows - row);
    const float *chunk_src = nullptr;
    if (use_source_f32)
    {
      chunk_src = aligned_f32_source_rows(plan, source, row);
      if (!chunk_src)
      {
        *error = "aligned F32 source became unavailable for " + plan.key;
        return false;
      }
    }
    else
    {
      decode_source_to_f32(plan, source, row, row_count, scratch.data());
      chunk_src = scratch.data();
    }
    const size_t written = libgguf_quantize_chunk(
        plan.qtype,
        chunk_src,
        encoded.data(),
        0,
        (int64_t)row_count,
        (int64_t)plan.n_per_row,
        imatrix);
    const uint64_t expected = row_count * (uint64_t)row_size;
    if (written != expected)
    {
      *error = "libgguf_quantize_chunk returned an unexpected byte count for " + plan.key;
      return false;
    }
    if (!write_all_fd(fd, encoded.data(), expected, error))
    {
      return false;
    }
  }

  return write_post_tensor_padding(fd, plan.expected_nbytes, alignment, error);
}

static bool write_safetensors_payload_native(
    const std::string &path,
    int fd,
    std::vector<native_tensor_plan> &plans,
    uint64_t alignment,
    uint64_t scratch_bytes,
    uint64_t *data_bytes_written,
    std::string *error)
{
  native_mapped_file mapped;
  if (!mapped.map(path, error))
  {
    return false;
  }

  *data_bytes_written = 0;
  for (native_tensor_plan &plan : plans)
  {
    if (plan.data_end > mapped.size)
    {
      *error = "safetensors byte range exceeds file size for " + plan.key;
      return false;
    }

    const size_t row_size = libgguf_row_size(plan.qtype, (int64_t)plan.n_per_row);
    uint64_t row_work_bytes = plan.n_per_row * sizeof(float) + (uint64_t)row_size;
    uint64_t rows_per_chunk = row_work_bytes == 0 ? 1 : scratch_bytes / row_work_bytes;
    if (rows_per_chunk == 0)
    {
      rows_per_chunk = 1;
    }
    rows_per_chunk = std::min<uint64_t>(rows_per_chunk, plan.n_rows);

    const float *imatrix = plan.has_imatrix_view ? (const float *)plan.imatrix_view.buf : nullptr;
    if (libgguf_quantize_requires_imatrix(plan.qtype) && imatrix == nullptr)
    {
      const bool imatrix_uses_source_f32 = aligned_f32_source_rows(plan, mapped.data, 0) != nullptr;
      if (imatrix_uses_source_f32)
      {
        rows_per_chunk = plan.n_rows;
      }
      std::vector<float> decode_scratch;
      if (!imatrix_uses_source_f32)
      {
        decode_scratch.resize((size_t)(rows_per_chunk * plan.n_per_row));
      }
      if (!compute_auto_imatrix(plan, mapped.data, rows_per_chunk, decode_scratch, error))
      {
        return false;
      }
      imatrix = plan.auto_imatrix.data();
    }

    if (!write_native_plan(fd, plan, mapped.data, alignment, scratch_bytes, imatrix, error))
    {
      return false;
    }
    *data_bytes_written += plan.expected_nbytes;
  }
  return true;
}

static PyObject *py_write_safetensors_payload(PyObject *, PyObject *args, PyObject *kwargs)
{
  static const char *kwlist[] = {"path", "fd", "plans", "alignment", "scratch_bytes", nullptr};

  PyObject *path_obj;
  int fd = -1;
  PyObject *plans_obj;
  unsigned long long alignment = 32;
  unsigned long long scratch_bytes = NATIVE_DEFAULT_SCRATCH_BYTES;

  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "OiOK|K",
          (char **)kwlist,
          &path_obj,
          &fd,
          &plans_obj,
          &alignment,
          &scratch_bytes))
  {
    return nullptr;
  }
  if (fd < 0)
  {
    PyErr_SetString(PyExc_ValueError, "fd must be a valid file descriptor");
    return nullptr;
  }
  if (alignment == 0)
  {
    PyErr_SetString(PyExc_ValueError, "alignment must be positive");
    return nullptr;
  }
  if (scratch_bytes == 0)
  {
    PyErr_SetString(PyExc_ValueError, "scratch_bytes must be positive");
    return nullptr;
  }

  const char *path_utf8 = PyUnicode_AsUTF8(path_obj);
  if (!path_utf8)
  {
    return nullptr;
  }
  std::string path(path_utf8);

  PyObject *plans_seq = PySequence_Fast(plans_obj, "plans must be a sequence");
  if (!plans_seq)
  {
    return nullptr;
  }
  const Py_ssize_t plan_count = PySequence_Fast_GET_SIZE(plans_seq);
  std::vector<native_tensor_plan> plans;
  plans.reserve((size_t)plan_count);
  PyObject **items = PySequence_Fast_ITEMS(plans_seq);
  for (Py_ssize_t i = 0; i < plan_count; ++i)
  {
    native_tensor_plan plan;
    if (!parse_native_plan(items[i], &plan))
    {
      Py_DECREF(plans_seq);
      release_native_plan_buffers(plans);
      if (plan.has_imatrix_view)
      {
        PyBuffer_Release(&plan.imatrix_view);
      }
      return nullptr;
    }
    plans.push_back(plan);
  }
  Py_DECREF(plans_seq);

  uint64_t data_bytes_written = 0;
  std::string error;
  bool ok = false;
  Py_BEGIN_ALLOW_THREADS
  ok = write_safetensors_payload_native(
      path,
      fd,
      plans,
      (uint64_t)alignment,
      (uint64_t)scratch_bytes,
      &data_bytes_written,
      &error);
  Py_END_ALLOW_THREADS

  release_native_plan_buffers(plans);

  if (!ok)
  {
    PyErr_SetString(PyExc_RuntimeError, error.c_str());
    return nullptr;
  }

  return PyLong_FromUnsignedLongLong((unsigned long long)data_bytes_written);
}

static PyObject *py_row_size(PyObject *, PyObject *args)
{
  int type;
  long long n_per_row;
  if (!PyArg_ParseTuple(args, "iL", &type, &n_per_row))
  {
    return nullptr;
  }

  const size_t result = libgguf_row_size((ggml_type)type, (int64_t)n_per_row);
  return PyLong_FromSize_t(result);
}

static PyObject *py_type_size(PyObject *, PyObject *args)
{
  int type;
  if (!PyArg_ParseTuple(args, "i", &type))
  {
    return nullptr;
  }

  const size_t result = libgguf_type_size((ggml_type)type);
  return PyLong_FromSize_t(result);
}

static PyObject *py_type_name(PyObject *, PyObject *args)
{
  int type;
  if (!PyArg_ParseTuple(args, "i", &type))
  {
    return nullptr;
  }

  return PyUnicode_FromString(libgguf_type_name((ggml_type)type));
}

static PyObject *py_quantize_requires_imatrix(PyObject *, PyObject *args)
{
  int type;
  if (!PyArg_ParseTuple(args, "i", &type))
  {
    return nullptr;
  }

  if (libgguf_quantize_requires_imatrix((ggml_type)type))
  {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

static PyObject *py_quantize_free(PyObject *, PyObject *)
{
  libgguf_quantize_free();
  Py_RETURN_NONE;
}

static PyObject *py_quantize_rows_raw(PyObject *, PyObject *args, PyObject *kwargs)
{
  static const char *kwlist[] = {"qtype", "src", "n_rows", "n_per_row", "imatrix", nullptr};

  int type;
  PyObject *src_obj;
  PyObject *n_rows_obj;
  PyObject *n_per_row_obj;
  PyObject *imatrix_obj = Py_None;

  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "iOOO|O",
          (char **)kwlist,
          &type,
          &src_obj,
          &n_rows_obj,
          &n_per_row_obj,
          &imatrix_obj))
  {
    return nullptr;
  }

  int64_t n_rows = 0;
  int64_t n_per_row = 0;
  if (!parse_int64(n_rows_obj, &n_rows) || !parse_int64(n_per_row_obj, &n_per_row))
  {
    return nullptr;
  }
  if (n_rows < 0 || n_per_row <= 0)
  {
    PyErr_SetString(PyExc_ValueError, "n_rows must be non-negative and n_per_row must be positive");
    return nullptr;
  }

  Py_buffer src_view;
  if (PyObject_GetBuffer(src_obj, &src_view, PyBUF_CONTIG_RO) < 0)
  {
    return nullptr;
  }

  Py_buffer imatrix_view;
  bool has_imatrix = false;
  const float *imatrix = nullptr;

  if (imatrix_obj != Py_None)
  {
    if (PyObject_GetBuffer(imatrix_obj, &imatrix_view, PyBUF_CONTIG_RO) < 0)
    {
      PyBuffer_Release(&src_view);
      return nullptr;
    }
    has_imatrix = true;
    imatrix = (const float *)imatrix_view.buf;
  }

  const size_t row_size = libgguf_row_size((ggml_type)type, n_per_row);
  if (row_size == 0)
  {
    PyErr_SetString(PyExc_ValueError, "unsupported quantization type or row width");
    if (has_imatrix)
    {
      PyBuffer_Release(&imatrix_view);
    }
    PyBuffer_Release(&src_view);
    return nullptr;
  }

  const uint64_t src_required = (uint64_t)n_rows * (uint64_t)n_per_row * sizeof(float);
  if ((uint64_t)src_view.len < src_required)
  {
    PyErr_SetString(PyExc_ValueError, "src buffer is smaller than n_rows * n_per_row float32 values");
    if (has_imatrix)
    {
      PyBuffer_Release(&imatrix_view);
    }
    PyBuffer_Release(&src_view);
    return nullptr;
  }

  if (libgguf_quantize_requires_imatrix((ggml_type)type))
  {
    if (!has_imatrix)
    {
      PyErr_SetString(PyExc_ValueError, "imatrix is required for this quantization type");
      PyBuffer_Release(&src_view);
      return nullptr;
    }
  }

  if (has_imatrix)
  {
    const uint64_t imatrix_required = (uint64_t)n_per_row * sizeof(float);
    if ((uint64_t)imatrix_view.len < imatrix_required)
    {
      PyErr_SetString(PyExc_ValueError, "imatrix buffer is smaller than n_per_row float32 values");
      PyBuffer_Release(&imatrix_view);
      PyBuffer_Release(&src_view);
      return nullptr;
    }
  }

  const uint64_t out_size_u64 = (uint64_t)n_rows * (uint64_t)row_size;
  if (out_size_u64 > (uint64_t)std::numeric_limits<Py_ssize_t>::max())
  {
    PyErr_SetString(PyExc_OverflowError, "quantized output is too large");
    if (has_imatrix)
    {
      PyBuffer_Release(&imatrix_view);
    }
    PyBuffer_Release(&src_view);
    return nullptr;
  }

  PyObject *out = PyBytes_FromStringAndSize(nullptr, (Py_ssize_t)out_size_u64);
  if (!out)
  {
    if (has_imatrix)
    {
      PyBuffer_Release(&imatrix_view);
    }
    PyBuffer_Release(&src_view);
    return nullptr;
  }

  size_t written = 0;
  Py_BEGIN_ALLOW_THREADS
  written = libgguf_quantize_chunk(
      (ggml_type)type,
      (const float *)src_view.buf,
      PyBytes_AS_STRING(out),
      0,
      n_rows,
      n_per_row,
      imatrix);
  Py_END_ALLOW_THREADS

  if (has_imatrix)
  {
    PyBuffer_Release(&imatrix_view);
  }
  PyBuffer_Release(&src_view);

  if (written != (size_t)out_size_u64)
  {
    Py_DECREF(out);
    PyErr_SetString(PyExc_RuntimeError, "libgguf_quantize_chunk returned an unexpected byte count");
    return nullptr;
  }

  return out;
}

static PyObject *py_quantize_rows_into_raw(PyObject *, PyObject *args, PyObject *kwargs)
{
  static const char *kwlist[] = {"qtype", "src", "dst", "n_rows", "n_per_row", "imatrix", nullptr};

  int type;
  PyObject *src_obj;
  PyObject *dst_obj;
  PyObject *n_rows_obj;
  PyObject *n_per_row_obj;
  PyObject *imatrix_obj = Py_None;

  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "iOOOO|O",
          (char **)kwlist,
          &type,
          &src_obj,
          &dst_obj,
          &n_rows_obj,
          &n_per_row_obj,
          &imatrix_obj))
  {
    return nullptr;
  }

  int64_t n_rows = 0;
  int64_t n_per_row = 0;
  if (!parse_int64(n_rows_obj, &n_rows) || !parse_int64(n_per_row_obj, &n_per_row))
  {
    return nullptr;
  }
  if (n_rows < 0 || n_per_row <= 0)
  {
    PyErr_SetString(PyExc_ValueError, "n_rows must be non-negative and n_per_row must be positive");
    return nullptr;
  }

  Py_buffer src_view;
  if (PyObject_GetBuffer(src_obj, &src_view, PyBUF_CONTIG_RO) < 0)
  {
    return nullptr;
  }

  Py_buffer dst_view;
  if (PyObject_GetBuffer(dst_obj, &dst_view, PyBUF_WRITABLE) < 0)
  {
    PyBuffer_Release(&src_view);
    return nullptr;
  }

  Py_buffer imatrix_view;
  bool has_imatrix = false;
  const float *imatrix = nullptr;

  if (imatrix_obj != Py_None)
  {
    if (PyObject_GetBuffer(imatrix_obj, &imatrix_view, PyBUF_CONTIG_RO) < 0)
    {
      PyBuffer_Release(&dst_view);
      PyBuffer_Release(&src_view);
      return nullptr;
    }
    has_imatrix = true;
    imatrix = (const float *)imatrix_view.buf;
  }

  const size_t row_size = libgguf_row_size((ggml_type)type, n_per_row);
  if (row_size == 0)
  {
    PyErr_SetString(PyExc_ValueError, "unsupported quantization type or row width");
    if (has_imatrix)
    {
      PyBuffer_Release(&imatrix_view);
    }
    PyBuffer_Release(&dst_view);
    PyBuffer_Release(&src_view);
    return nullptr;
  }

  const uint64_t src_required = (uint64_t)n_rows * (uint64_t)n_per_row * sizeof(float);
  if ((uint64_t)src_view.len < src_required)
  {
    PyErr_SetString(PyExc_ValueError, "src buffer is smaller than n_rows * n_per_row float32 values");
    if (has_imatrix)
    {
      PyBuffer_Release(&imatrix_view);
    }
    PyBuffer_Release(&dst_view);
    PyBuffer_Release(&src_view);
    return nullptr;
  }

  const uint64_t out_size_u64 = (uint64_t)n_rows * (uint64_t)row_size;
  if ((uint64_t)dst_view.len < out_size_u64)
  {
    PyErr_SetString(PyExc_ValueError, "dst buffer is smaller than n_rows * row_size bytes");
    if (has_imatrix)
    {
      PyBuffer_Release(&imatrix_view);
    }
    PyBuffer_Release(&dst_view);
    PyBuffer_Release(&src_view);
    return nullptr;
  }

  if (libgguf_quantize_requires_imatrix((ggml_type)type))
  {
    if (!has_imatrix)
    {
      PyErr_SetString(PyExc_ValueError, "imatrix is required for this quantization type");
      PyBuffer_Release(&dst_view);
      PyBuffer_Release(&src_view);
      return nullptr;
    }
  }

  if (has_imatrix)
  {
    const uint64_t imatrix_required = (uint64_t)n_per_row * sizeof(float);
    if ((uint64_t)imatrix_view.len < imatrix_required)
    {
      PyErr_SetString(PyExc_ValueError, "imatrix buffer is smaller than n_per_row float32 values");
      PyBuffer_Release(&imatrix_view);
      PyBuffer_Release(&dst_view);
      PyBuffer_Release(&src_view);
      return nullptr;
    }
  }

  size_t written = 0;
  Py_BEGIN_ALLOW_THREADS
  written = libgguf_quantize_chunk(
      (ggml_type)type,
      (const float *)src_view.buf,
      dst_view.buf,
      0,
      n_rows,
      n_per_row,
      imatrix);
  Py_END_ALLOW_THREADS

  if (has_imatrix)
  {
    PyBuffer_Release(&imatrix_view);
  }
  PyBuffer_Release(&dst_view);
  PyBuffer_Release(&src_view);

  if (written != (size_t)out_size_u64)
  {
    PyErr_SetString(PyExc_RuntimeError, "libgguf_quantize_chunk returned an unexpected byte count");
    return nullptr;
  }

  return PyLong_FromSize_t(written);
}

static PyObject *py_dequantize_rows_raw(PyObject *, PyObject *args, PyObject *kwargs)
{
  static const char *kwlist[] = {"qtype", "src", "n_rows", "n_per_row", nullptr};

  int type;
  PyObject *src_obj;
  PyObject *n_rows_obj;
  PyObject *n_per_row_obj;

  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "iOOO",
          (char **)kwlist,
          &type,
          &src_obj,
          &n_rows_obj,
          &n_per_row_obj))
  {
    return nullptr;
  }

  int64_t n_rows = 0;
  int64_t n_per_row = 0;
  if (!parse_int64(n_rows_obj, &n_rows) || !parse_int64(n_per_row_obj, &n_per_row))
  {
    return nullptr;
  }
  if (n_rows < 0 || n_per_row <= 0)
  {
    PyErr_SetString(PyExc_ValueError, "n_rows must be non-negative and n_per_row must be positive");
    return nullptr;
  }

  Py_buffer src_view;
  if (PyObject_GetBuffer(src_obj, &src_view, PyBUF_CONTIG_RO) < 0)
  {
    return nullptr;
  }

  const size_t row_size = libgguf_row_size((ggml_type)type, n_per_row);
  if (row_size == 0)
  {
    PyErr_SetString(PyExc_ValueError, "unsupported quantization type or row width");
    PyBuffer_Release(&src_view);
    return nullptr;
  }

  const uint64_t src_required = (uint64_t)n_rows * (uint64_t)row_size;
  if ((uint64_t)src_view.len < src_required)
  {
    PyErr_SetString(PyExc_ValueError, "src buffer is smaller than n_rows * row_size bytes");
    PyBuffer_Release(&src_view);
    return nullptr;
  }

  const uint64_t out_size_u64 = (uint64_t)n_rows * (uint64_t)n_per_row * sizeof(float);
  if (out_size_u64 > (uint64_t)std::numeric_limits<Py_ssize_t>::max())
  {
    PyErr_SetString(PyExc_OverflowError, "dequantized output is too large");
    PyBuffer_Release(&src_view);
    return nullptr;
  }

  PyObject *out = PyBytes_FromStringAndSize(nullptr, (Py_ssize_t)out_size_u64);
  if (!out)
  {
    PyBuffer_Release(&src_view);
    return nullptr;
  }

  size_t written = 0;
  Py_BEGIN_ALLOW_THREADS
  written = libgguf_dequantize_chunk(
      (ggml_type)type,
      src_view.buf,
      (float *)PyBytes_AS_STRING(out),
      0,
      n_rows,
      n_per_row);
  Py_END_ALLOW_THREADS
  PyBuffer_Release(&src_view);

  if (written != (size_t)out_size_u64)
  {
    Py_DECREF(out);
    PyErr_SetString(PyExc_RuntimeError, "libgguf_dequantize_chunk returned an unexpected byte count");
    return nullptr;
  }

  return out;
}

static PyObject *py_dequantize_rows_into_raw(PyObject *, PyObject *args, PyObject *kwargs)
{
  static const char *kwlist[] = {"qtype", "src", "dst", "n_rows", "n_per_row", nullptr};

  int type;
  PyObject *src_obj;
  PyObject *dst_obj;
  PyObject *n_rows_obj;
  PyObject *n_per_row_obj;

  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "iOOOO",
          (char **)kwlist,
          &type,
          &src_obj,
          &dst_obj,
          &n_rows_obj,
          &n_per_row_obj))
  {
    return nullptr;
  }

  int64_t n_rows = 0;
  int64_t n_per_row = 0;
  if (!parse_int64(n_rows_obj, &n_rows) || !parse_int64(n_per_row_obj, &n_per_row))
  {
    return nullptr;
  }
  if (n_rows < 0 || n_per_row <= 0)
  {
    PyErr_SetString(PyExc_ValueError, "n_rows must be non-negative and n_per_row must be positive");
    return nullptr;
  }

  Py_buffer src_view;
  if (PyObject_GetBuffer(src_obj, &src_view, PyBUF_CONTIG_RO) < 0)
  {
    return nullptr;
  }

  Py_buffer dst_view;
  if (PyObject_GetBuffer(dst_obj, &dst_view, PyBUF_WRITABLE) < 0)
  {
    PyBuffer_Release(&src_view);
    return nullptr;
  }

  const size_t row_size = libgguf_row_size((ggml_type)type, n_per_row);
  if (row_size == 0)
  {
    PyErr_SetString(PyExc_ValueError, "unsupported quantization type or row width");
    PyBuffer_Release(&dst_view);
    PyBuffer_Release(&src_view);
    return nullptr;
  }

  const uint64_t src_required = (uint64_t)n_rows * (uint64_t)row_size;
  if ((uint64_t)src_view.len < src_required)
  {
    PyErr_SetString(PyExc_ValueError, "src buffer is smaller than n_rows * row_size bytes");
    PyBuffer_Release(&dst_view);
    PyBuffer_Release(&src_view);
    return nullptr;
  }

  const uint64_t out_size_u64 = (uint64_t)n_rows * (uint64_t)n_per_row * sizeof(float);
  if ((uint64_t)dst_view.len < out_size_u64)
  {
    PyErr_SetString(PyExc_ValueError, "dst buffer is smaller than n_rows * n_per_row float32 values");
    PyBuffer_Release(&dst_view);
    PyBuffer_Release(&src_view);
    return nullptr;
  }

  size_t written = 0;
  Py_BEGIN_ALLOW_THREADS
  written = libgguf_dequantize_chunk(
      (ggml_type)type,
      src_view.buf,
      (float *)dst_view.buf,
      0,
      n_rows,
      n_per_row);
  Py_END_ALLOW_THREADS

  PyBuffer_Release(&dst_view);
  PyBuffer_Release(&src_view);

  if (written != (size_t)out_size_u64)
  {
    PyErr_SetString(PyExc_RuntimeError, "libgguf_dequantize_chunk returned an unexpected byte count");
    return nullptr;
  }

  return PyLong_FromSize_t(written);
}

static PyObject *py_q4_0_backend(PyObject *, PyObject *)
{
  return PyUnicode_FromString(libgguf_q4_0_backend());
}

static PyObject *py_dequant_backend(PyObject *, PyObject *args)
{
  int type = 0;
  if (!PyArg_ParseTuple(args, "i", &type))
    return nullptr;

  return PyUnicode_FromString(libgguf_dequant_backend(type));
}

static PyObject *py_dequant_cpu_supports_backend(PyObject *, PyObject *args)
{
  const char *backend = nullptr;
  if (!PyArg_ParseTuple(args, "s", &backend))
    return nullptr;

  if (libgguf_dequant_cpu_supports_backend(backend))
    Py_RETURN_TRUE;
  Py_RETURN_FALSE;
}

static PyObject *py_dequantize_for_backend(PyObject *, PyObject *args)
{
  int type = 0;
  const char *backend = nullptr;
  PyObject *src_obj = nullptr;
  PyObject *n_rows_obj = nullptr;
  PyObject *n_per_row_obj = nullptr;

  if (!PyArg_ParseTuple(args, "isOOO", &type, &backend, &src_obj, &n_rows_obj, &n_per_row_obj))
    return nullptr;

  int64_t nrows = PyLong_AsLongLong(n_rows_obj);
  if (nrows == -1 && PyErr_Occurred())
    return nullptr;
  int64_t n_per_row = PyLong_AsLongLong(n_per_row_obj);
  if (n_per_row == -1 && PyErr_Occurred())
    return nullptr;

  if (nrows < 0 || n_per_row < 0)
  {
    PyErr_SetString(PyExc_ValueError, "nrows and n_per_row must be non-negative");
    return nullptr;
  }

  Py_buffer src_view;
  if (PyObject_GetBuffer(src_obj, &src_view, PyBUF_CONTIG_RO) != 0)
    return nullptr;

  const size_t row_size = libgguf_row_size((ggml_type)type, n_per_row);
  const size_t expected_src = row_size * (size_t)nrows;
  if (row_size == 0)
  {
    PyBuffer_Release(&src_view);
    PyErr_SetString(PyExc_ValueError, "unsupported or invalid quantization type");
    return nullptr;
  }
  if ((size_t)src_view.len < expected_src)
  {
    PyBuffer_Release(&src_view);
    PyErr_SetString(PyExc_ValueError, "src buffer is smaller than required");
    return nullptr;
  }

  if (nrows != 0 && n_per_row > INT64_MAX / nrows)
  {
    PyBuffer_Release(&src_view);
    PyErr_SetString(PyExc_OverflowError, "dequantized output is too large");
    return nullptr;
  }
  const size_t values = (size_t)nrows * (size_t)n_per_row;
  if (values > PY_SSIZE_T_MAX / sizeof(float))
  {
    PyBuffer_Release(&src_view);
    PyErr_SetString(PyExc_OverflowError, "dequantized output is too large");
    return nullptr;
  }

  PyObject *out = PyBytes_FromStringAndSize(nullptr, (Py_ssize_t)(values * sizeof(float)));
  if (out == nullptr)
  {
    PyBuffer_Release(&src_view);
    return nullptr;
  }

  const size_t written = libgguf_dequantize_for_backend(
      type,
      backend,
      src_view.buf,
      (float *)PyBytes_AS_STRING(out),
      nrows,
      n_per_row);
  PyBuffer_Release(&src_view);

  if (written == 0 && values != 0)
  {
    Py_DECREF(out);
    PyErr_SetString(PyExc_ValueError, "unsupported dequant backend for this CPU");
    return nullptr;
  }
  if (written != values * sizeof(float))
  {
    Py_DECREF(out);
    PyErr_SetString(PyExc_RuntimeError, "libgguf_dequantize_for_backend returned an unexpected byte count");
    return nullptr;
  }

  return out;
}

static PyObject *py_q4_0_cpu_supports_backend(PyObject *, PyObject *args)
{
  const char *backend = nullptr;
  if (!PyArg_ParseTuple(args, "s", &backend))
  {
    return nullptr;
  }
  if (libgguf_q4_0_cpu_supports_backend(backend))
  {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

static PyObject *py_quantize_q4_0_for_backend(PyObject *, PyObject *args)
{
  return py_quantize_for_backend(args, GGML_TYPE_Q4_0, "unsupported Q4_0 backend for this CPU", libgguf_quantize_q4_0_for_backend);
}

static PyObject *py_q8_0_backend(PyObject *, PyObject *)
{
  return PyUnicode_FromString(libgguf_q8_0_backend());
}

static PyObject *py_q8_0_cpu_supports_backend(PyObject *, PyObject *args)
{
  const char *backend = nullptr;
  if (!PyArg_ParseTuple(args, "s", &backend))
  {
    return nullptr;
  }
  if (libgguf_q8_0_cpu_supports_backend(backend))
  {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

static PyObject *py_quantize_q8_0_for_backend(PyObject *, PyObject *args)
{
  return py_quantize_for_backend(args, GGML_TYPE_Q8_0, "unsupported Q8_0 backend for this CPU", libgguf_quantize_q8_0_for_backend);
}

static PyObject *py_cpu_supports_quant_backend(PyObject *args, int (*supports_backend)(const char *))
{
  const char *backend = nullptr;
  if (!PyArg_ParseTuple(args, "s", &backend))
  {
    return nullptr;
  }
  if (supports_backend(backend))
  {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

static PyObject *py_q1_0_backend(PyObject *, PyObject *)
{
  return PyUnicode_FromString(libgguf_q1_0_backend());
}

static PyObject *py_q1_0_cpu_supports_backend(PyObject *, PyObject *args)
{
  return py_cpu_supports_quant_backend(args, libgguf_q1_0_cpu_supports_backend);
}

static PyObject *py_quantize_q1_0_for_backend(PyObject *, PyObject *args)
{
  return py_quantize_for_backend(args, GGML_TYPE_Q1_0, "unsupported Q1_0 backend for this CPU", libgguf_quantize_q1_0_for_backend);
}

static PyObject *py_q4_1_backend(PyObject *, PyObject *)
{
  return PyUnicode_FromString(libgguf_q4_1_backend());
}

static PyObject *py_q4_1_cpu_supports_backend(PyObject *, PyObject *args)
{
  return py_cpu_supports_quant_backend(args, libgguf_q4_1_cpu_supports_backend);
}

static PyObject *py_quantize_q4_1_for_backend(PyObject *, PyObject *args)
{
  return py_quantize_for_backend(args, GGML_TYPE_Q4_1, "unsupported Q4_1 backend for this CPU", libgguf_quantize_q4_1_for_backend);
}

static PyObject *py_q5_0_backend(PyObject *, PyObject *)
{
  return PyUnicode_FromString(libgguf_q5_0_backend());
}

static PyObject *py_q5_0_cpu_supports_backend(PyObject *, PyObject *args)
{
  return py_cpu_supports_quant_backend(args, libgguf_q5_0_cpu_supports_backend);
}

static PyObject *py_quantize_q5_0_for_backend(PyObject *, PyObject *args)
{
  return py_quantize_for_backend(args, GGML_TYPE_Q5_0, "unsupported Q5_0 backend for this CPU", libgguf_quantize_q5_0_for_backend);
}

static PyObject *py_q5_1_backend(PyObject *, PyObject *)
{
  return PyUnicode_FromString(libgguf_q5_1_backend());
}

static PyObject *py_q5_1_cpu_supports_backend(PyObject *, PyObject *args)
{
  return py_cpu_supports_quant_backend(args, libgguf_q5_1_cpu_supports_backend);
}

static PyObject *py_quantize_q5_1_for_backend(PyObject *, PyObject *args)
{
  return py_quantize_for_backend(args, GGML_TYPE_Q5_1, "unsupported Q5_1 backend for this CPU", libgguf_quantize_q5_1_for_backend);
}

static PyObject *py_mxfp4_backend(PyObject *, PyObject *)
{
  return PyUnicode_FromString(libgguf_mxfp4_backend());
}

static PyObject *py_mxfp4_cpu_supports_backend(PyObject *, PyObject *args)
{
  return py_cpu_supports_quant_backend(args, libgguf_mxfp4_cpu_supports_backend);
}

static PyObject *py_quantize_mxfp4_for_backend(PyObject *, PyObject *args)
{
  return py_quantize_for_backend(args, GGML_TYPE_MXFP4, "unsupported MXFP4 backend for this CPU", libgguf_quantize_mxfp4_for_backend);
}

static PyObject *py_nvfp4_backend(PyObject *, PyObject *)
{
  return PyUnicode_FromString(libgguf_nvfp4_backend());
}

static PyObject *py_nvfp4_cpu_supports_backend(PyObject *, PyObject *args)
{
  return py_cpu_supports_quant_backend(args, libgguf_nvfp4_cpu_supports_backend);
}

static PyObject *py_quantize_nvfp4_for_backend(PyObject *, PyObject *args)
{
  return py_quantize_for_backend(args, GGML_TYPE_NVFP4, "unsupported NVFP4 backend for this CPU", libgguf_quantize_nvfp4_for_backend);
}

static PyObject *py_q2_k_backend(PyObject *, PyObject *)
{
  return PyUnicode_FromString(libgguf_q2_k_backend());
}

static PyObject *py_q2_k_cpu_supports_backend(PyObject *, PyObject *args)
{
  return py_cpu_supports_quant_backend(args, libgguf_q2_k_cpu_supports_backend);
}

static PyObject *py_quantize_q2_k_for_backend(PyObject *, PyObject *args)
{
  return py_quantize_for_backend(args, GGML_TYPE_Q2_K, "unsupported Q2_K backend for this CPU", libgguf_quantize_q2_k_for_backend);
}

static PyObject *py_q3_k_backend(PyObject *, PyObject *)
{
  return PyUnicode_FromString(libgguf_q3_k_backend());
}

static PyObject *py_q3_k_cpu_supports_backend(PyObject *, PyObject *args)
{
  return py_cpu_supports_quant_backend(args, libgguf_q3_k_cpu_supports_backend);
}

static PyObject *py_quantize_q3_k_for_backend(PyObject *, PyObject *args)
{
  return py_quantize_for_backend(args, GGML_TYPE_Q3_K, "unsupported Q3_K backend for this CPU", libgguf_quantize_q3_k_for_backend);
}

static PyObject *py_q4_k_backend(PyObject *, PyObject *)
{
  return PyUnicode_FromString(libgguf_q4_k_backend());
}

static PyObject *py_q4_k_cpu_supports_backend(PyObject *, PyObject *args)
{
  const char *backend = nullptr;
  if (!PyArg_ParseTuple(args, "s", &backend))
  {
    return nullptr;
  }
  if (libgguf_q4_k_cpu_supports_backend(backend))
  {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

static PyObject *py_quantize_q4_k_for_backend(PyObject *, PyObject *args)
{
  return py_quantize_for_backend(args, GGML_TYPE_Q4_K, "unsupported Q4_K backend for this CPU", libgguf_quantize_q4_k_for_backend);
}

static PyObject *py_q5_k_backend(PyObject *, PyObject *)
{
  return PyUnicode_FromString(libgguf_q5_k_backend());
}

static PyObject *py_q5_k_cpu_supports_backend(PyObject *, PyObject *args)
{
  return py_cpu_supports_quant_backend(args, libgguf_q5_k_cpu_supports_backend);
}

static PyObject *py_quantize_q5_k_for_backend(PyObject *, PyObject *args)
{
  return py_quantize_for_backend(args, GGML_TYPE_Q5_K, "unsupported Q5_K backend for this CPU", libgguf_quantize_q5_k_for_backend);
}

static PyObject *py_q6_k_backend(PyObject *, PyObject *)
{
  return PyUnicode_FromString(libgguf_q6_k_backend());
}

static PyObject *py_q6_k_cpu_supports_backend(PyObject *, PyObject *args)
{
  return py_cpu_supports_quant_backend(args, libgguf_q6_k_cpu_supports_backend);
}

static PyObject *py_quantize_q6_k_for_backend(PyObject *, PyObject *args)
{
  return py_quantize_for_backend(args, GGML_TYPE_Q6_K, "unsupported Q6_K backend for this CPU", libgguf_quantize_q6_k_for_backend);
}

static PyObject *py_iq4_nl_cpu_supports_backend(PyObject *, PyObject *args)
{
  const char *backend = nullptr;
  if (!PyArg_ParseTuple(args, "s", &backend))
  {
    return nullptr;
  }
  if (libgguf_iq4_nl_cpu_supports_backend(backend))
  {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

static PyObject *py_quantize_iq4_nl_for_backend(PyObject *, PyObject *args)
{
  return py_quantize_for_backend(args, GGML_TYPE_IQ4_NL, "unsupported IQ4_NL backend for this CPU", libgguf_quantize_iq4_nl_for_backend);
}

static PyObject *py_iq4_xs_cpu_supports_backend(PyObject *, PyObject *args)
{
  return py_cpu_supports_quant_backend(args, libgguf_iq4_xs_cpu_supports_backend);
}

static PyObject *py_quantize_iq4_xs_for_backend(PyObject *, PyObject *args)
{
  return py_quantize_for_backend(args, GGML_TYPE_IQ4_XS, "unsupported IQ4_XS backend for this CPU", libgguf_quantize_iq4_xs_for_backend);
}

static PyObject *py_tq1_0_backend(PyObject *, PyObject *)
{
  return PyUnicode_FromString(libgguf_tq1_0_backend());
}

static PyObject *py_tq1_0_cpu_supports_backend(PyObject *, PyObject *args)
{
  return py_cpu_supports_quant_backend(args, libgguf_tq1_0_cpu_supports_backend);
}

static PyObject *py_quantize_tq1_0_for_backend(PyObject *, PyObject *args)
{
  return py_quantize_for_backend(args, GGML_TYPE_TQ1_0, "unsupported TQ1_0 backend for this CPU", libgguf_quantize_tq1_0_for_backend);
}

static PyObject *py_tq2_0_backend(PyObject *, PyObject *)
{
  return PyUnicode_FromString(libgguf_tq2_0_backend());
}

static PyObject *py_tq2_0_cpu_supports_backend(PyObject *, PyObject *args)
{
  const char *backend = nullptr;
  if (!PyArg_ParseTuple(args, "s", &backend))
  {
    return nullptr;
  }
  if (libgguf_tq2_0_cpu_supports_backend(backend))
  {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

static PyObject *py_quantize_tq2_0_for_backend(PyObject *, PyObject *args)
{
  return py_quantize_for_backend(args, GGML_TYPE_TQ2_0, "unsupported TQ2_0 backend for this CPU", libgguf_quantize_tq2_0_for_backend);
}

static PyObject *py_common_quant_backend(PyObject *, PyObject *)
{
  return PyUnicode_FromString(libgguf_common_quant_backend());
}

static PyObject *py_common_quant_cpu_supports_backend(PyObject *, PyObject *args)
{
  const char *backend = nullptr;
  if (!PyArg_ParseTuple(args, "s", &backend))
  {
    return nullptr;
  }
  if (libgguf_common_quant_cpu_supports_backend(backend))
  {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

static PyObject *py_common_quant_set_backend(PyObject *, PyObject *args)
{
  const char *backend = nullptr;
  if (!PyArg_ParseTuple(args, "s", &backend))
  {
    return nullptr;
  }
  if (!libgguf_common_quant_set_backend(backend))
  {
    PyErr_SetString(PyExc_ValueError, "unsupported common quant backend for this CPU");
    return nullptr;
  }
  Py_RETURN_NONE;
}

static PyObject *py_common_quant_probe_for_backend(PyObject *, PyObject *args)
{
  const char *backend = nullptr;
  if (!PyArg_ParseTuple(args, "s", &backend))
  {
    return nullptr;
  }
  const uint64_t probe = libgguf_common_quant_probe_for_backend(backend);
  if (probe == 0)
  {
    PyErr_SetString(PyExc_ValueError, "unsupported common quant backend for this CPU");
    return nullptr;
  }
  return PyLong_FromUnsignedLongLong((unsigned long long)probe);
}

static PyObject *py_storage_backend(PyObject *, PyObject *)
{
  return PyUnicode_FromString(libgguf_storage_backend());
}

static PyObject *py_storage_cpu_supports_backend(PyObject *, PyObject *args)
{
  const char *backend = nullptr;
  if (!PyArg_ParseTuple(args, "s", &backend))
  {
    return nullptr;
  }
  if (libgguf_storage_cpu_supports_backend(backend))
  {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

static PyObject *py_storage_set_backend(PyObject *, PyObject *args)
{
  const char *backend = nullptr;
  if (!PyArg_ParseTuple(args, "s", &backend))
  {
    return nullptr;
  }
  if (!libgguf_storage_set_backend(backend))
  {
    PyErr_SetString(PyExc_ValueError, "unsupported storage backend for this CPU");
    return nullptr;
  }
  Py_RETURN_NONE;
}

static PyMethodDef module_methods[] = {
    {"row_size", py_row_size, METH_VARARGS, "Return encoded row byte size for a quantization type and row width."},
    {"type_size", py_type_size, METH_VARARGS, "Return GGUF block type byte size."},
    {"type_name", py_type_name, METH_VARARGS, "Return GGUF type name."},
    {"quantize_requires_imatrix", py_quantize_requires_imatrix, METH_VARARGS, "Return whether quantization requires an imatrix."},
    {"quantize_free", py_quantize_free, METH_NOARGS, "Free lazily initialized quantization tables."},
    {"quantize_rows_raw", (PyCFunction)py_quantize_rows_raw, METH_VARARGS | METH_KEYWORDS, "Quantize float32 rows from a contiguous buffer and return bytes."},
    {"quantize_rows_into_raw", (PyCFunction)py_quantize_rows_into_raw, METH_VARARGS | METH_KEYWORDS, "Quantize float32 rows into an existing writable buffer."},
    {"dequantize_rows_raw", (PyCFunction)py_dequantize_rows_raw, METH_VARARGS | METH_KEYWORDS, "Dequantize rows from a contiguous buffer and return float32 bytes."},
    {"dequantize_rows_into_raw", (PyCFunction)py_dequantize_rows_into_raw, METH_VARARGS | METH_KEYWORDS, "Dequantize rows into an existing writable float32 buffer."},
    {"write_safetensors_payload", (PyCFunction)py_write_safetensors_payload, METH_VARARGS | METH_KEYWORDS, "Stream safetensors tensor payloads directly into an open GGUF file descriptor."},
    {"_dequant_backend", py_dequant_backend, METH_VARARGS, "Return selected dequant backend for a qtype."},
    {"_dequant_cpu_supports_backend", py_dequant_cpu_supports_backend, METH_VARARGS, "Return whether the CPU supports a dequant backend."},
    {"_dequantize_for_backend", py_dequantize_for_backend, METH_VARARGS, "Dequantize with a selected backend for tests."},
    {"_q4_0_backend", py_q4_0_backend, METH_NOARGS, "Return selected Q4_0 backend."},
    {"_q4_0_cpu_supports_backend", py_q4_0_cpu_supports_backend, METH_VARARGS, "Return whether the CPU supports a Q4_0 backend."},
    {"_quantize_q4_0_for_backend", py_quantize_q4_0_for_backend, METH_VARARGS, "Quantize Q4_0 with a selected backend for tests."},
    {"_q8_0_backend", py_q8_0_backend, METH_NOARGS, "Return selected Q8_0 backend."},
    {"_q8_0_cpu_supports_backend", py_q8_0_cpu_supports_backend, METH_VARARGS, "Return whether the CPU supports a Q8_0 backend."},
    {"_quantize_q8_0_for_backend", py_quantize_q8_0_for_backend, METH_VARARGS, "Quantize Q8_0 with a selected backend for tests."},
    {"_q1_0_backend", py_q1_0_backend, METH_NOARGS, "Return selected Q1_0 backend."},
    {"_q1_0_cpu_supports_backend", py_q1_0_cpu_supports_backend, METH_VARARGS, "Return whether the CPU supports a Q1_0 backend."},
    {"_quantize_q1_0_for_backend", py_quantize_q1_0_for_backend, METH_VARARGS, "Quantize Q1_0 with a selected backend for tests."},
    {"_q4_1_backend", py_q4_1_backend, METH_NOARGS, "Return selected Q4_1 backend."},
    {"_q4_1_cpu_supports_backend", py_q4_1_cpu_supports_backend, METH_VARARGS, "Return whether the CPU supports a Q4_1 backend."},
    {"_quantize_q4_1_for_backend", py_quantize_q4_1_for_backend, METH_VARARGS, "Quantize Q4_1 with a selected backend for tests."},
    {"_q5_0_backend", py_q5_0_backend, METH_NOARGS, "Return selected Q5_0 backend."},
    {"_q5_0_cpu_supports_backend", py_q5_0_cpu_supports_backend, METH_VARARGS, "Return whether the CPU supports a Q5_0 backend."},
    {"_quantize_q5_0_for_backend", py_quantize_q5_0_for_backend, METH_VARARGS, "Quantize Q5_0 with a selected backend for tests."},
    {"_q5_1_backend", py_q5_1_backend, METH_NOARGS, "Return selected Q5_1 backend."},
    {"_q5_1_cpu_supports_backend", py_q5_1_cpu_supports_backend, METH_VARARGS, "Return whether the CPU supports a Q5_1 backend."},
    {"_quantize_q5_1_for_backend", py_quantize_q5_1_for_backend, METH_VARARGS, "Quantize Q5_1 with a selected backend for tests."},
    {"_mxfp4_backend", py_mxfp4_backend, METH_NOARGS, "Return selected MXFP4 backend."},
    {"_mxfp4_cpu_supports_backend", py_mxfp4_cpu_supports_backend, METH_VARARGS, "Return whether the CPU supports an MXFP4 backend."},
    {"_quantize_mxfp4_for_backend", py_quantize_mxfp4_for_backend, METH_VARARGS, "Quantize MXFP4 with a selected backend for tests."},
    {"_nvfp4_backend", py_nvfp4_backend, METH_NOARGS, "Return selected NVFP4 backend."},
    {"_nvfp4_cpu_supports_backend", py_nvfp4_cpu_supports_backend, METH_VARARGS, "Return whether the CPU supports an NVFP4 backend."},
    {"_quantize_nvfp4_for_backend", py_quantize_nvfp4_for_backend, METH_VARARGS, "Quantize NVFP4 with a selected backend for tests."},
    {"_q2_k_backend", py_q2_k_backend, METH_NOARGS, "Return selected Q2_K backend."},
    {"_q2_k_cpu_supports_backend", py_q2_k_cpu_supports_backend, METH_VARARGS, "Return whether the CPU supports a Q2_K backend."},
    {"_quantize_q2_k_for_backend", py_quantize_q2_k_for_backend, METH_VARARGS, "Quantize Q2_K with a selected backend for tests."},
    {"_q3_k_backend", py_q3_k_backend, METH_NOARGS, "Return selected Q3_K backend."},
    {"_q3_k_cpu_supports_backend", py_q3_k_cpu_supports_backend, METH_VARARGS, "Return whether the CPU supports a Q3_K backend."},
    {"_quantize_q3_k_for_backend", py_quantize_q3_k_for_backend, METH_VARARGS, "Quantize Q3_K with a selected backend for tests."},
    {"_q4_k_backend", py_q4_k_backend, METH_NOARGS, "Return selected Q4_K backend."},
    {"_q4_k_cpu_supports_backend", py_q4_k_cpu_supports_backend, METH_VARARGS, "Return whether the CPU supports a Q4_K backend."},
    {"_quantize_q4_k_for_backend", py_quantize_q4_k_for_backend, METH_VARARGS, "Quantize Q4_K with a selected backend for tests."},
    {"_q5_k_backend", py_q5_k_backend, METH_NOARGS, "Return selected Q5_K backend."},
    {"_q5_k_cpu_supports_backend", py_q5_k_cpu_supports_backend, METH_VARARGS, "Return whether the CPU supports a Q5_K backend."},
    {"_quantize_q5_k_for_backend", py_quantize_q5_k_for_backend, METH_VARARGS, "Quantize Q5_K with a selected backend for tests."},
    {"_q6_k_backend", py_q6_k_backend, METH_NOARGS, "Return selected Q6_K backend."},
    {"_q6_k_cpu_supports_backend", py_q6_k_cpu_supports_backend, METH_VARARGS, "Return whether the CPU supports a Q6_K backend."},
    {"_quantize_q6_k_for_backend", py_quantize_q6_k_for_backend, METH_VARARGS, "Quantize Q6_K with a selected backend for tests."},
    {"_iq4_nl_cpu_supports_backend", py_iq4_nl_cpu_supports_backend, METH_VARARGS, "Return whether the CPU supports an IQ4_NL backend."},
    {"_quantize_iq4_nl_for_backend", py_quantize_iq4_nl_for_backend, METH_VARARGS, "Quantize IQ4_NL with a selected backend for tests."},
    {"_iq4_xs_cpu_supports_backend", py_iq4_xs_cpu_supports_backend, METH_VARARGS, "Return whether the CPU supports an IQ4_XS backend."},
    {"_quantize_iq4_xs_for_backend", py_quantize_iq4_xs_for_backend, METH_VARARGS, "Quantize IQ4_XS with a selected backend for tests."},
    {"_tq1_0_backend", py_tq1_0_backend, METH_NOARGS, "Return selected TQ1_0 backend."},
    {"_tq1_0_cpu_supports_backend", py_tq1_0_cpu_supports_backend, METH_VARARGS, "Return whether the CPU supports a TQ1_0 backend."},
    {"_quantize_tq1_0_for_backend", py_quantize_tq1_0_for_backend, METH_VARARGS, "Quantize TQ1_0 with a selected backend for tests."},
    {"_tq2_0_backend", py_tq2_0_backend, METH_NOARGS, "Return selected TQ2_0 backend."},
    {"_tq2_0_cpu_supports_backend", py_tq2_0_cpu_supports_backend, METH_VARARGS, "Return whether the CPU supports a TQ2_0 backend."},
    {"_quantize_tq2_0_for_backend", py_quantize_tq2_0_for_backend, METH_VARARGS, "Quantize TQ2_0 with a selected backend for tests."},
    {"_common_quant_backend", py_common_quant_backend, METH_NOARGS, "Return selected common quant helper backend."},
    {"_common_quant_cpu_supports_backend", py_common_quant_cpu_supports_backend, METH_VARARGS, "Return whether the CPU supports a common quant helper backend."},
    {"_common_quant_set_backend", py_common_quant_set_backend, METH_VARARGS, "Select a common quant helper backend for tests."},
    {"_common_quant_probe_for_backend", py_common_quant_probe_for_backend, METH_VARARGS, "Return a parity probe hash for common quant helpers."},
    {"_storage_backend", py_storage_backend, METH_NOARGS, "Return selected storage conversion backend."},
    {"_storage_cpu_supports_backend", py_storage_cpu_supports_backend, METH_VARARGS, "Return whether the CPU supports a storage conversion backend."},
    {"_storage_set_backend", py_storage_set_backend, METH_VARARGS, "Select a storage conversion backend for tests."},
    {nullptr, nullptr, 0, nullptr},
};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "_libgguf",
    "CPython bindings for the vendored libgguf reference quantizers.",
    -1,
    module_methods,
};

PyMODINIT_FUNC PyInit__libgguf(void)
{
  return PyModule_Create(&module_def);
}
