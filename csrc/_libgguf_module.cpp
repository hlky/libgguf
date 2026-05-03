#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstdint>
#include <limits>

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
