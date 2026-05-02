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
    {"_q4_0_backend", py_q4_0_backend, METH_NOARGS, "Return selected Q4_0 backend."},
    {"_q4_0_cpu_supports_backend", py_q4_0_cpu_supports_backend, METH_VARARGS, "Return whether the CPU supports a Q4_0 backend."},
    {"_quantize_q4_0_for_backend", py_quantize_q4_0_for_backend, METH_VARARGS, "Quantize Q4_0 with a selected backend for tests."},
    {"_q8_0_backend", py_q8_0_backend, METH_NOARGS, "Return selected Q8_0 backend."},
    {"_q8_0_cpu_supports_backend", py_q8_0_cpu_supports_backend, METH_VARARGS, "Return whether the CPU supports a Q8_0 backend."},
    {"_quantize_q8_0_for_backend", py_quantize_q8_0_for_backend, METH_VARARGS, "Quantize Q8_0 with a selected backend for tests."},
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
