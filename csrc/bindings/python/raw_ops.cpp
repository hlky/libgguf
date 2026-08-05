#include "bindings/python/ops.h"

#include <libgguf/libgguf.h>

#include <cstdint>
#include <limits>
static bool parse_int64(PyObject *value, int64_t *out);

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


PyObject * py_row_size(PyObject *, PyObject *args)
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

PyObject * py_type_size(PyObject *, PyObject *args)
{
  int type;
  if (!PyArg_ParseTuple(args, "i", &type))
  {
    return nullptr;
  }

  const size_t result = libgguf_type_size((ggml_type)type);
  return PyLong_FromSize_t(result);
}

PyObject * py_type_name(PyObject *, PyObject *args)
{
  int type;
  if (!PyArg_ParseTuple(args, "i", &type))
  {
    return nullptr;
  }

  return PyUnicode_FromString(libgguf_type_name((ggml_type)type));
}

PyObject * py_quantize_requires_imatrix(PyObject *, PyObject *args)
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

PyObject * py_quantize_free(PyObject *, PyObject *)
{
  libgguf_quantize_free();
  Py_RETURN_NONE;
}

PyObject * py_quantize_rows_raw(PyObject *, PyObject *args, PyObject *kwargs)
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

PyObject * py_quantize_rows_into_raw(PyObject *, PyObject *args, PyObject *kwargs)
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

PyObject * py_dequantize_rows_raw(PyObject *, PyObject *args, PyObject *kwargs)
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

PyObject * py_dequantize_rows_into_raw(PyObject *, PyObject *args, PyObject *kwargs)
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

