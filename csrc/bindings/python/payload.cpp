#include "bindings/python/ops.h"

#include "conversion/safetensors_payload.h"
#include <libgguf/libgguf.h>

#include <cstdint>
#include <limits>
#include <string>
#include <vector>
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


static bool parse_native_plan(PyObject *item, safetensors_payload_plan *plan, Py_buffer *imatrix_view, bool *has_imatrix_view)
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
  if (!parse_payload_source_dtype(dtype, &plan->source_dtype))
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
  if (!checked_mul_u64(plan->n_values, (uint64_t)payload_source_dtype_size(plan->source_dtype), &expected_source_bytes))
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
      if (PyObject_GetBuffer(imatrix_obj, imatrix_view, PyBUF_CONTIG_RO) < 0)
      {
        Py_DECREF(imatrix_obj);
        return false;
      }
      *has_imatrix_view = true;
      plan->imatrix = (const float *)imatrix_view->buf;
      const uint64_t imatrix_required = plan->n_per_row * sizeof(float);
      if ((uint64_t)imatrix_view->len < imatrix_required)
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

PyObject * py_write_safetensors_payload(PyObject *, PyObject *args, PyObject *kwargs)
{
  static const char *kwlist[] = {"path", "fd", "plans", "alignment", "scratch_bytes", nullptr};

  PyObject *path_obj;
  int fd = -1;
  PyObject *plans_obj;
  unsigned long long alignment = 32;
  unsigned long long scratch_bytes = SAFETENSORS_PAYLOAD_DEFAULT_SCRATCH_BYTES;

  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "OiOK|K", (char **)kwlist,
          &path_obj, &fd, &plans_obj, &alignment, &scratch_bytes))
    return nullptr;
  if (fd < 0) {
    PyErr_SetString(PyExc_ValueError, "fd must be a valid file descriptor");
    return nullptr;
  }
  if (alignment == 0 || scratch_bytes == 0) {
    PyErr_SetString(PyExc_ValueError, alignment == 0 ? "alignment must be positive" : "scratch_bytes must be positive");
    return nullptr;
  }

  const char *path_utf8 = PyUnicode_AsUTF8(path_obj);
  if (!path_utf8) return nullptr;

  PyObject *plans_seq = PySequence_Fast(plans_obj, "plans must be a sequence");
  if (!plans_seq) return nullptr;

  const Py_ssize_t plan_count = PySequence_Fast_GET_SIZE(plans_seq);
  std::vector<safetensors_payload_plan> plans;
  std::vector<Py_buffer> imatrix_views;
  std::vector<bool> has_imatrix_views;
  plans.reserve((size_t)plan_count);
  imatrix_views.reserve((size_t)plan_count);
  has_imatrix_views.reserve((size_t)plan_count);

  PyObject **items = PySequence_Fast_ITEMS(plans_seq);
  for (Py_ssize_t i = 0; i < plan_count; ++i) {
    safetensors_payload_plan plan;
    Py_buffer imatrix_view{};
    bool has_imatrix_view = false;
    if (!parse_native_plan(items[i], &plan, &imatrix_view, &has_imatrix_view)) {
      Py_DECREF(plans_seq);
      if (has_imatrix_view) PyBuffer_Release(&imatrix_view);
      for (size_t j = 0; j < imatrix_views.size(); ++j)
        if (has_imatrix_views[j]) PyBuffer_Release(&imatrix_views[j]);
      return nullptr;
    }
    plans.push_back(std::move(plan));
    imatrix_views.push_back(imatrix_view);
    has_imatrix_views.push_back(has_imatrix_view);
  }
  Py_DECREF(plans_seq);

  uint64_t data_bytes_written = 0;
  std::string error;
  bool ok = false;
  Py_BEGIN_ALLOW_THREADS
  ok = write_safetensors_payload(
      path_utf8, fd, plans, (uint64_t)alignment, (uint64_t)scratch_bytes,
      &data_bytes_written, &error);
  Py_END_ALLOW_THREADS

  for (size_t i = 0; i < imatrix_views.size(); ++i)
    if (has_imatrix_views[i]) PyBuffer_Release(&imatrix_views[i]);

  if (!ok) {
    PyErr_SetString(PyExc_RuntimeError, error.c_str());
    return nullptr;
  }
  return PyLong_FromUnsignedLongLong((unsigned long long)data_bytes_written);
}
