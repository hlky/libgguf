#pragma once

#define PY_SSIZE_T_CLEAN
#include <Python.h>

PyObject * py_write_safetensors_payload(PyObject *, PyObject * args, PyObject * kwargs);
PyObject * py_row_size(PyObject *, PyObject * args);
PyObject * py_type_size(PyObject *, PyObject * args);
PyObject * py_type_name(PyObject *, PyObject * args);
PyObject * py_quantize_requires_imatrix(PyObject *, PyObject * args);
PyObject * py_quantize_free(PyObject *, PyObject *);
PyObject * py_quantize_rows_raw(PyObject *, PyObject * args, PyObject * kwargs);
PyObject * py_quantize_rows_into_raw(PyObject *, PyObject * args, PyObject * kwargs);
PyObject * py_dequantize_rows_raw(PyObject *, PyObject * args, PyObject * kwargs);
PyObject * py_dequantize_rows_into_raw(PyObject *, PyObject * args, PyObject * kwargs);
