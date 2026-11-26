#include "bindings/python/ops.h"
#include <libgguf/native_quant.h>
#include <limits>

namespace {
struct Buffer {
    Py_buffer view{};
    ~Buffer() { if (view.obj) PyBuffer_Release(&view); }
    bool get(PyObject* object) { return PyObject_GetBuffer(object, &view, PyBUF_CONTIG_RO) == 0; }
};

PyObject* convert(PyObject* args, bool encode) {
    int type = 0;
    Py_ssize_t rows = 0, columns = 0;
    PyObject* source = nullptr;
    PyObject* importance = Py_None;
    if (!PyArg_ParseTuple(args, "innO|O", &type, &rows, &columns, &source, &importance)) return nullptr;
    size_t row_bytes = 0;
    auto status = libgguf_native_row_size(static_cast<libgguf_native_format>(type),
        columns > 0 ? static_cast<size_t>(columns) : 0, &row_bytes);
    if (rows <= 0 || status != LIBGGUF_NATIVE_OK
        || row_bytes > static_cast<size_t>(PY_SSIZE_T_MAX) / static_cast<size_t>(rows)
        || columns > PY_SSIZE_T_MAX / rows / static_cast<Py_ssize_t>(sizeof(float))) {
        PyErr_SetString(PyExc_ValueError, "invalid native format or overflowing row geometry");
        return nullptr;
    }
    Buffer input, weights;
    if (!input.get(source) || (importance != Py_None && !weights.get(importance))) return nullptr;
    if (!encode && importance != Py_None) {
        PyErr_SetString(PyExc_ValueError, "decode does not accept importance");
        return nullptr;
    }
    const size_t output_size = encode ? row_bytes * rows : static_cast<size_t>(rows * columns) * sizeof(float);
    PyObject* result = PyBytes_FromStringAndSize(nullptr, static_cast<Py_ssize_t>(output_size));
    if (!result) return nullptr;
    auto* output = PyBytes_AS_STRING(result);
    Py_BEGIN_ALLOW_THREADS
    if (encode) {
        status = libgguf_native_encode_rows(static_cast<libgguf_native_format>(type),
            static_cast<const float*>(input.view.buf), input.view.len, rows, columns,
            static_cast<const float*>(weights.view.buf), weights.view.len, output, output_size);
    } else {
        status = libgguf_native_decode_rows(static_cast<libgguf_native_format>(type),
            input.view.buf, input.view.len, rows, columns, reinterpret_cast<float*>(output), output_size);
    }
    Py_END_ALLOW_THREADS
    if (status != LIBGGUF_NATIVE_OK) {
        Py_DECREF(result);
        PyErr_Format(PyExc_ValueError, "native conversion rejected its input (status %d)", status);
        return nullptr;
    }
    return result;
}
}

PyObject* py_native_quantize_rows_raw(PyObject*, PyObject* args) { return convert(args, true); }
PyObject* py_native_dequantize_rows_raw(PyObject*, PyObject* args) { return convert(args, false); }
