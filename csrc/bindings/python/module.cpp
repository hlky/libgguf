#include "bindings/python/ops.h"

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
    {nullptr, nullptr, 0, nullptr},
};

static PyModuleDef module_def = {
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