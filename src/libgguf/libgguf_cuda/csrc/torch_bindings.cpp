#include <optional>

#include <Python.h>
#include <torch/all.h>
#include <torch/library.h>

torch::Tensor dequantize(torch::Tensor W, int64_t type, int64_t m,
                              int64_t n,
                              std::optional<at::ScalarType> const &dtype);

TORCH_LIBRARY(_C_gguf, ops)
{
    ops.def(
        "dequantize(Tensor W, int type, SymInt m, SymInt n, ScalarType? "
        "dtype) -> Tensor");
    ops.impl("dequantize", torch::kCUDA, &dequantize);
}

static PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_C_gguf",
    nullptr,
    -1,
    nullptr,
};

PyMODINIT_FUNC PyInit__C_gguf(void)
{
    return PyModule_Create(&module);
}
