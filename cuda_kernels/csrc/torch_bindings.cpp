#include <optional>

#include <Python.h>
#include <torch/all.h>
#include <torch/library.h>

#if defined(_WIN32) && defined(USE_ROCM)
#pragma comment(linker, "/alternatename:__imp_??0ValueError@c10@@QEAA@USourceLocation@1@V?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@@Z=__imp_??0Error@c10@@QEAA@USourceLocation@1@V?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@@Z")
#endif

torch::Tensor ggml_dequantize(torch::Tensor W, int64_t type, int64_t m,
                              int64_t n,
                              std::optional<at::ScalarType> const &dtype);

TORCH_LIBRARY(_C_gguf, ops)
{
    ops.def(
        "ggml_dequantize(Tensor W, int type, SymInt m, SymInt n, ScalarType? "
        "dtype) -> Tensor");
    ops.impl("ggml_dequantize", torch::kCUDA, &ggml_dequantize);
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
