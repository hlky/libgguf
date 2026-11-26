#include <optional>

#include <Python.h>
#include <torch/all.h>
#include <torch/library.h>

torch::Tensor dequantize(torch::Tensor W, int64_t type, int64_t m,
                              int64_t n,
                              std::optional<at::ScalarType> const &dtype);
torch::Tensor quantize(torch::Tensor W, int64_t type, std::optional<torch::Tensor> const &imatrix);
void native_quantize_into(torch::Tensor,int64_t,torch::Tensor,torch::Tensor,std::optional<torch::Tensor> const&);
void native_dequantize_into(torch::Tensor,int64_t,int64_t,torch::Tensor,torch::Tensor);

TORCH_LIBRARY(_C_gguf, ops)
{
    ops.def(
        "dequantize(Tensor W, int type, SymInt m, SymInt n, ScalarType? "
        "dtype) -> Tensor");
    ops.def("quantize(Tensor W, int type, Tensor? imatrix=None) -> Tensor");
    ops.def("native_quantize_into(Tensor source, int format, Tensor(a!) output, Tensor(b!) status, Tensor? importance=None) -> ()");
    ops.def("native_dequantize_into(Tensor encoded, int format, SymInt columns, Tensor(a!) output, Tensor(b!) status) -> ()");
    ops.impl("dequantize", torch::kCUDA, &dequantize);
    ops.impl("quantize", torch::kCUDA, &quantize);
    ops.impl("native_quantize_into",torch::kCUDA,&native_quantize_into);
    ops.impl("native_dequantize_into",torch::kCUDA,&native_dequantize_into);
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
