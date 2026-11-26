#include <hip/hip_runtime.h>
#include <libgguf/native_gpu.h>
#include <libgguf/gpu/native_fragment.cuh>
#include <libgguf/native_quant.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void checked(hipError_t result) {
    if (result != hipSuccess) { std::fprintf(stderr, "%s\n", hipGetErrorString(result)); std::exit(1); }
}

__global__ void all_half_patterns(float* output) {
    const unsigned bits=blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned char encoded[]={static_cast<unsigned char>(bits),static_cast<unsigned char>(bits>>8)};
    output[bits]=libgguf::native_detail::read_half(encoded);
}

int main() {
    hipDeviceProp_t device{}; checked(hipGetDeviceProperties(&device, 0));
    std::printf("device=%s arch=%s\n", device.name, device.gcnArchName);
    float* half_device=nullptr;checked(hipMalloc(&half_device,65536*4));
    all_half_patterns<<<256,256>>>(half_device);checked(hipGetLastError());
    std::vector<float> half_actual(65536),half_expected(65536);
    checked(hipMemcpy(half_actual.data(),half_device,65536*4,hipMemcpyDeviceToHost));
    for(unsigned bits=0;bits<65536;++bits)half_expected[bits]=libgguf::gpu::fragment_detail::fp16_to_float(static_cast<unsigned short>(bits));
    if(std::memcmp(half_actual.data(),half_expected.data(),65536*4))return 19;
    checked(hipFree(half_device));
    constexpr int rows = 3, columns = 512, values = rows * columns;
    std::vector<float> input(values), importance(columns), expected(values), actual(values);
    for (int k = 0; k < values; ++k)
        input[k] = (static_cast<float>((k * 3571) % 4093) - 2046) / 257;
    for (int k = 0; k < 32; ++k) input[k] = 0;
    for (int k = 32; k < 64; ++k) input[k] *= 0x1p-20f;
    for (int k = 64; k < 96; ++k) input[k] *= 8192;
    for (int k = 0; k < columns; ++k) importance[k] = k < 64 ? 0.f : (k % 13) / 13.f;
    uint32_t* status = nullptr; uint32_t host_status = 99;
    hipStream_t stream{}; checked(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    checked(hipMalloc(&status, sizeof(uint32_t)));
    float *din = nullptr, *dw = nullptr, *dout = nullptr;
    unsigned char* packed = nullptr;
    checked(hipMalloc(&din, input.size() * sizeof(float)));
    checked(hipMalloc(&dw, importance.size() * sizeof(float)));
    checked(hipMalloc(&dout, actual.size() * sizeof(float)));
    checked(hipMalloc(&packed, values * 2));
    checked(hipMemcpy(din, input.data(), input.size() * sizeof(float), hipMemcpyHostToDevice));
    checked(hipMemcpy(dw, importance.data(), importance.size() * sizeof(float), hipMemcpyHostToDevice));
    for (int format = 0; format < 4; ++format) {
        size_t row_bytes = 0;
        const auto f = static_cast<libgguf_native_format>(format);
        if (libgguf_native_row_size(f, columns, &row_bytes) != 0) return 2;
        std::vector<unsigned char> host(row_bytes * rows), gpu(host.size());
        if (libgguf_native_encode_rows(f, input.data(), input.size() * 4, rows, columns,
            importance.data(), importance.size() * 4, host.data(), host.size()) != 0) return 3;
        if (libgguf_native_encode_gpu_into(f, din, input.size()*4, rows, columns,
            dw, importance.size()*4, packed, host.size(), status, stream) != 0) return 7;
        checked(hipStreamSynchronize(stream));
        checked(hipMemcpy(&host_status, status, 4, hipMemcpyDeviceToHost));
        if (host_status != 0) return 8;
        checked(hipGetLastError());
        checked(hipMemcpy(gpu.data(), packed, gpu.size(), hipMemcpyDeviceToHost));
        if (gpu != host) {
            for (size_t k = 0; k < host.size(); ++k) if (gpu[k] != host[k]) {
                std::fprintf(stderr, "format=%d byte=%zu CPU=%u GPU=%u\n", format, k, host[k], gpu[k]); break;
            }
            return 4;
        }
        if (libgguf_native_decode_rows(f, host.data(), host.size(), rows, columns,
            expected.data(), expected.size() * 4) != 0) return 5;
        if (libgguf_native_decode_gpu_into(f, packed, host.size(), rows, columns,
            dout, actual.size()*4, status, stream) != 0) return 9;
        checked(hipStreamSynchronize(stream));
        checked(hipMemcpy(&host_status, status, 4, hipMemcpyDeviceToHost));
        if (host_status != 0) return 10;
        checked(hipGetLastError());
        checked(hipMemcpy(actual.data(), dout, actual.size() * 4, hipMemcpyDeviceToHost));
        if (std::memcmp(expected.data(), actual.data(), actual.size() * 4)) return 6;
        hipGraph_t graph{}; hipGraphExec_t executable{};
        checked(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
        if (libgguf_native_encode_gpu_into(f, din, input.size()*4, rows, columns,
            dw, importance.size()*4, packed, host.size(), status, stream) != 0) return 16;
        if (libgguf_native_decode_gpu_into(f, packed, host.size(), rows, columns,
            dout, actual.size()*4, status, stream) != 0) return 17;
        checked(hipStreamEndCapture(stream, &graph));
        checked(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        for (int replay=0; replay<2; ++replay) {
            checked(hipGraphLaunch(executable, stream)); checked(hipStreamSynchronize(stream));
            checked(hipMemcpy(actual.data(), dout, actual.size()*4, hipMemcpyDeviceToHost));
            if (std::memcmp(expected.data(), actual.data(), actual.size()*4)) return 18;
        }
        checked(hipGraphExecDestroy(executable)); checked(hipGraphDestroy(graph));
        std::printf("format=%d encoder bytes=%zu and decoded values=%d exact\n", format, host.size(), values);
    }
    // Invalid input cannot publish partial bytes; subsequent valid work resets status.
    input[values-1] = __builtin_nanf("");
    checked(hipMemcpy(din, input.data(), input.size()*4, hipMemcpyHostToDevice));
    checked(hipMemset(packed, 0xa5, rows * (columns/32) * 18));
    if (libgguf_native_encode_gpu_into(LIBGGUF_NATIVE_S4G32, din, input.size()*4, rows, columns,
        nullptr, 0, packed, rows*(columns/32)*18, status, stream) != 0) return 11;
    checked(hipStreamSynchronize(stream));
    checked(hipMemcpy(&host_status, status, 4, hipMemcpyDeviceToHost));
    if (host_status != LIBGGUF_NATIVE_INVALID_VALUE) return 12;
    std::vector<unsigned char> guard(rows*(columns/32)*18);
    checked(hipMemcpy(guard.data(), packed, guard.size(), hipMemcpyDeviceToHost));
    for (auto value : guard) if (value != 0xa5) return 13;
    if (libgguf_native_decode_gpu_into(LIBGGUF_NATIVE_S4G32, packed, guard.size(), rows, columns,
        dout, actual.size()*4, status, stream) != 0) return 14;
    checked(hipStreamSynchronize(stream));
    checked(hipMemcpy(&host_status, status, 4, hipMemcpyDeviceToHost));
    if (host_status != LIBGGUF_NATIVE_INVALID_VALUE) return 15;
    checked(hipFree(status)); checked(hipStreamDestroy(stream));
    checked(hipFree(packed)); checked(hipFree(dout)); checked(hipFree(dw)); checked(hipFree(din));
}
