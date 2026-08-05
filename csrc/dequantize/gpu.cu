#include <libgguf/gpu/compat.h>

#include <libgguf/gpu/dequantize/kernels.h>

void dequantize_row(
    const void * x,
    void * y,
    int64_t type,
    int k,
    libgguf_gpu_dtype dtype,
    StreamType stream
) {
    switch (type) {
        case 2:
            dequantize_launch_q4_0(x, y, k, dtype, stream);
            return;
        case 3:
            dequantize_launch_q4_1(x, y, k, dtype, stream);
            return;
        case 6:
            dequantize_launch_q5_0(x, y, k, dtype, stream);
            return;
        case 7:
            dequantize_launch_q5_1(x, y, k, dtype, stream);
            return;
        case 8:
            dequantize_launch_q8_0(x, y, k, dtype, stream);
            return;
        case 10:
            dequantize_launch_q2_k(x, y, k, dtype, stream);
            return;
        case 11:
            dequantize_launch_q3_k(x, y, k, dtype, stream);
            return;
        case 12:
            dequantize_launch_q4_k(x, y, k, dtype, stream);
            return;
        case 13:
            dequantize_launch_q5_k(x, y, k, dtype, stream);
            return;
        case 14:
            dequantize_launch_q6_k(x, y, k, dtype, stream);
            return;
        case 16:
            dequantize_launch_iq2_xxs(x, y, k, dtype, stream);
            return;
        case 17:
            dequantize_launch_iq2_xs(x, y, k, dtype, stream);
            return;
        case 18:
            dequantize_launch_iq3_xxs(x, y, k, dtype, stream);
            return;
        case 19:
            dequantize_launch_iq1_s(x, y, k, dtype, stream);
            return;
        case 20:
            dequantize_launch_iq4_nl(x, y, k, dtype, stream);
            return;
        case 21:
            dequantize_launch_iq3_s(x, y, k, dtype, stream);
            return;
        case 22:
            dequantize_launch_iq2_s(x, y, k, dtype, stream);
            return;
        case 23:
            dequantize_launch_iq4_xs(x, y, k, dtype, stream);
            return;
        case 29:
            dequantize_launch_iq1_m(x, y, k, dtype, stream);
            return;
        case 30:
            dequantize_launch_bf16(x, y, k, dtype, stream);
            return;
        case 34:
            dequantize_launch_tq1_0(x, y, k, dtype, stream);
            return;
        case 35:
            dequantize_launch_tq2_0(x, y, k, dtype, stream);
            return;
        case 39:
            dequantize_launch_mxfp4(x, y, k, dtype, stream);
            return;
        case 40:
            dequantize_launch_nvfp4(x, y, k, dtype, stream);
            return;
        case 41:
            dequantize_launch_q1_0(x, y, k, dtype, stream);
            return;
        default:
            return;
    }
}
