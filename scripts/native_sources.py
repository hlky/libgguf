from __future__ import annotations

DEQUANT_BACKEND_QTYPES = [
    "q1_0",
    "q4_0",
    "q4_1",
    "q5_0",
    "q5_1",
    "q8_0",
    "q2_k",
    "q3_k",
    "q4_k",
    "q5_k",
    "q6_k",
    "iq2_xxs",
    "iq2_xs",
    "iq2_s",
    "iq3_xxs",
    "iq3_s",
    "iq1_s",
    "iq1_m",
    "iq4_nl",
    "iq4_xs",
    "tq1_0",
    "tq2_0",
    "mxfp4",
    "nvfp4",
]

DEQUANT_BACKEND_SOURCES = [
    f"csrc/dequant/{backend}/{qtype}.cpp"
    for qtype in DEQUANT_BACKEND_QTYPES
    for backend in ("sse2", "sse4_1", "avx2")
]

QUANT_BACKEND_QTYPES = [
    "q1_0",
    "q4_0",
    "q4_1",
    "q5_0",
    "q5_1",
    "q8_0",
    "q2_k",
    "q3_k",
    "q4_k",
    "q5_k",
    "q6_k",
    "iq2_xxs",
    "iq2_xs",
    "iq2_s",
    "iq3_xxs",
    "iq3_s",
    "iq1_s",
    "iq1_m",
    "iq4_nl",
    "iq4_xs",
    "tq1_0",
    "tq2_0",
    "mxfp4",
    "nvfp4",
]

QUANT_BACKEND_SOURCES = [
    f"csrc/quant/{backend}/{qtype}.cpp"
    for qtype in QUANT_BACKEND_QTYPES
    for backend in ("sse2", "sse4_1", "avx2")
]
QUANT_BACKEND_SOURCES.extend([f"csrc/quant/{qtype}.cpp" for qtype in QUANT_BACKEND_QTYPES])

NATIVE_SOURCES = [
    "csrc/libgguf.cpp",
    "csrc/common/libgguf_common.cpp",
    "csrc/common/libgguf_common_quant.cpp",
    "csrc/common/sse2/libgguf_common_quant.cpp",
    "csrc/common/sse4_1/libgguf_common_quant.cpp",
    "csrc/common/avx2/libgguf_common_quant.cpp",
    "csrc/common/libgguf_storage.cpp",
    "csrc/common/sse2/libgguf_storage.cpp",
    "csrc/common/sse4_1/libgguf_storage.cpp",
    "csrc/common/avx2/libgguf_storage.cpp",
    "csrc/common/libgguf_tables.cpp",
    "csrc/common/libgguf_iq_tables.cpp",
    "csrc/common/libgguf_cpu.cpp",
    "csrc/common/libgguf_validate.cpp",
    "csrc/dequant/dequant.cpp",
    "csrc/dequant/dequant_generic.cpp",
    *DEQUANT_BACKEND_SOURCES,
    *QUANT_BACKEND_SOURCES,
]
