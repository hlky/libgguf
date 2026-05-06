from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence


class GGMLQuantizationType(IntEnum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    IQ1_M = 29
    BF16 = 30
    TQ1_0 = 34
    TQ2_0 = 35
    MXFP4 = 39
    NVFP4 = 40
    Q1_0 = 41


QK_K = 256


@dataclass(frozen=True)
class GGMLFormatInfo:
    qtype: GGMLQuantizationType
    block_size: int
    type_size: int
    type_name: str
    is_quantized: bool


GGML_FORMAT_INFO: dict[GGMLQuantizationType, GGMLFormatInfo] = {
    GGMLQuantizationType.F32: GGMLFormatInfo(GGMLQuantizationType.F32, 1, 4, "f32", False),
    GGMLQuantizationType.F16: GGMLFormatInfo(GGMLQuantizationType.F16, 1, 2, "f16", False),
    GGMLQuantizationType.Q4_0: GGMLFormatInfo(GGMLQuantizationType.Q4_0, 32, 2 + 16, "q4_0", True),
    GGMLQuantizationType.Q4_1: GGMLFormatInfo(GGMLQuantizationType.Q4_1, 32, 2 + 2 + 16, "q4_1", True),
    GGMLQuantizationType.Q5_0: GGMLFormatInfo(GGMLQuantizationType.Q5_0, 32, 2 + 4 + 16, "q5_0", True),
    GGMLQuantizationType.Q5_1: GGMLFormatInfo(GGMLQuantizationType.Q5_1, 32, 2 + 2 + 4 + 16, "q5_1", True),
    GGMLQuantizationType.Q8_0: GGMLFormatInfo(GGMLQuantizationType.Q8_0, 32, 2 + 32, "q8_0", True),
    GGMLQuantizationType.Q8_1: GGMLFormatInfo(GGMLQuantizationType.Q8_1, 32, 4 + 4 + 32, "q8_1", True),
    GGMLQuantizationType.Q2_K: GGMLFormatInfo(GGMLQuantizationType.Q2_K, 256, 2 + 2 + QK_K // 16 + QK_K // 4, "q2_K", True),
    GGMLQuantizationType.Q3_K: GGMLFormatInfo(GGMLQuantizationType.Q3_K, 256, 2 + QK_K // 4 + QK_K // 8 + 12, "q3_K", True),
    GGMLQuantizationType.Q4_K: GGMLFormatInfo(GGMLQuantizationType.Q4_K, 256, 2 + 2 + QK_K // 2 + 12, "q4_K", True),
    GGMLQuantizationType.Q5_K: GGMLFormatInfo(GGMLQuantizationType.Q5_K, 256, 2 + 2 + QK_K // 2 + QK_K // 8 + 12, "q5_K", True),
    GGMLQuantizationType.Q6_K: GGMLFormatInfo(GGMLQuantizationType.Q6_K, 256, 2 + QK_K // 2 + QK_K // 4 + QK_K // 16, "q6_K", True),
    GGMLQuantizationType.Q8_K: GGMLFormatInfo(GGMLQuantizationType.Q8_K, 256, 4 + QK_K + QK_K // 8, "q8_K", True),
    GGMLQuantizationType.IQ2_XXS: GGMLFormatInfo(GGMLQuantizationType.IQ2_XXS, 256, 2 + QK_K // 4, "iq2_xxs", True),
    GGMLQuantizationType.IQ2_XS: GGMLFormatInfo(GGMLQuantizationType.IQ2_XS, 256, 2 + QK_K // 4 + QK_K // 32, "iq2_xs", True),
    GGMLQuantizationType.IQ3_XXS: GGMLFormatInfo(GGMLQuantizationType.IQ3_XXS, 256, 2 + QK_K // 4 + QK_K // 8, "iq3_xxs", True),
    GGMLQuantizationType.IQ1_S: GGMLFormatInfo(GGMLQuantizationType.IQ1_S, 256, 2 + QK_K // 8 + QK_K // 16, "iq1_s", True),
    GGMLQuantizationType.IQ4_NL: GGMLFormatInfo(GGMLQuantizationType.IQ4_NL, 32, 2 + 16, "iq4_nl", True),
    GGMLQuantizationType.IQ3_S: GGMLFormatInfo(GGMLQuantizationType.IQ3_S, 256, 2 + QK_K // 4 + QK_K // 8 + QK_K // 32 + 4, "iq3_s", True),
    GGMLQuantizationType.IQ2_S: GGMLFormatInfo(GGMLQuantizationType.IQ2_S, 256, 2 + QK_K // 4 + QK_K // 16, "iq2_s", True),
    GGMLQuantizationType.IQ4_XS: GGMLFormatInfo(GGMLQuantizationType.IQ4_XS, 256, 2 + 2 + QK_K // 2 + QK_K // 64, "iq4_xs", True),
    GGMLQuantizationType.I8: GGMLFormatInfo(GGMLQuantizationType.I8, 1, 1, "i8", False),
    GGMLQuantizationType.I16: GGMLFormatInfo(GGMLQuantizationType.I16, 1, 2, "i16", False),
    GGMLQuantizationType.I32: GGMLFormatInfo(GGMLQuantizationType.I32, 1, 4, "i32", False),
    GGMLQuantizationType.I64: GGMLFormatInfo(GGMLQuantizationType.I64, 1, 8, "i64", False),
    GGMLQuantizationType.F64: GGMLFormatInfo(GGMLQuantizationType.F64, 1, 8, "f64", False),
    GGMLQuantizationType.IQ1_M: GGMLFormatInfo(GGMLQuantizationType.IQ1_M, 256, QK_K // 8 + QK_K // 16 + QK_K // 32, "iq1_m", True),
    GGMLQuantizationType.BF16: GGMLFormatInfo(GGMLQuantizationType.BF16, 1, 2, "bf16", False),
    GGMLQuantizationType.TQ1_0: GGMLFormatInfo(GGMLQuantizationType.TQ1_0, 256, 2 + 4 * 13, "tq1_0", True),
    GGMLQuantizationType.TQ2_0: GGMLFormatInfo(GGMLQuantizationType.TQ2_0, 256, 2 + 64, "tq2_0", True),
    GGMLQuantizationType.MXFP4: GGMLFormatInfo(GGMLQuantizationType.MXFP4, 32, 1 + 16, "mxfp4", True),
    GGMLQuantizationType.NVFP4: GGMLFormatInfo(GGMLQuantizationType.NVFP4, 64, 4 + 32, "nvfp4", True),
    GGMLQuantizationType.Q1_0: GGMLFormatInfo(GGMLQuantizationType.Q1_0, 128, 2 + 16, "q1_0", True),
}

GGML_QUANT_SIZES: dict[GGMLQuantizationType, tuple[int, int]] = {
    qtype: (info.block_size, info.type_size) for qtype, info in GGML_FORMAT_INFO.items()
}


def quant_shape_to_byte_shape(
    shape: Sequence[int], quant_type: GGMLQuantizationType
) -> tuple[int, ...]:
    block_size, type_size = GGML_QUANT_SIZES[quant_type]
    if shape[-1] % block_size != 0:
        raise ValueError(
            f"Quantized tensor row size ({shape[-1]}) is not a multiple of "
            f"{quant_type.name} block size ({block_size})"
        )
    return (*shape[:-1], shape[-1] // block_size * type_size)


def quant_shape_from_byte_shape(
    shape: Sequence[int], quant_type: GGMLQuantizationType
) -> tuple[int, ...]:
    block_size, type_size = GGML_QUANT_SIZES[quant_type]
    if shape[-1] % type_size != 0:
        raise ValueError(
            f"Quantized tensor bytes per row ({shape[-1]}) is not a multiple of "
            f"{quant_type.name} type size ({type_size})"
        )
    return (*shape[:-1], shape[-1] // type_size * block_size)
