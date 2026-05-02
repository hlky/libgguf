from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence


class GGMLQuantizationType(IntEnum):
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    IQ1_M = 29
    TQ1_0 = 34
    TQ2_0 = 35
    MXFP4 = 39
    NVFP4 = 40
    Q1_0 = 41


QK_K = 256


@dataclass(frozen=True)
class GGMLFormatInfo:
    block_size: int
    type_size: int


GGML_FORMAT_INFO: dict[GGMLQuantizationType, GGMLFormatInfo] = {
    GGMLQuantizationType.Q4_0: GGMLFormatInfo(32, 2 + 16),
    GGMLQuantizationType.Q4_1: GGMLFormatInfo(32, 2 + 2 + 16),
    GGMLQuantizationType.Q5_0: GGMLFormatInfo(32, 2 + 4 + 16),
    GGMLQuantizationType.Q5_1: GGMLFormatInfo(32, 2 + 2 + 4 + 16),
    GGMLQuantizationType.Q8_0: GGMLFormatInfo(32, 2 + 32),
    GGMLQuantizationType.Q2_K: GGMLFormatInfo(256, 2 + 2 + QK_K // 16 + QK_K // 4),
    GGMLQuantizationType.Q3_K: GGMLFormatInfo(256, 2 + QK_K // 4 + QK_K // 8 + 12),
    GGMLQuantizationType.Q4_K: GGMLFormatInfo(256, 2 + 2 + QK_K // 2 + 12),
    GGMLQuantizationType.Q5_K: GGMLFormatInfo(256, 2 + 2 + QK_K // 2 + QK_K // 8 + 12),
    GGMLQuantizationType.Q6_K: GGMLFormatInfo(256, 2 + QK_K // 2 + QK_K // 4 + QK_K // 16),
    GGMLQuantizationType.IQ2_XXS: GGMLFormatInfo(256, 2 + QK_K // 4),
    GGMLQuantizationType.IQ2_XS: GGMLFormatInfo(256, 2 + QK_K // 4 + QK_K // 32),
    GGMLQuantizationType.IQ3_XXS: GGMLFormatInfo(256, 2 + QK_K // 4 + QK_K // 8),
    GGMLQuantizationType.IQ1_S: GGMLFormatInfo(256, 2 + QK_K // 8 + QK_K // 16),
    GGMLQuantizationType.IQ4_NL: GGMLFormatInfo(32, 2 + 16),
    GGMLQuantizationType.IQ3_S: GGMLFormatInfo(256, 2 + QK_K // 4 + QK_K // 8 + QK_K // 32 + 4),
    GGMLQuantizationType.IQ2_S: GGMLFormatInfo(256, 2 + QK_K // 4 + QK_K // 16),
    GGMLQuantizationType.IQ4_XS: GGMLFormatInfo(256, 2 + 2 + QK_K // 2 + QK_K // 64),
    GGMLQuantizationType.IQ1_M: GGMLFormatInfo(256, QK_K // 8 + QK_K // 16 + QK_K // 32),
    GGMLQuantizationType.TQ1_0: GGMLFormatInfo(256, 2 + 4 * 13),
    GGMLQuantizationType.TQ2_0: GGMLFormatInfo(256, 2 + 64),
    GGMLQuantizationType.MXFP4: GGMLFormatInfo(32, 1 + 16),
    GGMLQuantizationType.NVFP4: GGMLFormatInfo(64, 4 + 32),
    GGMLQuantizationType.Q1_0: GGMLFormatInfo(128, 2 + 16),
}


def quant_shape_to_byte_shape(
    shape: Sequence[int], quant_type: GGMLQuantizationType
) -> tuple[int, ...]:
    info = GGML_FORMAT_INFO[quant_type]
    if shape[-1] % info.block_size != 0:
        raise ValueError(f"{shape[-1]} is not a multiple of {quant_type.name} block size {info.block_size}")
    return (*shape[:-1], shape[-1] // info.block_size * info.type_size)
