from __future__ import annotations

import pytest
import torch

from .accuracy_utils import (
    ErrorThresholds,
    build_roundtrip_cases,
    compute_metrics,
    quantize_reference_tensor,
)

from libgguf import GGMLQuantizationType
from ..gguf_pt import dequantize, dequantize_functions


ROUNDTRIP_QTYPES = (
    GGMLQuantizationType.BF16,
    GGMLQuantizationType.Q1_0,
    GGMLQuantizationType.Q4_0,
    GGMLQuantizationType.Q4_1,
    GGMLQuantizationType.Q5_0,
    GGMLQuantizationType.Q5_1,
    GGMLQuantizationType.Q8_0,
    GGMLQuantizationType.Q2_K,
    GGMLQuantizationType.Q3_K,
    GGMLQuantizationType.Q4_K,
    GGMLQuantizationType.Q5_K,
    GGMLQuantizationType.Q6_K,
    GGMLQuantizationType.IQ2_XXS,
    GGMLQuantizationType.IQ2_XS,
    GGMLQuantizationType.IQ3_XXS,
    GGMLQuantizationType.IQ1_S,
    GGMLQuantizationType.IQ4_NL,
    GGMLQuantizationType.IQ3_S,
    GGMLQuantizationType.IQ2_S,
    GGMLQuantizationType.IQ4_XS,
    GGMLQuantizationType.IQ1_M,
    GGMLQuantizationType.TQ1_0,
    GGMLQuantizationType.TQ2_0,
    GGMLQuantizationType.MXFP4,
    GGMLQuantizationType.NVFP4,
)

ERROR_THRESHOLDS = {
    GGMLQuantizationType.BF16: ErrorThresholds(1.0e-4, 3.0e-4, 2.0e-3, 0.999_99),
    GGMLQuantizationType.Q1_0: ErrorThresholds(2.4e-2, 5.2e-2, 5.2e-1, 0.86),
    GGMLQuantizationType.Q4_0: ErrorThresholds(3.5e-3, 1.4e-2, 7.0e-2, 0.997),
    GGMLQuantizationType.Q4_1: ErrorThresholds(3.5e-3, 7.5e-3, 6.5e-2, 0.998),
    GGMLQuantizationType.Q5_0: ErrorThresholds(1.8e-3, 7.0e-3, 3.5e-2, 0.999_4),
    GGMLQuantizationType.Q5_1: ErrorThresholds(1.6e-3, 3.5e-3, 3.2e-2, 0.999_5),
    GGMLQuantizationType.Q8_0: ErrorThresholds(2.2e-4, 4.5e-4, 4.5e-3, 0.999_98),
    GGMLQuantizationType.Q2_K: ErrorThresholds(3.0e-2, 3.0e-1, 5.0e-1, 0.85),
    GGMLQuantizationType.Q3_K: ErrorThresholds(2.0e-2, 2.0e-1, 3.5e-1, 0.93),
    GGMLQuantizationType.Q4_K: ErrorThresholds(8.0e-3, 8.0e-2, 1.6e-1, 0.985),
    GGMLQuantizationType.Q5_K: ErrorThresholds(4.0e-3, 5.0e-2, 8.0e-2, 0.996),
    GGMLQuantizationType.Q6_K: ErrorThresholds(2.0e-3, 3.0e-2, 4.0e-2, 0.999),
    GGMLQuantizationType.IQ2_XXS: ErrorThresholds(4.0e-2, 5.0e-1, 6.5e-1, 0.75),
    GGMLQuantizationType.IQ2_XS: ErrorThresholds(4.0e-2, 5.0e-1, 6.5e-1, 0.75),
    GGMLQuantizationType.IQ3_XXS: ErrorThresholds(2.5e-2, 3.0e-1, 4.5e-1, 0.90),
    GGMLQuantizationType.IQ1_S: ErrorThresholds(6.0e-2, 7.0e-1, 9.0e-1, 0.45),
    GGMLQuantizationType.IQ4_NL: ErrorThresholds(1.0e-2, 1.0e-1, 2.0e-1, 0.98),
    GGMLQuantizationType.IQ3_S: ErrorThresholds(2.5e-2, 3.0e-1, 4.5e-1, 0.90),
    GGMLQuantizationType.IQ2_S: ErrorThresholds(4.0e-2, 5.0e-1, 6.5e-1, 0.75),
    GGMLQuantizationType.IQ4_XS: ErrorThresholds(1.0e-2, 1.0e-1, 2.0e-1, 0.98),
    GGMLQuantizationType.IQ1_M: ErrorThresholds(6.0e-2, 7.0e-1, 9.0e-1, 0.45),
    GGMLQuantizationType.TQ1_0: ErrorThresholds(1.7e-2, 3.2e-2, 5.2e-1, 0.91),
    GGMLQuantizationType.TQ2_0: ErrorThresholds(1.7e-2, 3.2e-2, 5.2e-1, 0.91),
    GGMLQuantizationType.MXFP4: ErrorThresholds(5.0e-3, 1.7e-2, 1.5e-1, 0.991),
    GGMLQuantizationType.NVFP4: ErrorThresholds(4.2e-3, 1.5e-2, 1.2e-1, 0.993),
}


def test_roundtrip_qtypes_cover_all_torch_dequantizers() -> None:
    assert set(ROUNDTRIP_QTYPES) == set(dequantize_functions)
    assert set(ERROR_THRESHOLDS) == set(ROUNDTRIP_QTYPES)


@pytest.mark.parametrize(("qtype", "tensor_name"), build_roundtrip_cases(ROUNDTRIP_QTYPES))
def test_torch_dequantize_matches_float16_reference(
    qtype: GGMLQuantizationType, tensor_name: str
) -> None:
    reference_fp16, quantized = quantize_reference_tensor(qtype, tensor_name)
    reference = reference_fp16.astype("float32", copy=False)

    quantized_torch = torch.from_numpy(quantized)
    dequantized = dequantize(
        quantized_torch, qtype, reference_fp16.shape, dtype=torch.float16
    )

    assert dequantized.shape == reference_fp16.shape
    expected_dtype = torch.float32 if qtype == GGMLQuantizationType.BF16 else torch.float16
    assert dequantized.dtype == expected_dtype

    metrics = compute_metrics(reference, dequantized.cpu().numpy().astype("float32"))
    limits = ERROR_THRESHOLDS[qtype]

    assert metrics["mean_abs"] <= limits.mean_abs, (
        f"{qtype.name} mean_abs={metrics['mean_abs']:.6g} "
        f"exceeded {limits.mean_abs:.6g} for {tensor_name}"
    )
    assert metrics["max_abs"] <= limits.max_abs, (
        f"{qtype.name} max_abs={metrics['max_abs']:.6g} "
        f"exceeded {limits.max_abs:.6g} for {tensor_name}"
    )
    assert metrics["nrmse"] <= limits.nrmse, (
        f"{qtype.name} nrmse={metrics['nrmse']:.6g} "
        f"exceeded {limits.nrmse:.6g} for {tensor_name}"
    )
    assert metrics["cosine"] >= limits.cosine_min, (
        f"{qtype.name} cosine={metrics['cosine']:.6g} "
        f"fell below {limits.cosine_min:.6g} for {tensor_name}"
    )
