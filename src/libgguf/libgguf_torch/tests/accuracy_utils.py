from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from libgguf.libgguf_numpy.gguf_np import GGMLQuantizationType, GGML_QUANT_SIZES, quantize


@dataclass(frozen=True)
class ErrorThresholds:
    mean_abs: float
    max_abs: float
    nrmse: float
    cosine_min: float


def build_reference_tensors() -> dict[str, np.ndarray]:
    torch.manual_seed(0)

    modules = {
        "linear_32_a": torch.nn.Linear(128, 96, bias=False),
        "linear_32_b": torch.nn.Linear(96, 64, bias=False),
        "conv2d_32_a": torch.nn.Conv2d(4, 32, kernel_size=(3, 32), bias=False),
        "conv2d_32_b": torch.nn.Conv2d(8, 16, kernel_size=(1, 32), bias=False),
        "linear_256_a": torch.nn.Linear(256, 64, bias=False),
        "linear_256_b": torch.nn.Linear(512, 32, bias=False),
        "conv2d_256_a": torch.nn.Conv2d(2, 8, kernel_size=(1, 256), bias=False),
        "conv2d_256_b": torch.nn.Conv2d(1, 4, kernel_size=(1, 512), bias=False),
    }

    references: dict[str, np.ndarray] = {}
    for module_name, module in modules.items():
        for tensor_name, tensor in module.state_dict().items():
            references[f"{module_name}.{tensor_name}"] = (
                tensor.detach().to(dtype=torch.float16).cpu().numpy()
            )

    return references


REFERENCE_TENSORS = build_reference_tensors()


def build_roundtrip_cases(qtypes: tuple[GGMLQuantizationType, ...]) -> list[object]:
    cases = []
    for qtype in qtypes:
        block_size, _ = GGML_QUANT_SIZES[qtype]
        for tensor_name, reference in REFERENCE_TENSORS.items():
            if reference.shape[-1] % block_size == 0:
                cases.append(
                    pytest.param(
                        qtype,
                        tensor_name,
                        id=f"{qtype.name}-{tensor_name}",
                    )
                )
    return cases


def compute_metrics(reference: np.ndarray, dequantized: np.ndarray) -> dict[str, float]:
    diff = dequantized - reference
    mean_abs = float(np.abs(diff).mean())
    max_abs = float(np.abs(diff).max())
    rmse = float(np.sqrt(np.mean(np.square(diff))))
    ref_rms = float(np.sqrt(np.mean(np.square(reference))))

    numerator = float(np.dot(reference.ravel(), dequantized.ravel()))
    denominator = float(np.linalg.norm(reference.ravel()) * np.linalg.norm(dequantized.ravel()))
    cosine = numerator / max(denominator, 1.0e-12)

    return {
        "mean_abs": mean_abs,
        "max_abs": max_abs,
        "nrmse": rmse / max(ref_rms, 1.0e-12),
        "cosine": cosine,
    }


def quantize_reference_tensor(
    qtype: GGMLQuantizationType, tensor_name: str
) -> tuple[np.ndarray, np.ndarray]:
    reference_fp16 = REFERENCE_TENSORS[tensor_name]
    quantized = quantize(reference_fp16, qtype)
    return reference_fp16, quantized


def quantized_row_shape(reference_shape: tuple[int, ...]) -> tuple[int, int]:
    row_count = int(np.prod(reference_shape[:-1], dtype=np.int64)) if len(reference_shape) > 1 else 1
    return row_count, reference_shape[-1]
