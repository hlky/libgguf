from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import libgguf
from libgguf import GGMLQuantizationType, GGML_QUANT_SIZES, _libgguf


BACKENDS = ("ref", "sse2", "sse4_1", "avx2")
SIMD_BACKENDS = BACKENDS[1:]

SUPPORTED_QUANTIZED_QTYPES = (
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
    GGMLQuantizationType.IQ2_S,
    GGMLQuantizationType.IQ3_XXS,
    GGMLQuantizationType.IQ3_S,
    GGMLQuantizationType.IQ1_S,
    GGMLQuantizationType.IQ1_M,
    GGMLQuantizationType.IQ4_NL,
    GGMLQuantizationType.IQ4_XS,
    GGMLQuantizationType.TQ1_0,
    GGMLQuantizationType.TQ2_0,
    GGMLQuantizationType.MXFP4,
    GGMLQuantizationType.NVFP4,
)

DIRECT_QUANTIZE_BACKEND_CASES: tuple[
    tuple[
        GGMLQuantizationType,
        Callable[[str], bool],
        Callable[[str, np.ndarray, int, int], bytes],
        Callable[[], str] | None,
    ],
    ...,
] = (
    (GGMLQuantizationType.Q1_0, _libgguf._q1_0_cpu_supports_backend, _libgguf._quantize_q1_0_for_backend, _libgguf._q1_0_backend),
    (GGMLQuantizationType.Q4_0, _libgguf._q4_0_cpu_supports_backend, _libgguf._quantize_q4_0_for_backend, _libgguf._q4_0_backend),
    (GGMLQuantizationType.Q4_1, _libgguf._q4_1_cpu_supports_backend, _libgguf._quantize_q4_1_for_backend, _libgguf._q4_1_backend),
    (GGMLQuantizationType.Q5_0, _libgguf._q5_0_cpu_supports_backend, _libgguf._quantize_q5_0_for_backend, _libgguf._q5_0_backend),
    (GGMLQuantizationType.Q5_1, _libgguf._q5_1_cpu_supports_backend, _libgguf._quantize_q5_1_for_backend, _libgguf._q5_1_backend),
    (GGMLQuantizationType.Q8_0, _libgguf._q8_0_cpu_supports_backend, _libgguf._quantize_q8_0_for_backend, _libgguf._q8_0_backend),
    (GGMLQuantizationType.Q2_K, _libgguf._q2_k_cpu_supports_backend, _libgguf._quantize_q2_k_for_backend, _libgguf._q2_k_backend),
    (GGMLQuantizationType.Q3_K, _libgguf._q3_k_cpu_supports_backend, _libgguf._quantize_q3_k_for_backend, _libgguf._q3_k_backend),
    (GGMLQuantizationType.Q4_K, _libgguf._q4_k_cpu_supports_backend, _libgguf._quantize_q4_k_for_backend, _libgguf._q4_k_backend),
    (GGMLQuantizationType.Q5_K, _libgguf._q5_k_cpu_supports_backend, _libgguf._quantize_q5_k_for_backend, _libgguf._q5_k_backend),
    (GGMLQuantizationType.Q6_K, _libgguf._q6_k_cpu_supports_backend, _libgguf._quantize_q6_k_for_backend, _libgguf._q6_k_backend),
    (GGMLQuantizationType.IQ4_NL, _libgguf._iq4_nl_cpu_supports_backend, _libgguf._quantize_iq4_nl_for_backend, None),
    (GGMLQuantizationType.IQ4_XS, _libgguf._iq4_xs_cpu_supports_backend, _libgguf._quantize_iq4_xs_for_backend, None),
    (GGMLQuantizationType.TQ1_0, _libgguf._tq1_0_cpu_supports_backend, _libgguf._quantize_tq1_0_for_backend, _libgguf._tq1_0_backend),
    (GGMLQuantizationType.TQ2_0, _libgguf._tq2_0_cpu_supports_backend, _libgguf._quantize_tq2_0_for_backend, _libgguf._tq2_0_backend),
    (GGMLQuantizationType.MXFP4, _libgguf._mxfp4_cpu_supports_backend, _libgguf._quantize_mxfp4_for_backend, _libgguf._mxfp4_backend),
    (GGMLQuantizationType.NVFP4, _libgguf._nvfp4_cpu_supports_backend, _libgguf._quantize_nvfp4_for_backend, _libgguf._nvfp4_backend),
)

COMMON_QUANT_CASES = (
    (GGMLQuantizationType.Q4_0, False),
    (GGMLQuantizationType.Q4_0, True),
    (GGMLQuantizationType.Q4_1, True),
    (GGMLQuantizationType.Q5_0, True),
    (GGMLQuantizationType.Q5_1, True),
    (GGMLQuantizationType.Q2_K, False),
    (GGMLQuantizationType.Q2_K, True),
    (GGMLQuantizationType.Q3_K, False),
    (GGMLQuantizationType.Q3_K, True),
    (GGMLQuantizationType.Q4_K, False),
    (GGMLQuantizationType.Q4_K, True),
    (GGMLQuantizationType.Q5_K, False),
    (GGMLQuantizationType.Q5_K, True),
    (GGMLQuantizationType.Q6_K, False),
    (GGMLQuantizationType.Q6_K, True),
)


def qtype_id(qtype: GGMLQuantizationType) -> str:
    return qtype.name


def direct_case_id(
    case: tuple[
        GGMLQuantizationType,
        Callable[[str], bool],
        Callable[[str, np.ndarray, int, int], bytes],
        Callable[[], str] | None,
    ]
) -> str:
    return case[0].name


def common_case_id(case: tuple[GGMLQuantizationType, bool]) -> str:
    qtype, weighted = case
    return f"{qtype.name}-{'weighted' if weighted else 'unweighted'}"


def build_rows(qtype: GGMLQuantizationType, *, rows: int = 6) -> np.ndarray:
    block_size, _ = GGML_QUANT_SIZES[qtype]
    width = block_size * (2 if block_size < 256 else 1)
    rng = np.random.default_rng(qtype.value * 17 + rows)
    data = np.stack(
        [
            np.linspace(-2.0, 2.0, width, dtype=np.float32),
            rng.normal(0.0, 0.75, width).astype(np.float32),
            np.zeros(width, dtype=np.float32),
            np.full(width, 0.25, dtype=np.float32),
            ((np.arange(width, dtype=np.float32) % 17.0) - 8.0) / np.float32(3.0),
            (rng.normal(0.0, 1.0, width) * np.float32(16.0)).astype(np.float32),
        ]
    )
    return np.ascontiguousarray(data[:rows], dtype=np.float32)


def computed_imatrix(rows: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.sum(rows * rows, axis=0, dtype=np.float32), dtype=np.float32)


@pytest.mark.parametrize("case", DIRECT_QUANTIZE_BACKEND_CASES, ids=direct_case_id)
def test_direct_quantize_backends_match_reference(
    case: tuple[
        GGMLQuantizationType,
        Callable[[str], bool],
        Callable[[str, np.ndarray, int, int], bytes],
        Callable[[], str] | None,
    ],
) -> None:
    qtype, supports_backend, quantize_for_backend, selected_backend = case
    rows = build_rows(qtype)

    expected = quantize_for_backend("ref", rows, rows.shape[0], rows.shape[1])
    assert isinstance(expected, bytes)
    assert len(expected) == rows.shape[0] * libgguf.row_size(qtype, rows.shape[1])

    if selected_backend is not None:
        backend = selected_backend()
        assert backend in BACKENDS
        assert supports_backend(backend)

    for backend in SIMD_BACKENDS:
        if not supports_backend(backend):
            continue
        actual = quantize_for_backend(backend, rows, rows.shape[0], rows.shape[1])
        assert actual == expected, f"{qtype.name}/{backend}"


def test_common_quant_probe_backends_match_reference() -> None:
    expected = _libgguf._common_quant_probe_for_backend("ref")

    for backend in SIMD_BACKENDS:
        if _libgguf._common_quant_cpu_supports_backend(backend):
            assert _libgguf._common_quant_probe_for_backend(backend) == expected, backend


def test_common_quant_backends_match_reference_for_public_quantize_rows_raw() -> None:
    def run_backend(backend: str) -> list[bytes]:
        _libgguf._common_quant_set_backend(backend)
        assert _libgguf._common_quant_backend() == backend
        parts: list[bytes] = []
        for qtype, weighted in COMMON_QUANT_CASES:
            rows = build_rows(qtype, rows=4)
            imatrix = np.linspace(0.25, 1.75, rows.shape[1], dtype=np.float32) if weighted else None
            parts.append(libgguf.quantize_rows_raw(qtype, rows, rows.shape[0], rows.shape[1], imatrix))
        return parts

    try:
        expected = run_backend("ref")
        for backend in SIMD_BACKENDS:
            if _libgguf._common_quant_cpu_supports_backend(backend):
                assert run_backend(backend) == expected, backend
    finally:
        _libgguf._common_quant_set_backend("ref")


def test_storage_backends_match_reference_for_bf16_store_rows() -> None:
    rows = np.array(
        [
            [0.0, -0.0, 1.0, -2.0, 0.1, 65504.0, 1.0e-8, np.inf, -np.inf],
            [np.nan, -3.5, 42.25, -1000.0, 3.1415927, 1.5, -7.25, 2.0e-4, -2.0e-4],
        ],
        dtype=np.float32,
    )

    try:
        _libgguf._storage_set_backend("ref")
        expected = libgguf.store_rows(rows, GGMLQuantizationType.BF16).tobytes()

        for backend in SIMD_BACKENDS:
            if not _libgguf._storage_cpu_supports_backend(backend):
                continue
            _libgguf._storage_set_backend(backend)
            assert _libgguf._storage_backend() == backend
            assert libgguf.store_rows(rows, GGMLQuantizationType.BF16).tobytes() == expected, backend
    finally:
        _libgguf._storage_set_backend("auto")


@pytest.mark.parametrize("qtype", SUPPORTED_QUANTIZED_QTYPES, ids=qtype_id)
def test_dequantize_backends_match_reference(qtype: GGMLQuantizationType) -> None:
    rows = build_rows(qtype)
    imatrix = computed_imatrix(rows) if libgguf.quantize_requires_imatrix(qtype) else None
    quantized = libgguf.quantize_rows(rows, qtype, imatrix=imatrix)

    selected_backend = _libgguf._dequant_backend(qtype)
    assert selected_backend in BACKENDS
    assert _libgguf._dequant_cpu_supports_backend(selected_backend)

    expected = _libgguf._dequantize_for_backend(qtype, "ref", quantized, rows.shape[0], rows.shape[1])
    for backend in SIMD_BACKENDS:
        if not _libgguf._dequant_cpu_supports_backend(backend):
            continue
        actual = _libgguf._dequantize_for_backend(qtype, backend, quantized, rows.shape[0], rows.shape[1])
        assert actual == expected, f"{qtype.name}/{backend}"
