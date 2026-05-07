from __future__ import annotations

import re
from pathlib import Path

import pytest

import libgguf
from libgguf import GGMLQuantizationType, GGML_QUANT_SIZES


ROOT = Path(__file__).resolve().parents[1]
PYTHON_API_DOC = ROOT / "docs" / "python-api.md"

STORAGE_QTYPES = {
    GGMLQuantizationType.F32,
    GGMLQuantizationType.F16,
    GGMLQuantizationType.BF16,
}
IMATRIX_QTYPES = {
    GGMLQuantizationType.IQ2_XXS,
    GGMLQuantizationType.IQ2_XS,
    GGMLQuantizationType.IQ1_S,
}


def qtype_id(qtype: GGMLQuantizationType) -> str:
    return qtype.name


def _doc_imatrix_qtypes() -> set[str]:
    text = PYTHON_API_DOC.read_text(encoding="utf-8")
    match = re.search(r"The qtypes that require imatrix weights are: (?P<qtypes>[^.]+)\.", text)
    if match is None:
        raise AssertionError(f"imatrix qtype contract was not found in {PYTHON_API_DOC}")
    return set(re.findall(r"`([A-Z][A-Z0-9_]+)`", match.group("qtypes")))


@pytest.mark.parametrize("qtype", GGMLQuantizationType, ids=qtype_id)
def test_native_type_names_match_python_qtype_names(qtype: GGMLQuantizationType) -> None:
    native_name = libgguf.type_name(qtype)

    if native_name == "unknown":
        assert libgguf.type_size(qtype) == 0
        return

    assert native_name.lower() == qtype.name.lower()


@pytest.mark.parametrize("qtype", GGMLQuantizationType, ids=qtype_id)
def test_native_qtype_sizes_match_python_metadata(qtype: GGMLQuantizationType) -> None:
    block_size, metadata_type_size = GGML_QUANT_SIZES[qtype]
    native_type_size = libgguf.type_size(qtype)
    native_row_size = libgguf.row_size(qtype, block_size)

    if native_type_size == 0:
        assert native_row_size == 0
        return

    assert native_type_size == metadata_type_size

    if native_row_size == 0:
        return

    if qtype in STORAGE_QTYPES:
        assert native_row_size == block_size * metadata_type_size
        assert libgguf.row_size(qtype, 7) == 7 * metadata_type_size
    else:
        assert native_row_size == metadata_type_size
        assert libgguf.row_size(qtype, block_size * 2) == metadata_type_size * 2


@pytest.mark.parametrize("qtype", GGMLQuantizationType, ids=qtype_id)
def test_quantize_requires_imatrix_matches_expected_qtypes(qtype: GGMLQuantizationType) -> None:
    assert libgguf.quantize_requires_imatrix(qtype) is (qtype in IMATRIX_QTYPES)
    assert libgguf.quantize_requires_imatrix(qtype.value) is (qtype in IMATRIX_QTYPES)


def test_imatrix_qtype_contract_matches_python_api_docs() -> None:
    assert _doc_imatrix_qtypes() == {qtype.name for qtype in IMATRIX_QTYPES}
