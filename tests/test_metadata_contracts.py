from __future__ import annotations

import pytest

import libgguf
from libgguf import GGMLQuantizationType, GGML_QUANT_SIZES


STORAGE_QTYPES = {
    GGMLQuantizationType.F32,
    GGMLQuantizationType.F16,
    GGMLQuantizationType.BF16,
}


def qtype_id(qtype: GGMLQuantizationType) -> str:
    return qtype.name


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
