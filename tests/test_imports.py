from __future__ import annotations

import libgguf


def test_top_level_exports_row_api() -> None:
    for name in (
        "row_size",
        "type_name",
        "type_size",
        "quantize_rows",
        "dequantize_rows",
        "store_rows",
    ):
        assert callable(getattr(libgguf, name))


def test_star_import_matches_public_exports() -> None:
    namespace: dict[str, object] = {}
    exec("from libgguf import *", namespace)

    assert set(libgguf.__all__).issubset(namespace)
    assert "convert_to_gguf" not in namespace
    assert "convert_safetensors_to_gguf_native" not in namespace
