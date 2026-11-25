from __future__ import annotations

import libgguf


def test_explicit_cpu_module_exports_row_api() -> None:
    from libgguf import cpu

    for name in (
        "row_size",
        "type_name",
        "type_size",
        "quantize_rows",
        "dequantize_rows",
        "store_rows",
    ):
        assert callable(getattr(cpu, name))


def test_top_level_does_not_restore_legacy_backend_wrappers() -> None:
    namespace: dict[str, object] = {}
    exec("from libgguf import *", namespace)

    assert not hasattr(libgguf, "quantize_rows")
    assert not hasattr(libgguf, "GGML_QUANT_SIZES")
    assert not hasattr(libgguf, "read_gguf_header")
    assert "convert_to_gguf" not in namespace
    assert "convert_safetensors_to_gguf_native" not in namespace
