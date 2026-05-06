from __future__ import annotations

import warnings

import numpy as np

from gguf_kernels.formats import GGMLQuantizationType


def has_ggml_reference() -> bool:
    try:
        import libgguf  # noqa: F401
    except ImportError:
        return False
    return True


def quantize_rows_with_ggml(data: np.ndarray, qtype: GGMLQuantizationType) -> np.ndarray:
    warnings.warn(
        "gguf_kernels.ggml_ref is deprecated; import libgguf and call "
        "libgguf.quantize_rows(...) instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    try:
        import libgguf
    except ImportError as exc:
        raise RuntimeError(
            "GGML-backed quantization now requires the separate `libgguf` package. "
            "Install it with `python -m pip install -e libgguf --no-build-isolation`."
        ) from exc

    return libgguf.quantize_rows(data, qtype)
