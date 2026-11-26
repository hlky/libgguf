import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from libgguf import native_numpy
from libgguf.native_formats import NativeQuantizationType as F


@pytest.mark.parametrize("format", list(F))
def test_compiled_python_native_cpu_backend(format):
    root = Path(__file__).parents[1]
    extensions = list((root / "build/quantization/ref").glob("_libgguf*.pyd"))
    if not extensions:
        extensions = list((root / "build/quantization/ref").glob("_libgguf*.so"))
    if not extensions:
        pytest.skip("compiled CPU Python extension has not been built")
    name = "libgguf._libgguf"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, extensions[0])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    from libgguf import native_cpu
    rng = np.random.default_rng(731)
    rows = rng.normal(size=(2, 256)).astype(np.float32)
    importance = rng.random(256, dtype=np.float32)
    importance[:64] = 0
    encoded = native_cpu.quantize_rows(rows, format, importance)
    np.testing.assert_array_equal(encoded, native_numpy.quantize_rows(rows, format, importance))
    np.testing.assert_array_equal(native_cpu.dequantize_rows(encoded, format, 256), native_numpy.dequantize_rows(encoded, format, 256))
    invalid = encoded.copy()
    invalid[-1, 1] = 0xfc
    with pytest.raises(ValueError):
        native_cpu.dequantize_rows(invalid, format, 256)
