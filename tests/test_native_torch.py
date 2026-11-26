import numpy as np
import pytest
import torch

from libgguf import native_numpy as ref, native_torch as actual
from libgguf.native_formats import NativeQuantizationType as F


@pytest.mark.parametrize("format", list(F))
@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_independent_torch_reference_and_fake_shapes(format, device):
    rng = np.random.default_rng(412)
    x = rng.normal(size=(2, 512)).astype(np.float32)
    x[0, :32] = 0
    w = rng.random(512, dtype=np.float32)
    w[:64] = 0
    expected = ref.quantize_rows(x, format, w)
    encoded = actual.quantize_rows(torch.from_numpy(x).to(device), format, torch.from_numpy(w).to(device))
    np.testing.assert_array_equal(encoded.cpu().numpy(), expected)
    decoded = actual.dequantize_rows(encoded, format, 512)
    np.testing.assert_array_equal(decoded.cpu().numpy(), ref.dequantize_rows(expected, format, 512))
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode():
        fake = actual.quantize_rows(torch.empty((2, 512), device=device), format)
        assert fake.shape == encoded.shape
        assert actual.dequantize_rows(fake, format, 512).shape == decoded.shape
