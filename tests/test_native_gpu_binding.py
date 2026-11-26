import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from libgguf import native_numpy
from libgguf.native_formats import NativeQuantizationType as F, native_row_size


@pytest.fixture(scope="module")
def gpu():
    path = os.environ.get("LIBGGUF_NATIVE_TORCH_EXTENSION")
    if not path or not Path(path).is_file() or not torch.cuda.is_available():
        pytest.skip("requires the newly built native Torch GPU extension and a GPU")
    name = "libgguf._C_gguf"
    spec = importlib.util.spec_from_file_location(name, path)
    extension = importlib.util.module_from_spec(spec)
    sys.modules[name] = extension
    spec.loader.exec_module(extension)
    from libgguf import native_gpu
    return native_gpu


@pytest.mark.parametrize("format", list(F))
def test_compiled_gpu_bytes_values_graph_and_fake_metadata(gpu, format):
    rng = np.random.default_rng(912)
    rows = rng.normal(size=(3, 512)).astype(np.float32)
    importance = rng.random(512, dtype=np.float32)
    importance[:64] = 0
    source = torch.from_numpy(rows).cuda()
    weights = torch.from_numpy(importance).cuda()
    output = torch.empty((3, native_row_size(format, 512)), dtype=torch.uint8, device="cuda")
    decoded = torch.empty_like(source)
    encode_status = torch.empty((1,), dtype=torch.int32, device="cuda")
    decode_status = torch.empty_like(encode_status)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        gpu.quantize_into(source, format, output, encode_status, weights)
        gpu.dequantize_into(output, format, 512, decoded, decode_status)
    stream.synchronize()
    assert encode_status.item() == decode_status.item() == 0
    expected = native_numpy.quantize_rows(rows, format, importance)
    np.testing.assert_array_equal(output.cpu().numpy(), expected)
    np.testing.assert_array_equal(decoded.cpu().numpy(), native_numpy.dequantize_rows(expected, format, 512))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gpu.quantize_into(source, format, output, encode_status, weights)
        gpu.dequantize_into(output, format, 512, decoded, decode_status)
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    assert encode_status.item() == decode_status.item() == 0
    np.testing.assert_array_equal(output.cpu().numpy(), expected)
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode():
        x = torch.empty((3, 512), device="cuda")
        y = torch.empty(output.shape, dtype=torch.uint8, device="cuda")
        s = torch.empty((1,), dtype=torch.int32, device="cuda")
        gpu.quantize_into(x, format, y, s)
        gpu.dequantize_into(y, format, 512, torch.empty_like(x), s)
        with pytest.raises(RuntimeError):
            gpu.dequantize_into(y, format, 512, torch.empty((3, 256), device="cuda"), s)


def test_compiled_gpu_invalid_values_do_not_partially_write_and_aliases_are_rejected(gpu):
    source = torch.ones((2, 64), device="cuda")
    output = torch.full((2, 36), 0xa5, dtype=torch.uint8, device="cuda")
    status = torch.zeros((1,), dtype=torch.int32, device="cuda")
    source[-1, -1] = float("nan")
    gpu.quantize_into(source, F.S4G32, output, status)
    assert status.item() == 4
    assert bool((output == 0xa5).all())
    decoded = torch.full_like(source, 123)
    gpu.dequantize_into(output, F.S4G32, 64, decoded, status)
    assert status.item() == 4
    assert bool((decoded == 123).all())
    source.fill_(1)
    alias = source.view(torch.uint8).flatten()[:72].reshape(2, 36)
    with pytest.raises(RuntimeError):
        gpu.quantize_into(source, F.S4G32, alias, status)
    with pytest.raises(RuntimeError):
        gpu.quantize_into(source, 2**32, output, status)
    with pytest.raises(RuntimeError):
        gpu.quantize_into(source, F.S4G32, output, status.cpu())


def test_compiled_gpu_dynamic_fake_contract_survives_torch_compile(gpu):
    def encode(source, output, status):
        gpu.quantize_into(source, F.S4G32, output, status)
        return output, status
    compiled = torch.compile(encode, backend="eager", fullgraph=True, dynamic=True)
    for rows in (2, 3):
        source = torch.ones((rows, 64), device="cuda")
        output = torch.empty((rows, 36), dtype=torch.uint8, device="cuda")
        status = torch.empty((1,), dtype=torch.int32, device="cuda")
        encoded, result = compiled(source, output, status)
        assert encoded is output and result is status
        assert result.item() == 0
