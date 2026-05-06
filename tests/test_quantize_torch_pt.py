from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from libgguf.quantize_pt import convert_to_gguf as convert_to_gguf_pt
from libgguf.quantize_torch_pt import convert_to_gguf as convert_to_gguf_torch

torch = pytest.importorskip("torch")
libgguf_torch = pytest.importorskip("libgguf.libgguf_torch")


def _write_safetensors(path: Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]]) -> None:
    payload = bytearray()
    header: dict[str, object] = {}
    for name, (dtype, shape, data) in tensors.items():
        begin = len(payload)
        payload += data
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [begin, len(payload)]}
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + payload)


def test_torch_pt_converter_uses_libgguf_torch_quantize(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key = "double_layers.3.modX.1.weight"
    rows = np.linspace(-2.0, 2.0, 64, dtype=np.float32).reshape(2, 32)
    src = tmp_path / "model.safetensors"
    native_out = tmp_path / "native.gguf"
    torch_out = tmp_path / "torch.gguf"
    _write_safetensors(src, {key: ("F32", rows.shape, rows.tobytes())})

    calls: list[str] = []
    devices: list[str] = []
    real_quantize = libgguf_torch.quantize

    def quantize_spy(data: torch.Tensor, qtype: object) -> torch.Tensor:
        calls.append(getattr(qtype, "name", str(qtype)))
        devices.append(data.device.type)
        return real_quantize(data, qtype)

    monkeypatch.setattr(libgguf_torch, "quantize", quantize_spy)

    convert_to_gguf_pt(src, native_out, "Q4_0", policy="uniform", overwrite=True)
    convert_to_gguf_torch(src, torch_out, "Q4_0", policy="uniform", overwrite=True, device="cpu")

    assert calls == ["Q4_0"]
    assert devices == ["cpu"]
    assert torch_out.read_bytes() == native_out.read_bytes()


def test_torch_pt_converter_can_compile_quantize(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is not available")

    key = "double_layers.3.modX.1.weight"
    rows = np.linspace(-2.0, 2.0, 64, dtype=np.float32).reshape(2, 32)
    src = tmp_path / "model.safetensors"
    torch_out = tmp_path / "torch.gguf"
    _write_safetensors(src, {key: ("F32", rows.shape, rows.tobytes())})

    compile_calls = 0

    def compile_spy(fn: object, *args: object, **kwargs: object) -> object:
        nonlocal compile_calls
        compile_calls += 1
        return fn

    monkeypatch.setattr(torch, "compile", compile_spy)

    convert_to_gguf_torch(src, torch_out, "Q4_0", policy="uniform", overwrite=True, compile=True)

    assert compile_calls == 1
    assert torch_out.is_file()


def test_torch_pt_converter_rejects_imatrix(tmp_path: Path) -> None:
    key = "double_layers.3.modX.1.weight"
    rows = np.linspace(-2.0, 2.0, 64, dtype=np.float32).reshape(2, 32)
    src = tmp_path / "model.safetensors"
    _write_safetensors(src, {key: ("F32", rows.shape, rows.tobytes())})

    with pytest.raises(ValueError, match="does not support explicit imatrix"):
        convert_to_gguf_torch(src, tmp_path / "out.gguf", "Q4_0", policy="uniform", imatrix={key: rows})


def test_torch_pt_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "libgguf.quantize_gguf_torch", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Torch-native libgguf_torch" in result.stdout
    assert "--device" in result.stdout
    assert "--compile" in result.stdout
