from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from libgguf.quantize import convert_safetensors_to_gguf_native


def _write_safetensors(path: Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]]) -> None:
    payload = bytearray()
    header: dict[str, object] = {}
    for name, (dtype, shape, data) in tensors.items():
        begin = len(payload)
        payload += data
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [begin, len(payload)]}
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + payload)


def _native_exe() -> Path:
    env = os.environ.get("LIBGGUF_QUANTIZE_GGUF_EXE")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))
    root = Path(__file__).resolve().parents[1]
    candidates.extend(root.glob("build/cmake/**/libgguf_quantize_gguf.exe"))
    candidates.extend(root.glob("build/cmake/**/libgguf_quantize_gguf"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    pytest.skip("libgguf_quantize_gguf executable is not built")


@pytest.mark.parametrize(
    "qtype",
    ["Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"],
)
def test_native_executable_matches_python_native_for_qk_types(tmp_path: Path, qtype: str) -> None:
    exe = _native_exe()
    key = "double_layers.3.modX.1.weight"
    rows = np.linspace(-2.0, 2.0, 512, dtype=np.float32).reshape(2, 256)
    src = tmp_path / "model.safetensors"
    expected = tmp_path / "expected.gguf"
    actual = tmp_path / "actual.gguf"
    _write_safetensors(src, {key: ("F32", rows.shape, rows.tobytes())})

    convert_safetensors_to_gguf_native(src, expected, qtype, policy="uniform", overwrite=True)
    result = subprocess.run(
        [str(exe), "--src", str(src), "--dst", str(actual), "--qtype", qtype, "--policy", "uniform", "--overwrite"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert actual.read_bytes() == expected.read_bytes()


def test_native_executable_supports_q1_0_without_python_gguf_support(tmp_path: Path) -> None:
    exe = _native_exe()
    key = "double_layers.3.modX.1.weight"
    rows = np.linspace(-2.0, 2.0, 512, dtype=np.float32).reshape(2, 256)
    src = tmp_path / "model.safetensors"
    actual = tmp_path / "actual.gguf"
    _write_safetensors(src, {key: ("F32", rows.shape, rows.tobytes())})

    result = subprocess.run(
        [str(exe), "--src", str(src), "--dst", str(actual), "--qtype", "Q1_0", "--policy", "uniform", "--overwrite"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    data = actual.read_bytes()
    assert data[:4] == b"GGUF"
    version, tensor_count, kv_count = struct.unpack_from("<IQQ", data, 4)
    assert version == 3
    assert tensor_count == 1
    assert kv_count >= 3
    assert "File type: MOSTLY_Q1_0" in result.stdout


def test_native_executable_threads_and_timings_flags_preserve_output(tmp_path: Path) -> None:
    exe = _native_exe()
    key = "double_layers.3.modX.1.weight"
    rows = np.linspace(-2.0, 2.0, 512, dtype=np.float32).reshape(2, 256)
    src = tmp_path / "model.safetensors"
    expected = tmp_path / "expected.gguf"
    actual = tmp_path / "actual.gguf"
    _write_safetensors(src, {key: ("F32", rows.shape, rows.tobytes())})

    convert_safetensors_to_gguf_native(src, expected, "Q4_0", policy="uniform", overwrite=True)
    result = subprocess.run(
        [
            str(exe),
            "--src",
            str(src),
            "--dst",
            str(actual),
            "--qtype",
            "Q4_0",
            "--policy",
            "uniform",
            "--threads",
            "2",
            "--timings",
            "--overwrite",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert actual.read_bytes() == expected.read_bytes()
    assert "Timings:" in result.stderr
    assert "threads=2" in result.stderr


def test_native_executable_handles_small_scratch_for_bf16_fused_path(tmp_path: Path) -> None:
    exe = _native_exe()
    key = "double_layers.3.modX.1.weight"
    rows = np.linspace(-2.0, 2.0, 1024, dtype=np.float32).reshape(4, 256)
    bf16 = (rows.view(np.uint32) >> np.uint32(16)).astype(np.uint16)
    src = tmp_path / "model.safetensors"
    expected = tmp_path / "expected.gguf"
    actual = tmp_path / "actual.gguf"
    _write_safetensors(src, {key: ("BF16", rows.shape, bf16.tobytes())})

    convert_safetensors_to_gguf_native(src, expected, "Q2_K", policy="uniform", overwrite=True)
    result = subprocess.run(
        [
            str(exe),
            "--src",
            str(src),
            "--dst",
            str(actual),
            "--qtype",
            "Q2_K",
            "--policy",
            "uniform",
            "--threads",
            "4",
            "--scratch-bytes",
            "4096",
            "--overwrite",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert actual.read_bytes() == expected.read_bytes()


@pytest.mark.parametrize("dtype,qtype", [("F16", "Q4_0"), ("BF16", "Q8_0")])
def test_native_executable_quantizes_half_precision_sources(tmp_path: Path, dtype: str, qtype: str) -> None:
    exe = _native_exe()
    key = "double_layers.3.modX.1.weight"
    rows = np.linspace(-2.0, 2.0, 512, dtype=np.float32).reshape(2, 256)
    if dtype == "F16":
        payload = rows.astype(np.float16).tobytes()
    else:
        payload = (rows.view(np.uint32) >> np.uint32(16)).astype(np.uint16).tobytes()
    src = tmp_path / "model.safetensors"
    expected = tmp_path / "expected.gguf"
    actual = tmp_path / "actual.gguf"
    _write_safetensors(src, {key: (dtype, rows.shape, payload)})

    convert_safetensors_to_gguf_native(src, expected, qtype, policy="uniform", overwrite=True)
    result = subprocess.run(
        [str(exe), "--src", str(src), "--dst", str(actual), "--qtype", qtype, "--policy", "uniform", "--overwrite"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert actual.read_bytes() == expected.read_bytes()


@pytest.mark.parametrize("qtype", ["Q4_K_S", "Q4_K_M", "Q5_K_M"])
def test_native_executable_matches_python_native_for_alias_file_types(tmp_path: Path, qtype: str) -> None:
    exe = _native_exe()
    key = "double_layers.3.modX.1.weight"
    rows = np.linspace(-1.5, 1.5, 512, dtype=np.float32).reshape(2, 256)
    src = tmp_path / "model.safetensors"
    expected = tmp_path / "expected.gguf"
    actual = tmp_path / "actual.gguf"
    _write_safetensors(src, {key: ("F32", rows.shape, rows.tobytes())})

    convert_safetensors_to_gguf_native(src, expected, qtype, policy="uniform", overwrite=True)
    result = subprocess.run(
        [str(exe), "--src", str(src), "--dst", str(actual), "--qtype", qtype, "--policy", "uniform", "--overwrite"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert actual.read_bytes() == expected.read_bytes()


def test_native_executable_rejects_q8_k_and_deferred_non_q(tmp_path: Path) -> None:
    exe = _native_exe()
    key = "double_layers.3.modX.1.weight"
    rows = np.zeros((2, 256), dtype=np.float32)
    src = tmp_path / "model.safetensors"
    _write_safetensors(src, {key: ("F32", rows.shape, rows.tobytes())})

    q8k = subprocess.run(
        [str(exe), "--src", str(src), "--qtype", "Q8_K"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert q8k.returncode != 0
    assert "Q8_K" in q8k.stderr

    iq = subprocess.run(
        [str(exe), "--src", str(src), "--qtype", "IQ4_NL"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert iq.returncode != 0
    assert "non-Q/K" in iq.stderr
