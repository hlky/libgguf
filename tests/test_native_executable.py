from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
import sysconfig
from pathlib import Path

import numpy as np
import pytest

from libgguf.quantize import convert_safetensors_to_gguf_native


def _native_exe_names() -> tuple[str, str]:
    base = "libgguf_quantize_gguf"
    if os.name == "nt":
        return (f"{base}.exe", base)
    return (base, f"{base}.exe")


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
    for name in _native_exe_names():
        candidates.extend(root.glob(f"build/cmake/**/{name}"))
    scripts_dir = sysconfig.get_path("scripts")
    if scripts_dir:
        candidates.extend(Path(scripts_dir) / name for name in _native_exe_names())
    for name in _native_exe_names():
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    pytest.skip("libgguf_quantize_gguf executable is not built")


def _documented_native_cli_options() -> set[str]:
    root = Path(__file__).resolve().parents[1]
    docs = (root / "docs" / "cli.md").read_text(encoding="utf-8")
    _, options_section = docs.split("Implemented native options:", 1)
    options_section, _ = options_section.split("The native executable currently supports", 1)
    options: set[str] = set()
    for line in options_section.splitlines():
        stripped = line.strip()
        if stripped.startswith("- `") and "`:" in stripped:
            options.add(stripped.split("`:", 1)[0][3:])
    return options


def test_native_exe_prefers_env_then_repo_build_then_current_env_scripts_then_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_test_file = tmp_path / "repo" / "tests" / "test_native_executable.py"
    fake_test_file.parent.mkdir(parents=True)
    fake_test_file.touch()
    repo_exe = tmp_path / "repo" / "build" / "cmake" / "out" / "libgguf_quantize_gguf"
    repo_exe.parent.mkdir(parents=True)
    repo_exe.touch()
    scripts_exe = tmp_path / "scripts" / "libgguf_quantize_gguf"
    scripts_exe.parent.mkdir()
    scripts_exe.touch()
    path_exe = tmp_path / "bin" / "libgguf_quantize_gguf"
    path_exe.parent.mkdir()
    path_exe.touch()
    env_exe = tmp_path / "env" / "libgguf_quantize_gguf"
    env_exe.parent.mkdir()
    env_exe.touch()
    monkeypatch.setitem(globals(), "__file__", str(fake_test_file))
    monkeypatch.setattr(sysconfig, "get_path", lambda name: str(scripts_exe.parent) if name == "scripts" else None)
    monkeypatch.setattr(shutil, "which", lambda name: str(path_exe))

    monkeypatch.setenv("LIBGGUF_QUANTIZE_GGUF_EXE", str(env_exe))
    assert _native_exe() == env_exe

    monkeypatch.delenv("LIBGGUF_QUANTIZE_GGUF_EXE")
    assert _native_exe() == repo_exe

    repo_exe.unlink()
    assert _native_exe() == scripts_exe

    scripts_exe.unlink()
    assert _native_exe() == path_exe


def test_native_exe_checks_windows_exe_name_in_current_env_scripts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_test_file = tmp_path / "repo" / "tests" / "test_native_executable.py"
    fake_test_file.parent.mkdir(parents=True)
    fake_test_file.touch()
    scripts_exe = tmp_path / "scripts" / "libgguf_quantize_gguf.exe"
    scripts_exe.parent.mkdir()
    scripts_exe.touch()
    monkeypatch.setitem(globals(), "__file__", str(fake_test_file))
    monkeypatch.delenv("LIBGGUF_QUANTIZE_GGUF_EXE", raising=False)
    monkeypatch.setattr(sysconfig, "get_path", lambda name: str(scripts_exe.parent) if name == "scripts" else None)
    monkeypatch.setattr(shutil, "which", lambda name: None)

    assert _native_exe() == scripts_exe


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


def test_native_executable_matches_python_native_for_dynamic_policy(tmp_path: Path) -> None:
    exe = _native_exe()
    rows = np.linspace(-2.0, 2.0, 512, dtype=np.float32).reshape(2, 256)
    src = tmp_path / "model.safetensors"
    expected = tmp_path / "expected.gguf"
    actual = tmp_path / "actual.gguf"
    _write_safetensors(
        src,
        {
            "double_layers.3.modX.1.weight": ("F32", rows.shape, rows.tobytes()),
            "double_layers.0.mlpC.c_proj.weight": ("F32", rows.shape, rows.tobytes()),
        },
    )

    convert_safetensors_to_gguf_native(src, expected, "Q5_K_M", policy="dynamic", overwrite=True)
    result = subprocess.run(
        [str(exe), "--src", str(src), "--dst", str(actual), "--qtype", "Q5_K_M", "--policy", "dynamic", "--overwrite"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert actual.read_bytes()[:4] == b"GGUF"
    assert expected.read_bytes()[:4] == b"GGUF"
    assert result.stdout.count("Q5_K=1") == 1
    assert "Q8_0=1" in result.stdout


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
    assert "cpu_convert=" in result.stderr
    assert "h2d=" in result.stderr
    assert "cuda_quant=" in result.stderr
    assert "d2h=" in result.stderr


def test_native_executable_help_lists_backend_flags() -> None:
    exe = _native_exe()

    result = subprocess.run([str(exe), "--help"], check=False, capture_output=True, text=True)

    assert result.returncode == 0
    assert "--backend cpu|cuda" in result.stdout
    assert "--cuda-fallback cpu" in result.stdout
    assert "--verify-cuda-tensors N" in result.stdout
    assert "--cuda-vram-bytes N" in result.stdout
    assert "--cpu-ram-bytes N" in result.stdout


@pytest.mark.parametrize(
    ("flag_group", "documented_spellings"),
    [
        ("policy", ["--policy comfy|dynamic|uniform"]),
        ("include/exclude", ["--include PATTERN", "--exclude PATTERN"]),
        ("scratch", ["--scratch-bytes N", "--cpu-ram-bytes N"]),
        ("timing", ["--threads N", "--timings"]),
        ("backend", ["--backend cpu|cuda"]),
        (
            "CUDA",
            [
                "--cuda-fallback cpu",
                "--verify-cuda-tensors N",
                "--cuda-vram-bytes N",
            ],
        ),
    ],
)
def test_native_executable_help_matches_documented_flag_groups(
    flag_group: str, documented_spellings: list[str]
) -> None:
    exe = _native_exe()
    documented_options = _documented_native_cli_options()

    result = subprocess.run([str(exe), "--help"], check=False, capture_output=True, text=True)

    assert result.returncode == 0
    missing_from_docs = [spelling for spelling in documented_spellings if spelling not in documented_options]
    missing_from_help = [spelling for spelling in documented_spellings if spelling not in result.stdout]
    assert not missing_from_docs, f"{flag_group} options missing from docs/cli.md: {missing_from_docs}"
    assert not missing_from_help, f"{flag_group} options missing from --help: {missing_from_help}"


def test_native_executable_backend_cpu_preserves_default_output(tmp_path: Path) -> None:
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
            "--backend",
            "cpu",
            "--overwrite",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert actual.read_bytes() == expected.read_bytes()


def test_native_executable_verify_cuda_requires_cuda_backend(tmp_path: Path) -> None:
    exe = _native_exe()
    key = "double_layers.3.modX.1.weight"
    rows = np.zeros((2, 256), dtype=np.float32)
    src = tmp_path / "model.safetensors"
    _write_safetensors(src, {key: ("F32", rows.shape, rows.tobytes())})

    result = subprocess.run(
        [str(exe), "--src", str(src), "--qtype", "Q4_0", "--verify-cuda-tensors", "1"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--verify-cuda-tensors requires --backend cuda" in result.stderr


def test_native_executable_cuda_vram_bytes_requires_cuda_backend(tmp_path: Path) -> None:
    exe = _native_exe()
    key = "double_layers.3.modX.1.weight"
    rows = np.zeros((2, 256), dtype=np.float32)
    src = tmp_path / "model.safetensors"
    _write_safetensors(src, {key: ("F32", rows.shape, rows.tobytes())})

    result = subprocess.run(
        [str(exe), "--src", str(src), "--qtype", "Q4_0", "--cuda-vram-bytes", "4096"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--cuda-vram-bytes requires --backend cuda" in result.stderr


def test_native_executable_cpu_ram_bytes_alias_preserves_output(tmp_path: Path) -> None:
    exe = _native_exe()
    key = "double_layers.3.modX.1.weight"
    rows = np.linspace(-2.0, 2.0, 1024, dtype=np.float32).reshape(4, 256)
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
            "--cpu-ram-bytes",
            "4096",
            "--overwrite",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert actual.read_bytes() == expected.read_bytes()


def test_native_executable_cuda_backend_cpu_only_failure_is_clear(tmp_path: Path) -> None:
    exe = _native_exe()
    key = "double_layers.3.modX.1.weight"
    rows = np.zeros((2, 256), dtype=np.float32)
    src = tmp_path / "model.safetensors"
    _write_safetensors(src, {key: ("F32", rows.shape, rows.tobytes())})

    result = subprocess.run(
        [str(exe), "--src", str(src), "--qtype", "Q4_0", "--backend", "cuda"],
        check=False,
        capture_output=True,
        text=True,
    )

    if result.returncode == 0 or "built without native CUDA support" not in result.stderr:
        pytest.skip("native CUDA converter support is present or failed before CPU-only support check")
    assert "CUDA backend requested" in result.stderr


@pytest.mark.parametrize("qtype", ["Q4_0", "Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"])
def test_native_executable_cuda_backend_matches_cpu_when_available(tmp_path: Path, qtype: str) -> None:
    exe = _native_exe()
    key = "double_layers.3.modX.1.weight"
    rows = np.linspace(-2.0, 2.0, 512, dtype=np.float32).reshape(2, 256)
    src = tmp_path / "model.safetensors"
    expected = tmp_path / "expected.gguf"
    actual = tmp_path / "actual.gguf"
    _write_safetensors(src, {key: ("F32", rows.shape, rows.tobytes())})

    convert_safetensors_to_gguf_native(src, expected, qtype, policy="uniform", overwrite=True)
    result = subprocess.run(
        [
            str(exe),
            "--src",
            str(src),
            "--dst",
            str(actual),
            "--qtype",
            qtype,
            "--policy",
            "uniform",
            "--backend",
            "cuda",
            "--cuda-vram-bytes",
            "4096",
            "--verify-cuda-tensors",
            "1",
            "--timings",
            "--overwrite",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    unavailable = (
        "built without native CUDA support",
        "failed to initialize CUDA backend",
        "CUDA driver",
        "CUDA-capable device",
    )
    if result.returncode != 0 and any(fragment in result.stderr for fragment in unavailable):
        pytest.skip("native CUDA converter support is unavailable")
    assert result.returncode == 0, result.stderr
    assert actual.read_bytes() == expected.read_bytes()
    assert "cuda_tensors=1" in result.stderr
    assert "cuda_verified=1" in result.stderr
    assert "cuda_vram=4096" in result.stderr
    assert "cuda_max_input=" in result.stderr
    assert "cuda_max_output=" in result.stderr


def test_native_executable_cuda_pipeline_matches_cpu_when_available(tmp_path: Path) -> None:
    exe = _native_exe()
    key = "double_layers.3.modX.1.weight"
    rows = np.linspace(-2.0, 2.0, 2048, dtype=np.float32).reshape(8, 256)
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
            "--backend",
            "cuda",
            "--cuda-vram-bytes",
            "1536",
            "--overwrite",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    unavailable = (
        "built without native CUDA support",
        "failed to initialize CUDA backend",
        "CUDA driver",
        "CUDA-capable device",
    )
    if result.returncode != 0 and any(fragment in result.stderr for fragment in unavailable):
        pytest.skip("native CUDA converter support is unavailable")
    assert result.returncode == 0, result.stderr
    assert actual.read_bytes() == expected.read_bytes()


def test_native_executable_cuda_unsupported_qtype_requires_fallback_when_available(tmp_path: Path) -> None:
    exe = _native_exe()
    key = "double_layers.3.modX.1.weight"
    rows = np.linspace(-2.0, 2.0, 512, dtype=np.float32).reshape(2, 256)
    src = tmp_path / "model.safetensors"
    expected = tmp_path / "expected.gguf"
    actual = tmp_path / "actual.gguf"
    _write_safetensors(src, {key: ("F32", rows.shape, rows.tobytes())})

    unsupported = subprocess.run(
        [
            str(exe),
            "--src",
            str(src),
            "--qtype",
            "Q4_1",
            "--policy",
            "uniform",
            "--backend",
            "cuda",
            "--overwrite",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    unavailable = (
        "built without native CUDA support",
        "failed to initialize CUDA backend",
        "CUDA driver",
        "CUDA-capable device",
    )
    if any(fragment in unsupported.stderr for fragment in unavailable):
        pytest.skip("native CUDA converter support is unavailable")
    assert unsupported.returncode != 0
    assert "does not support Q4_1" in unsupported.stderr
    assert "--cuda-fallback cpu" in unsupported.stderr

    convert_safetensors_to_gguf_native(src, expected, "Q4_1", policy="uniform", overwrite=True)
    fallback = subprocess.run(
        [
            str(exe),
            "--src",
            str(src),
            "--dst",
            str(actual),
            "--qtype",
            "Q4_1",
            "--policy",
            "uniform",
            "--backend",
            "cuda",
            "--cuda-fallback",
            "cpu",
            "--overwrite",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert fallback.returncode == 0, fallback.stderr
    assert actual.read_bytes() == expected.read_bytes()


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


def test_native_executable_rejects_non_safetensors(tmp_path: Path) -> None:
    exe = _native_exe()
    bad_src = tmp_path / "model.pt"
    bad_src.write_bytes(b"not a checkpoint")

    result = subprocess.run(
        [str(exe), "--src", str(bad_src), "--qtype", "Q4_0"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "only supports .safetensors" in result.stderr
