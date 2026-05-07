#!/usr/bin/env python3
from __future__ import annotations

import argparse
import configparser
from email.parser import Parser
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
import textwrap
import venv
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def _scripts_dir(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts" if os.name == "nt" else "bin")


def _python(venv_dir: Path) -> Path:
    return _scripts_dir(venv_dir) / ("python.exe" if os.name == "nt" else "python")


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print(f"+ cd {cwd}")
    print("+ " + " ".join(command))
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _build_wheel(python: str, tmp_dir: Path, no_build_isolation: bool) -> Path:
    wheelhouse = tmp_dir / "wheelhouse"
    wheelhouse.mkdir()
    command = [
        python,
        "-m",
        "pip",
        "wheel",
        str(ROOT),
        "--wheel-dir",
        str(wheelhouse),
        "--config-settings=cmake.define.LIBGGUF_BUILD_CUDA_KERNELS=OFF",
    ]
    if no_build_isolation:
        command.append("--no-build-isolation")
    _run(command, cwd=tmp_dir)

    wheels = sorted(wheelhouse.glob("libgguf-*.whl"))
    if len(wheels) != 1:
        names = ", ".join(wheel.name for wheel in wheels) or "none"
        raise RuntimeError(f"expected one libgguf wheel in {wheelhouse}, found: {names}")
    return wheels[0]


def _check_wheel_archive(wheel: Path) -> None:
    print(f"+ inspect wheel archive {wheel}")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        paths = [PurePosixPath(name) for name in names]
        entries = set(names)

        dist_info_dirs = sorted(
            {
                path.parts[0]
                for path in paths
                if len(path.parts) > 1
                and path.parts[0].startswith("libgguf-")
                and path.parts[0].endswith(".dist-info")
            }
        )
        errors: list[str] = []
        if len(dist_info_dirs) != 1:
            found = ", ".join(dist_info_dirs) or "none"
            errors.append(f"expected one libgguf dist-info directory, found: {found}")
            dist_info = None
        else:
            dist_info = dist_info_dirs[0]

        required_package_files = (
            "libgguf/__init__.py",
            "libgguf/_metadata.py",
            "libgguf/compare.py",
            "libgguf/imatrix.py",
            "libgguf/inspect.py",
            "libgguf/quantize.py",
            "libgguf/libgguf_numpy/__init__.py",
            "libgguf/libgguf_torch/__init__.py",
            "libgguf/libgguf_cuda/__init__.py",
        )
        for filename in required_package_files:
            if filename not in entries:
                errors.append(f"missing package file: {filename}")

        extension_files = [
            name
            for name in names
            if name.startswith("libgguf/_libgguf.") and PurePosixPath(name).suffix in {".so", ".pyd"}
        ]
        if len(extension_files) != 1:
            found = ", ".join(extension_files) or "none"
            errors.append(f"expected one native extension libgguf/_libgguf.*(.so|.pyd), found: {found}")

        native_scripts = [
            name
            for name in names
            if len(PurePosixPath(name).parts) == 3
            and PurePosixPath(name).parts[0].startswith("libgguf-")
            and PurePosixPath(name).parts[0].endswith(".data")
            and PurePosixPath(name).parts[1] == "scripts"
            and PurePosixPath(name).parts[2] in {"libgguf_quantize_gguf", "libgguf_quantize_gguf.exe"}
        ]
        if len(native_scripts) != 1:
            found = ", ".join(native_scripts) or "none"
            errors.append(f"expected one native executable script entry for libgguf_quantize_gguf, found: {found}")

        packaged_backend_tests = [
            name
            for name in names
            if len(PurePosixPath(name).parts) >= 4
            and PurePosixPath(name).parts[0] == "libgguf"
            and PurePosixPath(name).parts[1].startswith("libgguf_")
            and PurePosixPath(name).parts[2] == "tests"
        ]
        if packaged_backend_tests:
            errors.append("packaged backend tests found: " + ", ".join(sorted(packaged_backend_tests)))

        if dist_info is not None:
            required_dist_info_files = {
                f"{dist_info}/METADATA",
                f"{dist_info}/WHEEL",
                f"{dist_info}/RECORD",
                f"{dist_info}/entry_points.txt",
            }
            for filename in sorted(required_dist_info_files):
                if filename not in entries:
                    errors.append(f"missing dist-info file: {filename}")

            metadata_name = f"{dist_info}/METADATA"
            if metadata_name in entries:
                metadata = Parser().parsestr(archive.read(metadata_name).decode("utf-8"))
                if metadata.get("Name") != "libgguf":
                    errors.append(f"METADATA Name is {metadata.get('Name')!r}, expected 'libgguf'")
                if not metadata.get("Version"):
                    errors.append("METADATA Version is missing")
                if metadata.get("Requires-Python") != ">=3.10":
                    errors.append(
                        f"METADATA Requires-Python is {metadata.get('Requires-Python')!r}, expected '>=3.10'"
                    )

            wheel_metadata_name = f"{dist_info}/WHEEL"
            if wheel_metadata_name in entries:
                wheel_metadata = Parser().parsestr(archive.read(wheel_metadata_name).decode("utf-8"))
                if not wheel_metadata.get("Wheel-Version"):
                    errors.append("WHEEL Wheel-Version is missing")
                if not wheel_metadata.get_all("Tag"):
                    errors.append("WHEEL Tag is missing")

            entry_points_name = f"{dist_info}/entry_points.txt"
            if entry_points_name in entries:
                parser = configparser.ConfigParser()
                parser.optionxform = str
                parser.read_string(archive.read(entry_points_name).decode("utf-8"))
                console_scripts = parser["console_scripts"] if parser.has_section("console_scripts") else {}
                expected_console_scripts = {
                    "gguf-compare": "libgguf.compare:main",
                    "gguf-inspect": "libgguf.inspect:main",
                    "gguf-validate": "libgguf.inspect:validate_main",
                }
                for script_name, target in expected_console_scripts.items():
                    actual = console_scripts.get(script_name)
                    if actual != target:
                        errors.append(f"console script {script_name!r} is {actual!r}, expected {target!r}")

        if errors:
            message = "\n".join(f"- {error}" for error in errors)
            raise RuntimeError(f"wheel archive contract failed:\n{message}")

    print("Wheel archive contract passed")


def _create_venv(venv_dir: Path) -> Path:
    print(f"+ create venv {venv_dir}")
    venv.EnvBuilder(with_pip=True).create(venv_dir)
    return _python(venv_dir)


def _smoke_python(venv_python: Path, smoke_dir: Path) -> None:
    code = r"""
    import numpy as np
    import libgguf
    from libgguf import GGMLQuantizationType, load_imatrix, quantize_requires_imatrix

    rows = np.arange(64, dtype=np.float32).reshape(2, 32) / np.float32(16.0)
    encoded = libgguf.quantize_rows(rows, GGMLQuantizationType.Q8_0)
    decoded = libgguf.dequantize_rows(encoded, GGMLQuantizationType.Q8_0, n_per_row=32)

    assert encoded.dtype == np.uint8
    assert encoded.shape == (2, libgguf.row_size(GGMLQuantizationType.Q8_0, 32))
    assert decoded.dtype == np.float32
    assert decoded.shape == rows.shape
    assert np.all(np.isfinite(decoded))

    assert load_imatrix is libgguf.load_imatrix
    assert quantize_requires_imatrix is libgguf.quantize_requires_imatrix

    storage_rows = np.array([[0.0, -0.0, 1.0, -2.0], [0.5, -0.5, 3.5, 65504.0]], dtype=np.float32)
    bf16 = libgguf.store_rows(storage_rows, GGMLQuantizationType.BF16)
    storage_bits = storage_rows.view(np.uint32)
    expected_bf16 = ((storage_bits + (0x7FFF + ((storage_bits >> 16) & 1))) >> 16).astype(np.uint16)
    assert bf16.dtype == np.uint8
    assert bf16.shape == (2, libgguf.row_size(GGMLQuantizationType.BF16, 4))
    assert np.array_equal(bf16.view(np.uint16), expected_bf16)

    weighted_qtype = GGMLQuantizationType.IQ2_XXS
    weighted_rows = np.linspace(-1.0, 1.0, 256, dtype=np.float32).reshape(1, 256)
    imatrix = np.linspace(0.25, 1.75, 256, dtype=np.float32)
    weighted = libgguf.quantize_rows(weighted_rows, weighted_qtype, imatrix=imatrix)
    assert libgguf.quantize_requires_imatrix(weighted_qtype)
    assert weighted.dtype == np.uint8
    assert weighted.shape == (1, libgguf.row_size(weighted_qtype, 256))
    print("python smoke ok")
    """
    _run([str(venv_python), "-c", textwrap.dedent(code)], cwd=smoke_dir)


def _smoke_entry_points(venv_dir: Path, smoke_dir: Path) -> None:
    scripts_dir = _scripts_dir(venv_dir)
    path = os.pathsep.join([str(scripts_dir), os.environ.get("PATH", "")])
    env = {**os.environ, "PATH": path}

    for command_name in ("gguf-compare", "gguf-inspect", "gguf-validate"):
        _run([command_name, "--help"], cwd=smoke_dir, env=env)

    native_name = "libgguf_quantize_gguf"
    native_path = shutil.which(native_name, path=path)
    if native_path is None and os.name == "nt":
        native_path = shutil.which(f"{native_name}.exe", path=path)
    if native_path is None:
        raise RuntimeError(f"{native_name} was not installed on PATH")
    _run([native_path, "--help"], cwd=smoke_dir, env=env)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a CPU-only wheel, install it into a fresh venv, and run out-of-repo smoke checks."
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to build the wheel. Defaults to the current interpreter.",
    )
    parser.add_argument(
        "--no-build-isolation",
        action="store_true",
        help="Pass --no-build-isolation to pip wheel for local toolchain debugging.",
    )
    parser.add_argument(
        "--preserve-temp",
        action="store_true",
        help="Keep the temporary wheelhouse, venv, and smoke directory after the run.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    temp_parent = Path(tempfile.mkdtemp(prefix="libgguf-wheel-smoke-"))
    remove_temp = not args.preserve_temp
    print(f"Temporary workspace: {temp_parent}")
    if args.preserve_temp:
        print("Preserving temporary workspace after the run.")

    try:
        wheel = _build_wheel(args.python, temp_parent, args.no_build_isolation)
        _check_wheel_archive(wheel)
        venv_python = _create_venv(temp_parent / "venv")
        _run([str(venv_python), "-m", "pip", "install", str(wheel)], cwd=temp_parent)
        smoke_dir = temp_parent / "smoke"
        smoke_dir.mkdir()
        _smoke_python(venv_python, smoke_dir)
        _smoke_entry_points(temp_parent / "venv", smoke_dir)
        print(f"Wheel install smoke passed: {wheel}")
        return 0
    finally:
        if remove_temp:
            shutil.rmtree(temp_parent, ignore_errors=True)
        else:
            print(f"Temporary workspace preserved at: {temp_parent}")


if __name__ == "__main__":
    raise SystemExit(main())
