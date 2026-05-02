# AGENTS.md

## Project Overview

`libgguf` is a Python 3.10+ package that provides CPython bindings and command-line helpers for GGUF reference quantizers derived from llama.cpp. The package includes a native C++17 extension, Python conversion utilities, and tests for the extension, C ABI, imatrix loading, and safetensors-to-GGUF conversion.

## Repository Layout

- `__init__.py`, `imatrix.py`, `quantize.py`, and `quantize_gguf.py` are the Python package and CLI entry points.
- `_libgguf_module.cpp`, `libgguf.cpp`, and `libgguf.h` build the native quantizer extension.
- `scripts/build_libgguf.py` builds the standalone shared library used by C ABI callers and tests.
- `tests/` contains pytest coverage and local GGML type metadata used by the reference checks.
- `pyproject.toml`, `setup.py`, and `MANIFEST.in` define packaging and build behavior.

## Environment Setup

Use Python 3.10 or newer with a working C++17 compiler.

Install the package for local development:

```bash
python -m pip install -e . --no-build-isolation
```

Install test dependencies:

```bash
python -m pip install -e ".[test]" --no-build-isolation
```

Install optional direct quantization dependencies when working on `quantize.py` or `quantize_gguf.py`:

```bash
python -m pip install -e ".[quantize,test]" --no-build-isolation
```

## Build Commands

Build the CPython extension in place:

```bash
python setup.py build_ext --inplace
```

Build the standalone shared library:

```bash
python scripts/build_libgguf.py
```

On Windows this creates `libgguf.dll`; on Linux `libgguf.so`; on macOS `libgguf.dylib`.

## Test Commands

Run the full test suite:

```bash
python -m pytest
```

Run the native binding and C ABI tests:

```bash
python -m pytest tests/test_libgguf_reference.py
```

Run the direct GGUF conversion tests:

```bash
python -m pytest tests/test_quantize_gguf.py
```

`tests/test_quantize_gguf.py` skips unless `gguf`, `safetensors`, and `torch` are installed.

## Development Notes

- Keep native changes compatible with C++17 and the existing setuptools build flow.
- Preserve the Python package layout where the repository root is mapped to the `libgguf` package.
- Prefer NumPy arrays that are contiguous and explicitly typed when crossing the Python/native boundary.
- Keep quantization policy changes covered by focused pytest cases in `tests/test_quantize_gguf.py`.
- Keep extension or ABI changes covered by `tests/test_libgguf_reference.py`.
- Do not commit generated build outputs such as `build/`, `dist/`, extension binaries, shared libraries, egg-info, or pytest caches.

## CLI Smoke Test

After changing the conversion CLI, run a small conversion through the module entry point when optional dependencies are available:

```bash
python -m libgguf.quantize_gguf --src model.safetensors --dst model.gguf --qtype Q8_0
```

## .venv

Prefer `.venv` if it exists.
