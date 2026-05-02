# libgguf

Standalone CPython bindings and command-line helpers for minimized GGUF reference quantizers derived from llama.cpp.

## Install

Use Python 3.10+ with a C++17 compiler.

```bash
python -m pip install -e . --no-build-isolation
```

The base package depends on NumPy. Direct safetensors-to-GGUF conversion also needs optional runtime packages:

```bash
python -m pip install torch gguf safetensors
```

## Commands

Build the Python extension in place:

```bash
python setup.py build_ext --inplace
```

Build the standalone C ABI shared library:

```bash
python scripts/build_libgguf.py
```

Run tests:

```bash
python -m pytest
```

Run the conversion CLI:

```bash
python -m libgguf.quantize_gguf --src model.safetensors --dst model.gguf --qtype Q8_0
```

## Layout

- `__init__.py`, `imatrix.py`, `quantize.py`, and `quantize_gguf.py` expose the Python package and CLI.
- `_libgguf_module.cpp`, `libgguf.cpp`, and `libgguf.h` build the native quantizer extension.
- `scripts/build_libgguf.py` builds the standalone shared library for callers that use the C ABI directly.
- `tests/` contains standalone package, C ABI, imatrix, and conversion tests.
