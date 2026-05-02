# libgguf

Standalone CPython bindings and command-line helpers for minimized GGUF reference quantizers derived from llama.cpp.

The native extension builds as one portable artifact. On x86/x64 it includes runtime-dispatched Q4_0 and Q8_0 scalar, SSE2, and AVX2 paths where the compiler and CPU support them; AVX2 is compiled only for isolated AVX2 translation units, not as a global build mode.

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

See `docs/performance.md` for Q4_0 SIMD benchmark findings and the local benchmark method.

Run the conversion CLI:

```bash
python -m libgguf.quantize_gguf --src model.safetensors --dst model.gguf --qtype Q8_0
```

## Python API

Quantize rows into a newly allocated bytes object:

```python
import libgguf

Q8_0 = 8
raw = libgguf.quantize_rows_raw(Q8_0, rows, rows.shape[0], rows.shape[1])
```

Or reuse a preallocated writable buffer:

```python
Q8_0 = 8
dst = bytearray(libgguf.row_size(Q8_0, rows.shape[1]) * rows.shape[0])
written = libgguf.quantize_rows_into_raw(Q8_0, rows, dst, rows.shape[0], rows.shape[1])
```

## Layout

- `src/libgguf/` exposes the Python package and CLI.
- `include/libgguf.h` is the public C ABI header.
- `csrc/` contains the CPython extension shim and native quantizer implementation.
- `scripts/build_libgguf.py` builds the standalone shared library for callers that use the C ABI directly.
- `tests/` contains standalone package, C ABI, imatrix, and conversion tests.
