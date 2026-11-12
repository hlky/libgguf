# Source layout and builds

libgguf supplies format contracts and CPU/GPU primitives to the kernel libraries.
Its Python package remains `src/libgguf`.

| Path | Responsibility |
|---|---|
| `include/libgguf/common/` | Shared format and value types |
| `include/libgguf/gpu/` | Shared GPU semantic and storage contracts |
| `include/libgguf/rocm/<architecture>/` | Target instruction and fragment implementations |
| `src/core/` | Shared format implementation |
| `src/cpu/<target>/` | CPU primitives for ref, sse2, sse4_1, and avx2 |
| `src/gpu/` | Shared GPU implementation and runtime |
| `src/bindings/` | Python and Torch native bindings |
| `src/conversion/` | Conversion and serialization |
| `src/cli/` | Command line tools |

Each ROCm pack owns `architecture.h` and `pack.json`. Packs for gfx1151,
gfx1201, and gfx942 do not import each other. Numeric API revisions remain
metadata; canonical declarations have no revision suffixes.

For a CPU-only native build:

```console
cmake -S . -B build/cpu/ref -G Ninja -DLIBGGUF_CPU_TARGET=REF -DLIBGGUF_BUILD_PYTHON=OFF -DLIBGGUF_BUILD_TESTS=ON
cmake --build build/cpu/ref
ctest --test-dir build/cpu/ref --output-on-failure
```

For ROCm, also set `LIBGGUF_BUILD_GPU=ON`, `LIBGGUF_ENABLE_ROCM=ON`, and
`LIBGGUF_ARCHITECTURE=gfx942`. Select exactly one architecture and use a separate
build directory for it. GPU compilation alone does not qualify hardware behavior.
