# libgguf Documentation

libgguf is a standalone GGUF infrastructure library and toolkit for native C++, Python, NumPy, Torch, and CUDA quantization/dequantization workflows.

Start here:

- [Installation](installation.md): editable installs, extras, CMake options, and CUDA build notes.
- [CLI](cli.md): `gguf-inspect`, `gguf-validate`, `gguf-compare`, and the native `libgguf_quantize_gguf` conversion executable.
- [Python API](python-api.md): stable-ish row APIs, conversion helpers, GGUF inspection/validation, and experimental APIs.
- [CUDA](cuda.md): optional Torch CUDA extension, API shape, exactness goals, and limitations.
- [Packaging](packaging.md): editable/wheel build behavior, optional CUDA builds, and current limitations.
- [Qtypes](qtypes.md): GGUF qtype families and storage types.
- [Policy](policy.md): deterministic tensor planning for image-model conversion.
- [Benchmarks](benchmarks.md): representative RTX 3090 benchmark tables and metric definitions.
- [Correctness](correctness.md): byte exactness, decoded-output checks, and frozen golden fixtures.
- [Support Matrix](support-matrix.md): conservative backend support overview.
- [Ecosystem](ecosystem.md): upstream and adjacent project context.
- [Roadmap](roadmap.md): planned reader/writer, deeper validator, converter, CUDA, packaging, and integration work.
- [Development](development.md): build, test, benchmark, and optimization notes.
