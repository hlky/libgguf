# Roadmap

This page separates completed infrastructure from the next work areas, so the
roadmap stays useful during active development.

## Current Baseline

- Native CPU row quantization/dequantization kernels are available through the C
  API and top-level Python row helpers.
- NumPy, Torch, and optional Torch CUDA backends exist for integration and
  parity testing.
- The native `libgguf_quantize_gguf` converter writes low-memory safetensors to
  GGUF and has an experimental CUDA backend for selected Q/K tensor qtypes.
- The CUDA converter path includes reusable buffers, optional two-slot pinned
  host pipelining, timing buckets, chunk accounting, and CPU verification modes.
- `gguf-inspect`, `gguf-validate`, and `gguf-compare` expose lightweight public
  GGUF inspection, structural validation, and descriptor/content comparison.
- `docs/support-matrix.md` is generated from source/backend maps and checked by
  tests/CI.
- Frozen native CPU exactness fixtures live in `tests/golden/manifest.json`,
  alongside generated CPU-reference backend parity tests.

## Next Work

- Expose a first-class public GGUF writer API instead of relying on converter
  internals for write behavior.
- Deepen validator coverage beyond structural descriptor checks, especially for
  metadata contracts, alignment edge cases, and tensor payload consistency.
- Add source dtype GPU input paths for F16/BF16 so CUDA conversion does less
  CPU-side preparation before upload.
- Broaden native converter CUDA support beyond the current Q/K-focused routing,
  with clear fallback and verification behavior for additional qtype families.
- Expand frozen exactness coverage across more qtypes, edge cases, CPU backends,
  and converter-level fixtures.
- Continue CUDA IQ quantization optimization while preserving byte equality.
- Build a wheel publishing plan, including explicit CPU backend choices and a
  policy for optional CUDA artifacts.
- Run broader model-architecture conversion and compatibility sweeps for Flux,
  SD3, HiDream, Cosmos, Wan/HunyuanVideo, SDXL/SD1, LTXV, Aura, and Lumina2.
- Explore optional Diffusers and ComfyUI-GGUF integration points once the public
  reader/writer and converter contracts are less volatile.

## Recently Completed

- Native converter CUDA backend for selected Q/K qtypes.
- Converter pinned host pipeline and timing/counter reporting.
- Generated support matrix and CI check.
- Golden exactness manifest and update/check script.
- Public lightweight reader/inspection/validation/comparison CLIs.

The native converter already contains GGUF write and safetensors payload logic.
The reader/writer roadmap item is about exposing a cleaner public API so
downstream code does not need to depend on converter internals.
