# Roadmap

Planned or in-progress areas:

- First-class GGUF reader/writer API.
- GGUF validator.
- Native converter CUDA backend:
  - safetensors -> GPU upload -> CUDA quantization -> CPU encoded bytes -> GGUF write.
- Pinned host buffers and timing buckets for converter profiling.
- F16/BF16 source tensor GPU input paths.
- Golden exactness tests and fixtures.
- Generated support matrix from source/tests.
- CUDA IQ quantization polish.
- Packaging and wheels.
- Diffusers optional backend/integration exploration.
- ComfyUI-GGUF backend/tooling support or replacement exploration.

The native converter already contains GGUF write and safetensors payload logic, and `gguf-inspect` now exposes header/metadata/tensor descriptor inspection. The roadmap item is to expose a cleaner public reader/writer surface rather than asking downstream code to depend on converter internals.
