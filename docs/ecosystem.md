# Ecosystem Context

This page explains adjacent projects and references that shape libgguf. These links are context, not compatibility guarantees.

## Upstream GGUF/GGML

- [llama.cpp](https://github.com/ggml-org/llama.cpp): upstream GGML/GGUF ecosystem and reference behavior for GGUF format and quantization semantics.
- [gguf-py](https://github.com/ggml-org/llama.cpp/tree/master/gguf-py): upstream Python package for GGUF reading/writing and quantization support.
- [gguf-py constants](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/constants.py): qtype and format constants reference.
- [gguf-py writer](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/gguf_writer.py): writer reference relevant to planned first-class libgguf reader/writer APIs.

libgguf is not official llama.cpp. It vendors and adapts compatible GGUF/GGML behavior into standalone reusable native, Python, NumPy, Torch, and CUDA infrastructure.

## ComfyUI

- [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF): existing community ComfyUI GGUF inference/custom-node integration.
- [ComfyUI-GGUF tools](https://github.com/city96/ComfyUI-GGUF/tree/main/tools): existing conversion tooling, including Python conversion steps and patched llama.cpp quantization workflow.

libgguf may replace or support parts of the ComfyUI-GGUF backend/tooling stack. The goal is reusable infrastructure: native kernels, Python bindings, Torch/NumPy parity backends, CUDA kernels, low-memory conversion, and deterministic policy planning.

## Diffusers

- [Diffusers GGUF docs](https://huggingface.co/docs/diffusers/quantization/gguf)
- [Diffusers main GGUF docs](https://huggingface.co/docs/diffusers/main/quantization/gguf)

Current Diffusers docs describe GGUF loading through `from_single_file` model classes, not pipeline-level GGUF loading. The documented path keeps GGUF weights in a low-memory dtype, typically `torch.uint8`, and dynamically dequantizes/casts during module forward to the configured compute dtype. Diffusers currently lists `BF16`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, and K-family qtypes through `Q6_K` as supported.

Diffusers also documents optional optimized CUDA kernels through the [Hugging Face kernels](https://github.com/huggingface/kernels) package and notes that those kernels may introduce minor numerical or visual differences. libgguf can be positioned as a potential byte-exact optional GGUF backend/integration target for Diffusers, but no Diffusers integration is claimed in this repository yet.

## Model Distribution Context

- [city96/FLUX.1-dev-gguf](https://huggingface.co/city96/FLUX.1-dev-gguf): widely used FLUX.1-dev GGUF model repository and a useful real-world compatibility target.
- [city96 Hugging Face profile](https://huggingface.co/city96): broader GGUF image-model distribution context.

These repositories are useful examples for conversion and inference compatibility testing. Mentioning them does not imply endorsement or guaranteed support.

## Policy Research Context

- [Unsloth Dynamic 2.0 GGUFs](https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs)
- [Unsloth Dynamic GGUFs on Aider Polyglot](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot)

Unsloth Dynamic is useful background for tensor-level qtype decisions and quality-aware dynamic quantization. libgguf's `dynamic` policy is deterministic tensor planning and analysis work. It does not claim to reproduce Unsloth's calibration data, benchmark methodology, or reported scores.

## Format Background

- [DeepWiki GGUF file-format overview](https://deepwiki.com/ggml-org/llama.cpp/7.1-gguf-file-format)
- [GGUF format concept page](https://www.mintlify.com/ggml-org/llama.cpp/concepts/gguf-format)

These references are useful for high-level background on GGUF as a single-file, metadata-rich, mmap-friendly storage format with tensor descriptors and aligned tensor data. The source of truth for implementation compatibility remains upstream llama.cpp/GGML/GGUF behavior and libgguf's exactness tests.
