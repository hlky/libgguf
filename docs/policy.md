# Quantization Policy

Conversion policy in libgguf is deterministic tensor planning. It decides which tensors are quantized, skipped, stored at higher precision, or promoted to a different qtype based on tensor shape, name, architecture, policy, and explicit overrides.

It is not an automatic quality oracle. It encodes reproducible conversion rules that can be inspected and tested.

## Current Policies

| Policy | Behavior |
| --- | --- |
| `uniform` | Quantize eligible 2D weight tensors with the requested base qtype. |
| `comfy` | Apply architecture-aware skip and high-precision patterns, plus mixed qtype rules for selected tensor roles. |
| `dynamic` | Build on `comfy` and promote selected tensors by role and layer position. This includes ongoing investigation of Unsloth Dynamic-like behavior. |

The `dynamic` policy is libgguf's deterministic planning logic. It is informed by the broader idea of tensor-level qtype selection, including work such as [Unsloth Dynamic GGUFs](https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs), but it does not claim to reproduce Unsloth's calibration data, evaluation stack, or reported results.

Eligible tensors are generally 2D tensors whose names end in `weight`. Non-2D tensors and tensors below conversion thresholds are stored rather than quantized unless an implemented override changes the plan.

## Architecture Detection

The planner detects image-model architectures from tensor key patterns. Current templates include:

- Flux
- SD3
- Aura
- HiDream
- Cosmos
- HunyuanVideo / Wan-style models
- LTXV
- SDXL
- SD1
- Lumina2

Architecture detection drives skip patterns, high-precision patterns, and shape fixes for selected model families.

## Tensor Roles

The mixed and dynamic policies look for tensor role patterns such as:

- attention value projections;
- fused QKV projections;
- attention query/key/output projections;
- FFN up, gate, and down projections;
- architecture-specific high-precision or skipped tensors.

For K-family qtypes, policy promotion may move selected tensors upward, for example from `Q4_K` toward `Q5_K`, `Q6_K`, or `Q8_0`.

## Overrides

Implemented override controls:

- `--tensor-type PATTERN=QTYPE`: assign an explicit storage or quant qtype to matching tensors.
- `--include PATTERN`: force matching 2D tensors into quantization when possible.
- `--exclude PATTERN`: keep matching tensors unquantized.

Patterns use shell-style matching through `fnmatch` behavior.

Shape or qtype constraints can still force a fallback to storage. Fallback counts are reported by the conversion result.
