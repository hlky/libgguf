# GGUF Policy Analysis

Inputs:

- `reports/gguf_comparison/standard_results.json`
- `reports/gguf_comparison/ud_policy_analysis.json`
- `reports/gguf_comparison/policy_override_candidates.json`
- spot reruns after policy changes:
  - AuraFlow `Q3_K_M`: now byte-identical.
  - HiDream Fast `Q3_K_M`: tensor qtypes now match; payload still differs.

## Standard City96-Style Policy

The standard results contained two concrete qtype-policy gaps.

### HiDream Shared Expert Down Projection

Affected repos/variants:

- `city96/HiDream-I1-Dev-gguf`
- `city96/HiDream-I1-Fast-gguf`
- `city96/HiDream-I1-Full-gguf`

Affected tensors:

- `double_stream_blocks.*.block.ff_i.shared_experts.w2.weight`
- `single_stream_blocks.*.block.ff_i.shared_experts.w2.weight`

Required rule: treat `*shared_experts.w2.weight*` as an FFN-down tensor.

Expected target changes:

| File variant | Base tensor qtype | Reference tensor qtype |
|---|---|---|
| `Q3_K_M` | `Q3_K` | `Q4_K` |
| `Q4_0` | `Q4_0` | `Q4_1` |
| `Q4_K_S` | `Q4_K` | `Q5_K` |
| `Q4_K_M` | `Q4_K` | `Q6_K` |
| `Q5_0` | `Q5_0` | `Q5_1` |
| `Q5_K_M` | `Q5_K` | `Q6_K` |

This has been implemented in both Python and native executable policy code.

Post-change verification:

- `city96/HiDream-I1-Fast-gguf` `Q3_K_M` now has zero tensor qtype diffs.
- It remains non-byte-identical, so the remaining issue is payload-level, not qtype policy.

### AuraFlow Q3_K_M MLP Output Projection

Affected repo/variant:

- `city96/AuraFlow-v0.3-gguf` `Q3_K_M`

Affected tensors:

- `double_layers.*.mlpC.c_proj.weight`
- `double_layers.*.mlpX.c_proj.weight`

Required rule: for Aura only, when the file variant is `Q3_K_M`, promote these tensors from `Q3_K` to `Q4_K`.

This has been implemented in both Python and native executable policy code.

Post-change verification:

- `city96/AuraFlow-v0.3-gguf` `Q3_K_M` is now byte-identical to the HF reference.

## Not Policy Changes

### FLUX.1-schnell

The FLUX.1-schnell mismatches are dominated by non-quantized tensor storage type differences, not quantized tensor policy:

- local generated output uses `BF16=8, F32=464`.
- HF references use `F16`/`F32` for the skipped tensors.

The selected source candidate is `Comfy-Org/flux1-schnell/flux1-schnell.safetensors`, which is BF16. The likely original source is `black-forest-labs/FLUX.1-schnell/flux1-schnell.safetensors`, but that file is gated and currently returns `401 Unauthorized`.

Do not change global storage policy to match this result. The better fix is to validate against the gated source when credentials are available.

### HiDream Payload-Only Differences

HiDream `Q2_K`, `Q3_K_S`, and post-policy `Q3_K_M` still have payload differences with matching keys, metadata, tensor names, tensor shapes, and tensor qtypes.

For `HiDream-I1-Full Q3_K_S`, a fresh build of the city96 README tooling produced the same selected `Q3_K` tensor bytes as our generator, not the HF reference. That points to a different HF quantizer build/version/platform or slightly different source tensor contents, not a policy mismatch.

## Unsloth Dynamic Policy

UD variants have no explicit Unsloth metadata keys in the analyzed GGUF headers. Confidence is therefore `inferred_from_tensor_types`.

Analyzed UD entries:

| Repo | Variant | Changed tensors | Direction summary |
|---|---:|---:|---|
| `unsloth/ERNIE-Image-GGUF` | `UD-Q2_K` | 193 | 179 promoted, 14 demoted |
| `unsloth/ERNIE-Image-GGUF` | `UD-Q3_K_M` | 170 | 168 promoted, 2 demoted |
| `unsloth/ERNIE-Image-GGUF` | `UD-Q4_K_M` | 154 | 153 promoted, 1 demoted |
| `unsloth/ERNIE-Image-GGUF` | `UD-Q5_K_M` | 146 | 146 promoted |
| `unsloth/ERNIE-Image-Turbo-GGUF` | `UD-Q2_K` | 230 | 214 promoted, 16 demoted |
| `unsloth/ERNIE-Image-Turbo-GGUF` | `UD-Q3_K_M` | 206 | 195 promoted, 11 demoted |
| `unsloth/ERNIE-Image-Turbo-GGUF` | `UD-Q4_K_M` | 175 | 164 promoted, 11 demoted |
| `unsloth/ERNIE-Image-Turbo-GGUF` | `UD-Q5_K_M` | 184 | 172 promoted, 12 demoted |

The UD pattern is not a simple static tensor-class rule. It is layer- and tensor-specific. Frequent classes include:

- `self_attention.to_v.weight`: usually promoted strongly, often to `Q6_K` or `Q8_0`.
- `self_attention.to_q.weight`, `self_attention.to_k.weight`, `self_attention.to_out.0.weight`: often promoted, with layer-dependent target qtypes.
- `mlp.gate_proj.weight`, `mlp.up_proj.weight`, `mlp.linear_fc2.weight`: often promoted, but with variant- and layer-dependent exceptions.
- Turbo variants include more demotions, including late-layer `Q6_K -> Q5_K` in `UD-Q5_K_M`.

Concrete implementation recommendation:

1. Keep `UD-*` out of the normal `--policy comfy` path.
2. Add a separate `--policy unsloth-dynamic` only when we are ready to support it.
3. Implement UD as explicit per-repo/per-variant tensor override maps generated from `policy_override_candidates.json`, not as a small set of glob patterns.
4. Treat the current override maps as reproducibility targets for the exact analyzed repos, not as a general Unsloth algorithm.

The generated override manifest is `reports/gguf_comparison/policy_override_candidates.json`.
