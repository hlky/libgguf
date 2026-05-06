# Unsloth Dynamic Policy Findings

Inputs:

- `reports/gguf_comparison/policy_override_candidates.json`
- `reports/gguf_comparison/ud_policy_analysis.json`

Scope:

- `unsloth/ERNIE-Image-GGUF`
- `unsloth/ERNIE-Image-Turbo-GGUF`
- `UD-Q2_K`, `UD-Q3_K_M`, `UD-Q4_K_M`, `UD-Q5_K_M`

No explicit Unsloth policy metadata keys were found in the GGUF metadata. These findings are inferred from tensor qtype assignments relative to the matching non-UD base variant in the same repo.

## Summary

Unsloth Dynamic is not a normal mixed policy such as "all attention value tensors go higher" or "all FFN down tensors go higher".

It is a per-tensor override policy. The override choices are strongly concentrated in seven tensor roles:

- `self_attention.to_v.weight`
- `self_attention.to_out.0.weight`
- `self_attention.to_q.weight`
- `self_attention.to_k.weight`
- `mlp.linear_fc2.weight`
- `mlp.up_proj.weight`
- `mlp.gate_proj.weight`

The policy mostly promotes selected tensors above the corresponding normal base variant. A few tensors are demoted relative to the base mixed policy, which means the policy is not purely monotonic over the existing qtype map. The practical implementation should be a separate `unsloth-dynamic` policy backed by explicit tensor override maps.

## Main Behavior

### UD-Q2_K

`UD-Q2_K` is mostly a selective upgrade from `Q2_K` to `Q3_K`, with a smaller number of `Q4_K` and `Q6_K` assignments.

Observed transitions:

| Repo | Dominant transitions |
|---|---|
| ERNIE | `Q2_K -> Q3_K` 100 tensors, `Q2_K -> Q4_K` 45 tensors, attention-value `Q3_K -> Q6_K` 19 tensors |
| ERNIE Turbo | `Q2_K -> Q3_K` 133 tensors, `Q2_K -> Q4_K` 44 tensors, attention-value `Q3_K -> Q6_K` 25 tensors |

Interpretation:

- Most selected Q/K/MLP tensors are raised one step to `Q3_K`.
- Some output projections and MLP up/down tensors are raised to `Q4_K`.
- Attention value projections are treated as much more sensitive: many are raised to `Q6_K`.

### UD-Q3_K_M

`UD-Q3_K_M` is mostly a selective upgrade from `Q3_K` to `Q4_K` or `Q5_K`, while attention value tensors are usually raised to `Q6_K`.

Observed transitions:

| Repo | Dominant transitions |
|---|---|
| ERNIE | `Q3_K -> Q4_K` 84 tensors, `Q3_K -> Q5_K` 41 tensors, `Q4_K -> Q6_K` 30 tensors |
| ERNIE Turbo | `Q3_K -> Q4_K` 109 tensors, `Q3_K -> Q5_K` 41 tensors, `Q4_K -> Q6_K` 33 tensors |

Interpretation:

- Q/K/O and MLP tensors are split between `Q4_K` and `Q5_K`.
- Attention value projections are almost uniformly promoted to `Q6_K`.
- Turbo has more layer coverage than the non-Turbo model and includes a few non-monotonic exceptions such as `Q4_K -> Q3_K`.

### UD-Q4_K_M

`UD-Q4_K_M` is mostly a selective upgrade from `Q4_K` to `Q5_K` or `Q6_K`, with a small number of `Q8_0` assignments.

Observed transitions:

| Repo | Dominant transitions |
|---|---|
| ERNIE | `Q4_K -> Q5_K` 102 tensors, `Q4_K -> Q6_K` 34 tensors, `Q5_K -> Q6_K` 10 tensors |
| ERNIE Turbo | `Q4_K -> Q5_K` 102 tensors, `Q4_K -> Q6_K` 46 tensors, `Q5_K -> Q4_K` 11 tensors |

Interpretation:

- Most selected Q/K/O and MLP tensors move to `Q5_K`.
- A smaller subset moves to `Q6_K`.
- Attention value projections are no longer broadly changed because many are already high in the base mixed policy; the remaining changed value tensors usually become `Q8_0`.
- Turbo again has explicit non-monotonic exceptions, including `Q5_K -> Q4_K`.

### UD-Q5_K_M

`UD-Q5_K_M` is mostly a selective upgrade from `Q5_K` to `Q6_K`, with highly sensitive tensors raised to `Q8_0`.

Observed transitions:

| Repo | Dominant transitions |
|---|---|
| ERNIE | `Q5_K -> Q6_K` 97 tensors, `Q6_K -> Q8_0` 31 tensors, `Q5_K -> Q8_0` 18 tensors |
| ERNIE Turbo | `Q5_K -> Q6_K` 116 tensors, `Q6_K -> Q8_0` 33 tensors, `Q5_K -> Q8_0` 23 tensors |

Interpretation:

- Most selected Q/K/O and MLP tensors become `Q6_K`.
- Attention value tensors are strongly biased toward `Q8_0`.
- The earliest Turbo layers are especially protected; many Q/K/O and MLP tensors in layers 0-4 move to `Q8_0`.
- Turbo includes `Q6_K -> Q5_K` demotions for a small fixed-looking set, which confirms this is not expressible as a simple promotion-only policy.

## Layer Patterns

The same role can receive different qtypes depending on layer number. Examples:

- In ERNIE `UD-Q5_K_M`, attention value tensors are `Q8_0` across early, middle, and late layers: `0-1`, `9`, `14-15`, `17-20`, `22-35`.
- In ERNIE Turbo `UD-Q5_K_M`, early layers are heavily protected: Q/K/O and several MLP tensors in layers `0-4` are often `Q8_0`.
- In ERNIE `UD-Q3_K_M`, attention output tensors are raised to `Q5_K` in layers `0-1`, `15-20`, `22-24`, and `28-35`, while other changed output tensors are only `Q4_K`.
- In ERNIE `UD-Q4_K_M`, MLP up/down tensors are split between `Q5_K` and `Q6_K`; the `Q6_K` assignments are concentrated around layers `7`, `21`, and the late block range.

This layer selectivity is the strongest evidence that the policy was produced from an importance ranking or calibration process, not from static tensor-name classes alone.

## Concrete Implementation Guidance

Do not fold these rules into the existing `comfy` policy.

Recommended implementation:

1. Add a distinct policy name, for example `--policy unsloth-dynamic`.
2. Keep the normal base qtype policy as the starting point.
3. Apply exact tensor-name overrides after the normal qtype plan is computed.
4. Store overrides by architecture/model family, base qtype, and tensor name.
5. Treat the current ERNIE and ERNIE Turbo maps separately.

The generated override data in `policy_override_candidates.json` is already shaped for this implementation: each UD row has `repo_id`, `variant`, `base_variant`, and an `overrides` object mapping tensor name to target qtype.

## Confidence

Confidence is `inferred_from_tensor_types`.

The evidence is strong for reproducing these specific files because tensor names and target qtypes are explicit. The evidence is weak for generalizing to unseen models because no metadata declares the scoring method, threshold, calibration set, or layer importance formula.
