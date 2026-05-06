# UD vs Local Comfy Policy Analysis

Comparator: local `comfy` policy qtype assignment for each UD variant base qtype, then exact UD target qtype from `policy_override_candidates.json`.

## Overall Transition Summary

| Repo | Variant | Overrides | Same | Higher | Lower | Dominant transitions |
|---|---|---:|---:|---:|---:|---|
| `unsloth/ERNIE-Image-GGUF` | `UD-Q2_K` | 193 | 2 | 191 | 0 | `Q2_K->Q3_K` 112, `Q2_K->Q4_K` 45, `Q3_K->Q6_K` 22, `Q3_K->Q4_K` 11 |
| `unsloth/ERNIE-Image-GGUF` | `UD-Q3_K_M` | 170 | 2 | 168 | 0 | `Q3_K->Q4_K` 84, `Q3_K->Q5_K` 52, `Q5_K->Q6_K` 32, `Q3_K->Q3_K` 2 |
| `unsloth/ERNIE-Image-GGUF` | `UD-Q4_K_M` | 154 | 1 | 153 | 0 | `Q4_K->Q5_K` 102, `Q4_K->Q6_K` 44, `Q6_K->Q8_0` 7, `Q4_K->Q4_K` 1 |
| `unsloth/ERNIE-Image-GGUF` | `UD-Q5_K_M` | 146 | 0 | 146 | 0 | `Q5_K->Q6_K` 97, `Q5_K->Q8_0` 26, `Q6_K->Q8_0` 23 |
| `unsloth/ERNIE-Image-Turbo-GGUF` | `UD-Q2_K` | 230 | 9 | 221 | 0 | `Q2_K->Q3_K` 140, `Q2_K->Q4_K` 44, `Q3_K->Q6_K` 27, `Q2_K->Q2_K` 8 |
| `unsloth/ERNIE-Image-Turbo-GGUF` | `UD-Q3_K_M` | 206 | 11 | 195 | 0 | `Q3_K->Q4_K` 109, `Q3_K->Q5_K` 50, `Q5_K->Q6_K` 31, `Q3_K->Q3_K` 11 |
| `unsloth/ERNIE-Image-Turbo-GGUF` | `UD-Q4_K_M` | 175 | 11 | 164 | 0 | `Q4_K->Q5_K` 102, `Q4_K->Q6_K` 55, `Q4_K->Q4_K` 11, `Q4_K->Q8_0` 4 |
| `unsloth/ERNIE-Image-Turbo-GGUF` | `UD-Q5_K_M` | 184 | 12 | 172 | 0 | `Q5_K->Q6_K` 116, `Q5_K->Q8_0` 35, `Q6_K->Q8_0` 21, `Q5_K->Q5_K` 12 |

## Role / Transition Detail

### unsloth/ERNIE-Image-GGUF UD-Q2_K

| Role | Our comfy -> UD | Count | Layers |
|---|---|---:|---|
| `attn.k` | `Q2_K` -> `Q3_K` | 24 | 0-3, 14-30, 32-34 |
| `attn.k` | `Q2_K` -> `Q4_K` | 1 | 31 |
| `attn.out` | `Q2_K` -> `Q3_K` | 14 | 2-3, 6-14, 25-27 |
| `attn.out` | `Q2_K` -> `Q4_K` | 16 | 15-24, 28-33 |
| `attn.q` | `Q2_K` -> `Q3_K` | 24 | 0-3, 13-30, 32, 34 |
| `attn.q` | `Q2_K` -> `Q4_K` | 2 | 31, 33 |
| `attn.q` | `Q2_K` -> `Q5_K` | 1 | 35 |
| `attn.v` | `Q3_K` -> `Q4_K` | 11 | 2-3, 6-14 |
| `attn.v` | `Q3_K` -> `Q6_K` | 22 | 0, 15-35 |
| `mlp.fc2` | `Q2_K` -> `Q3_K` | 13 | 0-1, 8, 12-20, 22 |
| `mlp.fc2` | `Q2_K` -> `Q4_K` | 13 | 7, 21, 23-33 |
| `mlp.gate` | `Q2_K` -> `Q2_K` | 1 | 1 |
| `mlp.gate` | `Q2_K` -> `Q3_K` | 24 | 0, 7, 14-35 |
| `mlp.up` | `Q2_K` -> `Q2_K` | 1 | 1 |
| `mlp.up` | `Q2_K` -> `Q3_K` | 13 | 0, 8, 11-20, 22 |
| `mlp.up` | `Q2_K` -> `Q4_K` | 13 | 7, 21, 23-33 |

### unsloth/ERNIE-Image-GGUF UD-Q3_K_M

| Role | Our comfy -> UD | Count | Layers |
|---|---|---:|---|
| `attn.k` | `Q3_K` -> `Q4_K` | 19 | 2, 14-30, 32 |
| `attn.k` | `Q3_K` -> `Q5_K` | 3 | 31, 33, 35 |
| `attn.out` | `Q3_K` -> `Q4_K` | 13 | 2-3, 7-10, 12-14, 21, 25-27 |
| `attn.out` | `Q3_K` -> `Q5_K` | 19 | 0-1, 15-20, 22-24, 28-35 |
| `attn.q` | `Q3_K` -> `Q4_K` | 21 | 2-3, 14-26, 28-33 |
| `attn.q` | `Q3_K` -> `Q5_K` | 2 | 0, 35 |
| `attn.v` | `Q5_K` -> `Q6_K` | 32 | 0-2, 7-35 |
| `mlp.fc2` | `Q3_K` -> `Q3_K` | 1 | 1 |
| `mlp.fc2` | `Q3_K` -> `Q4_K` | 8 | 14-20, 22 |
| `mlp.fc2` | `Q3_K` -> `Q5_K` | 15 | 7, 21, 23-35 |
| `mlp.gate` | `Q3_K` -> `Q4_K` | 13 | 7, 20, 23-33 |
| `mlp.up` | `Q3_K` -> `Q3_K` | 1 | 1 |
| `mlp.up` | `Q3_K` -> `Q4_K` | 10 | 14-22, 25 |
| `mlp.up` | `Q3_K` -> `Q5_K` | 13 | 7, 23-24, 26-35 |

### unsloth/ERNIE-Image-GGUF UD-Q4_K_M

| Role | Our comfy -> UD | Count | Layers |
|---|---|---:|---|
| `attn.k` | `Q4_K` -> `Q5_K` | 24 | 2-3, 7, 13-33 |
| `attn.k` | `Q4_K` -> `Q6_K` | 1 | 0 |
| `attn.out` | `Q4_K` -> `Q5_K` | 17 | 2-3, 6-14, 20, 23, 25-27, 29 |
| `attn.out` | `Q4_K` -> `Q6_K` | 16 | 1, 15-19, 21-22, 24, 28, 30-35 |
| `attn.q` | `Q4_K` -> `Q5_K` | 23 | 2, 8, 13-33 |
| `attn.q` | `Q4_K` -> `Q6_K` | 3 | 0, 3, 35 |
| `attn.v` | `Q6_K` -> `Q8_0` | 7 | 0, 24, 31-35 |
| `mlp.fc2` | `Q4_K` -> `Q5_K` | 12 | 13-20, 22-25 |
| `mlp.fc2` | `Q4_K` -> `Q6_K` | 12 | 7, 21, 26-35 |
| `mlp.gate` | `Q4_K` -> `Q4_K` | 1 | 1 |
| `mlp.gate` | `Q4_K` -> `Q5_K` | 14 | 2, 7, 18, 21, 23-24, 26-33 |
| `mlp.up` | `Q4_K` -> `Q5_K` | 12 | 2, 14-22, 25-26 |
| `mlp.up` | `Q4_K` -> `Q6_K` | 12 | 7, 23-24, 27-35 |

### unsloth/ERNIE-Image-GGUF UD-Q5_K_M

| Role | Our comfy -> UD | Count | Layers |
|---|---|---:|---|
| `attn.k` | `Q5_K` -> `Q6_K` | 20 | 2-3, 15-31, 33 |
| `attn.k` | `Q5_K` -> `Q8_0` | 1 | 0 |
| `attn.out` | `Q5_K` -> `Q6_K` | 24 | 2, 4, 7-10, 12-23, 25-30 |
| `attn.out` | `Q5_K` -> `Q8_0` | 9 | 0-1, 3, 24, 31-35 |
| `attn.q` | `Q5_K` -> `Q6_K` | 18 | 3-4, 9, 14-17, 19-24, 26, 29-31, 33 |
| `attn.v` | `Q6_K` -> `Q8_0` | 23 | 0-1, 9, 14-15, 17-20, 22-35 |
| `mlp.fc2` | `Q5_K` -> `Q6_K` | 14 | 7, 14-26 |
| `mlp.fc2` | `Q5_K` -> `Q8_0` | 9 | 27-35 |
| `mlp.gate` | `Q5_K` -> `Q6_K` | 7 | 2, 28-33 |
| `mlp.up` | `Q5_K` -> `Q6_K` | 14 | 15-28 |
| `mlp.up` | `Q5_K` -> `Q8_0` | 7 | 7, 29-34 |

### unsloth/ERNIE-Image-Turbo-GGUF UD-Q2_K

| Role | Our comfy -> UD | Count | Layers |
|---|---|---:|---|
| `attn.k` | `Q2_K` -> `Q2_K` | 1 | 34 |
| `attn.k` | `Q2_K` -> `Q3_K` | 20 | 4-10, 12-14, 24-31, 33, 35 |
| `attn.k` | `Q2_K` -> `Q4_K` | 12 | 2-3, 11, 15-23 |
| `attn.k` | `Q2_K` -> `Q6_K` | 1 | 0 |
| `attn.out` | `Q2_K` -> `Q3_K` | 19 | 5-12, 14, 25-27, 29-35 |
| `attn.out` | `Q2_K` -> `Q4_K` | 15 | 2-4, 13, 15-24, 28 |
| `attn.out` | `Q2_K` -> `Q5_K` | 1 | 0 |
| `attn.q` | `Q2_K` -> `Q2_K` | 1 | 34 |
| `attn.q` | `Q2_K` -> `Q3_K` | 22 | 4-7, 9-10, 12-14, 21-31, 33, 35 |
| `attn.q` | `Q2_K` -> `Q4_K` | 10 | 2-3, 8, 11, 15-20 |
| `attn.q` | `Q2_K` -> `Q5_K` | 1 | 0 |
| `attn.v` | `Q3_K` -> `Q3_K` | 1 | 34 |
| `attn.v` | `Q3_K` -> `Q4_K` | 7 | 5-6, 29-33 |
| `attn.v` | `Q3_K` -> `Q6_K` | 27 | 0-4, 7-28 |
| `mlp.fc2` | `Q2_K` -> `Q2_K` | 2 | 34-35 |
| `mlp.fc2` | `Q2_K` -> `Q3_K` | 26 | 0, 2-15, 22-32 |
| `mlp.fc2` | `Q2_K` -> `Q4_K` | 6 | 16-21 |
| `mlp.gate` | `Q2_K` -> `Q2_K` | 2 | 34-35 |
| `mlp.gate` | `Q2_K` -> `Q3_K` | 25 | 1-7, 12-29 |
| `mlp.up` | `Q2_K` -> `Q2_K` | 2 | 34-35 |
| `mlp.up` | `Q2_K` -> `Q3_K` | 28 | 1-8, 12-20, 22-32 |
| `mlp.up` | `Q2_K` -> `Q4_K` | 1 | 21 |

### unsloth/ERNIE-Image-Turbo-GGUF UD-Q3_K_M

| Role | Our comfy -> UD | Count | Layers |
|---|---|---:|---|
| `attn.k` | `Q3_K` -> `Q3_K` | 2 | 34-35 |
| `attn.k` | `Q3_K` -> `Q4_K` | 17 | 5-7, 9-12, 14, 21, 23-29, 31 |
| `attn.k` | `Q3_K` -> `Q5_K` | 13 | 1-4, 8, 13, 15-20, 22 |
| `attn.k` | `Q3_K` -> `Q6_K` | 1 | 0 |
| `attn.out` | `Q3_K` -> `Q3_K` | 2 | 34-35 |
| `attn.out` | `Q3_K` -> `Q4_K` | 18 | 4-14, 25-31 |
| `attn.out` | `Q3_K` -> `Q5_K` | 12 | 2-3, 15-24 |
| `attn.out` | `Q3_K` -> `Q6_K` | 2 | 0-1 |
| `attn.q` | `Q3_K` -> `Q3_K` | 1 | 34 |
| `attn.q` | `Q3_K` -> `Q4_K` | 18 | 5-6, 9-12, 17, 19-29 |
| `attn.q` | `Q3_K` -> `Q5_K` | 11 | 0-1, 3-4, 7-8, 13-16, 18 |
| `attn.q` | `Q3_K` -> `Q6_K` | 1 | 2 |
| `attn.v` | `Q5_K` -> `Q6_K` | 31 | 1-31 |
| `attn.v` | `Q5_K` -> `Q8_0` | 1 | 0 |
| `mlp.fc2` | `Q3_K` -> `Q3_K` | 2 | 34-35 |
| `mlp.fc2` | `Q3_K` -> `Q4_K` | 19 | 3-7, 13-17, 22-30 |
| `mlp.fc2` | `Q3_K` -> `Q5_K` | 7 | 0-2, 18-21 |
| `mlp.gate` | `Q3_K` -> `Q3_K` | 2 | 34-35 |
| `mlp.gate` | `Q3_K` -> `Q4_K` | 14 | 3-7, 14-22 |
| `mlp.gate` | `Q3_K` -> `Q5_K` | 3 | 0-2 |
| `mlp.up` | `Q3_K` -> `Q3_K` | 2 | 34-35 |
| `mlp.up` | `Q3_K` -> `Q4_K` | 23 | 2-7, 12-17, 20-30 |
| `mlp.up` | `Q3_K` -> `Q5_K` | 4 | 0-1, 18-19 |

### unsloth/ERNIE-Image-Turbo-GGUF UD-Q4_K_M

| Role | Our comfy -> UD | Count | Layers |
|---|---|---:|---|
| `attn.k` | `Q4_K` -> `Q4_K` | 2 | 34-35 |
| `attn.k` | `Q4_K` -> `Q5_K` | 14 | 5, 7-10, 20, 23-29, 31 |
| `attn.k` | `Q4_K` -> `Q6_K` | 16 | 1-4, 6, 11-19, 21-22 |
| `attn.k` | `Q4_K` -> `Q8_0` | 1 | 0 |
| `attn.out` | `Q4_K` -> `Q4_K` | 2 | 34-35 |
| `attn.out` | `Q4_K` -> `Q5_K` | 17 | 4-6, 8-10, 12-14, 23, 25-31 |
| `attn.out` | `Q4_K` -> `Q6_K` | 13 | 1, 3, 7, 11, 15-22, 24 |
| `attn.out` | `Q4_K` -> `Q8_0` | 2 | 0, 2 |
| `attn.q` | `Q4_K` -> `Q4_K` | 1 | 34 |
| `attn.q` | `Q4_K` -> `Q5_K` | 18 | 7-9, 13-15, 17-28 |
| `attn.q` | `Q4_K` -> `Q6_K` | 10 | 1-6, 10-12, 16 |
| `attn.q` | `Q4_K` -> `Q8_0` | 1 | 0 |
| `attn.v` | `Q6_K` -> `Q8_0` | 3 | 0-2 |
| `mlp.fc2` | `Q4_K` -> `Q4_K` | 2 | 34-35 |
| `mlp.fc2` | `Q4_K` -> `Q5_K` | 17 | 4-7, 13-14, 18, 20, 22-30 |
| `mlp.fc2` | `Q4_K` -> `Q6_K` | 9 | 0-3, 15-17, 19, 21 |
| `mlp.gate` | `Q4_K` -> `Q4_K` | 2 | 34-35 |
| `mlp.gate` | `Q4_K` -> `Q5_K` | 11 | 4-7, 15-21 |
| `mlp.gate` | `Q4_K` -> `Q6_K` | 4 | 0-3 |
| `mlp.up` | `Q4_K` -> `Q4_K` | 2 | 34-35 |
| `mlp.up` | `Q4_K` -> `Q5_K` | 25 | 3-9, 12-29 |
| `mlp.up` | `Q4_K` -> `Q6_K` | 3 | 0-2 |

### unsloth/ERNIE-Image-Turbo-GGUF UD-Q5_K_M

| Role | Our comfy -> UD | Count | Layers |
|---|---|---:|---|
| `attn.k` | `Q5_K` -> `Q5_K` | 2 | 34-35 |
| `attn.k` | `Q5_K` -> `Q6_K` | 24 | 4, 6, 9-29, 31 |
| `attn.k` | `Q5_K` -> `Q8_0` | 6 | 0-3, 7-8 |
| `attn.out` | `Q5_K` -> `Q5_K` | 2 | 34-35 |
| `attn.out` | `Q5_K` -> `Q6_K` | 16 | 5, 8, 11-15, 20-28 |
| `attn.out` | `Q5_K` -> `Q8_0` | 13 | 0-4, 6-7, 9-10, 16-19 |
| `attn.q` | `Q5_K` -> `Q5_K` | 2 | 34-35 |
| `attn.q` | `Q5_K` -> `Q6_K` | 18 | 5-6, 9-13, 16-26 |
| `attn.q` | `Q5_K` -> `Q8_0` | 9 | 0-4, 7-8, 14-15 |
| `attn.v` | `Q6_K` -> `Q8_0` | 21 | 0-5, 7, 9-10, 12-20, 22-24 |
| `mlp.fc2` | `Q5_K` -> `Q5_K` | 2 | 34-35 |
| `mlp.fc2` | `Q5_K` -> `Q6_K` | 24 | 2-7, 11-12, 14-29 |
| `mlp.fc2` | `Q5_K` -> `Q8_0` | 2 | 0-1 |
| `mlp.gate` | `Q5_K` -> `Q5_K` | 2 | 34-35 |
| `mlp.gate` | `Q5_K` -> `Q6_K` | 11 | 3-8, 14, 17-20 |
| `mlp.gate` | `Q5_K` -> `Q8_0` | 3 | 0-2 |
| `mlp.up` | `Q5_K` -> `Q5_K` | 2 | 34-35 |
| `mlp.up` | `Q5_K` -> `Q6_K` | 23 | 2-7, 12-28 |
| `mlp.up` | `Q5_K` -> `Q8_0` | 2 | 0-1 |

## Adoptable Policy Shape

- Relative to local `comfy`, the UD maps are monotonic for the analyzed samples: entries are either promoted or left unchanged. No lower-than-comfy cases were found.
- The stable broad rule is: promote selected `to_q`, `to_k`, `to_out`, `mlp.up`, `mlp.linear_fc2`, and some `mlp.gate` tensors by one K step; promote `to_v` tensors more aggressively.
- For `UD-Q5_K_M`, a similar local policy would be: keep normal `to_v` `Q6_K` handling, then promote selected `to_v` to `Q8_0`; promote selected Q/K/O and MLP tensors from `Q5_K` to `Q6_K`; promote the most sensitive early-layer tensors to `Q8_0`.
- Exact layer/key checks are still required because the selected layer ranges differ by repo and variant, and some Turbo entries are intentionally left unchanged where nearby tensors are promoted.
