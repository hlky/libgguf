# Generalized Dynamic Policy vs UD Overrides

Comparator: implemented local `dynamic` policy against UD override targets from `policy_override_candidates.json`.

| Repo | Variant | Overrides | Exact dynamic matches | Match rate | Dominant misses |
|---|---|---:|---:|---:|---|
| `unsloth/ERNIE-Image-GGUF` | `UD-Q2_K` | 193 | 172 | 89.1% | `Q6_K->Q4_K` 11, `Q4_K->Q3_K` 4, `Q3_K->Q4_K` 3, `Q3_K->Q2_K` 2 |
| `unsloth/ERNIE-Image-GGUF` | `UD-Q3_K_M` | 170 | 155 | 91.2% | `Q4_K->Q5_K` 10, `Q5_K->Q4_K` 3, `Q4_K->Q3_K` 2 |
| `unsloth/ERNIE-Image-GGUF` | `UD-Q4_K_M` | 154 | 136 | 88.3% | `Q6_K->Q5_K` 9, `Q5_K->Q6_K` 8, `Q6_K->Q4_K` 1 |
| `unsloth/ERNIE-Image-GGUF` | `UD-Q5_K_M` | 146 | 138 | 94.5% | `Q6_K->Q8_0` 5, `Q8_0->Q6_K` 3 |
| `unsloth/ERNIE-Image-Turbo-GGUF` | `UD-Q2_K` | 230 | 153 | 66.5% | `Q3_K->Q4_K` 31, `Q4_K->Q3_K` 27, `Q3_K->Q2_K` 8, `Q6_K->Q4_K` 7 |
| `unsloth/ERNIE-Image-Turbo-GGUF` | `UD-Q3_K_M` | 206 | 131 | 63.6% | `Q4_K->Q5_K` 36, `Q5_K->Q4_K` 23, `Q4_K->Q3_K` 11, `Q5_K->Q6_K` 3 |
| `unsloth/ERNIE-Image-Turbo-GGUF` | `UD-Q4_K_M` | 175 | 101 | 57.7% | `Q5_K->Q6_K` 36, `Q6_K->Q5_K` 23, `Q5_K->Q4_K` 11, `Q6_K->Q8_0` 3 |
| `unsloth/ERNIE-Image-Turbo-GGUF` | `UD-Q5_K_M` | 184 | 142 | 77.2% | `Q6_K->Q8_0` 22, `Q6_K->Q5_K` 12, `Q8_0->Q6_K` 8 |
