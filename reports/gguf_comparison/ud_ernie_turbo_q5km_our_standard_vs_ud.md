# ERNIE Turbo UD-Q5_K_M vs Our Standard Policy

Source: `unsloth/ERNIE-Image-Turbo-GGUF` `UD-Q5_K_M`.

Comparator: local `comfy` policy for file type `Q5_K_M`, with base qtype `Q5_K`, computed by `libgguf.quantize._mixed_policy_qtype`.

## Transition Summary

| Our standard policy | UD policy | Count |
|---|---|---:|
| `Q5_K` | `Q6_K` | 116 |
| `Q5_K` | `Q8_0` | 35 |
| `Q6_K` | `Q8_0` | 21 |
| `Q5_K` | `Q5_K` | 12 |

## Role Summary

| Role | Our standard -> UD | Count |
|---|---|---:|
| `attn.k` | `Q5_K` -> `Q5_K` | 2 |
| `attn.k` | `Q5_K` -> `Q6_K` | 24 |
| `attn.k` | `Q5_K` -> `Q8_0` | 6 |
| `attn.out` | `Q5_K` -> `Q5_K` | 2 |
| `attn.out` | `Q5_K` -> `Q6_K` | 16 |
| `attn.out` | `Q5_K` -> `Q8_0` | 13 |
| `attn.q` | `Q5_K` -> `Q5_K` | 2 |
| `attn.q` | `Q5_K` -> `Q6_K` | 18 |
| `attn.q` | `Q5_K` -> `Q8_0` | 9 |
| `attn.v` | `Q6_K` -> `Q8_0` | 21 |
| `mlp.fc2` | `Q5_K` -> `Q5_K` | 2 |
| `mlp.fc2` | `Q5_K` -> `Q6_K` | 24 |
| `mlp.fc2` | `Q5_K` -> `Q8_0` | 2 |
| `mlp.gate` | `Q5_K` -> `Q5_K` | 2 |
| `mlp.gate` | `Q5_K` -> `Q6_K` | 11 |
| `mlp.gate` | `Q5_K` -> `Q8_0` | 3 |
| `mlp.up` | `Q5_K` -> `Q5_K` | 2 |
| `mlp.up` | `Q5_K` -> `Q6_K` | 23 |
| `mlp.up` | `Q5_K` -> `Q8_0` | 2 |

## Full Tensor Table

| Layer | Tensor role | Tensor | Our standard policy | UD policy |
|---:|---|---|---|---|
| 0 | `attn.k` | `layers.0.self_attention.to_k.weight` | `Q5_K` | `Q8_0` |
| 0 | `attn.out` | `layers.0.self_attention.to_out.0.weight` | `Q5_K` | `Q8_0` |
| 0 | `attn.q` | `layers.0.self_attention.to_q.weight` | `Q5_K` | `Q8_0` |
| 0 | `attn.v` | `layers.0.self_attention.to_v.weight` | `Q6_K` | `Q8_0` |
| 0 | `mlp.fc2` | `layers.0.mlp.linear_fc2.weight` | `Q5_K` | `Q8_0` |
| 0 | `mlp.gate` | `layers.0.mlp.gate_proj.weight` | `Q5_K` | `Q8_0` |
| 0 | `mlp.up` | `layers.0.mlp.up_proj.weight` | `Q5_K` | `Q8_0` |
| 1 | `attn.k` | `layers.1.self_attention.to_k.weight` | `Q5_K` | `Q8_0` |
| 1 | `attn.out` | `layers.1.self_attention.to_out.0.weight` | `Q5_K` | `Q8_0` |
| 1 | `attn.q` | `layers.1.self_attention.to_q.weight` | `Q5_K` | `Q8_0` |
| 1 | `attn.v` | `layers.1.self_attention.to_v.weight` | `Q6_K` | `Q8_0` |
| 1 | `mlp.fc2` | `layers.1.mlp.linear_fc2.weight` | `Q5_K` | `Q8_0` |
| 1 | `mlp.gate` | `layers.1.mlp.gate_proj.weight` | `Q5_K` | `Q8_0` |
| 1 | `mlp.up` | `layers.1.mlp.up_proj.weight` | `Q5_K` | `Q8_0` |
| 2 | `attn.k` | `layers.2.self_attention.to_k.weight` | `Q5_K` | `Q8_0` |
| 2 | `attn.out` | `layers.2.self_attention.to_out.0.weight` | `Q5_K` | `Q8_0` |
| 2 | `attn.q` | `layers.2.self_attention.to_q.weight` | `Q5_K` | `Q8_0` |
| 2 | `attn.v` | `layers.2.self_attention.to_v.weight` | `Q6_K` | `Q8_0` |
| 2 | `mlp.fc2` | `layers.2.mlp.linear_fc2.weight` | `Q5_K` | `Q6_K` |
| 2 | `mlp.gate` | `layers.2.mlp.gate_proj.weight` | `Q5_K` | `Q8_0` |
| 2 | `mlp.up` | `layers.2.mlp.up_proj.weight` | `Q5_K` | `Q6_K` |
| 3 | `attn.k` | `layers.3.self_attention.to_k.weight` | `Q5_K` | `Q8_0` |
| 3 | `attn.out` | `layers.3.self_attention.to_out.0.weight` | `Q5_K` | `Q8_0` |
| 3 | `attn.q` | `layers.3.self_attention.to_q.weight` | `Q5_K` | `Q8_0` |
| 3 | `attn.v` | `layers.3.self_attention.to_v.weight` | `Q6_K` | `Q8_0` |
| 3 | `mlp.fc2` | `layers.3.mlp.linear_fc2.weight` | `Q5_K` | `Q6_K` |
| 3 | `mlp.gate` | `layers.3.mlp.gate_proj.weight` | `Q5_K` | `Q6_K` |
| 3 | `mlp.up` | `layers.3.mlp.up_proj.weight` | `Q5_K` | `Q6_K` |
| 4 | `attn.k` | `layers.4.self_attention.to_k.weight` | `Q5_K` | `Q6_K` |
| 4 | `attn.out` | `layers.4.self_attention.to_out.0.weight` | `Q5_K` | `Q8_0` |
| 4 | `attn.q` | `layers.4.self_attention.to_q.weight` | `Q5_K` | `Q8_0` |
| 4 | `attn.v` | `layers.4.self_attention.to_v.weight` | `Q6_K` | `Q8_0` |
| 4 | `mlp.fc2` | `layers.4.mlp.linear_fc2.weight` | `Q5_K` | `Q6_K` |
| 4 | `mlp.gate` | `layers.4.mlp.gate_proj.weight` | `Q5_K` | `Q6_K` |
| 4 | `mlp.up` | `layers.4.mlp.up_proj.weight` | `Q5_K` | `Q6_K` |
| 5 | `attn.out` | `layers.5.self_attention.to_out.0.weight` | `Q5_K` | `Q6_K` |
| 5 | `attn.q` | `layers.5.self_attention.to_q.weight` | `Q5_K` | `Q6_K` |
| 5 | `attn.v` | `layers.5.self_attention.to_v.weight` | `Q6_K` | `Q8_0` |
| 5 | `mlp.fc2` | `layers.5.mlp.linear_fc2.weight` | `Q5_K` | `Q6_K` |
| 5 | `mlp.gate` | `layers.5.mlp.gate_proj.weight` | `Q5_K` | `Q6_K` |
| 5 | `mlp.up` | `layers.5.mlp.up_proj.weight` | `Q5_K` | `Q6_K` |
| 6 | `attn.k` | `layers.6.self_attention.to_k.weight` | `Q5_K` | `Q6_K` |
| 6 | `attn.out` | `layers.6.self_attention.to_out.0.weight` | `Q5_K` | `Q8_0` |
| 6 | `attn.q` | `layers.6.self_attention.to_q.weight` | `Q5_K` | `Q6_K` |
| 6 | `mlp.fc2` | `layers.6.mlp.linear_fc2.weight` | `Q5_K` | `Q6_K` |
| 6 | `mlp.gate` | `layers.6.mlp.gate_proj.weight` | `Q5_K` | `Q6_K` |
| 6 | `mlp.up` | `layers.6.mlp.up_proj.weight` | `Q5_K` | `Q6_K` |
| 7 | `attn.k` | `layers.7.self_attention.to_k.weight` | `Q5_K` | `Q8_0` |
| 7 | `attn.out` | `layers.7.self_attention.to_out.0.weight` | `Q5_K` | `Q8_0` |
| 7 | `attn.q` | `layers.7.self_attention.to_q.weight` | `Q5_K` | `Q8_0` |
| 7 | `attn.v` | `layers.7.self_attention.to_v.weight` | `Q6_K` | `Q8_0` |
| 7 | `mlp.fc2` | `layers.7.mlp.linear_fc2.weight` | `Q5_K` | `Q6_K` |
| 7 | `mlp.gate` | `layers.7.mlp.gate_proj.weight` | `Q5_K` | `Q6_K` |
| 7 | `mlp.up` | `layers.7.mlp.up_proj.weight` | `Q5_K` | `Q6_K` |
| 8 | `attn.k` | `layers.8.self_attention.to_k.weight` | `Q5_K` | `Q8_0` |
| 8 | `attn.out` | `layers.8.self_attention.to_out.0.weight` | `Q5_K` | `Q6_K` |
| 8 | `attn.q` | `layers.8.self_attention.to_q.weight` | `Q5_K` | `Q8_0` |
| 8 | `mlp.gate` | `layers.8.mlp.gate_proj.weight` | `Q5_K` | `Q6_K` |
| 9 | `attn.k` | `layers.9.self_attention.to_k.weight` | `Q5_K` | `Q6_K` |
| 9 | `attn.out` | `layers.9.self_attention.to_out.0.weight` | `Q5_K` | `Q8_0` |
| 9 | `attn.q` | `layers.9.self_attention.to_q.weight` | `Q5_K` | `Q6_K` |
| 9 | `attn.v` | `layers.9.self_attention.to_v.weight` | `Q6_K` | `Q8_0` |
| 10 | `attn.k` | `layers.10.self_attention.to_k.weight` | `Q5_K` | `Q6_K` |
| 10 | `attn.out` | `layers.10.self_attention.to_out.0.weight` | `Q5_K` | `Q8_0` |
| 10 | `attn.q` | `layers.10.self_attention.to_q.weight` | `Q5_K` | `Q6_K` |
| 10 | `attn.v` | `layers.10.self_attention.to_v.weight` | `Q6_K` | `Q8_0` |
| 11 | `attn.k` | `layers.11.self_attention.to_k.weight` | `Q5_K` | `Q6_K` |
| 11 | `attn.out` | `layers.11.self_attention.to_out.0.weight` | `Q5_K` | `Q6_K` |
| 11 | `attn.q` | `layers.11.self_attention.to_q.weight` | `Q5_K` | `Q6_K` |
| 11 | `mlp.fc2` | `layers.11.mlp.linear_fc2.weight` | `Q5_K` | `Q6_K` |
| 12 | `attn.k` | `layers.12.self_attention.to_k.weight` | `Q5_K` | `Q6_K` |
| 12 | `attn.out` | `layers.12.self_attention.to_out.0.weight` | `Q5_K` | `Q6_K` |
| 12 | `attn.q` | `layers.12.self_attention.to_q.weight` | `Q5_K` | `Q6_K` |
| 12 | `attn.v` | `layers.12.self_attention.to_v.weight` | `Q6_K` | `Q8_0` |
| 12 | `mlp.fc2` | `layers.12.mlp.linear_fc2.weight` | `Q5_K` | `Q6_K` |
| 12 | `mlp.up` | `layers.12.mlp.up_proj.weight` | `Q5_K` | `Q6_K` |
| 13 | `attn.k` | `layers.13.self_attention.to_k.weight` | `Q5_K` | `Q6_K` |
| 13 | `attn.out` | `layers.13.self_attention.to_out.0.weight` | `Q5_K` | `Q6_K` |
| 13 | `attn.q` | `layers.13.self_attention.to_q.weight` | `Q5_K` | `Q6_K` |
| 13 | `attn.v` | `layers.13.self_attention.to_v.weight` | `Q6_K` | `Q8_0` |
| 13 | `mlp.up` | `layers.13.mlp.up_proj.weight` | `Q5_K` | `Q6_K` |
| 14 | `attn.k` | `layers.14.self_attention.to_k.weight` | `Q5_K` | `Q6_K` |
| 14 | `attn.out` | `layers.14.self_attention.to_out.0.weight` | `Q5_K` | `Q6_K` |
| 14 | `attn.q` | `layers.14.self_attention.to_q.weight` | `Q5_K` | `Q8_0` |
| 14 | `attn.v` | `layers.14.self_attention.to_v.weight` | `Q6_K` | `Q8_0` |
| 14 | `mlp.fc2` | `layers.14.mlp.linear_fc2.weight` | `Q5_K` | `Q6_K` |
| 14 | `mlp.gate` | `layers.14.mlp.gate_proj.weight` | `Q5_K` | `Q6_K` |
| 14 | `mlp.up` | `layers.14.mlp.up_proj.weight` | `Q5_K` | `Q6_K` |
| 15 | `attn.k` | `layers.15.self_attention.to_k.weight` | `Q5_K` | `Q6_K` |
| 15 | `attn.out` | `layers.15.self_attention.to_out.0.weight` | `Q5_K` | `Q6_K` |
| 15 | `attn.q` | `layers.15.self_attention.to_q.weight` | `Q5_K` | `Q8_0` |
| 15 | `attn.v` | `layers.15.self_attention.to_v.weight` | `Q6_K` | `Q8_0` |
| 15 | `mlp.fc2` | `layers.15.mlp.linear_fc2.weight` | `Q5_K` | `Q6_K` |
| 15 | `mlp.up` | `layers.15.mlp.up_proj.weight` | `Q5_K` | `Q6_K` |
| 16 | `attn.k` | `layers.16.self_attention.to_k.weight` | `Q5_K` | `Q6_K` |
| 16 | `attn.out` | `layers.16.self_attention.to_out.0.weight` | `Q5_K` | `Q8_0` |
| 16 | `attn.q` | `layers.16.self_attention.to_q.weight` | `Q5_K` | `Q6_K` |
| 16 | `attn.v` | `layers.16.self_attention.to_v.weight` | `Q6_K` | `Q8_0` |
| 16 | `mlp.fc2` | `layers.16.mlp.linear_fc2.weight` | `Q5_K` | `Q6_K` |
| 16 | `mlp.up` | `layers.16.mlp.up_proj.weight` | `Q5_K` | `Q6_K` |
| 17 | `attn.k` | `layers.17.self_attention.to_k.weight` | `Q5_K` | `Q6_K` |
| 17 | `attn.out` | `layers.17.self_attention.to_out.0.weight` | `Q5_K` | `Q8_0` |
| 17 | `attn.q` | `layers.17.self_attention.to_q.weight` | `Q5_K` | `Q6_K` |
| 17 | `attn.v` | `layers.17.self_attention.to_v.weight` | `Q6_K` | `Q8_0` |
| 17 | `mlp.fc2` | `layers.17.mlp.linear_fc2.weight` | `Q5_K` | `Q6_K` |
| 17 | `mlp.gate` | `layers.17.mlp.gate_proj.weight` | `Q5_K` | `Q6_K` |
| 17 | `mlp.up` | `layers.17.mlp.up_proj.weight` | `Q5_K` | `Q6_K` |
| 18 | `attn.k` | `layers.18.self_attention.to_k.weight` | `Q5_K` | `Q6_K` |
| 18 | `attn.out` | `layers.18.self_attention.to_out.0.weight` | `Q5_K` | `Q8_0` |
| 18 | `attn.q` | `layers.18.self_attention.to_q.weight` | `Q5_K` | `Q6_K` |
| 18 | `attn.v` | `layers.18.self_attention.to_v.weight` | `Q6_K` | `Q8_0` |
| 18 | `mlp.fc2` | `layers.18.mlp.linear_fc2.weight` | `Q5_K` | `Q6_K` |
| 18 | `mlp.gate` | `layers.18.mlp.gate_proj.weight` | `Q5_K` | `Q6_K` |
| 18 | `mlp.up` | `layers.18.mlp.up_proj.weight` | `Q5_K` | `Q6_K` |
| 19 | `attn.k` | `layers.19.self_attention.to_k.weight` | `Q5_K` | `Q6_K` |
| 19 | `attn.out` | `layers.19.self_attention.to_out.0.weight` | `Q5_K` | `Q8_0` |
| 19 | `attn.q` | `layers.19.self_attention.to_q.weight` | `Q5_K` | `Q6_K` |
| 19 | `attn.v` | `layers.19.self_attention.to_v.weight` | `Q6_K` | `Q8_0` |
| 19 | `mlp.fc2` | `layers.19.mlp.linear_fc2.weight` | `Q5_K` | `Q6_K` |
| 19 | `mlp.gate` | `layers.19.mlp.gate_proj.weight` | `Q5_K` | `Q6_K` |
| 19 | `mlp.up` | `layers.19.mlp.up_proj.weight` | `Q5_K` | `Q6_K` |
| 20 | `attn.k` | `layers.20.self_attention.to_k.weight` | `Q5_K` | `Q6_K` |
| 20 | `attn.out` | `layers.20.self_attention.to_out.0.weight` | `Q5_K` | `Q6_K` |
| 20 | `attn.q` | `layers.20.self_attention.to_q.weight` | `Q5_K` | `Q6_K` |
| 20 | `attn.v` | `layers.20.self_attention.to_v.weight` | `Q6_K` | `Q8_0` |
| 20 | `mlp.fc2` | `layers.20.mlp.linear_fc2.weight` | `Q5_K` | `Q6_K` |
| 20 | `mlp.gate` | `layers.20.mlp.gate_proj.weight` | `Q5_K` | `Q6_K` |
| 20 | `mlp.up` | `layers.20.mlp.up_proj.weight` | `Q5_K` | `Q6_K` |
| 21 | `attn.k` | `layers.21.self_attention.to_k.weight` | `Q5_K` | `Q6_K` |
| 21 | `attn.out` | `layers.21.self_attention.to_out.0.weight` | `Q5_K` | `Q6_K` |
| 21 | `attn.q` | `layers.21.self_attention.to_q.weight` | `Q5_K` | `Q6_K` |
| 21 | `mlp.fc2` | `layers.21.mlp.linear_fc2.weight` | `Q5_K` | `Q6_K` |
| 21 | `mlp.up` | `layers.21.mlp.up_proj.weight` | `Q5_K` | `Q6_K` |
| 22 | `attn.k` | `layers.22.self_attention.to_k.weight` | `Q5_K` | `Q6_K` |
| 22 | `attn.out` | `layers.22.self_attention.to_out.0.weight` | `Q5_K` | `Q6_K` |
| 22 | `attn.q` | `layers.22.self_attention.to_q.weight` | `Q5_K` | `Q6_K` |
| 22 | `attn.v` | `layers.22.self_attention.to_v.weight` | `Q6_K` | `Q8_0` |
| 22 | `mlp.fc2` | `layers.22.mlp.linear_fc2.weight` | `Q5_K` | `Q6_K` |
| 22 | `mlp.up` | `layers.22.mlp.up_proj.weight` | `Q5_K` | `Q6_K` |
| 23 | `attn.k` | `layers.23.self_attention.to_k.weight` | `Q5_K` | `Q6_K` |
| 23 | `attn.out` | `layers.23.self_attention.to_out.0.weight` | `Q5_K` | `Q6_K` |
| 23 | `attn.q` | `layers.23.self_attention.to_q.weight` | `Q5_K` | `Q6_K` |
| 23 | `attn.v` | `layers.23.self_attention.to_v.weight` | `Q6_K` | `Q8_0` |
| 23 | `mlp.fc2` | `layers.23.mlp.linear_fc2.weight` | `Q5_K` | `Q6_K` |
| 23 | `mlp.up` | `layers.23.mlp.up_proj.weight` | `Q5_K` | `Q6_K` |
| 24 | `attn.k` | `layers.24.self_attention.to_k.weight` | `Q5_K` | `Q6_K` |
| 24 | `attn.out` | `layers.24.self_attention.to_out.0.weight` | `Q5_K` | `Q6_K` |
| 24 | `attn.q` | `layers.24.self_attention.to_q.weight` | `Q5_K` | `Q6_K` |
| 24 | `attn.v` | `layers.24.self_attention.to_v.weight` | `Q6_K` | `Q8_0` |
| 24 | `mlp.fc2` | `layers.24.mlp.linear_fc2.weight` | `Q5_K` | `Q6_K` |
| 24 | `mlp.up` | `layers.24.mlp.up_proj.weight` | `Q5_K` | `Q6_K` |
| 25 | `attn.k` | `layers.25.self_attention.to_k.weight` | `Q5_K` | `Q6_K` |
| 25 | `attn.out` | `layers.25.self_attention.to_out.0.weight` | `Q5_K` | `Q6_K` |
| 25 | `attn.q` | `layers.25.self_attention.to_q.weight` | `Q5_K` | `Q6_K` |
| 25 | `mlp.fc2` | `layers.25.mlp.linear_fc2.weight` | `Q5_K` | `Q6_K` |
| 25 | `mlp.up` | `layers.25.mlp.up_proj.weight` | `Q5_K` | `Q6_K` |
| 26 | `attn.k` | `layers.26.self_attention.to_k.weight` | `Q5_K` | `Q6_K` |
| 26 | `attn.out` | `layers.26.self_attention.to_out.0.weight` | `Q5_K` | `Q6_K` |
| 26 | `attn.q` | `layers.26.self_attention.to_q.weight` | `Q5_K` | `Q6_K` |
| 26 | `mlp.fc2` | `layers.26.mlp.linear_fc2.weight` | `Q5_K` | `Q6_K` |
| 26 | `mlp.up` | `layers.26.mlp.up_proj.weight` | `Q5_K` | `Q6_K` |
| 27 | `attn.k` | `layers.27.self_attention.to_k.weight` | `Q5_K` | `Q6_K` |
| 27 | `attn.out` | `layers.27.self_attention.to_out.0.weight` | `Q5_K` | `Q6_K` |
| 27 | `mlp.fc2` | `layers.27.mlp.linear_fc2.weight` | `Q5_K` | `Q6_K` |
| 27 | `mlp.up` | `layers.27.mlp.up_proj.weight` | `Q5_K` | `Q6_K` |
| 28 | `attn.k` | `layers.28.self_attention.to_k.weight` | `Q5_K` | `Q6_K` |
| 28 | `attn.out` | `layers.28.self_attention.to_out.0.weight` | `Q5_K` | `Q6_K` |
| 28 | `mlp.fc2` | `layers.28.mlp.linear_fc2.weight` | `Q5_K` | `Q6_K` |
| 28 | `mlp.up` | `layers.28.mlp.up_proj.weight` | `Q5_K` | `Q6_K` |
| 29 | `attn.k` | `layers.29.self_attention.to_k.weight` | `Q5_K` | `Q6_K` |
| 29 | `mlp.fc2` | `layers.29.mlp.linear_fc2.weight` | `Q5_K` | `Q6_K` |
| 31 | `attn.k` | `layers.31.self_attention.to_k.weight` | `Q5_K` | `Q6_K` |
| 34 | `attn.k` | `layers.34.self_attention.to_k.weight` | `Q5_K` | `Q5_K` |
| 34 | `attn.out` | `layers.34.self_attention.to_out.0.weight` | `Q5_K` | `Q5_K` |
| 34 | `attn.q` | `layers.34.self_attention.to_q.weight` | `Q5_K` | `Q5_K` |
| 34 | `mlp.fc2` | `layers.34.mlp.linear_fc2.weight` | `Q5_K` | `Q5_K` |
| 34 | `mlp.gate` | `layers.34.mlp.gate_proj.weight` | `Q5_K` | `Q5_K` |
| 34 | `mlp.up` | `layers.34.mlp.up_proj.weight` | `Q5_K` | `Q5_K` |
| 35 | `attn.k` | `layers.35.self_attention.to_k.weight` | `Q5_K` | `Q5_K` |
| 35 | `attn.out` | `layers.35.self_attention.to_out.0.weight` | `Q5_K` | `Q5_K` |
| 35 | `attn.q` | `layers.35.self_attention.to_q.weight` | `Q5_K` | `Q5_K` |
| 35 | `mlp.fc2` | `layers.35.mlp.linear_fc2.weight` | `Q5_K` | `Q5_K` |
| 35 | `mlp.gate` | `layers.35.mlp.gate_proj.weight` | `Q5_K` | `Q5_K` |
| 35 | `mlp.up` | `layers.35.mlp.up_proj.weight` | `Q5_K` | `Q5_K` |
