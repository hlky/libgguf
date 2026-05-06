# ERNIE Dynamic Mismatch Investigation

Comparator: extended `dynamic` policy versus ERNIE UD override targets.

## Summary

The extended policy improves base ERNIE by adding mid/late output and MLP up/down boosts with a final-tail guard. Turbo remains less exact because its UD layer bands differ from base ERNIE.

## unsloth/ERNIE-Image-GGUF

### UD-Q2_K (172/193, 89.1% exact)

| Role | Dynamic | UD | Count | Layers | Direction |
|---|---|---|---:|---|---|
| `attn.k` | `Q3_K` | `Q4_K` | 1 | 31 | under |
| `attn.k` | `Q4_K` | `Q3_K` | 2 | 0-1 | over |
| `attn.q` | `Q3_K` | `Q4_K` | 2 | 31,33 | under |
| `attn.q` | `Q3_K` | `Q5_K` | 1 | 35 | under |
| `attn.q` | `Q4_K` | `Q3_K` | 2 | 0-1 | over |
| `attn.v` | `Q6_K` | `Q4_K` | 11 | 2-3,6-14 | over |
| `mlp.gate` | `Q3_K` | `Q2_K` | 1 | 1 | over |
| `mlp.up` | `Q3_K` | `Q2_K` | 1 | 1 | over |

Target map by role/layer qtype:

| Role | UD qtype | Layers |
|---|---|---|
| `attn.k` | `Q3_K` | 0-3,14-30,32-34 |
| `attn.k` | `Q4_K` | 31 |
| `attn.out` | `Q3_K` | 2-3,6-14,25-27 |
| `attn.out` | `Q4_K` | 15-24,28-33 |
| `attn.q` | `Q3_K` | 0-3,13-30,32,34 |
| `attn.q` | `Q4_K` | 31,33 |
| `attn.q` | `Q5_K` | 35 |
| `attn.v` | `Q4_K` | 2-3,6-14 |
| `attn.v` | `Q6_K` | 0,15-35 |
| `mlp.fc2` | `Q3_K` | 0-1,8,12-20,22 |
| `mlp.fc2` | `Q4_K` | 7,21,23-33 |
| `mlp.gate` | `Q2_K` | 1 |
| `mlp.gate` | `Q3_K` | 0,7,14-35 |
| `mlp.up` | `Q2_K` | 1 |
| `mlp.up` | `Q3_K` | 0,8,11-20,22 |
| `mlp.up` | `Q4_K` | 7,21,23-33 |

### UD-Q3_K_M (155/170, 91.2% exact)

| Role | Dynamic | UD | Count | Layers | Direction |
|---|---|---|---:|---|---|
| `attn.k` | `Q4_K` | `Q5_K` | 3 | 31,33,35 | under |
| `attn.out` | `Q4_K` | `Q5_K` | 2 | 34-35 | under |
| `attn.out` | `Q5_K` | `Q4_K` | 1 | 21 | over |
| `attn.q` | `Q4_K` | `Q5_K` | 1 | 35 | under |
| `mlp.fc2` | `Q4_K` | `Q3_K` | 1 | 1 | over |
| `mlp.fc2` | `Q4_K` | `Q5_K` | 2 | 34-35 | under |
| `mlp.up` | `Q4_K` | `Q3_K` | 1 | 1 | over |
| `mlp.up` | `Q4_K` | `Q5_K` | 2 | 34-35 | under |
| `mlp.up` | `Q5_K` | `Q4_K` | 2 | 21,25 | over |

Target map by role/layer qtype:

| Role | UD qtype | Layers |
|---|---|---|
| `attn.k` | `Q4_K` | 2,14-30,32 |
| `attn.k` | `Q5_K` | 31,33,35 |
| `attn.out` | `Q4_K` | 2-3,7-10,12-14,21,25-27 |
| `attn.out` | `Q5_K` | 0-1,15-20,22-24,28-35 |
| `attn.q` | `Q4_K` | 2-3,14-26,28-33 |
| `attn.q` | `Q5_K` | 0,35 |
| `attn.v` | `Q6_K` | 0-2,7-35 |
| `mlp.fc2` | `Q3_K` | 1 |
| `mlp.fc2` | `Q4_K` | 14-20,22 |
| `mlp.fc2` | `Q5_K` | 7,21,23-35 |
| `mlp.gate` | `Q4_K` | 7,20,23-33 |
| `mlp.up` | `Q3_K` | 1 |
| `mlp.up` | `Q4_K` | 14-22,25 |
| `mlp.up` | `Q5_K` | 7,23-24,26-35 |

### UD-Q4_K_M (136/154, 88.3% exact)

| Role | Dynamic | UD | Count | Layers | Direction |
|---|---|---|---:|---|---|
| `attn.out` | `Q5_K` | `Q6_K` | 2 | 34-35 | under |
| `attn.out` | `Q6_K` | `Q5_K` | 3 | 20,23,29 | over |
| `attn.q` | `Q5_K` | `Q6_K` | 2 | 3,35 | under |
| `mlp.fc2` | `Q5_K` | `Q6_K` | 2 | 34-35 | under |
| `mlp.fc2` | `Q6_K` | `Q5_K` | 3 | 23-25 | over |
| `mlp.gate` | `Q6_K` | `Q4_K` | 1 | 1 | over |
| `mlp.up` | `Q5_K` | `Q6_K` | 2 | 34-35 | under |
| `mlp.up` | `Q6_K` | `Q5_K` | 3 | 21,25-26 | over |

Target map by role/layer qtype:

| Role | UD qtype | Layers |
|---|---|---|
| `attn.k` | `Q5_K` | 2-3,7,13-33 |
| `attn.k` | `Q6_K` | 0 |
| `attn.out` | `Q5_K` | 2-3,6-14,20,23,25-27,29 |
| `attn.out` | `Q6_K` | 1,15-19,21-22,24,28,30-35 |
| `attn.q` | `Q5_K` | 2,8,13-33 |
| `attn.q` | `Q6_K` | 0,3,35 |
| `attn.v` | `Q8_0` | 0,24,31-35 |
| `mlp.fc2` | `Q5_K` | 13-20,22-25 |
| `mlp.fc2` | `Q6_K` | 7,21,26-35 |
| `mlp.gate` | `Q4_K` | 1 |
| `mlp.gate` | `Q5_K` | 2,7,18,21,23-24,26-33 |
| `mlp.up` | `Q5_K` | 2,14-22,25-26 |
| `mlp.up` | `Q6_K` | 7,23-24,27-35 |

### UD-Q5_K_M (138/146, 94.5% exact)

| Role | Dynamic | UD | Count | Layers | Direction |
|---|---|---|---:|---|---|
| `attn.out` | `Q6_K` | `Q8_0` | 2 | 34-35 | under |
| `mlp.fc2` | `Q6_K` | `Q8_0` | 2 | 34-35 | under |
| `mlp.fc2` | `Q8_0` | `Q6_K` | 1 | 7 | over |
| `mlp.up` | `Q6_K` | `Q8_0` | 1 | 34 | under |
| `mlp.up` | `Q8_0` | `Q6_K` | 2 | 27-28 | over |

Target map by role/layer qtype:

| Role | UD qtype | Layers |
|---|---|---|
| `attn.k` | `Q6_K` | 2-3,15-31,33 |
| `attn.k` | `Q8_0` | 0 |
| `attn.out` | `Q6_K` | 2,4,7-10,12-23,25-30 |
| `attn.out` | `Q8_0` | 0-1,3,24,31-35 |
| `attn.q` | `Q6_K` | 3-4,9,14-17,19-24,26,29-31,33 |
| `attn.v` | `Q8_0` | 0-1,9,14-15,17-20,22-35 |
| `mlp.fc2` | `Q6_K` | 7,14-26 |
| `mlp.fc2` | `Q8_0` | 27-35 |
| `mlp.gate` | `Q6_K` | 2,28-33 |
| `mlp.up` | `Q6_K` | 15-28 |
| `mlp.up` | `Q8_0` | 7,29-34 |

## unsloth/ERNIE-Image-Turbo-GGUF

### UD-Q2_K (153/230, 66.5% exact)

| Role | Dynamic | UD | Count | Layers | Direction |
|---|---|---|---:|---|---|
| `attn.k` | `Q3_K` | `Q2_K` | 1 | 34 | over |
| `attn.k` | `Q3_K` | `Q4_K` | 12 | 2-3,11,15-23 | under |
| `attn.k` | `Q4_K` | `Q6_K` | 1 | 0 | under |
| `attn.out` | `Q3_K` | `Q4_K` | 4 | 2-4,13 | under |
| `attn.out` | `Q4_K` | `Q3_K` | 5 | 29-33 | over |
| `attn.out` | `Q4_K` | `Q5_K` | 1 | 0 | under |
| `attn.q` | `Q3_K` | `Q2_K` | 1 | 34 | over |
| `attn.q` | `Q3_K` | `Q4_K` | 10 | 2-3,8,11,15-20 | under |
| `attn.q` | `Q4_K` | `Q5_K` | 1 | 0 | under |
| `attn.v` | `Q6_K` | `Q3_K` | 1 | 34 | over |
| `attn.v` | `Q6_K` | `Q4_K` | 7 | 5-6,29-33 | over |
| `mlp.fc2` | `Q3_K` | `Q2_K` | 2 | 34-35 | over |
| `mlp.fc2` | `Q3_K` | `Q4_K` | 5 | 16-20 | under |
| `mlp.fc2` | `Q4_K` | `Q3_K` | 11 | 7,23-32 | over |
| `mlp.gate` | `Q3_K` | `Q2_K` | 2 | 34-35 | over |
| `mlp.up` | `Q3_K` | `Q2_K` | 2 | 34-35 | over |
| `mlp.up` | `Q4_K` | `Q3_K` | 11 | 7,23-32 | over |

Target map by role/layer qtype:

| Role | UD qtype | Layers |
|---|---|---|
| `attn.k` | `Q2_K` | 34 |
| `attn.k` | `Q3_K` | 4-10,12-14,24-31,33,35 |
| `attn.k` | `Q4_K` | 2-3,11,15-23 |
| `attn.k` | `Q6_K` | 0 |
| `attn.out` | `Q3_K` | 5-12,14,25-27,29-35 |
| `attn.out` | `Q4_K` | 2-4,13,15-24,28 |
| `attn.out` | `Q5_K` | 0 |
| `attn.q` | `Q2_K` | 34 |
| `attn.q` | `Q3_K` | 4-7,9-10,12-14,21-31,33,35 |
| `attn.q` | `Q4_K` | 2-3,8,11,15-20 |
| `attn.q` | `Q5_K` | 0 |
| `attn.v` | `Q3_K` | 34 |
| `attn.v` | `Q4_K` | 5-6,29-33 |
| `attn.v` | `Q6_K` | 0-4,7-28 |
| `mlp.fc2` | `Q2_K` | 34-35 |
| `mlp.fc2` | `Q3_K` | 0,2-15,22-32 |
| `mlp.fc2` | `Q4_K` | 16-21 |
| `mlp.gate` | `Q2_K` | 34-35 |
| `mlp.gate` | `Q3_K` | 1-7,12-29 |
| `mlp.up` | `Q2_K` | 34-35 |
| `mlp.up` | `Q3_K` | 1-8,12-20,22-32 |
| `mlp.up` | `Q4_K` | 21 |

### UD-Q3_K_M (131/206, 63.6% exact)

| Role | Dynamic | UD | Count | Layers | Direction |
|---|---|---|---:|---|---|
| `attn.k` | `Q4_K` | `Q3_K` | 2 | 34-35 | over |
| `attn.k` | `Q4_K` | `Q5_K` | 12 | 2-4,8,13,15-20,22 | under |
| `attn.k` | `Q5_K` | `Q6_K` | 1 | 0 | under |
| `attn.out` | `Q4_K` | `Q3_K` | 2 | 34-35 | over |
| `attn.out` | `Q4_K` | `Q5_K` | 2 | 2-3 | under |
| `attn.out` | `Q5_K` | `Q4_K` | 4 | 28-31 | over |
| `attn.out` | `Q5_K` | `Q6_K` | 2 | 0-1 | under |
| `attn.q` | `Q4_K` | `Q3_K` | 1 | 34 | over |
| `attn.q` | `Q4_K` | `Q5_K` | 9 | 3-4,7-8,13-16,18 | under |
| `attn.q` | `Q4_K` | `Q6_K` | 1 | 2 | under |
| `attn.v` | `Q6_K` | `Q8_0` | 1 | 0 | under |
| `mlp.fc2` | `Q4_K` | `Q3_K` | 2 | 34-35 | over |
| `mlp.fc2` | `Q4_K` | `Q5_K` | 6 | 0-2,18-20 | under |
| `mlp.fc2` | `Q5_K` | `Q4_K` | 9 | 7,23-30 | over |
| `mlp.gate` | `Q4_K` | `Q3_K` | 2 | 34-35 | over |
| `mlp.gate` | `Q4_K` | `Q5_K` | 3 | 0-2 | under |
| `mlp.up` | `Q4_K` | `Q3_K` | 2 | 34-35 | over |
| `mlp.up` | `Q4_K` | `Q5_K` | 4 | 0-1,18-19 | under |
| `mlp.up` | `Q5_K` | `Q4_K` | 10 | 7,21,23-30 | over |

Target map by role/layer qtype:

| Role | UD qtype | Layers |
|---|---|---|
| `attn.k` | `Q3_K` | 34-35 |
| `attn.k` | `Q4_K` | 5-7,9-12,14,21,23-29,31 |
| `attn.k` | `Q5_K` | 1-4,8,13,15-20,22 |
| `attn.k` | `Q6_K` | 0 |
| `attn.out` | `Q3_K` | 34-35 |
| `attn.out` | `Q4_K` | 4-14,25-31 |
| `attn.out` | `Q5_K` | 2-3,15-24 |
| `attn.out` | `Q6_K` | 0-1 |
| `attn.q` | `Q3_K` | 34 |
| `attn.q` | `Q4_K` | 5-6,9-12,17,19-29 |
| `attn.q` | `Q5_K` | 0-1,3-4,7-8,13-16,18 |
| `attn.q` | `Q6_K` | 2 |
| `attn.v` | `Q6_K` | 1-31 |
| `attn.v` | `Q8_0` | 0 |
| `mlp.fc2` | `Q3_K` | 34-35 |
| `mlp.fc2` | `Q4_K` | 3-7,13-17,22-30 |
| `mlp.fc2` | `Q5_K` | 0-2,18-21 |
| `mlp.gate` | `Q3_K` | 34-35 |
| `mlp.gate` | `Q4_K` | 3-7,14-22 |
| `mlp.gate` | `Q5_K` | 0-2 |
| `mlp.up` | `Q3_K` | 34-35 |
| `mlp.up` | `Q4_K` | 2-7,12-17,20-30 |
| `mlp.up` | `Q5_K` | 0-1,18-19 |

### UD-Q4_K_M (101/175, 57.7% exact)

| Role | Dynamic | UD | Count | Layers | Direction |
|---|---|---|---:|---|---|
| `attn.k` | `Q5_K` | `Q4_K` | 2 | 34-35 | over |
| `attn.k` | `Q5_K` | `Q6_K` | 15 | 2-4,6,11-19,21-22 | under |
| `attn.k` | `Q6_K` | `Q8_0` | 1 | 0 | under |
| `attn.out` | `Q5_K` | `Q4_K` | 2 | 34-35 | over |
| `attn.out` | `Q5_K` | `Q6_K` | 3 | 3,7,11 | under |
| `attn.out` | `Q5_K` | `Q8_0` | 1 | 2 | under |
| `attn.out` | `Q6_K` | `Q5_K` | 5 | 23,28-31 | over |
| `attn.out` | `Q6_K` | `Q8_0` | 1 | 0 | under |
| `attn.q` | `Q5_K` | `Q4_K` | 1 | 34 | over |
| `attn.q` | `Q5_K` | `Q6_K` | 9 | 2-6,10-12,16 | under |
| `attn.q` | `Q6_K` | `Q8_0` | 1 | 0 | under |
| `mlp.fc2` | `Q5_K` | `Q4_K` | 2 | 34-35 | over |
| `mlp.fc2` | `Q5_K` | `Q6_K` | 6 | 2-3,15-17,19 | under |
| `mlp.fc2` | `Q6_K` | `Q5_K` | 9 | 7,23-30 | over |
| `mlp.gate` | `Q5_K` | `Q4_K` | 2 | 34-35 | over |
| `mlp.gate` | `Q5_K` | `Q6_K` | 2 | 2-3 | under |
| `mlp.up` | `Q5_K` | `Q4_K` | 2 | 34-35 | over |
| `mlp.up` | `Q5_K` | `Q6_K` | 1 | 2 | under |
| `mlp.up` | `Q6_K` | `Q5_K` | 9 | 7,21,23-29 | over |

Target map by role/layer qtype:

| Role | UD qtype | Layers |
|---|---|---|
| `attn.k` | `Q4_K` | 34-35 |
| `attn.k` | `Q5_K` | 5,7-10,20,23-29,31 |
| `attn.k` | `Q6_K` | 1-4,6,11-19,21-22 |
| `attn.k` | `Q8_0` | 0 |
| `attn.out` | `Q4_K` | 34-35 |
| `attn.out` | `Q5_K` | 4-6,8-10,12-14,23,25-31 |
| `attn.out` | `Q6_K` | 1,3,7,11,15-22,24 |
| `attn.out` | `Q8_0` | 0,2 |
| `attn.q` | `Q4_K` | 34 |
| `attn.q` | `Q5_K` | 7-9,13-15,17-28 |
| `attn.q` | `Q6_K` | 1-6,10-12,16 |
| `attn.q` | `Q8_0` | 0 |
| `attn.v` | `Q8_0` | 0-2 |
| `mlp.fc2` | `Q4_K` | 34-35 |
| `mlp.fc2` | `Q5_K` | 4-7,13-14,18,20,22-30 |
| `mlp.fc2` | `Q6_K` | 0-3,15-17,19,21 |
| `mlp.gate` | `Q4_K` | 34-35 |
| `mlp.gate` | `Q5_K` | 4-7,15-21 |
| `mlp.gate` | `Q6_K` | 0-3 |
| `mlp.up` | `Q4_K` | 34-35 |
| `mlp.up` | `Q5_K` | 3-9,12-29 |
| `mlp.up` | `Q6_K` | 0-2 |

### UD-Q5_K_M (142/184, 77.2% exact)

| Role | Dynamic | UD | Count | Layers | Direction |
|---|---|---|---:|---|---|
| `attn.k` | `Q6_K` | `Q5_K` | 2 | 34-35 | over |
| `attn.k` | `Q6_K` | `Q8_0` | 4 | 2-3,7-8 | under |
| `attn.out` | `Q6_K` | `Q5_K` | 2 | 34-35 | over |
| `attn.out` | `Q6_K` | `Q8_0` | 10 | 2,4,6-7,9-10,16-19 | under |
| `attn.out` | `Q8_0` | `Q6_K` | 1 | 24 | over |
| `attn.q` | `Q6_K` | `Q5_K` | 2 | 34-35 | over |
| `attn.q` | `Q6_K` | `Q8_0` | 7 | 2-4,7-8,14-15 | under |
| `mlp.fc2` | `Q6_K` | `Q5_K` | 2 | 34-35 | over |
| `mlp.fc2` | `Q8_0` | `Q6_K` | 4 | 7,27-29 | over |
| `mlp.gate` | `Q6_K` | `Q5_K` | 2 | 34-35 | over |
| `mlp.gate` | `Q6_K` | `Q8_0` | 1 | 2 | under |
| `mlp.up` | `Q6_K` | `Q5_K` | 2 | 34-35 | over |
| `mlp.up` | `Q8_0` | `Q6_K` | 3 | 7,27-28 | over |

Target map by role/layer qtype:

| Role | UD qtype | Layers |
|---|---|---|
| `attn.k` | `Q5_K` | 34-35 |
| `attn.k` | `Q6_K` | 4,6,9-29,31 |
| `attn.k` | `Q8_0` | 0-3,7-8 |
| `attn.out` | `Q5_K` | 34-35 |
| `attn.out` | `Q6_K` | 5,8,11-15,20-28 |
| `attn.out` | `Q8_0` | 0-4,6-7,9-10,16-19 |
| `attn.q` | `Q5_K` | 34-35 |
| `attn.q` | `Q6_K` | 5-6,9-13,16-26 |
| `attn.q` | `Q8_0` | 0-4,7-8,14-15 |
| `attn.v` | `Q8_0` | 0-5,7,9-10,12-20,22-24 |
| `mlp.fc2` | `Q5_K` | 34-35 |
| `mlp.fc2` | `Q6_K` | 2-7,11-12,14-29 |
| `mlp.fc2` | `Q8_0` | 0-1 |
| `mlp.gate` | `Q5_K` | 34-35 |
| `mlp.gate` | `Q6_K` | 3-8,14,17-20 |
| `mlp.gate` | `Q8_0` | 0-2 |
| `mlp.up` | `Q5_K` | 34-35 |
| `mlp.up` | `Q6_K` | 2-7,12-28 |
| `mlp.up` | `Q8_0` | 0-1 |
