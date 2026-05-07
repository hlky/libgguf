# Correctness

The native CPU path is the reference path for libgguf.

## Byte Exactness

For quantization, byte exactness means:

```text
same input + same qtype + same shape + same imatrix behavior -> identical encoded bytes
```

This is stricter than numerical closeness after dequantization. It is important because GGUF quantization formats are byte-level storage formats, and tiny changes in scalar search order, rounding, or packing can change the encoded result.

## Dequantization Checks

For dequantization, tests compare decoded values for a fixed output dtype. Some paths compare exact decoded bytes; others use value equality or tight tolerances depending on dtype and backend behavior.

## Current Test Strategy

Current checks generate CPU reference outputs at test time and compare other implementations against those outputs. This covers:

- native SIMD/threaded backend parity with the reference backend;
- Python row API coverage;
- NumPy backend parity;
- Torch backend parity;
- CUDA quantization byte equality when CUDA is available;
- CUDA dequantization decoded-output parity when CUDA is available;
- converter policy and native executable behavior.

Frozen golden fixtures live at `tests/golden/manifest.json` so native CPU
encoded-byte drift is visible even when generated reference tests still agree
with the current source tree. Generated CPU-reference tests remain useful for
backend parity; the frozen manifest is specifically for native CPU storage hash
stability.

## Exactness Checker Command

Run deterministic edge-case row checks:

```bash
python bench/check_exact.py --qtypes Q4_K,Q5_K,IQ2_XS --shapes 4x4096
```

Write or compare a JSON fixture:

```bash
python bench/check_exact.py --qtypes Q4_K,Q5_K,IQ2_XS --shapes 4x4096 \
  --write-json reports/exactness/qk_iq2xs.json
python bench/check_exact.py --qtypes Q4_K,Q5_K,IQ2_XS --shapes 4x4096 \
  --expect-json reports/exactness/qk_iq2xs.json
```

Check or update frozen goldens:

```bash
python scripts/update_golden.py --check
python scripts/update_golden.py
```

## Recommended Edge Cases

Golden and generated checks should include:

- all zeros;
- constants;
- absmax ties;
- outliers;
- random normal values;
- random uniform values;
- tiny values;
- large values;
- imatrix qtypes with explicit weights;
- shapes spanning one block, multiple blocks, and multiple rows.

CUDA optimization should preserve scalar accumulation order where byte-sensitive searches depend on that order. Any warp or CTA parallelism that changes reduction order needs an exactness proof and tests.
