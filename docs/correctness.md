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

Frozen golden fixtures are planned so correctness does not rely only on generated references from the current source tree.

## Planned Golden Fixture Command

The command below is planned documentation for a future fixture checker; it does not currently exist in the repo:

```bash
python bench/check_exact.py --qtypes Q4_K,Q5_K,IQ2_XS --shapes 4x4096
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
