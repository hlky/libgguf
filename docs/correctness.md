# Correctness

The compiled `REF` CPU backend is the reference implementation.

For quantization, parity means identical encoded bytes for the same input,
qtype, row shape, and importance matrix. Numerical similarity after decoding is
not sufficient because GGUF qtypes are byte-level storage formats.

For dequantization, tests compare decoded values using exact equality or a
dtype-appropriate tolerance.

The test suite covers:

- native CPU row operations;
- Python row and metadata APIs;
- NumPy and Torch parity;
- native GPU quantization and dequantization parity when a GPU build is
  installed;
- Torch compilation separately when Triton is available;
- GGUF inspection, validation, and comparison.

Run the installed package tests from the project directory:

```bash
python -m pytest tests
```
