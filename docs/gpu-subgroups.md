# Logical quantizer subgroups on AMD wave64

GGUF's 32-value block grouping is independent of a GPU's physical wave size.
The same Q8_0, Q4_0/Q5_0 extrema helpers, Q5_1, TQ and MXFP4 quantizers keep
logical groups of 32 on gfx942. NVFP4 keeps its eight-lane reduction groups and
broadcasts within the containing logical group of 32.

Every shuffle in these paths specifies its logical width. Q5_0/Q5_1 ballot
packing uses `gpu/subgroup.cuh` to select the current logical group's 32 bits
from the physical ballot before extracting the low 16 high-bit flags. Masks
remain native-width until that extraction. One-dimensional quantizer thread
blocks and complete logical groups are required, as in the existing launchers.
An odd final block count may deactivate one half of a wave64; it must not mix
the remaining block's scale or packed bits with another group's values.

The runtime-free fragment interfaces and GGUF block layouts are unchanged.
The Q3_K shared-memory path retains its physical ballot and native-width
synchronization mask; it does not pack that mask into GGUF bytes. This change
does not establish parity for that path or the other K/IQ quantizers.

## Compile-only bring-up

```powershell
python scripts/compile_gpu_quantizers.py `
  --sdk H:/rocm-sdk/10.1.0/therock-dist-windows-multiarch-10.1.0 `
  --output build/cdna3-quantizers --architectures gfx942 gfx1201 --jobs 4
```

The script compiles all 24 production quantizer translation units for each
target, retaining `-ffp-contract=off`. It saves commands, compiler version,
header/source/object hashes, logs and exit codes. It loads no binaries and
executes no GPU kernels. A Windows compile is not Linux ABI qualification;
CUDA compilation is also outside this check.

MI300X byte parity against the REF CPU backend remains required. Focus on the
upper logical group with very different scales, high-bit patterns and signs;
zero blocks, first-index extrema ties, rounding boundaries and odd block counts.
No numerical, performance or full-package qualification is claimed here.
