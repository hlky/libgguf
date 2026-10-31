# CDNA3 / gfx942 experimental pack

Physical wave64 retains logical groups of 32 for format-defined blocks, shuffles and Q5 ballot packing. No file format or block size changes. The gfx942 quantizer parity executable compares all 24 production formats with REF CPU bytes, exercises upper half-waves/tails and checks allocation guards. It also supplies optional native timings.

Select `LIBGGUF_ARCHITECTURE=gfx942` at CMake configuration time (libgguf uses `LIBGGUF_ROCM_ARCH=gfx942`). Use a separate build/install directory from gfx1201. A checked entry rejects the wrong architecture/wave size; no automatic runtime candidate selection is added.

Windows ROCm 10.1 compilation is a smoke check, not Linux or numerical qualification. No GPU tests or benchmarks ran during implementation. The local kernel superrepo supplies pinned Linux checkout/build/qualification/benchmark commands in `docs/mi300x/HARDWARE.md` and `tools/mi300x_linux.py`. Experimental commits must remain on their branch until hardware evidence justifies promotion to main.

## Pipeline candidates added 2026-10-31

`libgguf/gpu/gfx942/decode_pipeline.cuh` adds four explicit coalesced decoder schedules: 128/256 threads x 1/2 groups per physical wave. Format-defined decode lanes first populate LDS, then the physical wave writes contiguous four-float packs. The IQ4_NL adapter admits natural 32-value rows. All 24 formats use the canonical decoder; encoded GGUF bytes and production quantizers are unchanged. Checked `launch_decode<Threads,GroupsPerWave>` and trusted `launch_decode<Threads,GroupsPerWave,false>` are distinct. Counts must be complete decode groups. The existing gfx942 parity executable qualifies every schedule with guards and REF values and optionally times them separately from quantization.

All candidates remain unqualified until the pinned Linux MI300X suites pass. Local evidence is bounded compilation and linking, with no GPU execution.
