# CDNA3 / gfx942 experimental pack

Physical wave64 retains logical groups of 32 for format-defined blocks, shuffles and Q5 ballot packing. No file format or block size changes. The gfx942 quantizer parity executable compares all 24 production formats with REF CPU bytes, exercises upper half-waves/tails and checks allocation guards. It also supplies optional native timings.

Select `LIBGGUF_ARCHITECTURE=gfx942` at CMake configuration time (libgguf uses `LIBGGUF_ROCM_ARCH=gfx942`). Use a separate build/install directory from gfx1201. A checked entry rejects the wrong architecture/wave size; no automatic runtime candidate selection is added.

Windows ROCm 10.1 compilation is a smoke check, not Linux or numerical qualification. No GPU tests or benchmarks ran during implementation. The local kernel superrepo supplies pinned Linux checkout/build/qualification/benchmark commands in `docs/mi300x/HARDWARE.md` and `tools/mi300x_linux.py`. Experimental commits must remain on their branch until hardware evidence justifies promotion to main.
