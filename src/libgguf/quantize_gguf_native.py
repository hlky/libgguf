from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from .quantize import NATIVE_DEFAULT_SCRATCH_BYTES, convert_safetensors_to_gguf_native, parse_qtype, parse_tensor_qtype


def _parse_tensor_type(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected PATTERN=QTYPE")
    pattern, qtype = value.split("=", 1)
    if not pattern or not qtype:
        raise argparse.ArgumentTypeError("expected non-empty PATTERN=QTYPE")
    parse_tensor_qtype(qtype)
    return pattern, qtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Native safetensors-only quantization of diffusion models to GGUF")
    parser.add_argument("--src", required=True, help="Source safetensors model")
    parser.add_argument("--qtype", required=True, help="Output file type, e.g. Q4_K_S, Q4_K_M, Q4_K, Q8_0")
    parser.add_argument("--dst", help="Output GGUF path")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output file")
    parser.add_argument("--policy", choices=("comfy", "dynamic", "uniform"), default="comfy", help="Tensor selection policy")
    parser.add_argument("--imatrix", help="llama.cpp imatrix file")
    parser.add_argument("--tensor-type", action="append", type=_parse_tensor_type, default=[], metavar="PATTERN=QTYPE")
    parser.add_argument("--include", action="append", default=[], metavar="PATTERN", help="Force matching tensors into quantization when possible")
    parser.add_argument("--exclude", action="append", default=[], metavar="PATTERN", help="Keep matching tensors unquantized")
    parser.add_argument(
        "--scratch-bytes",
        type=int,
        default=NATIVE_DEFAULT_SCRATCH_BYTES,
        help=f"Native scratch buffer target in bytes (default: {NATIVE_DEFAULT_SCRATCH_BYTES})",
    )
    args = parser.parse_args()

    src_path = Path(args.src)
    if not src_path.is_file():
        parser.error(f"invalid source file: {args.src}")
    if src_path.suffix.lower() != ".safetensors":
        parser.error("quantize-gguf-native only supports .safetensors inputs")
    if args.scratch_bytes <= 0:
        parser.error("--scratch-bytes must be positive")
    parse_qtype(args.qtype)
    return args


def main() -> None:
    args = parse_args()
    started = perf_counter()
    result = convert_safetensors_to_gguf_native(
        args.src,
        args.dst,
        args.qtype,
        policy=args.policy,
        overwrite=args.overwrite,
        imatrix=args.imatrix,
        tensor_overrides=args.tensor_type,
        include=args.include,
        exclude=args.exclude,
        scratch_bytes=args.scratch_bytes,
    )
    elapsed = perf_counter() - started

    counts = ", ".join(f"{name}={count}" for name, count in sorted(result.tensor_type_counts.items()))
    print(f"Wrote {result.output_path}")
    print(f"Architecture: {result.arch}")
    print(f"File type: {result.file_type}")
    print(f"Tensor types: {counts}")
    print(f"Time taken: {elapsed:.2f}s")
    if result.fallback_counts:
        fallbacks = ", ".join(f"{name}={count}" for name, count in sorted(result.fallback_counts.items()))
        print(f"Fallbacks: {fallbacks}")


if __name__ == "__main__":
    main()
