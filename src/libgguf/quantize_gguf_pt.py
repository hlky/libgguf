from __future__ import annotations

import argparse
from pathlib import Path

from .quantize_pt import convert_to_gguf, parse_qtype, parse_tensor_qtype


def _parse_tensor_type(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected PATTERN=QTYPE")
    pattern, qtype = value.split("=", 1)
    if not pattern or not qtype:
        raise argparse.ArgumentTypeError("expected non-empty PATTERN=QTYPE")
    parse_tensor_qtype(qtype)
    return pattern, qtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Torch-backed quantization of safetensors/ckpt diffusion models to GGUF")
    parser.add_argument("--src", required=True, help="Source safetensors/ckpt model")
    parser.add_argument("--qtype", required=True, help="Output file type, e.g. Q4_K_S, Q4_K_M, Q4_K, Q8_0")
    parser.add_argument("--dst", help="Output GGUF path")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output file")
    parser.add_argument("--policy", choices=("comfy", "dynamic", "uniform"), default="comfy", help="Tensor selection policy")
    parser.add_argument("--imatrix", help="llama.cpp imatrix file")
    parser.add_argument("--tensor-type", action="append", type=_parse_tensor_type, default=[], metavar="PATTERN=QTYPE")
    parser.add_argument("--include", action="append", default=[], metavar="PATTERN", help="Force matching tensors into quantization when possible")
    parser.add_argument("--exclude", action="append", default=[], metavar="PATTERN", help="Keep matching tensors unquantized")
    args = parser.parse_args()

    if not Path(args.src).is_file():
        parser.error(f"invalid source file: {args.src}")
    parse_qtype(args.qtype)
    return args


def main() -> None:
    args = parse_args()
    result = convert_to_gguf(
        args.src,
        args.dst,
        args.qtype,
        policy=args.policy,
        overwrite=args.overwrite,
        imatrix=args.imatrix,
        tensor_overrides=args.tensor_type,
        include=args.include,
        exclude=args.exclude,
    )

    counts = ", ".join(f"{name}={count}" for name, count in sorted(result.tensor_type_counts.items()))
    print(f"Wrote {result.output_path}")
    print(f"Architecture: {result.arch}")
    print(f"File type: {result.file_type}")
    print(f"Tensor types: {counts}")
    if result.fallback_counts:
        fallbacks = ", ".join(f"{name}={count}" for name, count in sorted(result.fallback_counts.items()))
        print(f"Fallbacks: {fallbacks}")


if __name__ == "__main__":
    main()
