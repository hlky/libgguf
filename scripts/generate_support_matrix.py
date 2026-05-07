from __future__ import annotations

import argparse
from pathlib import Path

from libgguf import GGMLQuantizationType
from libgguf.libgguf_numpy.libgguf_numpy import _type_traits as numpy_type_traits
from libgguf.libgguf_torch.libgguf_torch import dequantize_functions, quantize_functions


ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = ROOT / "docs" / "support-matrix.md"
CUDA_CSRC = ROOT / "src" / "libgguf" / "libgguf_cuda" / "csrc"

STORAGE_QTYPES = (
    GGMLQuantizationType.F32,
    GGMLQuantizationType.F16,
    GGMLQuantizationType.BF16,
)

QUANTIZED_QTYPES = (
    GGMLQuantizationType.Q1_0,
    GGMLQuantizationType.Q4_0,
    GGMLQuantizationType.Q4_1,
    GGMLQuantizationType.Q5_0,
    GGMLQuantizationType.Q5_1,
    GGMLQuantizationType.Q8_0,
    GGMLQuantizationType.Q2_K,
    GGMLQuantizationType.Q3_K,
    GGMLQuantizationType.Q4_K,
    GGMLQuantizationType.Q5_K,
    GGMLQuantizationType.Q6_K,
    GGMLQuantizationType.IQ1_S,
    GGMLQuantizationType.IQ1_M,
    GGMLQuantizationType.IQ2_XXS,
    GGMLQuantizationType.IQ2_XS,
    GGMLQuantizationType.IQ2_S,
    GGMLQuantizationType.IQ3_XXS,
    GGMLQuantizationType.IQ3_S,
    GGMLQuantizationType.IQ4_NL,
    GGMLQuantizationType.IQ4_XS,
    GGMLQuantizationType.TQ1_0,
    GGMLQuantizationType.TQ2_0,
    GGMLQuantizationType.MXFP4,
    GGMLQuantizationType.NVFP4,
)

MATRIX_QTYPES = (*STORAGE_QTYPES, *QUANTIZED_QTYPES)


def _cuda_stem(qtype: GGMLQuantizationType) -> str:
    return qtype.name.lower()


def _cuda_quant_status(qtype: GGMLQuantizationType) -> str:
    if (CUDA_CSRC / f"cuda_quantize_{_cuda_stem(qtype)}.cu").is_file():
        return "experimental"
    return "unknown"


def _cuda_dequant_status(qtype: GGMLQuantizationType) -> str:
    if (CUDA_CSRC / f"cuda_dequantize_{_cuda_stem(qtype)}.cu").is_file():
        return "experimental"
    return "unknown"


def _numpy_quant_status(qtype: GGMLQuantizationType) -> str:
    if qtype in STORAGE_QTYPES:
        return "yes (storage)"
    if qtype in numpy_type_traits:
        return "yes"
    return "unknown"


def _numpy_dequant_status(qtype: GGMLQuantizationType) -> str:
    if qtype in STORAGE_QTYPES:
        return "yes (storage)"
    if qtype in numpy_type_traits:
        return "yes"
    return "unknown"


def _torch_quant_status(qtype: GGMLQuantizationType) -> str:
    if qtype in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
        return "yes (storage)"
    if qtype in quantize_functions:
        return "yes" if qtype not in STORAGE_QTYPES else "yes (storage)"
    return "unknown"


def _torch_dequant_status(qtype: GGMLQuantizationType) -> str:
    if qtype in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
        return "yes (storage)"
    if qtype in dequantize_functions:
        return "yes" if qtype not in STORAGE_QTYPES else "yes (storage)"
    return "unknown"


def generate_markdown() -> str:
    lines = [
        "# Support Matrix",
        "",
        "This matrix is generated from visible source/backend maps by `scripts/generate_support_matrix.py`. It distinguishes row/backends from converter executables. `yes` means implemented in visible code and covered by the current style of tests or source layout. `experimental` means implemented but optional, young, or explicitly subject to change. `planned` means intended but not currently public. `unknown` means not claimed here.",
        "",
        "| qtype | native CPU quant | native CPU dequant | NumPy quant | NumPy dequant | Torch quant | Torch dequant | CUDA quant | CUDA dequant |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for qtype in MATRIX_QTYPES:
        native_quant = "yes (storage)" if qtype in STORAGE_QTYPES else "yes"
        native_dequant = "unknown" if qtype in STORAGE_QTYPES else "yes"
        lines.append(
            f"| `{qtype.name}` | {native_quant} | {native_dequant} | "
            f"{_numpy_quant_status(qtype)} | {_numpy_dequant_status(qtype)} | "
            f"{_torch_quant_status(qtype)} | {_torch_dequant_status(qtype)} | "
            f"{_cuda_quant_status(qtype)} | {_cuda_dequant_status(qtype)} |"
        )

    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- The native row APIs support the broad qtype list above.",
            "- The native executable `libgguf_quantize_gguf` is currently Q/K-focused and rejects IQ/TQ/MXFP4/NVFP4 output families.",
            "- `Q8_1`, `Q8_K`, integer storage types, and `F64` are present in enum metadata but are not claimed as supported row quantization targets here.",
            "- CUDA is optional and depends on a successful Torch/CUDA extension build.",
            "- Regenerate this file with `python scripts/generate_support_matrix.py --write`.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate or check docs/support-matrix.md.")
    parser.add_argument("--write", action="store_true", help="Rewrite docs/support-matrix.md")
    parser.add_argument("--check", action="store_true", help="Fail if docs/support-matrix.md is out of date")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    content = generate_markdown()
    if args.write:
        DOC_PATH.write_text(content, encoding="utf-8")
    if args.check and DOC_PATH.read_text(encoding="utf-8") != content:
        raise SystemExit("docs/support-matrix.md is out of date; run scripts/generate_support_matrix.py --write")
    if not args.write and not args.check:
        print(content, end="")


if __name__ == "__main__":
    main()
