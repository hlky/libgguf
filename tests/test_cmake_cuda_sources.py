from __future__ import annotations

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CUDA_CSRC = ROOT / "src" / "libgguf" / "libgguf_cuda" / "csrc"
CUDA_QUANTIZE_TEST = ROOT / "tests" / "backends" / "libgguf_cuda" / "test_cuda_quantize.py"
CUDA_DEQUANTIZE_TEST = ROOT / "tests" / "backends" / "libgguf_cuda" / "test_cuda_dequantize.py"
CUDA_OPS = ROOT / "src" / "libgguf" / "libgguf_cuda" / "ops.py"
GGML_METADATA = ROOT / "src" / "libgguf" / "_metadata.py"
NATIVE_CONVERTER = ROOT / "csrc" / "quantize_gguf.cpp"
CLI_DOC = ROOT / "docs" / "cli.md"
CUDA_DOC = ROOT / "docs" / "cuda.md"

NATIVE_CONVERTER_CUDA_QTYPES = {"Q4_0", "Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"}


def _listed_cuda_build_sources() -> set[Path]:
    cmake_text = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    pattern = re.compile(r"\$\{CMAKE_CURRENT_SOURCE_DIR\}/(src/libgguf/libgguf_cuda/csrc/[^\"\s]+\.(?:cu|cpp))")
    return {ROOT / match for match in pattern.findall(cmake_text)}


def _cuda_qtype_sources(prefix: str) -> set[str]:
    qtypes = set()
    for path in CUDA_CSRC.glob(f"{prefix}_*.cu"):
        stem = path.stem.removeprefix(f"{prefix}_")
        if stem == "kernels":
            continue
        qtypes.add(stem.upper())
    return qtypes


def _qtype_attrs(node: ast.AST) -> set[str]:
    qtypes = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute) and child.attr.isupper():
            qtypes.add(child.attr)
    return qtypes


def _python_assignment_qtypes(path: Path, assignment_name: str) -> set[str]:
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == assignment_name for target in node.targets):
            return _qtype_attrs(node.value)
    raise AssertionError(f"{assignment_name} was not found in {path}")


def _fake_quantize_qtypes() -> set[str]:
    module = ast.parse(CUDA_OPS.read_text(encoding="utf-8"), filename=str(CUDA_OPS))
    for node in ast.walk(module):
        if not isinstance(node, ast.Compare):
            continue
        if not (
            isinstance(node.left, ast.Name)
            and node.left.id == "qtype"
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.NotIn)
        ):
            continue
        return _qtype_attrs(node.comparators[0])
    raise AssertionError(f"fake quantize qtype allowlist was not found in {CUDA_OPS}")


def _ggml_type_value_names() -> dict[int, str]:
    module = ast.parse(GGML_METADATA.read_text(encoding="utf-8"), filename=str(GGML_METADATA))
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "GGMLQuantizationType":
            names = {}
            for item in node.body:
                if isinstance(item, ast.Assign) and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name):
                    value = ast.literal_eval(item.value)
                    names[value] = item.targets[0].id
            return names
    raise AssertionError(f"GGMLQuantizationType was not found in {GGML_METADATA}")


def _dequant_dispatch_qtypes() -> set[str]:
    value_names = _ggml_type_value_names()
    text = (CUDA_CSRC / "cuda_dequantize_kernels.cu").read_text(encoding="utf-8")
    return {value_names[int(value)] for value in re.findall(r"\bcase\s+(\d+)\s*:", text)}


def _native_converter_cuda_qtypes() -> set[str]:
    text = NATIVE_CONVERTER.read_text(encoding="utf-8")
    match = re.search(
        r"bool\s+converter_cuda_supported_qtype\(ggml_type qtype\)\s*\{\s*switch\s*\(qtype\)\s*\{(?P<body>.*?)\n\s*\}\n\s*\}",
        text,
        flags=re.DOTALL,
    )
    if match is None:
        raise AssertionError(f"converter_cuda_supported_qtype was not found in {NATIVE_CONVERTER}")
    return set(re.findall(r"\bcase\s+GGML_TYPE_([A-Z0-9_]+)\s*:", match.group("body")))


def _native_converter_doc_qtypes(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    for sentence in re.findall(r"[^.\n]*\bconverter\b[^.\n]*\.", text):
        if "CUDA" not in sentence:
            continue
        qtypes = set(re.findall(r"`([A-Z][A-Z0-9_]+)`", sentence))
        if qtypes:
            return qtypes
    raise AssertionError(f"native CUDA converter qtype set was not found in {path}")


def test_cuda_build_sources_are_listed_in_cmake() -> None:
    actual_sources = set(CUDA_CSRC.glob("*.cu")) | set(CUDA_CSRC.glob("*.cpp"))
    listed_sources = _listed_cuda_build_sources()

    assert actual_sources - listed_sources == set()
    assert listed_sources - actual_sources == set()


def test_cuda_quantize_source_qtypes_match_support_contracts() -> None:
    source_qtypes = _cuda_qtype_sources("cuda_quantize")

    assert source_qtypes == _python_assignment_qtypes(CUDA_QUANTIZE_TEST, "CUDA_QUANT_QTYPES")
    assert source_qtypes == _fake_quantize_qtypes()


def test_cuda_dequantize_source_qtypes_match_support_contracts() -> None:
    source_qtypes = _cuda_qtype_sources("cuda_dequantize")
    generic_test_qtypes = _python_assignment_qtypes(CUDA_DEQUANTIZE_TEST, "CUDA_DEQUANT_QTYPES")

    assert source_qtypes - {"BF16"} == generic_test_qtypes
    assert "BF16" in source_qtypes
    assert source_qtypes == _dequant_dispatch_qtypes()


def test_native_converter_cuda_qtypes_match_docs() -> None:
    source_qtypes = _native_converter_cuda_qtypes()

    assert source_qtypes == NATIVE_CONVERTER_CUDA_QTYPES
    assert source_qtypes == _native_converter_doc_qtypes(CLI_DOC)
    assert source_qtypes == _native_converter_doc_qtypes(CUDA_DOC)
