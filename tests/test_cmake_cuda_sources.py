from __future__ import annotations

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CUDA_CSRC = ROOT / "src" / "libgguf" / "libgguf_cuda" / "csrc"
CUDA_QUANTIZE_TEST = ROOT / "tests" / "backends" / "libgguf_cuda" / "test_cuda_quantize.py"
CUDA_DEQUANTIZE_TEST = ROOT / "tests" / "backends" / "libgguf_cuda" / "test_cuda_dequantize.py"
CUDA_FAKE_OPS_TEST = ROOT / "tests" / "backends" / "libgguf_cuda" / "test_cuda_fake_ops.py"
CUDA_OPS = ROOT / "src" / "libgguf" / "libgguf_cuda" / "ops.py"
GGML_METADATA = ROOT / "src" / "libgguf" / "_metadata.py"
NATIVE_API = ROOT / "csrc" / "libgguf.cpp"
NATIVE_CONVERTER = ROOT / "csrc" / "quantize_gguf.cpp"
CLI_DOC = ROOT / "docs" / "cli.md"
CUDA_DOC = ROOT / "docs" / "cuda.md"
PYTHON_API_DOC = ROOT / "docs" / "python-api.md"

NATIVE_CONVERTER_CUDA_QTYPES = {"Q4_0", "Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"}
IMATRIX_QTYPES = {"IQ2_XXS", "IQ2_XS", "IQ1_S"}


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


def _is_quantize_requires_imatrix_call(node: ast.AST, loop_variable: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "quantize_requires_imatrix"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "libgguf"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == loop_variable
    )


def _python_cuda_imatrix_qtypes(path: Path) -> set[str]:
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "CUDA_IMATRIX_QTYPES" for target in node.targets):
            continue
        if _qtype_attrs(node.value):
            return _qtype_attrs(node.value)
        if not (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "tuple"
            and len(node.value.args) == 1
            and isinstance(node.value.args[0], ast.GeneratorExp)
        ):
            raise AssertionError(f"CUDA_IMATRIX_QTYPES has an unsupported shape in {path}")
        generator = node.value.args[0]
        if not (
            isinstance(generator.elt, ast.Name)
            and len(generator.generators) == 1
            and isinstance(generator.generators[0].target, ast.Name)
            and isinstance(generator.generators[0].iter, ast.Name)
            and generator.generators[0].iter.id == "CUDA_QUANT_QTYPES"
            and len(generator.generators[0].ifs) == 1
        ):
            raise AssertionError(f"CUDA_IMATRIX_QTYPES has an unsupported generator in {path}")
        loop_variable = generator.generators[0].target.id
        if generator.elt.id != loop_variable or not _is_quantize_requires_imatrix_call(
            generator.generators[0].ifs[0], loop_variable
        ):
            raise AssertionError(f"CUDA_IMATRIX_QTYPES does not use libgguf.quantize_requires_imatrix in {path}")
        return _python_assignment_qtypes(path, "CUDA_QUANT_QTYPES") & IMATRIX_QTYPES
    raise AssertionError(f"CUDA_IMATRIX_QTYPES was not found in {path}")


def _fake_quantize_qtypes() -> set[str]:
    module = ast.parse(CUDA_OPS.read_text(encoding="utf-8"), filename=str(CUDA_OPS))
    fake_quantize = next(
        (
            node
            for node in ast.walk(module)
            if isinstance(node, ast.FunctionDef) and node.name == "_quantize_fake"
        ),
        None,
    )
    if fake_quantize is None:
        raise AssertionError(f"fake quantize implementation was not found in {CUDA_OPS}")
    for node in ast.walk(fake_quantize):
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


def _c_function_cases(path: Path, function_name: str) -> set[str]:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        rf"\b{re.escape(function_name)}\s*\([^)]*\)\s*\{{(?P<body>.*?)\n\}}",
        text,
        flags=re.DOTALL,
    )
    if match is None:
        raise AssertionError(f"{function_name} was not found in {path}")
    return set(re.findall(r"\bcase\s+GGML_TYPE_([A-Z0-9_]+)\s*:", match.group("body")))


def _c_referenced_qtypes(path: Path, function_name: str) -> set[str]:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        rf"\b{re.escape(function_name)}\s*\([^)]*\)\s*\{{(?P<body>.*?)\n\}}",
        text,
        flags=re.DOTALL,
    )
    if match is None:
        raise AssertionError(f"{function_name} was not found in {path}")
    return set(re.findall(r"\bGGML_TYPE_([A-Z0-9_]+)\b", match.group("body")))


def _cuda_launch_declaration_qtypes(path: Path, prefix: str) -> set[str]:
    text = path.read_text(encoding="utf-8")
    return {
        name.upper()
        for name in re.findall(rf"\bgguf_cuda_{prefix}_launch_([a-z0-9_]+)\s*\(", text)
    }


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


def _cuda_extension_doc_qtypes(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"CUDA quantize/dequantize kernels are present for:\n(?P<body>.*?)\nThere is also a BF16",
        text,
        flags=re.DOTALL,
    )
    if match is None:
        raise AssertionError(f"CUDA extension qtype coverage list was not found in {path}")
    return set(re.findall(r"`([A-Z][A-Z0-9_]+)`", match.group("body")))


def _imatrix_doc_qtypes(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"The qtypes that require imatrix weights are: (?P<qtypes>[^.]+)\.", text)
    if match is None:
        raise AssertionError(f"imatrix qtype contract was not found in {path}")
    return set(re.findall(r"`([A-Z][A-Z0-9_]+)`", match.group("qtypes")))


def test_cuda_build_sources_are_listed_in_cmake() -> None:
    actual_sources = set(CUDA_CSRC.glob("*.cu")) | set(CUDA_CSRC.glob("*.cpp"))
    listed_sources = _listed_cuda_build_sources()

    assert actual_sources - listed_sources == set()
    assert listed_sources - actual_sources == set()


def test_cuda_quantize_source_qtypes_match_support_contracts() -> None:
    source_qtypes = _cuda_qtype_sources("cuda_quantize")

    assert source_qtypes == _python_assignment_qtypes(CUDA_QUANTIZE_TEST, "CUDA_QUANT_QTYPES")
    assert source_qtypes == _python_assignment_qtypes(CUDA_FAKE_OPS_TEST, "CUDA_QUANT_QTYPES")
    assert source_qtypes == _fake_quantize_qtypes()
    assert source_qtypes == _c_function_cases(
        CUDA_CSRC / "cuda_quantize_kernels.cu", "gguf_cuda_quantize_row_size_for_type"
    )
    assert source_qtypes == _c_function_cases(
        CUDA_CSRC / "cuda_quantize_kernels.cu", "gguf_cuda_quantize_block_size_for_type"
    )
    assert source_qtypes == _c_function_cases(
        CUDA_CSRC / "cuda_quantize_kernels.cu", "gguf_cuda_quantize_row"
    )
    assert source_qtypes == _cuda_launch_declaration_qtypes(CUDA_CSRC / "cuda_quantize_kernels.h", "quantize")
    assert source_qtypes == _cuda_extension_doc_qtypes(CUDA_DOC)


def test_imatrix_qtypes_match_native_cuda_tests_and_docs() -> None:
    assert _c_referenced_qtypes(NATIVE_API, "libgguf_quantize_requires_imatrix") == IMATRIX_QTYPES
    assert (
        _c_function_cases(CUDA_CSRC / "cuda_quantize_kernels.cu", "gguf_cuda_quantize_type_needs_imatrix")
        == IMATRIX_QTYPES
    )
    assert _python_cuda_imatrix_qtypes(CUDA_QUANTIZE_TEST) == IMATRIX_QTYPES
    assert _python_cuda_imatrix_qtypes(CUDA_FAKE_OPS_TEST) == IMATRIX_QTYPES
    assert _imatrix_doc_qtypes(PYTHON_API_DOC) == IMATRIX_QTYPES


def test_cuda_dequantize_source_qtypes_match_support_contracts() -> None:
    source_qtypes = _cuda_qtype_sources("cuda_dequantize")
    generic_test_qtypes = _python_assignment_qtypes(CUDA_DEQUANTIZE_TEST, "CUDA_DEQUANT_QTYPES")

    assert source_qtypes - {"BF16"} == generic_test_qtypes
    assert source_qtypes == _python_assignment_qtypes(CUDA_FAKE_OPS_TEST, "CUDA_DEQUANT_QTYPES")
    assert "BF16" in source_qtypes
    assert source_qtypes == _dequant_dispatch_qtypes()
    assert source_qtypes == _cuda_launch_declaration_qtypes(CUDA_CSRC / "cuda_dequantize_kernels.h", "dequantize")
    assert source_qtypes - {"BF16"} == _cuda_extension_doc_qtypes(CUDA_DOC)


def test_native_converter_cuda_qtypes_match_docs() -> None:
    source_qtypes = _native_converter_cuda_qtypes()

    assert source_qtypes == NATIVE_CONVERTER_CUDA_QTYPES
    assert source_qtypes == _native_converter_doc_qtypes(CLI_DOC)
    assert source_qtypes == _native_converter_doc_qtypes(CUDA_DOC)
