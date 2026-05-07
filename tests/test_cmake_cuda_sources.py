from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CUDA_CSRC = ROOT / "src" / "libgguf" / "libgguf_cuda" / "csrc"


def _listed_cuda_build_sources() -> set[Path]:
    cmake_text = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    pattern = re.compile(r"\$\{CMAKE_CURRENT_SOURCE_DIR\}/(src/libgguf/libgguf_cuda/csrc/[^\"\s]+\.(?:cu|cpp))")
    return {ROOT / match for match in pattern.findall(cmake_text)}


def test_cuda_build_sources_are_listed_in_cmake() -> None:
    actual_sources = set(CUDA_CSRC.glob("*.cu")) | set(CUDA_CSRC.glob("*.cpp"))
    listed_sources = _listed_cuda_build_sources()

    assert actual_sources - listed_sources == set()
    assert listed_sources - actual_sources == set()
