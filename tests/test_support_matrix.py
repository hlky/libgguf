from __future__ import annotations

from pathlib import Path

from scripts.generate_support_matrix import DOC_PATH, generate_markdown


def test_support_matrix_is_generated() -> None:
    assert Path(DOC_PATH).read_text(encoding="utf-8") == generate_markdown()
