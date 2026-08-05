from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize("module", ("libgguf.inspect", "libgguf.compare"))
def test_cli_module_help(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
