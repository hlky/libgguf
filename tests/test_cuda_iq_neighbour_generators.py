from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import generate_cuda_iq1_neighbours
from scripts import generate_cuda_iq2_neighbours
from scripts import generate_cuda_iq3_neighbours


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    (generate_cuda_iq1_neighbours, "scripts/generate_cuda_iq1_neighbours.py"),
    (generate_cuda_iq2_neighbours, "scripts/generate_cuda_iq2_neighbours.py"),
    (generate_cuda_iq3_neighbours, "scripts/generate_cuda_iq3_neighbours.py"),
)


@pytest.mark.parametrize(("module", "script"), SCRIPTS)
def test_cuda_iq_neighbour_header_check_mode_is_current(module: object, script: str) -> None:
    env = os.environ.copy()
    # Keep src/ off PYTHONPATH so wheel-install test runs do not shadow the
    # installed native extension when the generator runs in a subprocess.
    pythonpath = [str(ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)

    result = subprocess.run(
        [sys.executable, script, "--check"],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout


@pytest.mark.parametrize(("module", "script"), SCRIPTS)
def test_cuda_iq_neighbour_check_mode_does_not_rewrite(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    module: object,
    script: str,
) -> None:
    header = tmp_path / Path(script).name.replace("generate_cuda_", "libgguf_cuda_").replace(".py", ".cuh")
    stale_content = "stale tracked header\n"
    header.write_text(stale_content, encoding="utf-8")

    monkeypatch.setattr(module, "HEADER_PATH", header)
    monkeypatch.setattr(module, "generate_header", lambda: "fresh generated header\n")

    with pytest.raises(SystemExit):
        module.main(["--check"])  # type: ignore[attr-defined]

    assert header.read_text(encoding="utf-8") == stale_content
