from __future__ import annotations

import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    tomllib = None


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_SCRIPT_NAMES = {
    "gguf-compare",
    "gguf-inspect",
    "gguf-validate",
}

RETIRED_QUANTIZE_MODULES = (
    "quantize_gguf",
    "quantize_gguf_native",
    "quantize_gguf_pt",
    "quantize_gguf_torch",
    "quantize_pt",
    "quantize_torch_pt",
)

RETIRED_DOC_PATTERNS = {
    "quantize_gguf": re.compile(r"(?<!libgguf_)\bquantize_gguf\b(?!\.cpp)"),
    **{
        name: re.compile(rf"\b{re.escape(name)}\b")
        for name in RETIRED_QUANTIZE_MODULES
        if name != "quantize_gguf"
    },
}


def _project_scripts() -> dict[str, str]:
    if tomllib is None:
        pytest.skip("stdlib tomllib is not available")

    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]["scripts"]


def _run_python(code: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code), *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _assert_success(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, result.stderr or result.stdout


def test_project_scripts_include_expected_entry_points() -> None:
    scripts = _project_scripts()

    assert scripts.keys() == EXPECTED_SCRIPT_NAMES
    assert scripts["gguf-compare"] == "libgguf.compare:main"
    assert scripts["gguf-inspect"] == "libgguf.inspect:main"
    assert scripts["gguf-validate"] == "libgguf.inspect:validate_main"


def test_retired_quantize_modules_are_not_in_source_package() -> None:
    package_dir = ROOT / "src" / "libgguf"
    for module_name in RETIRED_QUANTIZE_MODULES:
        assert not (package_dir / f"{module_name}.py").exists()


def test_public_docs_do_not_advertise_retired_python_quantize_clis() -> None:
    docs = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    offenders: list[str] = []
    for path in docs:
        text = path.read_text(encoding="utf-8")
        for name, pattern in RETIRED_DOC_PATTERNS.items():
            if pattern.search(text):
                offenders.append(f"{path.relative_to(ROOT)} mentions {name}")

    assert offenders == []


def test_inspect_entry_point_targets_resolve_without_conversion_extras() -> None:
    _project_scripts()

    result = _run_python(
        """
        import importlib
        from pathlib import Path
        import sys
        import tomllib

        blocked = {"gguf", "safetensors", "torch", "tqdm"}

        class Blocker:
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".", 1)[0] in blocked:
                    raise ImportError(f"blocked optional dependency: {fullname}")
                return None

        sys.meta_path.insert(0, Blocker())
        scripts = tomllib.loads(Path("pyproject.toml").read_text())["project"]["scripts"]

        for script_name in ("gguf-compare", "gguf-inspect", "gguf-validate"):
            module_name, separator, qualname = scripts[script_name].partition(":")
            assert separator, f"{script_name} target has no callable separator"
            module = importlib.import_module(module_name)
            target = module
            for part in qualname.split("."):
                target = getattr(target, part)
            assert callable(target), f"{script_name} target is not callable"
        """
    )

    _assert_success(result)


@pytest.mark.parametrize(
    ("mode", "needle"),
    [
        ("compare-main", "Compare GGUF tensor descriptors and optional exact content"),
        ("inspect-module", "Inspect GGUF metadata and tensor descriptors"),
        ("validate-main", "Validate GGUF structure without reading tensor payload bytes"),
    ],
)
def test_inspect_help_paths_work_without_conversion_extras(mode: str, needle: str) -> None:
    _project_scripts()

    result = _run_python(
        """
        import runpy
        import sys

        blocked = {"gguf", "safetensors", "torch", "tqdm"}

        class Blocker:
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".", 1)[0] in blocked:
                    raise ImportError(f"blocked optional dependency: {fullname}")
                return None

        sys.meta_path.insert(0, Blocker())

        if sys.argv[1] == "inspect-module":
            sys.argv = ["python -m libgguf.inspect", "--help"]
            runpy.run_module("libgguf.inspect", run_name="__main__", alter_sys=True)
        elif sys.argv[1] == "compare-main":
            from libgguf.compare import main

            main(["--help"])
        elif sys.argv[1] == "validate-main":
            from libgguf.inspect import validate_main

            validate_main(["--help"])
        else:
            raise AssertionError(f"unknown mode: {sys.argv[1]}")
        """,
        mode,
    )

    _assert_success(result)
    assert needle in result.stdout
