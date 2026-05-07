from __future__ import annotations

import subprocess
import sys
import textwrap


BLOCKED_OPTIONAL_DEPS = {"gguf", "safetensors", "torch", "tqdm"}


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        check=False,
        capture_output=True,
        text=True,
    )


def _assert_success(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, result.stderr or result.stdout


def test_top_level_import_does_not_eagerly_import_conversion_module() -> None:
    result = _run_python(
        f"""
        import sys

        blocked = {BLOCKED_OPTIONAL_DEPS!r}

        class Blocker:
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".", 1)[0] in blocked:
                    raise ImportError(fullname)
                return None

        sys.meta_path.insert(0, Blocker())
        import libgguf
        assert "libgguf.quantize" not in sys.modules, "libgguf.quantize imported"
        assert "libgguf.inspect" not in sys.modules, "libgguf.inspect imported"
        for name in blocked:
            assert name not in sys.modules, f"{{name}} imported"
        """
    )

    _assert_success(result)


def test_star_import_is_safe_without_conversion_extras() -> None:
    result = _run_python(
        f"""
        import sys

        blocked = {BLOCKED_OPTIONAL_DEPS!r}

        class Blocker:
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".", 1)[0] in blocked:
                    raise ImportError(fullname)
                return None

        sys.meta_path.insert(0, Blocker())
        namespace = {{}}
        exec("from libgguf import *", namespace)

        assert namespace["QuantResult"].__name__ == "QuantResult"
        assert callable(namespace["convert_to_gguf"])
        assert callable(namespace["convert_safetensors_to_gguf_native"])
        for name in blocked:
            assert name not in sys.modules, f"{{name}} imported"
        """
    )

    _assert_success(result)


def test_top_level_conversion_helpers_are_lazy_exports() -> None:
    import libgguf

    assert callable(libgguf.convert_to_gguf)
    assert callable(libgguf.convert_safetensors_to_gguf_native)
