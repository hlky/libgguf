from __future__ import annotations

import subprocess
import sys


def test_top_level_import_does_not_eagerly_import_conversion_module() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "class Blocker:\n"
                "    def find_spec(self, fullname, path=None, target=None):\n"
                "        if fullname.split('.')[0] in {'gguf', 'safetensors', 'tqdm'}:\n"
                "            raise ImportError(fullname)\n"
                "        return None\n"
                "sys.meta_path.insert(0, Blocker()); "
                "import libgguf; "
                "assert 'libgguf.quantize' not in sys.modules, 'libgguf.quantize imported'"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_top_level_conversion_helpers_are_lazy_exports() -> None:
    import libgguf

    assert callable(libgguf.convert_to_gguf)
    assert callable(libgguf.convert_safetensors_to_gguf_native)
