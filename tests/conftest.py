from __future__ import annotations

from pathlib import Path
import importlib.machinery
import importlib.util
import sys


ROOT = Path(__file__).resolve().parents[1]

for path in (ROOT / "src", ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

LOCAL_EXTENSION_DIR = ROOT / "src" / "libgguf"
for suffix in importlib.machinery.EXTENSION_SUFFIXES:
    local_extension = LOCAL_EXTENSION_DIR / f"_libgguf{suffix}"
    if not local_extension.exists():
        continue
    spec = importlib.util.spec_from_file_location("libgguf._libgguf", local_extension)
    if spec is not None and spec.loader is not None:
        module = importlib.util.module_from_spec(spec)
        sys.modules["libgguf._libgguf"] = module
        spec.loader.exec_module(module)
    break
