from __future__ import annotations

from pathlib import Path
import importlib.machinery
import importlib.util
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PACKAGE_DIR = SRC / "libgguf"


def _candidate_extension_dirs() -> list[Path]:
    dirs = [PACKAGE_DIR]
    repo_paths = {ROOT.resolve(), SRC.resolve()}

    for entry in sys.path:
        if not entry:
            continue

        path = Path(entry).resolve()
        if path in repo_paths:
            continue
        dirs.append(path / "libgguf")

    return dirs


def _preload_native_extension() -> None:
    if "libgguf._libgguf" in sys.modules:
        return

    for directory in _candidate_extension_dirs():
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            extension = directory / f"_libgguf{suffix}"
            if not extension.exists():
                continue

            spec = importlib.util.spec_from_file_location("libgguf._libgguf", extension)
            if spec is None or spec.loader is None:
                continue

            module = importlib.util.module_from_spec(spec)
            sys.modules["libgguf._libgguf"] = module
            spec.loader.exec_module(module)
            return


_preload_native_extension()

for path in (SRC, ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)
