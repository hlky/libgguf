from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys

from setuptools._distutils.ccompiler import new_compiler
from setuptools._distutils.sysconfig import customize_compiler


def build_shared_lib(output: Path, build_dir: Path) -> Path:
    root = Path(__file__).resolve().parents[1]

    compiler = new_compiler()
    customize_compiler(compiler)

    compile_args = []
    link_args = []
    if compiler.compiler_type == "msvc":
        compile_args.extend(["/O2", "/EHsc", "/std:c++17", "/D_CRT_SECURE_NO_WARNINGS"])
        if os.environ.get("LIBGGUF_AVX2") == "1":
            compile_args.append("/arch:AVX2")
    else:
        compile_args.extend(["-O3", "-std=c++17", "-fPIC", "-pthread"])
        link_args.extend(["-lm", "-pthread"])
        if os.environ.get("LIBGGUF_AVX2") == "1":
            compile_args.append("-mavx2")

    build_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    objects = compiler.compile(
        sources=[str(root / "libgguf.cpp")],
        output_dir=str(build_dir),
        include_dirs=[str(root)],
        macros=[("NDEBUG", "1")],
        extra_postargs=compile_args,
    )
    compiler.link_shared_object(
        objects=objects,
        output_filename=str(output),
        extra_postargs=link_args,
    )
    return output


def default_output_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    if sys.platform == "win32":
        lib_name = "libgguf.dll"
    elif sys.platform == "darwin":
        lib_name = "libgguf.dylib"
    else:
        lib_name = "libgguf.so"
    return root / lib_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the libgguf C++ reference shared library")
    parser.add_argument("--output", type=Path, default=default_output_path(), help="Shared library output path")
    parser.add_argument("--build-dir", type=Path, default=Path("build/libgguf"), help="Object file build directory")
    parser.add_argument("--clean", action="store_true", help="Delete the build directory before compiling")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    build_dir = args.build_dir if args.build_dir.is_absolute() else root / args.build_dir
    output = args.output if args.output.is_absolute() else root / args.output

    if args.clean and build_dir.exists():
        shutil.rmtree(build_dir)

    built = build_shared_lib(output=output, build_dir=build_dir)
    print(built)


if __name__ == "__main__":
    main()
