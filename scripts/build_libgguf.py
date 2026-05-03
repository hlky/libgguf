from __future__ import annotations

import argparse
import os
import platform
from pathlib import Path
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

from setuptools._distutils.ccompiler import new_compiler
from setuptools._distutils.sysconfig import customize_compiler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.native_sources import NATIVE_SOURCES


def is_x86_build() -> bool:
    machine = platform.machine().lower()
    return machine in {"amd64", "x86_64", "x86", "i386", "i686"}


def build_shared_lib(output: Path, build_dir: Path, jobs: int | None = None) -> Path:
    root = ROOT

    compiler = new_compiler()
    customize_compiler(compiler)

    compile_args = []
    link_args = []
    if compiler.compiler_type == "msvc":
        compile_args.extend(["/O2", "/EHsc", "/std:c++17", "/D_CRT_SECURE_NO_WARNINGS"])
    else:
        compile_args.extend(["-O3", "-std=c++17", "-fPIC", "-pthread"])
        link_args.extend(["-lm", "-pthread"])

    build_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    x86_build = is_x86_build()

    def compile_one(source: str) -> list[str]:
        source_args = list(compile_args)
        source_name = Path(source).name
        is_dequant_backend = source_name.startswith("dequant_")
        is_quant_backend = source_name.startswith("quant_") and (
            source_name.endswith("_sse2.cpp")
            or source_name.endswith("_sse4_1.cpp")
            or source_name.endswith("_avx2.cpp")
        )
        is_common_quant_backend = source_name.startswith("libgguf_common_quant_") and (
            source_name.endswith("_sse2.cpp")
            or source_name.endswith("_sse4_1.cpp")
            or source_name.endswith("_avx2.cpp")
        )
        is_common_storage_backend = source_name.startswith("libgguf_storage_") and (
            source_name.endswith("_sse2.cpp")
            or source_name.endswith("_sse4_1.cpp")
            or source_name.endswith("_avx2.cpp")
        )
        if x86_build and (
            (is_dequant_backend and source_name.endswith("_avx2.cpp"))
            or (is_quant_backend and source_name.endswith("_avx2.cpp"))
            or (is_common_quant_backend and source_name.endswith("_avx2.cpp"))
            or (is_common_storage_backend and source_name.endswith("_avx2.cpp"))
        ):
            source_args.append("/arch:AVX2" if compiler.compiler_type == "msvc" else "-mavx2")
        elif (
            x86_build
            and (
                (is_dequant_backend and (source_name.endswith("_sse2.cpp") or source_name.endswith("_sse4_1.cpp")))
                or is_quant_backend
                or is_common_quant_backend
                or is_common_storage_backend
            )
            and compiler.compiler_type != "msvc"
        ):
            source_args.append("-msse4.1" if source_name.endswith("_sse4_1.cpp") else "-msse2")
        return compiler.compile(
            sources=[str(root / source)],
            output_dir=str(build_dir),
            include_dirs=[str(root / "include"), str(root / "csrc"), str(root / "csrc" / "common")],
            macros=[("NDEBUG", "1")],
            extra_postargs=source_args,
        )

    build_jobs = jobs or int(os.environ.get("LIBGGUF_BUILD_JOBS") or os.environ.get("MAX_JOBS") or os.cpu_count() or 1)
    if build_jobs <= 1:
        objects = []
        for source in NATIVE_SOURCES:
            objects.extend(compile_one(source))
    else:
        objects = []
        with ThreadPoolExecutor(max_workers=build_jobs) as executor:
            for compiled in executor.map(compile_one, NATIVE_SOURCES):
                objects.extend(compiled)

    compiler.link_shared_object(
        objects=objects,
        output_filename=str(output),
        extra_postargs=link_args,
    )
    return output


def default_output_path() -> Path:
    root = ROOT
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
    parser.add_argument("-j", "--jobs", type=int, default=None, help="Number of compile jobs")
    parser.add_argument("--clean", action="store_true", help="Delete the build directory before compiling")
    args = parser.parse_args()

    root = ROOT
    build_dir = args.build_dir if args.build_dir.is_absolute() else root / args.build_dir
    output = args.output if args.output.is_absolute() else root / args.output

    if args.clean and build_dir.exists():
        shutil.rmtree(build_dir)

    built = build_shared_lib(output=output, build_dir=build_dir, jobs=args.jobs)
    print(built)


if __name__ == "__main__":
    main()
