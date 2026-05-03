from __future__ import annotations

import importlib.util
import os
import platform
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

ROOT = Path(__file__).resolve().parent
NATIVE_SOURCES_SPEC = importlib.util.spec_from_file_location(
    "libgguf_native_sources",
    ROOT / "scripts" / "native_sources.py",
)
if NATIVE_SOURCES_SPEC is None or NATIVE_SOURCES_SPEC.loader is None:
    raise RuntimeError("failed to load native source list")
NATIVE_SOURCES_MODULE = importlib.util.module_from_spec(NATIVE_SOURCES_SPEC)
NATIVE_SOURCES_SPEC.loader.exec_module(NATIVE_SOURCES_MODULE)
NATIVE_SOURCES = NATIVE_SOURCES_MODULE.NATIVE_SOURCES


def _is_x86_build() -> bool:
    machine = platform.machine().lower()
    return machine in {"amd64", "x86_64", "x86", "i386", "i686"}


class BuildExt(build_ext):
    def build_extensions(self) -> None:
        original_compile = self.compiler.compile
        is_x86_build = _is_x86_build()

        def source_flags(source: str) -> list[str]:
            name = Path(source).name
            is_dequant_backend = name.startswith("dequant_")
            is_quant_backend = name.startswith("quant_") and (
                name.endswith("_sse2.cpp") or name.endswith("_sse4_1.cpp") or name.endswith("_avx2.cpp")
            )
            is_common_quant_backend = name.startswith("libgguf_common_quant_") and (
                name.endswith("_sse2.cpp") or name.endswith("_sse4_1.cpp") or name.endswith("_avx2.cpp")
            )
            is_common_storage_backend = name.startswith("libgguf_storage_") and (
                name.endswith("_sse2.cpp") or name.endswith("_sse4_1.cpp") or name.endswith("_avx2.cpp")
            )
            if is_x86_build and (
                (is_dequant_backend and name.endswith("_avx2.cpp"))
                or (is_quant_backend and name.endswith("_avx2.cpp"))
                or (is_common_quant_backend and name.endswith("_avx2.cpp"))
                or (is_common_storage_backend and name.endswith("_avx2.cpp"))
            ):
                if self.compiler.compiler_type == "msvc":
                    return ["/arch:AVX2"]
                return ["-mavx2"]
            if (
                is_x86_build
                and (
                    (is_dequant_backend and name.endswith("_sse2.cpp"))
                    or (is_quant_backend and name.endswith("_sse2.cpp"))
                    or (is_common_quant_backend and name.endswith("_sse2.cpp"))
                    or (is_common_storage_backend and name.endswith("_sse2.cpp"))
                )
                and self.compiler.compiler_type != "msvc"
            ):
                return ["-msse2"]
            if (
                is_x86_build
                and (
                    (is_dequant_backend and name.endswith("_sse4_1.cpp"))
                    or (is_quant_backend and name.endswith("_sse4_1.cpp"))
                    or (is_common_quant_backend and name.endswith("_sse4_1.cpp"))
                    or (is_common_storage_backend and name.endswith("_sse4_1.cpp"))
                )
                and self.compiler.compiler_type != "msvc"
            ):
                return ["-msse4.1"]
            return []

        def compile_with_source_flags(
            sources,
            output_dir=None,
            macros=None,
            include_dirs=None,
            debug=0,
            extra_preargs=None,
            extra_postargs=None,
            depends=None,
        ):
            base_postargs = list(extra_postargs or [])
            jobs = self.parallel or int(os.environ.get("LIBGGUF_BUILD_JOBS") or os.environ.get("MAX_JOBS") or os.cpu_count() or 1)

            def compile_one(source: str):
                return original_compile(
                    [source],
                    output_dir=output_dir,
                    macros=macros,
                    include_dirs=include_dirs,
                    debug=debug,
                    extra_preargs=extra_preargs,
                    extra_postargs=base_postargs + source_flags(source),
                    depends=depends,
                )

            if jobs <= 1 or len(sources) <= 1:
                objects = []
                for source in sources:
                    objects.extend(compile_one(source))
                return objects

            objects = []
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                for compiled in executor.map(compile_one, sources):
                    objects.extend(compiled)
            return objects

        self.compiler.compile = compile_with_source_flags
        for ext in self.extensions:
            if self.compiler.compiler_type == "msvc":
                ext.extra_compile_args = ["/O2", "/EHsc", "/std:c++17", "/D_CRT_SECURE_NO_WARNINGS"]
            else:
                ext.extra_compile_args = ["-O3", "-std=c++17", "-pthread"]
                ext.extra_link_args = ["-lm", "-pthread"]
        super().build_extensions()


setup(
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=[
        Extension(
            "libgguf._libgguf",
            sources=["csrc/_libgguf_module.cpp", *NATIVE_SOURCES],
            include_dirs=["include", "csrc", "csrc/common"],
            language="c++",
        )
    ],
    cmdclass={"build_ext": BuildExt},
)
