from __future__ import annotations

import platform
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

from scripts.native_sources import NATIVE_SOURCES


def _is_x86_build() -> bool:
    machine = platform.machine().lower()
    return machine in {"amd64", "x86_64", "x86", "i386", "i686"}


class BuildExt(build_ext):
    def build_extensions(self) -> None:
        original_compile = self.compiler.compile
        is_x86_build = _is_x86_build()

        def source_flags(source: str) -> list[str]:
            name = Path(source).name
            if is_x86_build and name in {
                "dequant_q4_0_avx2.cpp",
                "dequant_q8_0_avx2.cpp",
                "quant_q4_0_avx2.cpp",
                "quant_q8_0_avx2.cpp",
            }:
                if self.compiler.compiler_type == "msvc":
                    return ["/arch:AVX2"]
                return ["-mavx2"]
            if (
                is_x86_build
                and name in {
                    "dequant_q4_0_sse2.cpp",
                    "dequant_q8_0_sse2.cpp",
                    "quant_q4_0_sse2.cpp",
                    "quant_q8_0_sse2.cpp",
                }
                and self.compiler.compiler_type != "msvc"
            ):
                return ["-msse2"]
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
            objects = []
            base_postargs = list(extra_postargs or [])
            for source in sources:
                objects.extend(
                    original_compile(
                        [source],
                        output_dir=output_dir,
                        macros=macros,
                        include_dirs=include_dirs,
                        debug=debug,
                        extra_preargs=extra_preargs,
                        extra_postargs=base_postargs + source_flags(source),
                        depends=depends,
                    )
                )
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
