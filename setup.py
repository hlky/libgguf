from __future__ import annotations

import os

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class BuildExt(build_ext):
    def build_extensions(self) -> None:
        for ext in self.extensions:
            if self.compiler.compiler_type == "msvc":
                ext.extra_compile_args = ["/O2", "/EHsc", "/std:c++17", "/D_CRT_SECURE_NO_WARNINGS"]
                if os.environ.get("LIBGGUF_AVX2") == "1":
                    ext.extra_compile_args.append("/arch:AVX2")
            else:
                ext.extra_compile_args = ["-O3", "-std=c++17", "-pthread"]
                ext.extra_link_args = ["-lm", "-pthread"]
                if os.environ.get("LIBGGUF_AVX2") == "1":
                    ext.extra_compile_args.append("-mavx2")
        super().build_extensions()


setup(
    packages=["libgguf"],
    package_dir={"libgguf": "."},
    ext_modules=[
        Extension(
            "libgguf._libgguf",
            sources=["_libgguf_module.cpp", "libgguf.cpp"],
            include_dirs=["."],
            language="c++",
        )
    ],
    cmdclass={"build_ext": BuildExt},
)
