from __future__ import annotations

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class BuildExt(build_ext):
    def build_extensions(self) -> None:
        for ext in self.extensions:
            if self.compiler.compiler_type == "msvc":
                ext.extra_compile_args = ["/O3", "/std:c++17"]
            else:
                ext.extra_compile_args = ["-O3", "-std=c++17"]
                ext.extra_link_args = ["-lm"]
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
