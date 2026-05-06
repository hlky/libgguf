#!/usr/bin/env python3
from pathlib import Path

import numpy as np

from libgguf.libgguf_numpy.libgguf_numpy import _get_iq1_s_lookup


def _format_array(values: list[int], suffix: str = "u", per_line: int = 12) -> str:
    lines = []
    for i in range(0, len(values), per_line):
        chunk = values[i : i + per_line]
        lines.append("    " + ", ".join(f"{value}{suffix}" for value in chunk) + ",")
    return "\n".join(lines)


def _emit() -> str:
    kmap, neighbours, _ = _get_iq1_s_lookup()
    offsets = [0]
    flat: list[int] = []
    direct = [int(value) for value in kmap]
    for key in range(len(kmap)):
        values = [int(value) for value in neighbours[key]]
        flat.extend(values)
        offsets.append(len(flat))

    return f"""static const __device__ short iq1s_grid_direct[IQ1_GRID_LOOKUP_SIZE] = {{
{_format_array(direct, "", 16)}
}};

static const __device__ uint32_t iq1s_neighbour_offsets[IQ1_GRID_LOOKUP_SIZE + 1] = {{
{_format_array(offsets, "u", 8)}
}};

static const __device__ uint16_t iq1s_neighbours[{len(flat)}] = {{
{_format_array(flat, "u", 12)}
}};
"""


def main() -> None:
    path = Path(__file__).resolve().parents[1] / "src/libgguf/libgguf_cuda/csrc/libgguf_cuda_iq1_neighbours.cuh"
    output = "#pragma once\n\n#ifdef GGUF_CUDA_USE_IQ1_NEIGHBOURS\n"
    output += "#define IQ1_GRID_LOOKUP_SIZE 65536\n\n"
    output += _emit()
    output += "#endif\n"
    path.write_text(output)


if __name__ == "__main__":
    main()
