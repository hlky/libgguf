from __future__ import annotations

from pathlib import Path

from libgguf.libgguf_numpy.libgguf_numpy import _get_iq2_s_lookup, _get_iq2_xs_lookup, _get_iq2_xxs_lookup


def _base3_to_base4_key(key: int) -> int:
    out = 0
    for i in range(8):
        out |= (key % 3) << (2 * i)
        key //= 3
    return out


def _format_array(values: list[int], *, suffix: str = "", per_line: int = 16) -> str:
    lines: list[str] = []
    for i in range(0, len(values), per_line):
        chunk = values[i : i + per_line]
        lines.append("    " + ", ".join(f"{value}{suffix}" for value in chunk))
    return ",\n".join(lines)


def _emit(name: str, lookup: tuple) -> str:
    kmap, neighbours, _ = lookup
    offsets = [0]
    flat: list[int] = []
    for key in range(3**8):
        base4_key = _base3_to_base4_key(key)
        if kmap[base4_key] >= 0:
            offsets.append(len(flat))
            continue
        values = [int(value) for value in neighbours[base4_key]]
        flat.extend(values)
        offsets.append(len(flat))

    return f"""static const __device__ uint32_t {name}_neighbour_offsets[IQ2_GRID_LOOKUP_SIZE + 1] = {{
{_format_array(offsets, suffix="u", per_line=12)}
}};

static const __device__ uint16_t {name}_neighbours[{len(flat)}] = {{
{_format_array(flat, suffix="u", per_line=16)}
}};
"""


def main() -> None:
    output = """#pragma once

#ifdef GGUF_CUDA_USE_IQ2_XXS_NEIGHBOURS
"""
    output += _emit("iq2xxs", _get_iq2_xxs_lookup())
    output += """#endif

#ifdef GGUF_CUDA_USE_IQ2_XS_NEIGHBOURS
"""
    output += _emit("iq2xs", _get_iq2_xs_lookup())
    output += """#endif

#ifdef GGUF_CUDA_USE_IQ2_S_NEIGHBOURS
"""
    output += _emit("iq2s", _get_iq2_s_lookup())
    output += "#endif\n"
    path = Path(__file__).resolve().parents[1] / "src/libgguf/libgguf_cuda/csrc/libgguf_cuda_iq2_neighbours.cuh"
    path.write_text(output)


if __name__ == "__main__":
    main()
