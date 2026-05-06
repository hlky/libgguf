from __future__ import annotations

from pathlib import Path

from libgguf.libgguf_numpy.libgguf_numpy import _get_iq3_s_lookup, _get_iq3_xxs_lookup


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
    for key in range(8**4):
        if kmap[key] >= 0:
            offsets.append(len(flat))
            continue
        values = [int(value) for value in neighbours[key]]
        flat.extend(values)
        offsets.append(len(flat))

    return f"""static const __device__ uint32_t {name}_neighbour_offsets[IQ3_GRID_LOOKUP_SIZE + 1] = {{
{_format_array(offsets, suffix="u", per_line=12)}
}};

static const __device__ uint16_t {name}_neighbours[{len(flat)}] = {{
{_format_array(flat, suffix="u", per_line=16)}
}};
"""


def main() -> None:
    output = "#pragma once\n\n"
    output += _emit("iq3xxs", _get_iq3_xxs_lookup())
    output += "\n"
    output += _emit("iq3xs", _get_iq3_s_lookup())

    path = Path(__file__).resolve().parents[1] / "src/libgguf/libgguf_cuda/csrc/libgguf_cuda_iq3_neighbours.cuh"
    path.write_text(output)


if __name__ == "__main__":
    main()
