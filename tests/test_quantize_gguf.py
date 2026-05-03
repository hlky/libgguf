from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import struct

import numpy as np
import pytest

pytest.importorskip("gguf")
pytest.importorskip("libgguf")
pytest.importorskip("safetensors")

import gguf
import libgguf
import torch
from safetensors.torch import save_file

from libgguf.quantize import convert_to_gguf, parse_qtype, parse_tensor_qtype, strip_prefix


def _write_bf16_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    entries = {}
    data = bytearray()
    for name, values in tensors.items():
        raw = np.ascontiguousarray(values, dtype=np.uint16).tobytes()
        begin = len(data)
        data.extend(raw)
        entries[name] = {
            "dtype": "BF16",
            "shape": list(values.shape),
            "data_offsets": [begin, len(data)],
        }
    header = json.dumps(entries, separators=(",", ":")).encode("utf-8")
    padding = (-((8 + len(header)) % 8)) % 8
    json_bytes = header + (b" " * padding)
    path.write_bytes(struct.pack("<Q", len(json_bytes)) + json_bytes + data)


def _float32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    high = bits >> 16
    rounded = np.where(
        (bits & np.uint32(0x7FFFFFFF)) > np.uint32(0x7F800000),
        high | np.uint32(64),
        (bits + (np.uint32(0x7FFF) + (high & np.uint32(1)))) >> np.uint32(16),
    )
    return rounded.astype(np.uint16)


def _tensor_types(path: Path) -> dict[str, gguf.GGMLQuantizationType]:
    reader = gguf.GGUFReader(path)
    return {tensor.name: tensor.tensor_type for tensor in reader.tensors}


def _field_string(reader: gguf.GGUFReader, key: str) -> str:
    field = reader.get_field(key)
    return str(field.parts[field.data[-1]], encoding="utf-8")


def _field_int(reader: gguf.GGUFReader, key: str) -> int:
    field = reader.get_field(key)
    return int(field.parts[field.data[-1]].item())


def test_parse_qtype_aliases_and_rejects_unsupported() -> None:
    assert parse_qtype("Q3_K") == ("Q3_K_M", "Q3_K")
    assert parse_qtype("Q4_K") == ("Q4_K_M", "Q4_K")
    assert parse_qtype("Q4_K_S") == ("Q4_K_S", "Q4_K")
    assert parse_qtype("Q4_K_M") == ("Q4_K_M", "Q4_K")
    assert parse_qtype("Q5_K") == ("Q5_K_M", "Q5_K")
    assert parse_qtype("MXFP4") == ("MXFP4_MOE", "MXFP4")
    assert parse_qtype("Q8_0") == ("Q8_0", "Q8_0")
    assert parse_tensor_qtype("F16") == "F16"
    assert parse_tensor_qtype("Q4_K_S") == "Q4_K"
    assert parse_tensor_qtype("MXFP4_MOE") == "MXFP4"

    with pytest.raises(ValueError, match="Unsupported"):
        parse_qtype("Q8_1")


def test_strip_prefix_keeps_only_model_payload() -> None:
    state = {
        "model.diffusion_model.double_blocks.0.img_attn.proj.weight": torch.zeros(2, 2),
        "first_stage_model.weight": torch.ones(2, 2),
    }

    assert strip_prefix(state) == {"double_blocks.0.img_attn.proj.weight": state["model.diffusion_model.double_blocks.0.img_attn.proj.weight"]}


def test_direct_quantization_writes_readable_gguf_with_mixed_policy(tmp_path: Path) -> None:
    src = tmp_path / "flux.safetensors"
    dst = tmp_path / "flux.gguf"
    save_file(
        {
            "model.diffusion_model.double_blocks.0.img_attn.proj.weight": torch.linspace(-1, 1, 512).reshape(2, 256),
            "model.diffusion_model.block.ffn_down.weight": torch.linspace(-1, 1, 512).reshape(2, 256),
            "model.diffusion_model.txt_in.weight": torch.ones(2, 256),
        },
        src,
    )

    result = convert_to_gguf(src, dst, "Q4_K_S")
    reader = gguf.GGUFReader(dst)
    types = _tensor_types(dst)

    assert result.arch == "flux"
    assert result.file_type == "MOSTLY_Q4_K_S"
    assert _field_string(reader, "general.architecture") == "flux"
    assert _field_int(reader, "general.file_type") == gguf.LlamaFileType.MOSTLY_Q4_K_S.value
    assert types["double_blocks.0.img_attn.proj.weight"] == gguf.GGMLQuantizationType.Q4_K
    assert types["block.ffn_down.weight"] == gguf.GGMLQuantizationType.Q5_K
    assert types["txt_in.weight"] == gguf.GGMLQuantizationType.F32


def test_direct_quantization_uses_native_storage_for_unquantized_tensors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "flux.safetensors"
    dst = tmp_path / "flux.gguf"
    save_file(
        {
            "double_blocks.0.img_attn.proj.weight": torch.ones(2, 256),
            "txt_in.weight": torch.ones(2, 256),
        },
        src,
    )

    calls: list[str] = []
    original_store_rows = libgguf.store_rows

    def store_rows_spy(data: np.ndarray, qtype: object) -> np.ndarray:
        calls.append(getattr(qtype, "name", str(qtype)))
        return original_store_rows(data, qtype)

    monkeypatch.setattr(libgguf, "store_rows", store_rows_spy)

    convert_to_gguf(src, dst, "Q8_0")

    assert "F32" in calls


def test_5d_tensor_is_written_without_sidecar_fix_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "wan.safetensors"
    dst = tmp_path / "wan.gguf"
    save_file(
        {
            "blocks.0.self_attn.norm_q.weight": torch.ones(4),
            "text_embedding.2.weight": torch.ones(2, 256),
            "head.modulation": torch.ones(4),
            "patch_embedding.proj.weight": torch.arange(32, dtype=torch.float32).reshape(1, 2, 2, 2, 4),
        },
        src,
    )

    convert_to_gguf(src, dst, "Q8_0")
    reader = gguf.GGUFReader(dst)
    tensors = {tensor.name: tensor for tensor in reader.tensors}

    assert "patch_embedding.proj.weight" in tensors
    assert tensors["patch_embedding.proj.weight"].tensor_type == gguf.GGMLQuantizationType.F32
    assert tuple(reversed([int(dim) for dim in tensors["patch_embedding.proj.weight"].shape])) == (1, 2, 2, 2, 4)
    assert tensors["patch_embedding.proj.weight"].data.shape == (1, 2, 2, 2, 4)
    assert not (tmp_path / "fix_5d_tensors_wan.safetensors").exists()


def test_safetensors_conversion_does_not_use_eager_load_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import safetensors.torch

    src = tmp_path / "flux.safetensors"
    dst = tmp_path / "flux.gguf"
    save_file({"double_blocks.0.img_attn.proj.weight": torch.ones(2, 256)}, src)

    def fail_load_file(*args: object, **kwargs: object) -> None:
        raise AssertionError("safetensors conversion should use safe_open")

    monkeypatch.setattr(safetensors.torch, "load_file", fail_load_file)

    convert_to_gguf(src, dst, "Q8_0")

    assert dst.exists()


def test_safetensors_f32_f16_conversion_does_not_import_torch(tmp_path: Path) -> None:
    src = tmp_path / "flux_numpy.safetensors"
    dst = tmp_path / "flux_numpy.gguf"
    script = f"""
import builtins
from pathlib import Path
import numpy as np
from safetensors.numpy import save_file

real_import = builtins.__import__
def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "torch" or name.startswith("torch."):
        raise ImportError("torch import blocked")
    return real_import(name, globals, locals, fromlist, level)

save_file({{
    "double_blocks.0.img_attn.proj.weight": np.linspace(-1, 1, 64, dtype=np.float32).reshape(2, 32),
    "txt_in.weight": np.ones((2, 32), dtype=np.float16),
}}, {str(src)!r})

builtins.__import__ = blocked_import
from libgguf.quantize import convert_to_gguf
convert_to_gguf({str(src)!r}, {str(dst)!r}, "Q8_0")
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)

    assert dst.exists()
    types = _tensor_types(dst)
    assert types["double_blocks.0.img_attn.proj.weight"] == gguf.GGMLQuantizationType.Q8_0
    assert types["txt_in.weight"] == gguf.GGMLQuantizationType.F16


def test_safetensors_bf16_storage_uses_raw_local_reader(tmp_path: Path) -> None:
    src = tmp_path / "flux_bf16.safetensors"
    dst = tmp_path / "flux_bf16.gguf"
    source = {
        "double_blocks.0.img_attn.proj.weight": _float32_to_bf16_bits(np.linspace(-1, 1, 64, dtype=np.float32).reshape(2, 32)),
        "txt_in.weight": _float32_to_bf16_bits(np.array([[1.0, -2.0, 3.5, 0.25]], dtype=np.float32)),
    }
    _write_bf16_safetensors(src, source)

    convert_to_gguf(src, dst, "Q8_0", tensor_overrides={"txt_in.weight": "BF16"})

    reader = gguf.GGUFReader(dst)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    assert tensors["txt_in.weight"].tensor_type == gguf.GGMLQuantizationType.BF16
    np.testing.assert_array_equal(tensors["txt_in.weight"].data.reshape(-1), source["txt_in.weight"].view(np.uint8).reshape(-1))


def test_safetensors_bf16_quantization_converts_to_float32_once_loaded(tmp_path: Path) -> None:
    src = tmp_path / "flux_bf16_quant.safetensors"
    dst = tmp_path / "flux_bf16_quant.gguf"
    rows = np.linspace(-1, 1, 64, dtype=np.float32).reshape(2, 32)
    _write_bf16_safetensors(src, {"double_blocks.0.img_attn.proj.weight": _float32_to_bf16_bits(rows)})

    convert_to_gguf(src, dst, "Q8_0", policy="uniform")

    reader = gguf.GGUFReader(dst)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    tensor = tensors["double_blocks.0.img_attn.proj.weight"]
    assert tensor.tensor_type == gguf.GGMLQuantizationType.Q8_0
    dequantized = libgguf.dequantize_rows(tensor.data, libgguf.GGMLQuantizationType.Q8_0)
    assert dequantized.shape == rows.shape
    assert np.all(np.isfinite(dequantized))


def test_shape_fix_metadata_is_written(tmp_path: Path) -> None:
    src = tmp_path / "sd1.safetensors"
    dst = tmp_path / "sd1.gguf"
    save_file(
        {
            "down_blocks.0.downsamplers.0.conv.weight": torch.arange(512, dtype=torch.float32).reshape(16, 16, 2),
        },
        src,
    )

    convert_to_gguf(src, dst, "Q8_0")
    reader = gguf.GGUFReader(dst)

    assert reader.get_field("comfy.gguf.orig_shape.down_blocks.0.downsamplers.0.conv.weight") is not None


def test_overrides_force_tensor_outcomes(tmp_path: Path) -> None:
    src = tmp_path / "flux.safetensors"
    dst = tmp_path / "flux.gguf"
    save_file(
        {
            "double_blocks.0.img_attn.proj.weight": torch.ones(2, 256),
            "other.weight": torch.ones(2, 32),
            "skip.weight": torch.ones(2, 256),
        },
        src,
    )

    convert_to_gguf(
        src,
        dst,
        "Q4_K_S",
        policy="uniform",
        exclude=("skip.*",),
        tensor_overrides={"other.weight": "Q8_0", "skip.weight": "F16"},
    )
    types = _tensor_types(dst)

    assert types["double_blocks.0.img_attn.proj.weight"] == gguf.GGMLQuantizationType.Q4_K
    assert types["other.weight"] == gguf.GGMLQuantizationType.Q8_0
    assert types["skip.weight"] == gguf.GGMLQuantizationType.F16


def test_cli_accepts_required_arguments(tmp_path: Path) -> None:
    src = tmp_path / "flux.safetensors"
    dst = tmp_path / "flux.gguf"
    save_file({"double_blocks.0.img_attn.proj.weight": torch.ones(2, 32)}, src)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "libgguf.quantize_gguf",
            "--src",
            str(src),
            "--dst",
            str(dst),
            "--qtype",
            "Q8_0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Wrote" in completed.stdout
    assert dst.exists()
