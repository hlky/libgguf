from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("gguf")
pytest.importorskip("libgguf")
pytest.importorskip("safetensors")

import gguf
import torch
from safetensors.torch import save_file

from libgguf.quantize import convert_to_gguf, parse_qtype, parse_tensor_qtype, strip_prefix


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
    assert parse_qtype("Q4_K") == ("Q4_K_M", "Q4_K")
    assert parse_qtype("Q4_K_S") == ("Q4_K_S", "Q4_K")
    assert parse_qtype("Q4_K_M") == ("Q4_K_M", "Q4_K")
    assert parse_qtype("Q8_0") == ("Q8_0", "Q8_0")
    assert parse_tensor_qtype("F16") == "F16"
    assert parse_tensor_qtype("Q4_K_S") == "Q4_K"

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
