from __future__ import annotations

import struct
import subprocess
import sys
from pathlib import Path

import gguf

from scripts import compare_hf_gguf as compare


def _gguf_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _minimal_gguf(path: Path, qtype: int, *, long_numeric_metadata: bool = False) -> None:
    data = bytearray()
    data += b"GGUF"
    data += struct.pack("<IQQ", 3, 1, 3 if long_numeric_metadata else 2)
    data += _gguf_string("general.architecture")
    data += struct.pack("<I", 8)
    data += _gguf_string("test-arch")
    data += _gguf_string("general.quantization_version")
    data += struct.pack("<II", 4, 2)
    if long_numeric_metadata:
        data += _gguf_string("test.long_array")
        data += struct.pack("<IIQ", 9, 4, 129)
        data += struct.pack("<" + "I" * 129, *range(129))
    data += _gguf_string("blocks.0.attn_v.weight")
    data += struct.pack("<I", 2)
    data += struct.pack("<QQ", 256, 2)
    data += struct.pack("<IQ", qtype, 0)
    path.write_bytes(data)


def test_importing_compare_does_not_import_libgguf() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "import scripts.compare_hf_gguf\n"
                "imported = [name for name in sys.modules if name == 'libgguf' or name.startswith('libgguf.')]\n"
                "if imported:\n"
                "    raise SystemExit(f'libgguf imported at module import: {imported}')\n"
            ),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_gguf_header_from_prefix_grows_until_header_available(tmp_path: Path) -> None:
    gguf_path = tmp_path / "minimal.gguf"
    _minimal_gguf(gguf_path, int(gguf.GGMLQuantizationType.Q4_0))
    data = gguf_path.read_bytes()
    calls: list[int] = []

    def fetch_prefix(size: int) -> bytes:
        calls.append(size)
        return data[:size]

    header = compare._gguf_header_from_prefix(fetch_prefix, initial_bytes=16)

    assert calls[0] == 16
    assert len(calls) > 1
    assert calls == [16 * (2**index) for index in range(len(calls))]
    assert header["tensor_count"] == 1
    assert header["tensors"][0]["name"] == "blocks.0.attn_v.weight"


def test_qtype_filtering_excludes_float_iq_and_ud_from_standard() -> None:
    assert compare.is_standard_comparison_variant("Q4_0")
    assert compare.is_standard_comparison_variant("Q5_K_M")
    assert not compare.is_standard_comparison_variant("BF16")
    assert not compare.is_standard_comparison_variant("F16")
    assert not compare.is_standard_comparison_variant("F32")
    assert not compare.is_standard_comparison_variant("IQ4_NL")
    assert not compare.is_standard_comparison_variant("UD-Q5_K_M")


def test_ud_detection_maps_to_base_qtype_but_routes_to_analysis() -> None:
    assert compare.is_ud_variant("UD-Q5_K_M")
    assert compare.base_qtype_variant("UD-Q5_K_M") == "Q5_K_M"
    assert compare.is_ud_analysis_variant("UD-Q5_K_M")
    assert not compare.is_standard_comparison_variant("UD-Q5_K_M")


def test_local_gguf_header_extracts_metadata_and_tensor_qtypes(tmp_path: Path) -> None:
    gguf_path = tmp_path / "minimal.gguf"
    _minimal_gguf(gguf_path, int(gguf.GGMLQuantizationType.Q4_0))

    header = compare.local_gguf_header(gguf_path, initial_bytes=64)

    assert header["version"] == 3
    assert header["tensor_count"] == 1
    assert header["metadata"]["general.architecture"]["value"] == "test-arch"
    assert header["metadata"]["general.quantization_version"]["value"] == 2
    assert header["tensors"] == [
        {
            "name": "blocks.0.attn_v.weight",
            "shape": [256, 2],
            "qtype": "Q4_0",
            "qtype_value": int(gguf.GGMLQuantizationType.Q4_0),
            "offset": 0,
        }
    ]


def test_local_gguf_header_preserves_large_numeric_array_summary(tmp_path: Path) -> None:
    gguf_path = tmp_path / "minimal.gguf"
    _minimal_gguf(gguf_path, int(gguf.GGMLQuantizationType.Q4_0), long_numeric_metadata=True)

    header = compare.local_gguf_header(gguf_path, initial_bytes=64)

    assert header["metadata"]["test.long_array"] == {
        "type": "ARRAY",
        "array_type": "UINT32",
        "length": 129,
        "value": "<129 items>",
    }


def test_standard_entries_skip_excluded_and_ud_variants() -> None:
    repo = compare.RepoInventory(
        repo_id="test/model-gguf",
        base_models=["test/model"],
        gguf_files=[
            compare.FileInfo("model-F16.gguf", 1, "F16", "F16"),
            compare.FileInfo("model-IQ4_NL.gguf", 1, "IQ4_NL", "IQ4_NL"),
            compare.FileInfo("model-Q4_0.gguf", 1, "Q4_0", "Q4_0"),
            compare.FileInfo("model-UD-Q4_K_M.gguf", 1, "Q4_K_M", "UD-Q4_K_M"),
        ],
        qtypes=[],
        qtype_variants=[],
        qtype_groups={},
        source_candidates=[compare.SourceCandidate("Comfy-Org/test", "diffusion_models/model_bf16.safetensors", 1, 10)],
    )

    entries = compare._standard_entries([repo], None, None)

    assert [entry["variant"] for entry in entries] == ["Q4_0"]
    assert entries[0]["source_candidate"]["path"] == "diffusion_models/model_bf16.safetensors"


def test_ud_entries_pair_with_base_variant() -> None:
    base = compare.FileInfo("model-Q5_K_M.gguf", 1, "Q5_K_M", "Q5_K_M")
    ud = compare.FileInfo("model-UD-Q5_K_M.gguf", 1, "Q5_K_M", "UD-Q5_K_M")
    repo = compare.RepoInventory(
        repo_id="test/repo",
        base_models=[],
        gguf_files=[base, ud, compare.FileInfo("model-Q4_0.gguf", 1, "Q4_0", "Q4_0")],
        qtypes=[],
        qtype_variants=[],
        qtype_groups={},
        source_candidates=[],
    )

    entries = compare._ud_entries([repo], None, None)

    assert entries == [(repo, ud, base)]


def test_standard_resume_key_includes_source_and_status_policy() -> None:
    entry = {
        "repo_id": "repo/a",
        "gguf_file": "model-Q4_0.gguf",
        "variant": "Q4_0",
        "source_candidate": {"repo_id": "source/a", "path": "diffusion_models/model.safetensors"},
    }
    same_reference_different_source = {
        **entry,
        "source_candidate": {"repo_id": "source/b", "path": "diffusion_models/model.safetensors"},
    }

    assert compare._standard_result_key(entry) != compare._standard_result_key(same_reference_different_source)
    assert compare._can_resume_standard_result({"status": "byte_identical"}, execute=True)
    assert compare._can_resume_standard_result({"status": "key_mismatch"}, execute=True)
    assert compare._can_resume_standard_result({"status": "key_match_not_executed"}, execute=False)
    assert not compare._can_resume_standard_result({"status": "key_match_not_executed"}, execute=True)


def test_source_selection_rejects_shards_and_prefers_single_file_base_model() -> None:
    repo = compare.RepoInventory(
        repo_id="city96/AuraFlow-v0.3-gguf",
        base_models=["fal/AuraFlow-v0.3"],
        gguf_files=[],
        qtypes=[],
        qtype_variants=[],
        qtype_groups={},
        source_candidates=[
            compare.SourceCandidate(
                "fal/AuraFlow-v0.3",
                "transformer/diffusion_pytorch_model-00001-of-00002.fp16.safetensors",
                1,
                8,
            ),
            compare.SourceCandidate("fal/AuraFlow-v0.3", "aura_flow_0.3.safetensors", 1, 7),
            compare.SourceCandidate("Comfy-Org/HunyuanVideo_1.5_repackaged", "split_files/diffusion_models/capybara_v0.1.safetensors", 1, 2),
        ],
    )

    selected = compare.select_source_candidate(repo)

    assert selected is not None
    assert selected.repo_id == "fal/AuraFlow-v0.3"
    assert selected.path == "aura_flow_0.3.safetensors"
