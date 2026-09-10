from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
from pathlib import Path

import libgguf
import pytest


def test_mixtral_conversion_preserves_decoder_names_and_head(tmp_path: Path) -> None:
    executable = os.environ.get("LIBGGUF_QUANTIZE_EXECUTABLE") or shutil.which("libgguf_quantize_gguf")
    if executable is None:
        pytest.skip("native converter is not installed")
    keys = [
        "model.embed_tokens.weight",
        "model.layers.0.block_sparse_moe.gate.weight",
        "model.layers.0.block_sparse_moe.experts.0.w1.weight",
        "lm_head.weight",
    ]
    payload = struct.pack("<512H", *([0x3F80] * 512))
    header = {
        key: {"dtype": "BF16", "shape": [2, 256], "data_offsets": [i * len(payload), (i + 1) * len(payload)]}
        for i, key in enumerate(keys)
    }
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    source = tmp_path / "mixtral.safetensors"
    source.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload * len(keys))
    destination = tmp_path / "mixtral.gguf"
    backend = os.environ.get("LIBGGUF_TEST_CONVERSION_BACKEND", "cpu")
    verification = ["--verify-gpu-tensors", "all"] if backend == "gpu" else []
    completed = subprocess.run(
        [
            executable,
            "--src",
            str(source),
            "--dst",
            str(destination),
            "--qtype",
            "Q4_K_S",
            "--policy",
            "uniform",
            "--backend",
            backend,
            *verification,
            "--tensor-type",
            "*.block_sparse_moe.gate.weight=BF16",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = libgguf.open_gguf(destination)
    assert result.metadata["general.architecture"].value == "mixtral"
    tensors = {tensor.name: tensor for tensor in result.tensors}
    assert set(tensors) == set(keys)
    assert tensors[keys[1]].qtype == "BF16"
    assert tensors["lm_head.weight"].qtype == "Q4_K"
