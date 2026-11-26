import json
from pathlib import Path
import struct
import subprocess
import os

import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import save_file, load_file


def converter():
    path = Path(os.environ.get("LIBGGUF_NATIVE_CONVERTER", Path(__file__).parents[1] / "build/native/libgguf_quantize_native.exe"))
    if not path.exists():
        pytest.skip("native converter executable has not been built")
    return path


def run(source, destination):
    return subprocess.run([str(converter()), "--src", str(source), "--dst", str(destination), "--format", "S4G64"], capture_output=True)


def test_native_converter_accepts_unicode_filesystem_paths(tmp_path):
    source=tmp_path/"模型入力.safetensors"
    destination=tmp_path/"重み🦖.safetensors"
    save_file({"weight":np.ones((2,64),dtype=np.float32)},source)
    result=run(source,destination)
    assert result.returncode==0, result.stderr
    assert load_file(destination)["weight"].shape==(2,34)


def test_native_converter_preserves_names_metadata_plain_tensors_and_never_overwrites(tmp_path):
    source, destination = tmp_path / "source.safetensors", tmp_path / "native.safetensors"
    values = {"layer.\u00e9\U0001f996.weight": np.arange(512, dtype=np.float32).reshape(2, 256), "bias": np.array([1, -2], dtype=np.float16)}
    metadata = {"format": "pt", "note": "escaped quotes \" and \u00e9 \U0001f996"}
    save_file(values, source, metadata=metadata)
    assert run(source, destination).returncode == 0
    assert set(load_file(destination)) == set(values)
    np.testing.assert_array_equal(load_file(destination)["bias"], values["bias"])
    with safe_open(destination, framework="numpy") as file:
        for name, value in metadata.items():
            assert file.metadata()[name] == value
        descriptor = json.loads(file.metadata()["libgguf.quantization.v1"])
        assert descriptor["tensors"]["layer.\u00e9\U0001f996.weight"]["logical_shape"] == [2, 256]
    before = destination.read_bytes()
    assert run(source, destination).returncode != 0
    assert destination.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp-*"))


@pytest.mark.parametrize("header", [
    b'{"w":{"dtype":"F32","dtype":"F16","shape":[1,64],"data_offsets":[0,256]}}',
    b'{"w":{"dtype":"F32","shape":[1,64],"data_offsets":[0,256]},"w":{"dtype":"F32","shape":[1,64],"data_offsets":[0,256]}}',
    b'{"w":{"dtype":"F32","shape":[1,64],"data_offsets":[18446744073709551615,255]}}',
    b'{} trailing', b'{"w":', b'{"w":{},',
    b'{"bad\x01name":{}}', b'{"bad\xffname":{}}',
    b'{"w":{"dtype":"F32","shape":[01,64],"data_offsets":[0,256]}}',
    b'{"__metadata__":{"note":"\\ud800"}}',
    b'{"a":{"dtype":"F32","shape":[64],"data_offsets":[0,256]},"b":{"dtype":"F32","shape":[64],"data_offsets":[0,256]}}',
])
def test_native_converter_rejects_malformed_source_without_publishing(tmp_path, header):
    source, destination = tmp_path / "invalid.safetensors", tmp_path / "output.safetensors"
    source.write_bytes(struct.pack("<Q", len(header)) + header + bytes(256))
    assert run(source, destination).returncode != 0
    assert not destination.exists()
    assert not list(tmp_path.glob("*.tmp-*"))
