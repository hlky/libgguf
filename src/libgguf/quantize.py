from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from fnmatch import fnmatchcase
import json
import logging
import os
from pathlib import Path
import struct
from typing import Any, Iterator, Mapping, Protocol, Sequence

import numpy as np
from tqdm import tqdm

from .imatrix import load_imatrix
from ._metadata import GGMLQuantizationType, LlamaFileType, GGML_QUANT_SIZES

try:
    import gguf
    from safetensors import safe_open
except ImportError as exc:
    raise ImportError(
        "Direct GGUF quantization requires gguf, safetensors, and the editable libgguf package."
    ) from exc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


QUANTIZATION_THRESHOLD = 1024
REARRANGE_THRESHOLD = 512
MAX_TENSOR_NAME_LENGTH = 127
NATIVE_DEFAULT_SCRATCH_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class QuantResult:
    output_path: Path
    arch: str
    file_type: str
    tensor_type_counts: dict[str, int]
    fallback_counts: dict[str, int]


@dataclass(frozen=True)
class TensorPlan:
    key: str
    source_key: str
    source_shape: tuple[int, ...]
    source_dtype: str
    write_shape: tuple[int, ...]
    target_qtype: Any
    quantize: bool
    imatrix: np.ndarray | None


class TensorSource(Protocol):
    def keys(self) -> Sequence[str]: ...

    def tensor_meta(self, key: str) -> tuple[tuple[int, ...], str]: ...

    def load_tensor(self, key: str) -> Any: ...


class ModelTemplate:
    arch = "invalid"
    shape_fix = False
    keys_detect: Sequence[Sequence[str]] = ()
    keys_banned: Sequence[str] = ()
    keys_hiprec: Sequence[str] = ()
    keys_ignore: Sequence[str] = ()


class ModelFlux(ModelTemplate):
    arch = "flux"
    keys_detect = (
        ("transformer_blocks.0.attn.norm_added_k.weight",),
        ("double_blocks.0.img_attn.proj.weight",),
    )
    keys_banned = ("transformer_blocks.0.attn.norm_added_k.weight",)


class ModelSD3(ModelTemplate):
    arch = "sd3"
    keys_detect = (
        ("transformer_blocks.0.attn.add_q_proj.weight",),
        ("joint_blocks.0.x_block.attn.qkv.weight",),
    )
    keys_banned = ("transformer_blocks.0.attn.add_q_proj.weight",)


class ModelAura(ModelTemplate):
    arch = "aura"
    keys_detect = (
        ("double_layers.3.modX.1.weight",),
        ("joint_transformer_blocks.3.ff_context.out_projection.weight",),
    )
    keys_banned = ("joint_transformer_blocks.3.ff_context.out_projection.weight",)


class ModelHiDream(ModelTemplate):
    arch = "hidream"
    keys_detect = (("caption_projection.0.linear.weight", "double_stream_blocks.0.block.ff_i.shared_experts.w3.weight"),)
    keys_hiprec = (".ff_i.gate.weight", "img_emb.emb_pos")


class CosmosPredict2(ModelTemplate):
    arch = "cosmos"
    keys_detect = (("blocks.0.mlp.layer1.weight", "blocks.0.adaln_modulation_cross_attn.1.weight"),)
    keys_hiprec = ("pos_embedder",)
    keys_ignore = ("_extra_state", "accum_")


class ModelHyVid(ModelTemplate):
    arch = "hyvid"
    keys_detect = (("double_blocks.0.img_attn_proj.weight", "txt_in.individual_token_refiner.blocks.1.self_attn_qkv.weight"),)


class ModelWan(ModelHyVid):
    arch = "wan"
    keys_detect = (("blocks.0.self_attn.norm_q.weight", "text_embedding.2.weight", "head.modulation"),)
    keys_hiprec = (".modulation",)


class ModelLTXV(ModelTemplate):
    arch = "ltxv"
    keys_detect = (("adaln_single.emb.timestep_embedder.linear_2.weight", "transformer_blocks.27.scale_shift_table", "caption_projection.linear_2.weight"),)
    keys_hiprec = ("scale_shift_table",)


class ModelSDXL(ModelTemplate):
    arch = "sdxl"
    shape_fix = True
    keys_detect = (
        ("down_blocks.0.downsamplers.0.conv.weight", "add_embedding.linear_1.weight"),
        ("input_blocks.3.0.op.weight", "input_blocks.6.0.op.weight", "output_blocks.2.2.conv.weight", "output_blocks.5.2.conv.weight"),
        ("label_emb.0.0.weight",),
    )


class ModelSD1(ModelTemplate):
    arch = "sd1"
    shape_fix = True
    keys_detect = (
        ("down_blocks.0.downsamplers.0.conv.weight",),
        ("input_blocks.3.0.op.weight", "input_blocks.6.0.op.weight", "input_blocks.9.0.op.weight", "output_blocks.2.1.conv.weight", "output_blocks.5.2.conv.weight", "output_blocks.8.2.conv.weight"),
    )


class ModelLumina2(ModelTemplate):
    arch = "lumina2"
    keys_detect = (("cap_embedder.1.weight", "context_refiner.0.attention.qkv.weight"),)


ARCH_LIST = (
    ModelFlux,
    ModelSD3,
    ModelAura,
    ModelHiDream,
    CosmosPredict2,
    ModelLTXV,
    ModelHyVid,
    ModelWan,
    ModelSDXL,
    ModelSD1,
    ModelLumina2,
)

IMAGE_ARCHES = {arch.arch for arch in ARCH_LIST}

SUPPORTED_QUANT_QTYPES = {
    "Q1_0",
    "Q4_0",
    "Q4_1",
    "Q5_0",
    "Q5_1",
    "Q8_0",
    "Q2_K",
    "Q3_K",
    "Q4_K",
    "Q5_K",
    "Q6_K",
    "IQ2_XXS",
    "IQ2_XS",
    "IQ2_S",
    "IQ3_XXS",
    "IQ3_S",
    "IQ1_S",
    "IQ1_M",
    "IQ4_NL",
    "IQ4_XS",
    "TQ1_0",
    "TQ2_0",
    "MXFP4",
    "NVFP4",
}
SUPPORTED_STORAGE_QTYPES = SUPPORTED_QUANT_QTYPES | {"F32", "F16", "BF16"}

FILE_TYPE_ALIASES = {
    "Q3_K": "Q3_K_M",
    "Q4_K": "Q4_K_M",
    "Q5_K": "Q5_K_M",
    "MXFP4": "MXFP4_MOE",
}

FILE_TYPE_TO_TENSOR_QTYPE = {
    "Q3_K_S": "Q3_K",
    "Q3_K_M": "Q3_K",
    "Q3_K_L": "Q3_K",
    "Q4_K_S": "Q4_K",
    "Q4_K_M": "Q4_K",
    "Q5_K_S": "Q5_K",
    "Q5_K_M": "Q5_K",
    "Q2_K_S": "Q2_K",
    "MXFP4_MOE": "MXFP4",
    "IQ2_M": "IQ2_S",
    "IQ3_M": "IQ3_S",
}

ARCH_SKIP_PATTERNS = {
    "flux": ("txt_in.*", "img_in.*", "time_in.*", "vector_in.*", "guidance_in.*", "final_layer.*"),
    "sd1": ("class_embedding.*", "time_embedding.*", "add_embedding.*", "time_embed.*", "label_emb.*", "conv_in.*", "conv_out.*", "input_blocks.0.0.weight", "out.2.weight"),
    "sdxl": ("class_embedding.*", "time_embedding.*", "add_embedding.*", "time_embed.*", "label_emb.*", "conv_in.*", "conv_out.*", "input_blocks.0.0.weight", "out.2.weight"),
    "sd3": ("final_layer.*", "time_text_embed.*", "context_embedder.*", "t_embedder.*", "y_embedder.*", "x_embedder.*", "proj_out.weight", "pos_embed"),
    "aura": ("t_embedder.*", "init_x_linear.*", "modF.1.weight", "cond_seq_linear.weight", "final_linear.weight", "positional_encoding", "register_tokens"),
    "ltxv": ("adaln_single.*", "caption_projection.*", "patchify_proj.*", "proj_out.*", "*scale_shift_table*"),
    "hyvid": ("txt_in.*", "img_in.*", "time_in.*", "vector_in.*", "guidance_in.*", "final_layer.*"),
    "wan": ("*modulation.*", "patch_embedding.*", "text_embedding.*", "time_projection.*", "time_embedding.*", "img_emb.*", "head.*"),
    "hidream": ("p_embedder.*", "t_embedder.*", "x_embedder.*", "final_layer.*", "*.ff_i.gate.weight", "caption_projection.*"),
    "cosmos": ("p_embedder.*", "t_embedder.*", "t_embedding_norm.*", "x_embedder.*", "pos_embedder.*", "final_layer.*"),
    "lumina2": ("t_embedder.*", "x_embedder.*", "final_layer.*", "cap_embedder.*", "context_refiner.*", "noise_refiner.*"),
}

ATTENTION_VALUE_PATTERNS = (
    "*attn_v.weight*",
    "*.to_v.weight*",
    "*.v.weight*",
    "*.attn.w1v.weight*",
    "*.attn.w2v.weight*",
    "*_attn.v_proj.weight*",
)
FUSED_QKV_PATTERNS = ("*attn_qkv.weight*", "*attn.qkv.weight*", "*attention.qkv.weight*")
FFN_DOWN_PATTERNS = (
    "*ffn_down*",
    "*experts.*.w2.weight*",
    "*.ffn.2.weight*",
    "*.ff.net.2.weight*",
    "*.mlp.layer2.weight*",
    "*.adaln_modulation_mlp.2.weight*",
    "*.feed_forward.w2.weight*",
)


def _matches(name: str, patterns: Sequence[str] | None) -> bool:
    return bool(patterns) and any(fnmatchcase(name, pattern) for pattern in patterns)


def _is_model_arch(model: type[ModelTemplate], state_dict: Mapping[str, Any]) -> bool:
    matched = False
    invalid = False
    for match_list in model.keys_detect:
        if all(key in state_dict for key in match_list):
            matched = True
            invalid = any(key in state_dict for key in model.keys_banned)
            break
    if invalid:
        raise AssertionError("Model architecture not allowed for conversion; use the reference checkpoint key format")
    return matched


def detect_arch(state_dict: Mapping[str, Any] | set[str]) -> ModelTemplate:
    keys = state_dict if isinstance(state_dict, set) else set(state_dict)
    for arch in ARCH_LIST:
        if _is_model_arch(arch, keys):
            return arch()
    raise AssertionError("Unknown model architecture")


def strip_prefix(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    prefix = None
    for pfx in ("model.diffusion_model.", "model."):
        if any(key.startswith(pfx) for key in state_dict):
            prefix = pfx
            break
    if prefix is None:
        for pfx in ("net.",):
            if all(key.startswith(pfx) for key in state_dict):
                prefix = pfx
                break
    if prefix is None:
        return dict(state_dict)

    logging.info("State dict prefix found: %r", prefix)
    return {key.removeprefix(prefix): value for key, value in state_dict.items() if key.startswith(prefix)}


def load_state_dict(path: str | os.PathLike[str]) -> dict[str, Any]:
    path_str = os.fspath(path)
    if path_str.endswith((".ckpt", ".pt", ".bin", ".pth")):
        try:
            import torch
        except ImportError as exc:
            raise ImportError("Loading Torch checkpoint formats requires torch.") from exc

        state_dict = torch.load(path_str, map_location="cpu", weights_only=True)
        for subkey in ("model", "module"):
            if subkey in state_dict:
                state_dict = state_dict[subkey]
                break
        if len(state_dict) < 20:
            raise RuntimeError(f"pt subkey load failed: {state_dict.keys()}")
    else:
        from safetensors.numpy import load_file

        state_dict = load_file(path_str)
    return strip_prefix(state_dict)


def _strip_prefix_keys(keys: Sequence[str]) -> tuple[dict[str, str], str | None]:
    prefix = None
    for pfx in ("model.diffusion_model.", "model."):
        if any(key.startswith(pfx) for key in keys):
            prefix = pfx
            break
    if prefix is None:
        for pfx in ("net.",):
            if all(key.startswith(pfx) for key in keys):
                prefix = pfx
                break

    if prefix is None:
        return {key: key for key in keys}, None

    logging.info("State dict prefix found: %r", prefix)
    return {key.removeprefix(prefix): key for key in keys if key.startswith(prefix)}, prefix


def _is_safetensors_path(path: Path) -> bool:
    return path.suffix.lower() == ".safetensors"


_NUMPY_DTYPE_TAGS: dict[np.dtype[Any], str] = {
    np.dtype(np.float64): "F64",
    np.dtype(np.float32): "F32",
    np.dtype(np.float16): "F16",
    np.dtype(np.int64): "I64",
    np.dtype(np.int32): "I32",
    np.dtype(np.int16): "I16",
    np.dtype(np.int8): "I8",
    np.dtype(np.uint8): "U8",
    np.dtype(np.bool_): "BOOL",
}


def _dtype_tag(dtype: Any) -> str:
    if isinstance(dtype, str):
        return dtype
    try:
        return _NUMPY_DTYPE_TAGS[np.dtype(dtype)]
    except TypeError:
        pass
    except KeyError as exc:
        raise ValueError(f"Unsupported NumPy dtype: {dtype}") from exc

    name = str(dtype)
    torch_tags = {
        "torch.float64": "F64",
        "torch.float32": "F32",
        "torch.float16": "F16",
        "torch.bfloat16": "BF16",
        "torch.int64": "I64",
        "torch.int32": "I32",
        "torch.int16": "I16",
        "torch.int8": "I8",
        "torch.uint8": "U8",
        "torch.bool": "BOOL",
    }
    try:
        return torch_tags[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported tensor dtype: {dtype}") from exc


def _read_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as f:
        header_len_data = f.read(8)
        if len(header_len_data) != 8:
            raise ValueError(f"Invalid safetensors file: {path}")
        header_len = struct.unpack("<Q", header_len_data)[0]
        header = json.loads(f.read(header_len))
    return 8 + header_len, header


def _load_bf16_safetensors_tensor(file_bytes: np.ndarray, data_start: int, header: Mapping[str, Any], key: str) -> np.ndarray:
    info = header[key]
    if info.get("dtype") != "BF16":
        raise ValueError(f"Expected BF16 safetensors tensor for {key!r}")
    shape = tuple(int(dim) for dim in info["shape"])
    begin, end = (int(offset) for offset in info["data_offsets"])
    nbytes = end - begin
    if nbytes != int(np.prod(shape, dtype=np.int64)) * 2:
        raise ValueError(f"Invalid BF16 byte length for tensor {key!r}")
    tensor_bytes = file_bytes[data_start + begin:data_start + end]
    if tensor_bytes.nbytes != nbytes:
        raise ValueError(f"Invalid BF16 byte range for tensor {key!r}")
    return tensor_bytes.view(np.uint16).reshape(shape)


def parse_qtype(qtype: str | Any) -> tuple[str, str]:
    if not isinstance(qtype, str):
        name = getattr(qtype, "name", None)
        if name is None:
            raise ValueError(f"Unsupported quantization type: {qtype!r}")
        qtype = name
    file_type = FILE_TYPE_ALIASES.get(qtype.upper(), qtype.upper())
    tensor_qtype = FILE_TYPE_TO_TENSOR_QTYPE.get(file_type, file_type)
    if tensor_qtype not in SUPPORTED_QUANT_QTYPES:
        raise ValueError(f"Unsupported direct quantization type: {qtype}")
    return file_type, tensor_qtype


def parse_tensor_qtype(qtype: str | Any) -> str:
    if not isinstance(qtype, str):
        name = getattr(qtype, "name", None)
        if name is None:
            raise ValueError(f"Unsupported tensor type: {qtype!r}")
        qtype = name
    qtype_name = FILE_TYPE_TO_TENSOR_QTYPE.get(FILE_TYPE_ALIASES.get(qtype.upper(), qtype.upper()), qtype.upper())
    if qtype_name not in SUPPORTED_STORAGE_QTYPES:
        raise ValueError(f"Unsupported GGML tensor type: {qtype}")
    return qtype_name


def _enum_member(enum_cls: Any, name: str) -> Any:
    try:
        return enum_cls[name]
    except KeyError as exc:
        raise ValueError(f"Installed gguf package does not provide {enum_cls.__name__}.{name}") from exc


def _file_type_enum(file_type_name: str) -> Any:
    return _enum_member(LlamaFileType, f"MOSTLY_{file_type_name}")


def _tensor_qtype_enum(qtype_name: str) -> Any:
    return _enum_member(GGMLQuantizationType, qtype_name)


def _default_output_path(src: Path, file_type_name: str) -> Path:
    return src.with_name(f"{src.stem}-{file_type_name}.gguf")


def _is_quantized_qtype(qtype: Any) -> bool:
    if qtype in {GGMLQuantizationType.F32, GGMLQuantizationType.F16, GGMLQuantizationType.BF16}:
        return False
    return True


def _base_storage_type_for_meta(
    key: str,
    dtype: str,
    ndim: int,
    model_arch: ModelTemplate,
    n_params: int,
) -> Any:
    if dtype == "BF16":
        data_qtype = GGMLQuantizationType.BF16
    else:
        data_qtype = GGMLQuantizationType.F16

    if dtype in {"F32", "BF16"}:
        if ndim == 1 or n_params <= QUANTIZATION_THRESHOLD or any(marker in key for marker in model_arch.keys_hiprec):
            data_qtype = GGMLQuantizationType.F32
    return data_qtype


def _bf16_bits_to_float32(tensor: np.ndarray) -> np.ndarray:
    return (tensor.astype(np.uint32) << np.uint32(16)).view(np.float32).reshape(tensor.shape)


def _is_direct_storage_array(tensor: Any, dtype: Any) -> bool:
    return (
        isinstance(tensor, np.ndarray)
        and tensor.dtype == np.dtype(dtype)
        and tensor.dtype.isnative
        and tensor.flags.c_contiguous
    )


def _to_numpy_for_qtype(tensor: Any, qtype: Any, source_dtype: str | None = None) -> np.ndarray:
    if isinstance(tensor, np.ndarray):
        if source_dtype == "BF16" and tensor.dtype == np.dtype(np.uint16):
            tensor = _bf16_bits_to_float32(tensor)
        if qtype == GGMLQuantizationType.F32:
            return np.ascontiguousarray(tensor, dtype=np.float32)
        if qtype == GGMLQuantizationType.F16:
            return np.ascontiguousarray(tensor, dtype=np.float16)
        return np.ascontiguousarray(tensor, dtype=np.float32)

    try:
        import torch
    except ImportError as exc:
        raise TypeError(f"Unsupported tensor object without torch installed: {type(tensor)!r}") from exc

    if not isinstance(tensor, torch.Tensor):
        return np.ascontiguousarray(tensor, dtype=np.float32)
    if qtype == GGMLQuantizationType.F32:
        return tensor.to(torch.float32).numpy()
    if qtype == GGMLQuantizationType.F16:
        return tensor.to(torch.float16).numpy()
    if qtype == GGMLQuantizationType.BF16:
        return tensor.to(torch.float32).numpy()
    return tensor.to(torch.float32).numpy()


def _native_store_rows(data: np.ndarray, qtype: Any) -> np.ndarray:
    import libgguf

    return libgguf.store_rows(data, qtype)


def _native_quantize_rows(data: np.ndarray, qtype: Any, imatrix: np.ndarray | None) -> np.ndarray:
    import libgguf

    return libgguf.quantize_rows(data, qtype, imatrix=imatrix)


def _policy_allows_quant_shape(key: str, shape: tuple[int, ...], model_arch: ModelTemplate, policy: str) -> bool:
    if policy not in {"comfy", "uniform"}:
        raise ValueError("policy must be 'comfy' or 'uniform'")
    if len(shape) != 2:
        return False
    if not key.endswith("weight"):
        return False
    if policy == "uniform":
        return True
    if model_arch.arch in IMAGE_ARCHES and _matches(key, ARCH_SKIP_PATTERNS.get(model_arch.arch, ())):
        return False
    return True


def _mixed_policy_qtype(base_file_type: str, base_qtype_name: str, key: str, counters: dict[str, int]) -> str:
    qtype = base_qtype_name
    if _matches(key, ATTENTION_VALUE_PATTERNS):
        if base_file_type == "Q2_K":
            qtype = "Q3_K"
        elif base_file_type == "Q3_K_M":
            qtype = "Q5_K" if counters["attention_value"] < 2 else "Q4_K"
        elif base_file_type == "Q3_K_L":
            qtype = "Q5_K"
        elif base_file_type in {"Q4_K_M", "Q5_K_M"}:
            qtype = "Q6_K"
        elif base_file_type == "Q4_K_S" and counters["attention_value"] < 4:
            qtype = "Q5_K"
        counters["attention_value"] += 1
    elif _matches(key, FUSED_QKV_PATTERNS):
        if base_file_type in {"Q3_K_M", "Q3_K_L"}:
            qtype = "Q4_K"
        elif base_file_type == "Q4_K_M":
            qtype = "Q5_K"
        elif base_file_type == "Q5_K_M":
            qtype = "Q6_K"
    elif _matches(key, FFN_DOWN_PATTERNS):
        if base_file_type == "Q3_K_M":
            qtype = "Q4_K"
        elif base_file_type == "Q3_K_L":
            qtype = "Q5_K"
        elif base_file_type == "Q4_K_S":
            qtype = "Q5_K"
        elif base_file_type in {"Q4_K_M", "Q5_K_M"}:
            qtype = "Q6_K"
        elif base_file_type == "Q4_0":
            qtype = "Q4_1"
        elif base_file_type == "Q5_0":
            qtype = "Q5_1"
        counters["ffn_down"] += 1
    return qtype


def _normalize_overrides(overrides: Mapping[str, str] | Sequence[tuple[str, str]] | None) -> list[tuple[str, str]]:
    if overrides is None:
        return []
    items = overrides.items() if isinstance(overrides, Mapping) else overrides
    return [(pattern, qtype.upper()) for pattern, qtype in items]


def _override_qtype(key: str, overrides: Sequence[tuple[str, str]]) -> str | None:
    for pattern, qtype in overrides:
        if fnmatchcase(key, pattern):
            return qtype
    return None


def _shape_fix_shape(writer: gguf.GGUFWriter, key: str, shape: tuple[int, ...], model_arch: ModelTemplate) -> tuple[int, ...]:
    n_params = int(np.prod(shape, dtype=np.int64))
    if (
        model_arch.shape_fix
        and len(shape) > 1
        and n_params >= REARRANGE_THRESHOLD
        and n_params % 256 == 0
        and shape[-1] % 256 != 0
    ):
        writer.add_array(f"comfy.gguf.orig_shape.{key}", tuple(int(dim) for dim in shape))
        return (n_params // 256, 256)
    return shape


def _qtype_nbytes(shape: tuple[int, ...], qtype: Any) -> int:
    n_params = int(np.prod(shape, dtype=np.int64))
    block_size, type_size = GGML_QUANT_SIZES[qtype]
    if n_params % block_size != 0:
        raise ValueError(f"Tensor with shape {shape} cannot be stored as {qtype.name}")
    return n_params // block_size * type_size


def _writer_info_shape(shape: tuple[int, ...], qtype: Any, nbytes: int) -> tuple[int, ...]:
    if qtype in {GGMLQuantizationType.F32, GGMLQuantizationType.F16}:
        return shape
    if not shape:
        return (nbytes,)
    return (*shape[:-1], nbytes // max(1, int(np.prod(shape[:-1], dtype=np.int64))))


def _writer_info_dtype(qtype: Any) -> np.dtype[Any]:
    if qtype == GGMLQuantizationType.F32:
        return np.dtype(np.float32)
    if qtype == GGMLQuantizationType.F16:
        return np.dtype(np.float16)
    return np.dtype(np.uint8)


def _add_tensor_info(writer: gguf.GGUFWriter, key: str, shape: tuple[int, ...], qtype: Any) -> None:
    nbytes = _qtype_nbytes(shape, qtype)
    writer.add_tensor_info(
        key,
        _writer_info_shape(shape, qtype, nbytes),
        _writer_info_dtype(qtype),
        nbytes,
        raw_dtype=qtype,
    )


def _unquantized_tensor_data(tensor: Any, source_dtype: str, qtype: Any) -> np.ndarray:
    if qtype == GGMLQuantizationType.F32 and _is_direct_storage_array(tensor, np.float32):
        return tensor
    if qtype == GGMLQuantizationType.F16 and _is_direct_storage_array(tensor, np.float16):
        return tensor
    if qtype == GGMLQuantizationType.BF16 and source_dtype == "BF16" and _is_direct_storage_array(tensor, np.uint16):
        return tensor

    data = _to_numpy_for_qtype(tensor, qtype, source_dtype=source_dtype)
    if qtype in {GGMLQuantizationType.F32, GGMLQuantizationType.F16, GGMLQuantizationType.BF16}:
        return _native_store_rows(data, qtype)
    raise ValueError(f"Unknown unquantized type: {qtype}")


def _quantized_tensor_data(tensor: Any, source_dtype: str, qtype: Any, imatrix: np.ndarray | None) -> np.ndarray:
    data = _to_numpy_for_qtype(tensor, qtype, source_dtype=source_dtype)
    return _native_quantize_rows(data, qtype, imatrix=imatrix)


@contextmanager
def _open_tensor_source(path: Path) -> Iterator[tuple[Mapping[str, str], TensorSource]]:
    if _is_safetensors_path(path):
        data_start, header = _read_safetensors_header(path)
        file_bytes = np.memmap(path, dtype=np.uint8, mode="r")
        with safe_open(os.fspath(path), framework="np") as handle:
            keys = list(handle.keys())
            key_map, _ = _strip_prefix_keys(keys)

            def load_bf16_tensor(source_key: str) -> np.ndarray:
                return _load_bf16_safetensors_tensor(file_bytes, data_start, header, source_key)

            class SafetensorsSource:
                def keys(self) -> Sequence[str]:
                    return tuple(key_map)

                def tensor_meta(self, key: str) -> tuple[tuple[int, ...], str]:
                    tensor_slice = handle.get_slice(key_map[key])
                    return tuple(int(dim) for dim in tensor_slice.get_shape()), tensor_slice.get_dtype()

                def load_tensor(self, key: str) -> Any:
                    source_key = key_map[key]
                    if header[source_key]["dtype"] == "BF16":
                        return load_bf16_tensor(source_key)
                    return handle.get_tensor(source_key)

            yield key_map, SafetensorsSource()
        return

    state_dict = load_state_dict(path)

    class EagerSource:
        def keys(self) -> Sequence[str]:
            return tuple(state_dict)

        def tensor_meta(self, key: str) -> tuple[tuple[int, ...], str]:
            tensor = state_dict[key]
            return tuple(int(dim) for dim in tensor.shape), _dtype_tag(tensor.dtype)

        def load_tensor(self, key: str) -> Any:
            return state_dict[key]

    yield {key: key for key in state_dict}, EagerSource()


@contextmanager
def _open_safetensors_metadata_source(path: Path) -> Iterator[tuple[Mapping[str, str], TensorSource]]:
    if not _is_safetensors_path(path):
        raise ValueError("Native GGUF conversion only supports .safetensors inputs")

    with safe_open(os.fspath(path), framework="np") as handle:
        keys = list(handle.keys())
        key_map, _ = _strip_prefix_keys(keys)

        class SafetensorsMetadataSource:
            def keys(self) -> Sequence[str]:
                return tuple(key_map)

            def tensor_meta(self, key: str) -> tuple[tuple[int, ...], str]:
                tensor_slice = handle.get_slice(key_map[key])
                return tuple(int(dim) for dim in tensor_slice.get_shape()), tensor_slice.get_dtype()

            def load_tensor(self, key: str) -> Any:
                raise RuntimeError("Metadata-only safetensors source cannot load tensor data")

        yield key_map, SafetensorsMetadataSource()


def _prepare_conversion(
    file_type: Any,
    file_type_name: str,
    base_qtype_name: str,
    key_map: Mapping[str, str],
    tensor_source: TensorSource,
    *,
    policy: str,
    imatrix_data: Mapping[str, np.ndarray],
    overrides: Sequence[tuple[str, str]],
    include: Sequence[str] | None,
    exclude: Sequence[str] | None,
) -> tuple[gguf.GGUFWriter, ModelTemplate, list[TensorPlan], dict[str, int], dict[str, int]]:
    keys = tuple(tensor_source.keys())
    model_arch = detect_arch(set(keys))
    name_lengths = sorted(((key, len(key)) for key in keys), key=lambda item: item[1], reverse=True)
    if name_lengths and name_lengths[0][1] > MAX_TENSOR_NAME_LENGTH:
        bad = ", ".join(f"{key!r} ({length})" for key, length in name_lengths if length > MAX_TENSOR_NAME_LENGTH)
        raise ValueError(f"Can only handle tensor names up to {MAX_TENSOR_NAME_LENGTH} characters. Tensors exceeding the limit: {bad}")

    writer = gguf.GGUFWriter(path=None, arch=model_arch.arch)
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    writer.add_file_type(file_type)

    tensor_counts: dict[str, int] = {}
    fallback_counts: dict[str, int] = {}
    counters = {"attention_value": 0, "ffn_down": 0}
    plans: list[TensorPlan] = []

    for key in keys:
        if any(marker in key for marker in model_arch.keys_ignore):
            logging.info("Filtering ignored key: %r", key)
            continue

        source_shape, source_dtype = tensor_source.tensor_meta(key)
        n_params = int(np.prod(source_shape, dtype=np.int64))
        storage_qtype = _base_storage_type_for_meta(key, source_dtype, len(source_shape), model_arch, n_params)
        write_shape = _shape_fix_shape(writer, key, source_shape, model_arch)

        quantize = _policy_allows_quant_shape(key, write_shape, model_arch, policy)
        if _matches(key, include) and len(write_shape) == 2:
            quantize = True
        if _matches(key, exclude):
            quantize = False

        forced_qtype = _override_qtype(key, overrides)
        target_qtype = storage_qtype
        if forced_qtype is not None:
            forced_tensor_qtype_name = parse_tensor_qtype(forced_qtype)
            target_qtype = _tensor_qtype_enum(forced_tensor_qtype_name)
            quantize = _is_quantized_qtype(target_qtype)
        elif quantize:
            target_name = base_qtype_name
            if policy == "comfy":
                target_name = _mixed_policy_qtype(file_type_name, base_qtype_name, key, counters)
            target_qtype = _tensor_qtype_enum(target_name)

        if quantize and _is_quantized_qtype(target_qtype):
            block_size = GGML_QUANT_SIZES[target_qtype][0]
            if write_shape[-1] % block_size != 0:
                fallback_counts[target_qtype.name] = fallback_counts.get(target_qtype.name, 0) + 1
                target_qtype = GGMLQuantizationType.F16
                quantize = False

        plan = TensorPlan(
            key=key,
            source_key=key_map[key],
            source_shape=source_shape,
            source_dtype=source_dtype,
            write_shape=write_shape,
            target_qtype=target_qtype,
            quantize=quantize and _is_quantized_qtype(target_qtype),
            imatrix=imatrix_data.get(key),
        )
        plans.append(plan)
        _add_tensor_info(writer, key, write_shape, target_qtype)
        tensor_counts[target_qtype.name] = tensor_counts.get(target_qtype.name, 0) + 1
        logging.info("%s -> %s, shape=%s", key, target_qtype.name, write_shape)

    return writer, model_arch, plans, tensor_counts, fallback_counts


def convert_to_gguf(
    src: str | os.PathLike[str],
    dst: str | os.PathLike[str] | None = None,
    qtype: str | Any = "Q4_K_S",
    *,
    policy: str = "comfy",
    overwrite: bool = False,
    imatrix: str | os.PathLike[str] | Mapping[str, np.ndarray] | None = None,
    tensor_overrides: Mapping[str, str] | Sequence[tuple[str, str]] | None = None,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
) -> QuantResult:
    src_path = Path(src)
    file_type_name, base_qtype_name = parse_qtype(qtype)
    file_type = _file_type_enum(file_type_name)
    dst_path = Path(dst) if dst is not None else _default_output_path(src_path, file_type_name)

    if dst_path.exists() and not overwrite:
        raise OSError(f"Output exists and overwriting is disabled: {dst_path}")

    imatrix_data = load_imatrix(imatrix) if isinstance(imatrix, (str, os.PathLike)) else dict(imatrix or {})
    overrides = _normalize_overrides(tensor_overrides)

    with _open_tensor_source(src_path) as (key_map, tensor_source):
        writer, model_arch, plans, tensor_counts, fallback_counts = _prepare_conversion(
            file_type,
            file_type_name,
            base_qtype_name,
            key_map,
            tensor_source,
            policy=policy,
            imatrix_data=imatrix_data,
            overrides=overrides,
            include=include,
            exclude=exclude,
        )

        writer.write_header_to_file(path=dst_path)
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()

        try:
            for plan in tqdm(plans):
                tensor = tensor_source.load_tensor(plan.key)
                if plan.write_shape != plan.source_shape:
                    if plan.source_dtype == "BF16" and isinstance(tensor, np.ndarray) and tensor.dtype == np.dtype(np.uint16):
                        tensor = tensor.reshape(plan.write_shape)
                    else:
                        tensor = _to_numpy_for_qtype(tensor, plan.target_qtype, source_dtype=plan.source_dtype).reshape(plan.write_shape)

                if plan.quantize:
                    data = _quantized_tensor_data(tensor, plan.source_dtype, plan.target_qtype, plan.imatrix)
                else:
                    data = _unquantized_tensor_data(tensor, plan.source_dtype, plan.target_qtype)
                writer.write_tensor_data(data)
                del data, tensor
        finally:
            writer.close()

    return QuantResult(
        output_path=dst_path,
        arch=model_arch.arch,
        file_type=file_type.name,
        tensor_type_counts=tensor_counts,
        fallback_counts=fallback_counts,
    )


def _native_payload_plan(plan: TensorPlan, header: Mapping[str, Any], data_start: int) -> dict[str, Any]:
    info = header[plan.source_key]
    if info.get("dtype") != plan.source_dtype:
        raise ValueError(f"Plan/source dtype mismatch for tensor {plan.key!r}")
    begin, end = (int(offset) for offset in info["data_offsets"])
    imatrix = None
    if plan.imatrix is not None:
        imatrix = np.ascontiguousarray(plan.imatrix, dtype=np.float32)
    return {
        "key": plan.key,
        "source_dtype": plan.source_dtype,
        "source_shape": plan.source_shape,
        "write_shape": plan.write_shape,
        "qtype": int(plan.target_qtype),
        "nbytes": _qtype_nbytes(plan.write_shape, plan.target_qtype),
        "data_begin": int(data_start + begin),
        "data_end": int(data_start + end),
        "imatrix": imatrix,
    }


def convert_safetensors_to_gguf_native(
    src: str | os.PathLike[str],
    dst: str | os.PathLike[str] | None = None,
    qtype: str | Any = "Q4_K_S",
    *,
    policy: str = "comfy",
    overwrite: bool = False,
    imatrix: str | os.PathLike[str] | Mapping[str, np.ndarray] | None = None,
    tensor_overrides: Mapping[str, str] | Sequence[tuple[str, str]] | None = None,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    scratch_bytes: int = NATIVE_DEFAULT_SCRATCH_BYTES,
) -> QuantResult:
    from . import _libgguf

    src_path = Path(src)
    if not _is_safetensors_path(src_path):
        raise ValueError("Native GGUF conversion only supports .safetensors inputs")

    file_type_name, base_qtype_name = parse_qtype(qtype)
    file_type = _file_type_enum(file_type_name)
    dst_path = Path(dst) if dst is not None else _default_output_path(src_path, file_type_name)

    if dst_path.exists() and not overwrite:
        raise OSError(f"Output exists and overwriting is disabled: {dst_path}")

    data_start, header = _read_safetensors_header(src_path)
    imatrix_data = load_imatrix(imatrix) if isinstance(imatrix, (str, os.PathLike)) else dict(imatrix or {})
    overrides = _normalize_overrides(tensor_overrides)

    with _open_safetensors_metadata_source(src_path) as (key_map, tensor_source):
        writer, model_arch, plans, tensor_counts, fallback_counts = _prepare_conversion(
            file_type,
            file_type_name,
            base_qtype_name,
            key_map,
            tensor_source,
            policy=policy,
            imatrix_data=imatrix_data,
            overrides=overrides,
            include=include,
            exclude=exclude,
        )

        native_plans = [_native_payload_plan(plan, header, data_start) for plan in plans]

        writer.write_header_to_file(path=dst_path)
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()

        try:
            if writer.fout is None or len(writer.fout) != 1:
                raise ValueError("Native GGUF conversion currently supports single-file outputs only")
            fout = writer.fout[0]
            fout.flush()
            _libgguf.write_safetensors_payload(
                os.fspath(src_path),
                fout.fileno(),
                native_plans,
                int(getattr(writer, "data_alignment", 32)),
                scratch_bytes=int(scratch_bytes),
            )
            fout.flush()
            for tensors in writer.tensors:
                tensors.clear()
        finally:
            writer.close()

    return QuantResult(
        output_path=dst_path,
        arch=model_arch.arch,
        file_type=file_type.name,
        tensor_type_counts=tensor_counts,
        fallback_counts=fallback_counts,
    )


__all__ = [
    "QuantResult",
    "NATIVE_DEFAULT_SCRATCH_BYTES",
    "convert_safetensors_to_gguf_native",
    "convert_to_gguf",
    "detect_arch",
    "load_state_dict",
    "parse_qtype",
    "parse_tensor_qtype",
    "strip_prefix",
]
