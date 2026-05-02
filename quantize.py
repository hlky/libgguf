from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
import logging
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


QUANTIZATION_THRESHOLD = 1024
REARRANGE_THRESHOLD = 512
MAX_TENSOR_NAME_LENGTH = 127


@dataclass(frozen=True)
class QuantResult:
    output_path: Path
    arch: str
    file_type: str
    tensor_type_counts: dict[str, int]
    fallback_counts: dict[str, int]


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
    "Q4_K": "Q4_K_M",
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


def _lazy_imports() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        import gguf
        import libgguf
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError(
            "Direct GGUF quantization requires torch, gguf, safetensors, and the editable libgguf package."
        ) from exc
    return torch, gguf, libgguf, load_file


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
    torch, _, _, load_file = _lazy_imports()
    path_str = os.fspath(path)
    if path_str.endswith((".ckpt", ".pt", ".bin", ".pth")):
        state_dict = torch.load(path_str, map_location="cpu", weights_only=True)
        for subkey in ("model", "module"):
            if subkey in state_dict:
                state_dict = state_dict[subkey]
                break
        if len(state_dict) < 20:
            raise RuntimeError(f"pt subkey load failed: {state_dict.keys()}")
    else:
        state_dict = load_file(path_str)
    return strip_prefix(state_dict)


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


def _file_type_enum(gguf: Any, file_type_name: str) -> Any:
    return _enum_member(gguf.LlamaFileType, f"MOSTLY_{file_type_name}")


def _tensor_qtype_enum(gguf: Any, qtype_name: str) -> Any:
    return _enum_member(gguf.GGMLQuantizationType, qtype_name)


def _default_output_path(src: Path, file_type_name: str) -> Path:
    return src.with_name(f"{src.stem}-{file_type_name}.gguf")


def _is_quantized_qtype(gguf: Any, qtype: Any) -> bool:
    if qtype in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16}:
        return False
    return True


def _base_storage_type(torch: Any, gguf: Any, key: str, tensor: Any, model_arch: ModelTemplate, n_params: int) -> Any:
    if tensor.dtype == torch.bfloat16:
        data_qtype = gguf.GGMLQuantizationType.BF16
    else:
        data_qtype = gguf.GGMLQuantizationType.F16

    if tensor.dtype in (torch.float32, torch.bfloat16):
        if tensor.ndim == 1 or n_params <= QUANTIZATION_THRESHOLD or any(marker in key for marker in model_arch.keys_hiprec):
            data_qtype = gguf.GGMLQuantizationType.F32
    return data_qtype


def _to_numpy_for_qtype(torch: Any, gguf: Any, tensor: Any, qtype: Any) -> np.ndarray:
    if qtype == gguf.GGMLQuantizationType.F32:
        return tensor.to(torch.float32).numpy()
    if qtype == gguf.GGMLQuantizationType.F16:
        return tensor.to(torch.float16).numpy()
    if qtype == gguf.GGMLQuantizationType.BF16:
        return tensor.to(torch.float32).numpy()
    return tensor.to(torch.float32).numpy()


def _policy_allows_quant(key: str, tensor: Any, model_arch: ModelTemplate, policy: str) -> bool:
    if policy not in {"comfy", "uniform"}:
        raise ValueError("policy must be 'comfy' or 'uniform'")
    if tensor.ndim != 2:
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


def _maybe_shape_fix(writer: Any, gguf: Any, key: str, data: np.ndarray, model_arch: ModelTemplate) -> np.ndarray:
    n_params = int(np.prod(data.shape, dtype=np.int64))
    if (
        model_arch.shape_fix
        and data.ndim > 1
        and n_params >= REARRANGE_THRESHOLD
        and n_params % 256 == 0
        and data.shape[-1] % 256 != 0
    ):
        orig_shape = data.shape
        data = data.reshape(n_params // 256, 256)
        writer.add_array(f"comfy.gguf.orig_shape.{key}", tuple(int(dim) for dim in orig_shape))
    return data


def _add_unquantized_tensor(writer: Any, gguf: Any, torch: Any, key: str, tensor: Any, qtype: Any) -> None:
    data = _to_numpy_for_qtype(torch, gguf, tensor, qtype)
    data = gguf.quants.quantize(data, qtype)
    writer.add_tensor(key, data, raw_dtype=qtype)


def _add_quantized_tensor(
    writer: Any,
    gguf: Any,
    libgguf: Any,
    torch: Any,
    key: str,
    tensor: Any,
    qtype: Any,
    imatrix: np.ndarray | None,
) -> None:
    data = _to_numpy_for_qtype(torch, gguf, tensor, qtype)
    quantized = libgguf.quantize_rows(data, qtype, imatrix=imatrix)
    writer.add_tensor(key, quantized, raw_dtype=qtype)


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
    torch, gguf, libgguf, _ = _lazy_imports()
    src_path = Path(src)
    file_type_name, base_qtype_name = parse_qtype(qtype)
    base_qtype = _tensor_qtype_enum(gguf, base_qtype_name)
    file_type = _file_type_enum(gguf, file_type_name)
    dst_path = Path(dst) if dst is not None else _default_output_path(src_path, file_type_name)

    if dst_path.exists() and not overwrite:
        raise OSError(f"Output exists and overwriting is disabled: {dst_path}")

    state_dict = load_state_dict(src_path)
    model_arch = detect_arch(state_dict)
    name_lengths = sorted(((key, len(key)) for key in state_dict), key=lambda item: item[1], reverse=True)
    if name_lengths and name_lengths[0][1] > MAX_TENSOR_NAME_LENGTH:
        bad = ", ".join(f"{key!r} ({length})" for key, length in name_lengths if length > MAX_TENSOR_NAME_LENGTH)
        raise ValueError(f"Can only handle tensor names up to {MAX_TENSOR_NAME_LENGTH} characters. Tensors exceeding the limit: {bad}")

    imatrix_data = libgguf.load_imatrix(imatrix) if isinstance(imatrix, (str, os.PathLike)) else dict(imatrix or {})
    overrides = _normalize_overrides(tensor_overrides)
    writer = gguf.GGUFWriter(path=None, arch=model_arch.arch)
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    writer.add_file_type(file_type)

    tensor_counts: dict[str, int] = {}
    fallback_counts: dict[str, int] = {}
    counters = {"attention_value": 0, "ffn_down": 0}

    for key, tensor in state_dict.items():
        if any(marker in key for marker in model_arch.keys_ignore):
            logging.info("Filtering ignored key: %r", key)
            continue

        n_params = int(tensor.numel())
        storage_qtype = _base_storage_type(torch, gguf, key, tensor, model_arch, n_params)
        data_for_shape = _to_numpy_for_qtype(torch, gguf, tensor, storage_qtype)
        data_for_shape = _maybe_shape_fix(writer, gguf, key, data_for_shape, model_arch)
        if data_for_shape.shape != tuple(tensor.shape):
            tensor = torch.from_numpy(data_for_shape)

        quantize = _policy_allows_quant(key, tensor, model_arch, policy)
        if _matches(key, include) and tensor.ndim == 2:
            quantize = True
        if _matches(key, exclude):
            quantize = False

        forced_qtype = _override_qtype(key, overrides)
        target_qtype = storage_qtype
        if forced_qtype is not None:
            forced_tensor_qtype_name = parse_tensor_qtype(forced_qtype)
            target_qtype = _tensor_qtype_enum(gguf, forced_tensor_qtype_name)
            quantize = _is_quantized_qtype(gguf, target_qtype)
        elif quantize:
            target_name = base_qtype_name
            if policy == "comfy":
                target_name = _mixed_policy_qtype(file_type_name, base_qtype_name, key, counters)
            target_qtype = _tensor_qtype_enum(gguf, target_name)

        if quantize and _is_quantized_qtype(gguf, target_qtype):
            block_size = gguf.GGML_QUANT_SIZES[target_qtype][0]
            if tensor.shape[-1] % block_size != 0:
                fallback_counts[target_qtype.name] = fallback_counts.get(target_qtype.name, 0) + 1
                target_qtype = gguf.GGMLQuantizationType.F16
                quantize = False

        if quantize and _is_quantized_qtype(gguf, target_qtype):
            tensor_imatrix = imatrix_data.get(key)
            _add_quantized_tensor(writer, gguf, libgguf, torch, key, tensor, target_qtype, tensor_imatrix)
        else:
            _add_unquantized_tensor(writer, gguf, torch, key, tensor, target_qtype)

        tensor_counts[target_qtype.name] = tensor_counts.get(target_qtype.name, 0) + 1
        logging.info("%s -> %s, shape=%s", key, target_qtype.name, tuple(tensor.shape))

    writer.write_header_to_file(path=dst_path)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
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
    "convert_to_gguf",
    "detect_arch",
    "load_state_dict",
    "parse_qtype",
    "parse_tensor_qtype",
    "strip_prefix",
]
