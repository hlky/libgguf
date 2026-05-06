from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.file_download import hf_hub_url
import requests


LOGGER = logging.getLogger("compare_hf_gguf")

DEFAULT_FULL_INVENTORY = Path(".cache/gguf_inventory_full.json")
DEFAULT_REPORT_DIR = Path("reports/gguf_comparison")

UNSLOTH_REPOS = (
    "unsloth/Z-Image-Turbo-GGUF",
    "unsloth/Z-Image-GGUF",
    "unsloth/Qwen-Image-2512-GGUF",
    "unsloth/ERNIE-Image-Turbo-GGUF",
    "unsloth/ERNIE-Image-GGUF",
)

QTYPE_NAMES = (
    "BF16",
    "F16",
    "F32",
    "Q1_0",
    "Q2_K_S",
    "Q2_K",
    "Q3_K_S",
    "Q3_K_M",
    "Q3_K_L",
    "Q3_K",
    "Q4_0",
    "Q4_1",
    "Q4_K_S",
    "Q4_K_M",
    "Q4_K",
    "Q5_0",
    "Q5_1",
    "Q5_K_S",
    "Q5_K_M",
    "Q5_K",
    "Q6_K",
    "Q8_0",
    "Q8_K",
    "IQ1_S",
    "IQ1_M",
    "IQ2_XXS",
    "IQ2_XS",
    "IQ2_S",
    "IQ2_M",
    "IQ3_XXS",
    "IQ3_S",
    "IQ3_M",
    "IQ4_NL",
    "IQ4_XS",
    "TQ1_0",
    "TQ2_0",
    "MXFP4",
    "MXFP4_MOE",
    "NVFP4",
)

EXE_QTYPES = {
    "Q1_0",
    "Q4_0",
    "Q4_1",
    "Q5_0",
    "Q5_1",
    "Q8_0",
    "Q2_K",
    "Q3_K",
    "Q3_K_S",
    "Q3_K_M",
    "Q3_K_L",
    "Q4_K",
    "Q4_K_S",
    "Q4_K_M",
    "Q5_K",
    "Q5_K_S",
    "Q5_K_M",
    "Q6_K",
}

FLOAT_QTYPE_VARIANTS = {"BF16", "F16", "F32"}
UD_QTYPE_VARIANTS = {"UD-Q2_K", "UD-Q3_K_M", "UD-Q4_K_M", "UD-Q5_K_M"}
STANDARD_QTYPE_VARIANTS = {
    "Q2_K",
    "Q3_K_L",
    "Q3_K_M",
    "Q3_K_S",
    "Q4_0",
    "Q4_1",
    "Q4_K_M",
    "Q4_K_S",
    "Q5_0",
    "Q5_1",
    "Q5_K_M",
    "Q5_K_S",
    "Q6_K",
    "Q8_0",
}

SOURCE_REJECT_PARTS = (
    "clip",
    "controlnet",
    "loras/",
    "model_patches/",
    "prompt-enhancer",
    "text_encoders/",
    "vae/",
)
SOURCE_ACCEPT_PARTS = ("diffusion_models/", "transformer", "unet")
SOURCE_VARIANT_TOKENS = {
    "canny",
    "control",
    "depth",
    "distill",
    "distilled",
    "edit",
    "fill",
    "fun",
    "inp",
    "instantx",
    "kontext",
    "layered",
    "lora",
    "patch",
    "redux",
    "turbo",
}
SOURCE_DTYPE_PREFERENCE = ("bf16", "fp16", "f16")
SOURCE_DTYPE_PENALTY = ("fp8", "nvfp4", "fp4")
SOURCE_CANDIDATE_LIMIT = 30

SOURCE_ALIAS_OVERRIDES = {
    "city96/FLUX.1-dev-gguf": {"flux1dev"},
    "city96/FLUX.1-schnell-gguf": {"flux1schnell"},
    "city96/FLUX.2-dev-gguf": {"flux2dev"},
    "city96/Qwen-Image-gguf": {"qwenimage"},
    "city96/t5-v1_1-xxl-encoder-gguf": {"t5xxl", "t5v11xxl"},
    "city96/umt5-xxl-encoder-gguf": {"umt5xxl"},
    "unsloth/ERNIE-Image-GGUF": {"ernieimage"},
    "unsloth/ERNIE-Image-Turbo-GGUF": {"ernieimageturbo"},
    "unsloth/Qwen-Image-2512-GGUF": {"qwenimage2512"},
    "unsloth/Z-Image-GGUF": {"zimage"},
    "unsloth/Z-Image-Turbo-GGUF": {"zimageturbo"},
}


GGUF_VALUE_TYPE_NAMES = {
    0: "UINT8",
    1: "INT8",
    2: "UINT16",
    3: "INT16",
    4: "UINT32",
    5: "INT32",
    6: "FLOAT32",
    7: "BOOL",
    8: "STRING",
    9: "ARRAY",
    10: "UINT64",
    11: "INT64",
    12: "FLOAT64",
}

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


@dataclass(frozen=True)
class FileInfo:
    path: str
    size: int | None
    qtype: str | None
    qtype_variant: str | None


@dataclass(frozen=True)
class SourceCandidate:
    repo_id: str
    path: str
    size: int | None
    score: int


@dataclass(frozen=True)
class RepoInventory:
    repo_id: str
    base_models: list[str]
    gguf_files: list[FileInfo]
    qtypes: list[str]
    qtype_variants: list[str]
    qtype_groups: dict[str, list[str]]
    source_candidates: list[SourceCandidate]


def _repo_file_size(sibling: Any) -> int | None:
    size = getattr(sibling, "size", None)
    if size is not None:
        return int(size)
    blob = getattr(sibling, "blob_id", None)
    lfs = getattr(sibling, "lfs", None)
    if isinstance(lfs, dict) and lfs.get("size") is not None:
        return int(lfs["size"])
    if isinstance(blob, dict) and blob.get("size") is not None:
        return int(blob["size"])
    return None


def _qtype_from_name(path: str) -> str | None:
    upper = Path(path).stem.upper().replace("-", "_")
    for qtype in sorted(QTYPE_NAMES, key=len, reverse=True):
        if re.search(rf"(^|_){re.escape(qtype)}($|_)", upper):
            return qtype
    return None


def _qtype_variant_from_name(path: str) -> str | None:
    qtype = _qtype_from_name(path)
    if qtype is None:
        return None
    stem = Path(path).stem.upper().replace("-", "_")
    match = re.search(rf"(^|_)(UD_)?{re.escape(qtype)}($|_)", stem)
    if match and match.group(2):
        return f"UD-{qtype}"
    return qtype


def base_qtype_variant(qtype_variant: str) -> str:
    return qtype_variant.removeprefix("UD-")


def is_ud_variant(qtype_variant: str | None) -> bool:
    return bool(qtype_variant and qtype_variant.startswith("UD-"))


def is_excluded_variant(qtype_variant: str | None) -> bool:
    return qtype_variant is None or qtype_variant in FLOAT_QTYPE_VARIANTS or qtype_variant.startswith("IQ")


def is_standard_comparison_variant(qtype_variant: str | None) -> bool:
    return bool(qtype_variant in STANDARD_QTYPE_VARIANTS)


def is_ud_analysis_variant(qtype_variant: str | None) -> bool:
    return bool(qtype_variant in UD_QTYPE_VARIANTS)


def _qtype_group(qtype: str) -> str:
    if qtype in {"F16", "F32", "BF16"}:
        return "float"
    if qtype.startswith("IQ"):
        return "importance"
    if qtype.startswith("TQ"):
        return "ternary"
    if qtype in {"MXFP4", "MXFP4_MOE", "NVFP4"}:
        return "float4"
    if "_K" in qtype:
        return "k_quant"
    if qtype.startswith("Q"):
        return "legacy_q"
    return "other"


def _base_models(tags: Iterable[str] | None) -> list[str]:
    out: list[str] = []
    for tag in tags or ():
        if tag.startswith("base_model:") and not tag.startswith("base_model:quantized:"):
            out.append(tag.split(":", 1)[1])
    return sorted(set(out))


def _tokens(value: str) -> set[str]:
    return {part for part in re.split(r"[^a-z0-9]+", value.lower()) if len(part) >= 2 or part == "z"}


def _compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _source_score(repo_id: str, base_models: Iterable[str], source_repo: str, source_path: str) -> int:
    target_tokens = _tokens(repo_id)
    compact_targets = {_compact(repo_id.rsplit("/", 1)[-1].removesuffix("-gguf").removesuffix("_gguf"))}
    for base_model in base_models:
        target_tokens |= _tokens(base_model)
        compact_targets.add(_compact(base_model.rsplit("/", 1)[-1]))
    source_tokens = _tokens(source_repo) | _tokens(source_path)
    compact_source = _compact(f"{source_repo}/{source_path}")
    score = len(target_tokens & source_tokens)
    if any(target and target in compact_source for target in compact_targets):
        score += 4
    if "single" in source_path.lower():
        score += 2
    if any(term in source_path.lower() for term in ("diffusion_models", "unet", "transformer")):
        score += 1
    return score


def _list_city96_repos(api: HfApi) -> list[str]:
    LOGGER.info("Listing city96 GGUF repos")
    repos = []
    for model in api.list_models(author="city96", limit=200, full=True):
        tags = set(model.tags or ())
        if "gguf" in tags or model.modelId.lower().endswith("-gguf") or model.modelId.lower().endswith("_gguf"):
            repos.append(model.modelId)
    out = sorted(set(repos))
    LOGGER.info("Found %d city96 GGUF repos", len(out))
    return out


def _comfy_safetensors(api: HfApi, limit: int = 200) -> list[tuple[str, str, int | None]]:
    LOGGER.info("Scanning Comfy-Org safetensors candidates (limit=%d repos)", limit)
    files: list[tuple[str, str, int | None]] = []
    for model in api.list_models(author="Comfy-Org", limit=limit):
        info = api.repo_info(model.modelId, repo_type="model", files_metadata=True)
        for sibling in info.siblings or ():
            path = sibling.rfilename
            if path.lower().endswith(".safetensors"):
                files.append((model.modelId, path, _repo_file_size(sibling)))
    LOGGER.info("Found %d Comfy-Org safetensors candidates", len(files))
    return files


def _repo_safetensors(api: HfApi, repo_id: str) -> list[tuple[str, str, int | None]]:
    try:
        info = api.repo_info(repo_id, repo_type="model", files_metadata=True)
    except Exception as exc:
        LOGGER.debug("Could not inspect source repo %s: %s", repo_id, exc)
        return []
    files: list[tuple[str, str, int | None]] = []
    for sibling in info.siblings or ():
        path = sibling.rfilename
        if path.lower().endswith(".safetensors"):
            files.append((repo_id, path, _repo_file_size(sibling)))
    return files


def _searched_safetensors(api: HfApi, query: str, *, limit: int = 5) -> list[tuple[str, str, int | None]]:
    files: list[tuple[str, str, int | None]] = []
    try:
        models = list(api.list_models(search=query, limit=limit))
    except Exception as exc:
        LOGGER.debug("HF source search failed for %r: %s", query, exc)
        return []
    for model in models:
        files.extend(_repo_safetensors(api, model.modelId))
    return files


def _extra_source_safetensors(api: HfApi, base_models: Iterable[str]) -> list[tuple[str, str, int | None]]:
    files: list[tuple[str, str, int | None]] = []
    seen_repos: set[str] = set()
    for base_model in base_models:
        if base_model in seen_repos:
            continue
        seen_repos.add(base_model)
        direct = _repo_safetensors(api, base_model)
        files.extend(direct)
        if direct:
            LOGGER.debug("Found %d direct base-model safetensors in %s", len(direct), base_model)
            continue
        query = base_model.rsplit("/", 1)[-1]
        searched = _searched_safetensors(api, query)
        LOGGER.debug("Found %d searched safetensors for %s", len(searched), query)
        files.extend(searched)
    return files


def build_inventory(include_city96: bool, repos: list[str], comfy_limit: int) -> list[RepoInventory]:
    api = HfApi()
    repo_ids = list(repos)
    if include_city96:
        repo_ids.extend(_list_city96_repos(api))
    repo_ids = sorted(set(repo_ids))
    LOGGER.info("Building inventory for %d GGUF repos", len(repo_ids))
    comfy_files = _comfy_safetensors(api, limit=comfy_limit)
    extra_source_cache: dict[tuple[str, ...], list[tuple[str, str, int | None]]] = {}
    inventory: list[RepoInventory] = []

    for repo_id in repo_ids:
        LOGGER.info("Reading HF metadata for %s", repo_id)
        info = api.repo_info(repo_id, repo_type="model", files_metadata=True)
        base_models = _base_models(info.tags)
        gguf_files = [
            FileInfo(
                sibling.rfilename,
                _repo_file_size(sibling),
                _qtype_from_name(sibling.rfilename),
                _qtype_variant_from_name(sibling.rfilename),
            )
            for sibling in info.siblings or ()
            if sibling.rfilename.lower().endswith(".gguf")
        ]
        qtypes = sorted({file.qtype for file in gguf_files if file.qtype})
        qtype_variants = sorted({file.qtype_variant for file in gguf_files if file.qtype_variant})
        groups: dict[str, list[str]] = {}
        for qtype in qtypes:
            groups.setdefault(_qtype_group(qtype), []).append(qtype)

        base_key = tuple(base_models)
        if base_key not in extra_source_cache:
            extra_source_cache[base_key] = _extra_source_safetensors(api, base_models)
        source_files = list(comfy_files) + extra_source_cache[base_key]
        candidates = [
            SourceCandidate(source_repo, path, size, _source_score(repo_id, base_models, source_repo, path))
            for source_repo, path, size in source_files
        ]
        candidates = [candidate for candidate in candidates if candidate.score > 0]
        candidates.sort(key=lambda item: (-item.score, item.repo_id, item.path))

        inventory.append(
            RepoInventory(
                repo_id=repo_id,
                base_models=base_models,
                gguf_files=gguf_files,
                qtypes=qtypes,
                qtype_variants=qtype_variants,
                qtype_groups={key: groups[key] for key in sorted(groups)},
                source_candidates=candidates[:SOURCE_CANDIDATE_LIMIT],
            )
        )
        LOGGER.info(
            "Inventory %s: %d GGUF files, %d qtype variants, %d source candidates",
            repo_id,
            len(gguf_files),
            len(qtype_variants),
            len(candidates[:SOURCE_CANDIDATE_LIMIT]),
        )
    return inventory


def _file_info_from_json(data: MappingLike) -> FileInfo:
    return FileInfo(
        path=str(data["path"]),
        size=data.get("size"),
        qtype=data.get("qtype"),
        qtype_variant=data.get("qtype_variant") or data.get("qtype"),
    )


def _source_candidate_from_json(data: MappingLike) -> SourceCandidate:
    return SourceCandidate(
        repo_id=str(data["repo_id"]),
        path=str(data["path"]),
        size=data.get("size"),
        score=int(data.get("score", 0)),
    )


def _repo_inventory_from_json(data: MappingLike) -> RepoInventory:
    gguf_files = [_file_info_from_json(item) for item in data.get("gguf_files", [])]
    if not gguf_files:
        # reports/gguf_qtype_inventory.json is a compact report without filenames.
        # Comparison workflows need per-file GGUF names, so callers should provide
        # the full inventory or let load_full_inventory regenerate it.
        raise ValueError("Inventory does not include gguf_files; use .cache/gguf_inventory_full.json or regenerate inventory")
    return RepoInventory(
        repo_id=str(data["repo_id"]),
        base_models=list(data.get("base_models", [])),
        gguf_files=gguf_files,
        qtypes=list(data.get("qtypes", [])),
        qtype_variants=list(data.get("qtype_variants", [])),
        qtype_groups=dict(data.get("qtype_groups", {})),
        source_candidates=[_source_candidate_from_json(item) for item in data.get("source_candidates", [])],
    )


def load_full_inventory(path: Path | None, *, comfy_limit: int = 200) -> list[RepoInventory]:
    inventory_path = path or DEFAULT_FULL_INVENTORY
    if inventory_path.is_file():
        LOGGER.info("Loading full inventory from %s", inventory_path)
        raw = json.loads(inventory_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "repos" in raw:
            raise ValueError(f"{inventory_path} is a compact report; use full inventory JSON with gguf_files")
        inventory = [_repo_inventory_from_json(item) for item in raw]
        LOGGER.info("Loaded %d repos from inventory", len(inventory))
        return inventory

    LOGGER.info("Inventory %s not found; regenerating", inventory_path)
    inventory = build_inventory(include_city96=True, repos=list(UNSLOTH_REPOS), comfy_limit=comfy_limit)
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(json.dumps([asdict(item) for item in inventory], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return inventory


def _source_score_for_comparison(candidate: SourceCandidate) -> int:
    path = candidate.path.lower().replace("\\", "/")
    score = candidate.score
    if _is_sharded_safetensors(path):
        score -= 1_000
    if any(part in path for part in SOURCE_REJECT_PARTS):
        score -= 100
    if any(part in path for part in SOURCE_ACCEPT_PARTS):
        score += 20
    if any(dtype in path for dtype in SOURCE_DTYPE_PREFERENCE) or path.endswith(".safetensors"):
        score += 3
    if any(dtype in path for dtype in SOURCE_DTYPE_PENALTY):
        score -= 10
    return score


def _is_sharded_safetensors(path: str) -> bool:
    return bool(re.search(r"-\d{5}-of-\d{5}(?:\.[^.]+)?\.safetensors$", path.lower()))


def _target_aliases(repo: RepoInventory) -> set[str]:
    aliases = set(SOURCE_ALIAS_OVERRIDES.get(repo.repo_id, set()))
    repo_slug = repo.repo_id.rsplit("/", 1)[-1].lower()
    for suffix in ("-gguf", "_gguf", "-encoder"):
        repo_slug = repo_slug.removesuffix(suffix)
    aliases.add(_compact(repo_slug))
    for base in repo.base_models:
        aliases.add(_compact(base.rsplit("/", 1)[-1]))
    return {alias for alias in aliases if alias}


def _source_has_strong_model_match(repo: RepoInventory, candidate: SourceCandidate) -> bool:
    compact_source = _compact(f"{candidate.repo_id}/{candidate.path}")
    return any(alias in compact_source for alias in _target_aliases(repo))


def _source_variant_penalty(repo: RepoInventory, candidate: SourceCandidate) -> int:
    target_tokens = _tokens(repo.repo_id) | {token for base in repo.base_models for token in _tokens(base)}
    source_tokens = _tokens(candidate.path)
    unexpected = (source_tokens & SOURCE_VARIANT_TOKENS) - target_tokens
    return 20 * len(unexpected)


def _source_selection_score(repo: RepoInventory, candidate: SourceCandidate) -> int:
    if not _source_has_strong_model_match(repo, candidate):
        return -10_000 + _source_score_for_comparison(candidate)
    path = candidate.path.lower().replace("\\", "/")
    score = _source_score_for_comparison(candidate)
    score -= _source_variant_penalty(repo, candidate)
    aliases = _target_aliases(repo)
    stem = _compact(Path(path).stem)
    if stem in aliases:
        score += 60
    if any(stem == f"{alias}{dtype}" for alias in aliases for dtype in SOURCE_DTYPE_PREFERENCE):
        score += 50
    if any(stem.startswith(alias) for alias in aliases):
        score += 10
    if "/" not in path:
        score += 20
    return score


def select_source_candidate(repo: RepoInventory) -> SourceCandidate | None:
    candidates = sorted(repo.source_candidates, key=lambda item: (-_source_selection_score(repo, item), item.repo_id, item.path))
    for candidate in candidates:
        score = _source_selection_score(repo, candidate)
        if score > 0:
            LOGGER.debug("Selected source for %s: %s/%s (score=%d)", repo.repo_id, candidate.repo_id, candidate.path, score)
            return candidate
    LOGGER.warning("No usable source candidate for %s", repo.repo_id)
    return None


MappingLike = dict[str, Any]


def _strip_key_prefixes(keys: Iterable[str]) -> set[str]:
    keys = set(keys)
    for prefix in ("model.diffusion_model.", "model."):
        if any(key.startswith(prefix) for key in keys):
            return {key.removeprefix(prefix) for key in keys if key.startswith(prefix)}
    if keys and all(key.startswith("net.") for key in keys):
        return {key.removeprefix("net.") for key in keys}
    return keys


def safetensors_keys(path: Path) -> set[str]:
    from safetensors import safe_open

    with safe_open(os.fspath(path), framework="np") as handle:
        return _strip_key_prefixes(handle.keys())


def gguf_keys(path: Path) -> set[str]:
    import gguf

    reader = gguf.GGUFReader(os.fspath(path), "r")
    return {tensor.name for tensor in reader.tensors}


def _remote_bytes(repo_id: str, filename: str, start: int, end: int) -> bytes:
    url = hf_hub_url(repo_id, filename, repo_type="model")
    LOGGER.debug("Fetching bytes %d-%d from %s/%s", start, end - 1, repo_id, filename)
    response = requests.get(url, headers={"Range": f"bytes={start}-{end - 1}"}, timeout=60)
    response.raise_for_status()
    return response.content


def _remote_prefix(repo_id: str, filename: str, size: int) -> bytes:
    return _remote_bytes(repo_id, filename, 0, size)


def remote_safetensors_keys(repo_id: str, filename: str) -> set[str]:
    import struct

    LOGGER.info("Reading remote safetensors header for %s/%s", repo_id, filename)
    first = _remote_prefix(repo_id, filename, 8)
    header_len = struct.unpack("<Q", first)[0]
    header = json.loads(_remote_bytes(repo_id, filename, 8, 8 + header_len))
    keys = _strip_key_prefixes(key for key in header if key != "__metadata__")
    LOGGER.info("Read %d safetensors keys from %s/%s", len(keys), repo_id, filename)
    return keys


class _NeedMoreData(Exception):
    pass


class _GGUFHeaderParser:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def read(self, size: int) -> bytes:
        end = self.pos + size
        if end > len(self.data):
            raise _NeedMoreData
        out = self.data[self.pos:end]
        self.pos = end
        return out

    def u32(self) -> int:
        import struct

        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        import struct

        return struct.unpack("<Q", self.read(8))[0]

    def string(self) -> str:
        size = self.u64()
        return self.read(size).decode("utf-8")

    def scalar(self, value_type: int) -> Any:
        import struct

        if value_type in {0, 1, 7}:
            raw = self.read(1)
            if value_type == 0:
                return raw[0]
            if value_type == 1:
                return struct.unpack("<b", raw)[0]
            return bool(raw[0])
        elif value_type in {2, 3}:
            raw = self.read(2)
            return struct.unpack("<H" if value_type == 2 else "<h", raw)[0]
        elif value_type in {4, 5, 6}:
            raw = self.read(4)
            if value_type == 4:
                return struct.unpack("<I", raw)[0]
            if value_type == 5:
                return struct.unpack("<i", raw)[0]
            return struct.unpack("<f", raw)[0]
        elif value_type in {10, 11, 12}:
            raw = self.read(8)
            if value_type == 10:
                return struct.unpack("<Q", raw)[0]
            if value_type == 11:
                return struct.unpack("<q", raw)[0]
            return struct.unpack("<d", raw)[0]
        elif value_type == 8:
            return self.string()
        raise ValueError(f"Unsupported GGUF value type {value_type}")

    def value(self) -> dict[str, Any]:
        value_type = self.u32()
        if value_type != 9:
            return {"type": GGUF_VALUE_TYPE_NAMES.get(value_type, str(value_type)), "value": self.scalar(value_type)}
        array_type = self.u32()
        array_len = self.u64()
        values: list[Any] | None = []
        if array_type == 8:
            values = [self.string() for _ in range(array_len)]
        else:
            widths = {0: 1, 1: 1, 7: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 10: 8, 11: 8, 12: 8}
            try:
                width = widths[array_type]
            except KeyError as exc:
                raise ValueError(f"Unsupported GGUF array type {array_type}") from exc
            if array_len <= 128:
                values = [self.scalar(array_type) for _ in range(array_len)]
            else:
                self.read(width * array_len)
                values = None
        return {
            "type": "ARRAY",
            "array_type": GGUF_VALUE_TYPE_NAMES.get(array_type, str(array_type)),
            "length": array_len,
            "value": values if values is not None else f"<{array_len} items>",
        }


def _parse_gguf_header_keys(data: bytes) -> set[str]:
    return {tensor["name"] for tensor in _parse_gguf_header(data)["tensors"]}


def _ggml_qtype_name(value: int) -> str:
    try:
        import gguf

        return gguf.GGMLQuantizationType(value).name
    except Exception:
        return str(value)


def _parse_gguf_header(data: bytes) -> dict[str, Any]:
    parser = _GGUFHeaderParser(data)
    if parser.read(4) != b"GGUF":
        raise ValueError("Not a GGUF file")
    version = parser.u32()
    if version != 3:
        raise ValueError(f"Unsupported GGUF version {version}")
    tensor_count = parser.u64()
    kv_count = parser.u64()
    metadata: dict[str, Any] = {}
    for _ in range(kv_count):
        key = parser.string()
        metadata[key] = parser.value()
    tensors: list[dict[str, Any]] = []
    for _ in range(tensor_count):
        name = parser.string()
        n_dims = parser.u32()
        shape = [parser.u64() for _ in range(n_dims)]
        qtype_value = parser.u32()
        offset = parser.u64()
        tensors.append(
            {
                "name": name,
                "shape": shape,
                "qtype": _ggml_qtype_name(qtype_value),
                "qtype_value": qtype_value,
                "offset": offset,
            }
        )
    return {"version": version, "tensor_count": tensor_count, "metadata": metadata, "tensors": tensors}


def remote_gguf_keys(repo_id: str, filename: str, initial_bytes: int = 1024 * 1024) -> set[str]:
    return {tensor["name"] for tensor in remote_gguf_header(repo_id, filename, initial_bytes=initial_bytes)["tensors"]}


def _gguf_header_from_prefix(fetch_prefix: Any, initial_bytes: int = 1024 * 1024) -> dict[str, Any]:
    size = initial_bytes
    while size <= 128 * 1024 * 1024:
        LOGGER.debug("Reading GGUF header prefix: %d bytes", size)
        data = fetch_prefix(size)
        try:
            return _parse_gguf_header(data)
        except _NeedMoreData:
            LOGGER.debug("GGUF header prefix too small at %d bytes; doubling", size)
            size *= 2
    raise RuntimeError("GGUF header is larger than 128 MiB")


def remote_gguf_header(repo_id: str, filename: str, initial_bytes: int = 1024 * 1024) -> dict[str, Any]:
    LOGGER.info("Reading remote GGUF header for %s/%s", repo_id, filename)
    header = _gguf_header_from_prefix(lambda size: _remote_prefix(repo_id, filename, size), initial_bytes=initial_bytes)
    LOGGER.info("Read GGUF header for %s/%s: %d tensors, %d metadata keys", repo_id, filename, len(header["tensors"]), len(header["metadata"]))
    return header


def local_gguf_header(path: Path, initial_bytes: int = 1024 * 1024) -> dict[str, Any]:
    LOGGER.debug("Reading local GGUF header for %s", path)
    def read_prefix(size: int) -> bytes:
        with path.open("rb") as handle:
            return handle.read(size)

    header = _gguf_header_from_prefix(read_prefix, initial_bytes=initial_bytes)
    LOGGER.debug("Read local GGUF header for %s: %d tensors", path, len(header["tensors"]))
    return header


def validate_keys(source: Path, gguf_path: Path) -> dict[str, Any]:
    src_keys = safetensors_keys(source)
    out_keys = gguf_keys(gguf_path)
    return {
        "source": os.fspath(source),
        "gguf": os.fspath(gguf_path),
        "match": src_keys == out_keys,
        "source_count": len(src_keys),
        "gguf_count": len(out_keys),
        "missing_in_gguf": sorted(src_keys - out_keys)[:50],
        "extra_in_gguf": sorted(out_keys - src_keys)[:50],
    }


def validate_remote_keys(source_repo: str, source_file: str, gguf_repo: str, gguf_file: str) -> dict[str, Any]:
    LOGGER.info("Validating remote keys: %s/%s -> %s/%s", source_repo, source_file, gguf_repo, gguf_file)
    src_keys = remote_safetensors_keys(source_repo, source_file)
    out_keys = remote_gguf_keys(gguf_repo, gguf_file)
    LOGGER.info("Remote key validation %s: source=%d gguf=%d", "matched" if src_keys == out_keys else "mismatched", len(src_keys), len(out_keys))
    return {
        "source": f"{source_repo}/{source_file}",
        "gguf": f"{gguf_repo}/{gguf_file}",
        "match": src_keys == out_keys,
        "source_count": len(src_keys),
        "gguf_count": len(out_keys),
        "missing_in_gguf": sorted(src_keys - out_keys)[:50],
        "extra_in_gguf": sorted(out_keys - src_keys)[:50],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(repo_id: str, filename: str, cache_dir: Path) -> Path:
    LOGGER.info("Downloading/caching %s/%s", repo_id, filename)
    path = hf_hub_download(repo_id, filename, repo_type="model", cache_dir=os.fspath(cache_dir))
    LOGGER.info("Using local file %s", path)
    return Path(path)


def _find_exe(explicit: str | None) -> Path:
    if explicit:
        exe = Path(explicit)
        if exe.is_file():
            return exe
        raise FileNotFoundError(exe)
    root = Path(__file__).resolve().parents[1]
    for pattern in ("build/cmake/**/libgguf_quantize_gguf.exe", "build/cmake/**/libgguf_quantize_gguf"):
        for exe in root.glob(pattern):
            if exe.is_file():
                return exe
    raise FileNotFoundError("libgguf_quantize_gguf executable is not built")


def compare_case(args: argparse.Namespace) -> dict[str, Any]:
    LOGGER.info("Starting comparison: %s/%s", args.gguf_repo, args.gguf_file)
    source = _download(args.source_repo, args.source_file, args.cache_dir)
    reference = _download(args.gguf_repo, args.gguf_file, args.cache_dir)
    qtype = args.qtype or _qtype_from_name(args.gguf_file)
    if not qtype:
        raise ValueError(f"Could not infer qtype from {args.gguf_file}; pass --qtype")
    if qtype not in EXE_QTYPES:
        raise ValueError(f"{qtype} is not supported by the native executable")

    generated = args.output_dir / f"{Path(args.source_file).stem}-{qtype}.gguf"
    generated.parent.mkdir(parents=True, exist_ok=True)
    exe = _find_exe(args.exe)
    command = [
        os.fspath(exe),
        "--src",
        os.fspath(source),
        "--dst",
        os.fspath(generated),
        "--qtype",
        qtype,
        "--policy",
        args.policy,
        "--overwrite",
    ]
    LOGGER.info("Running converter for %s with qtype=%s policy=%s", args.gguf_file, qtype, args.policy)
    LOGGER.debug("Command: %s", command)
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        LOGGER.error("Conversion failed for %s with return code %d.\nstdout: %s\nstderr: %s\n", args.gguf_file, result.returncode, result.stdout, result.stderr)
        return {
            "ok": False,
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    key_check = validate_keys(source, reference)
    generated_key_check = validate_keys(source, generated)
    ref_hash = _sha256(reference)
    gen_hash = _sha256(generated)
    LOGGER.info(
        "Comparison complete for %s: byte_identical=%s",
        args.gguf_file,
        ref_hash == gen_hash,
    )
    return {
        "ok": ref_hash == gen_hash and key_check["match"] and generated_key_check["match"],
        "qtype": qtype,
        "source": os.fspath(source),
        "reference": os.fspath(reference),
        "generated": os.fspath(generated),
        "reference_sha256": ref_hash,
        "generated_sha256": gen_hash,
        "byte_identical": ref_hash == gen_hash,
        "reference_key_check": key_check,
        "generated_key_check": generated_key_check,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _tensor_map(header: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {tensor["name"]: tensor for tensor in header["tensors"]}


def _metadata_simple(header: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, info in header["metadata"].items():
        value = info.get("value")
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        elif isinstance(value, list) and len(value) <= 16:
            out[key] = value
        else:
            out[key] = {"type": info.get("type"), "length": info.get("length")}
    return out


def _metadata_diff(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_meta = _metadata_simple(left)
    right_meta = _metadata_simple(right)
    left_keys = set(left_meta)
    right_keys = set(right_meta)
    changed = {
        key: {"reference": left_meta[key], "generated": right_meta[key]}
        for key in sorted(left_keys & right_keys)
        if left_meta[key] != right_meta[key]
    }
    return {
        "missing_in_generated": sorted(left_keys - right_keys),
        "extra_in_generated": sorted(right_keys - left_keys),
        "changed": changed,
    }


def _tensor_info_diff(reference_header: dict[str, Any], generated_header: dict[str, Any]) -> dict[str, Any]:
    reference = _tensor_map(reference_header)
    generated = _tensor_map(generated_header)
    shared = sorted(set(reference) & set(generated))
    qtype_diffs = [
        {"name": name, "reference": reference[name]["qtype"], "generated": generated[name]["qtype"]}
        for name in shared
        if reference[name]["qtype"] != generated[name]["qtype"]
    ]
    shape_diffs = [
        {"name": name, "reference": reference[name]["shape"], "generated": generated[name]["shape"]}
        for name in shared
        if reference[name]["shape"] != generated[name]["shape"]
    ]
    return {
        "tensor_names_match": set(reference) == set(generated),
        "missing_in_generated": sorted(set(reference) - set(generated))[:50],
        "extra_in_generated": sorted(set(generated) - set(reference))[:50],
        "tensor_qtypes_match": not qtype_diffs,
        "tensor_qtype_diffs": qtype_diffs[:100],
        "tensor_shapes_match": not shape_diffs,
        "tensor_shape_diffs": shape_diffs[:100],
    }


def _matching_gguf_file(repo: RepoInventory, qtype_variant: str) -> FileInfo | None:
    for file in repo.gguf_files:
        if file.qtype_variant == qtype_variant:
            return file
    return None


def _standard_entries(inventory: list[RepoInventory], repo_filter: set[str] | None, variant_filter: set[str] | None) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for repo in inventory:
        if repo_filter and repo.repo_id not in repo_filter:
            continue
        source = select_source_candidate(repo)
        for file in repo.gguf_files:
            variant = file.qtype_variant
            if not is_standard_comparison_variant(variant):
                continue
            if variant_filter and variant not in variant_filter:
                continue
            entries.append(
                {
                    "repo_id": repo.repo_id,
                    "gguf_file": file.path,
                    "variant": variant,
                    "base_qtype": base_qtype_variant(variant or ""),
                    "size": file.size,
                    "source_candidate": asdict(source) if source else None,
                    "status": "planned" if source else "no_source_candidate",
                }
            )
    return entries


def _safe_remote_key_check(source: SourceCandidate, repo_id: str, gguf_file: str) -> dict[str, Any]:
    try:
        return validate_remote_keys(source.repo_id, source.path, repo_id, gguf_file)
    except Exception as exc:
        LOGGER.exception("Remote key validation failed for %s/%s", repo_id, gguf_file)
        return {"match": False, "error": str(exc), "source": f"{source.repo_id}/{source.path}", "gguf": f"{repo_id}/{gguf_file}"}


def _run_standard_entry(entry: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    LOGGER.info("Processing standard entry %s %s", entry["repo_id"], entry["variant"])
    source_data = entry.get("source_candidate")
    if not source_data:
        LOGGER.warning("Skipping %s %s: no source candidate", entry["repo_id"], entry["variant"])
        return {**entry, "status": "no_source_candidate"}
    source = _source_candidate_from_json(source_data)
    key_check = _safe_remote_key_check(source, entry["repo_id"], entry["gguf_file"])
    if not key_check.get("match"):
        LOGGER.warning("Skipping conversion for %s %s: key mismatch", entry["repo_id"], entry["variant"])
        return {**entry, "status": "key_mismatch", "key_check": key_check}
    if not args.execute:
        LOGGER.info("Key check matched for %s %s; execution disabled", entry["repo_id"], entry["variant"])
        return {**entry, "status": "key_match_not_executed", "key_check": key_check}

    compare_args = argparse.Namespace(
        source_repo=source.repo_id,
        source_file=source.path,
        gguf_repo=entry["repo_id"],
        gguf_file=entry["gguf_file"],
        qtype=entry["variant"],
        policy=args.policy,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        exe=args.exe,
    )
    result = compare_case(compare_args)
    if not result.get("ok") and result.get("reference") and result.get("generated"):
        LOGGER.info("Collecting metadata/tensor diffs for %s %s", entry["repo_id"], entry["variant"])
        reference_header = local_gguf_header(Path(result["reference"]))
        generated_header = local_gguf_header(Path(result["generated"]))
        result["metadata_diff"] = _metadata_diff(reference_header, generated_header)
        result["tensor_info_diff"] = _tensor_info_diff(reference_header, generated_header)
    status = "byte_identical" if result.get("byte_identical") else "different"
    if not result.get("ok") and "returncode" in result:
        status = "conversion_failed"
    return {**entry, "status": status, "key_check": key_check, "comparison": result}


def _standard_result_key(entry: dict[str, Any]) -> str:
    source = entry.get("source_candidate") or {}
    return "\n".join(
        [
            str(entry.get("repo_id", "")),
            str(entry.get("gguf_file", "")),
            str(entry.get("variant", "")),
            str(source.get("repo_id", "")),
            str(source.get("path", "")),
        ]
    )


def _load_existing_standard_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        LOGGER.warning("Ignoring invalid existing standard results JSON: %s", path)
        return {}
    if not isinstance(raw, list):
        LOGGER.warning("Ignoring existing standard results with unexpected shape: %s", path)
        return {}
    results = {_standard_result_key(item): item for item in raw if isinstance(item, dict)}
    LOGGER.info("Loaded %d existing standard results from %s", len(results), path)
    return results


RESUMABLE_STANDARD_STATUSES = {
    "byte_identical",
    "different",
    "conversion_failed",
    "key_mismatch",
    "no_source_candidate",
}


def _can_resume_standard_result(result: dict[str, Any], *, execute: bool) -> bool:
    status = result.get("status")
    if status in RESUMABLE_STANDARD_STATUSES:
        return True
    if not execute and status == "key_match_not_executed":
        return True
    return False


def _write_text_report(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _comparison_summary(standard_results: list[dict[str, Any]], ud_results: list[dict[str, Any]]) -> str:
    lines = ["# GGUF Comparison Summary", "", "## Standard Q/K", "", "| Repo | Variant | Status | Reference | Source |", "|---|---|---|---|---|"]
    for item in standard_results:
        source = item.get("source_candidate") or {}
        source_name = f"{source.get('repo_id', '')}/{source.get('path', '')}" if source else ""
        lines.append(f"| `{item['repo_id']}` | `{item['variant']}` | `{item['status']}` | `{item['gguf_file']}` | `{source_name}` |")
    lines.extend(["", "## Unsloth Dynamic", "", "| Repo | Variant | Confidence | Changed tensors | Base reference |", "|---|---|---|---|---|"])
    for item in ud_results:
        lines.append(
            f"| `{item['repo_id']}` | `{item['variant']}` | `{item.get('confidence', '')}` | "
            f"{item.get('changed_tensor_count', 0)} | `{item.get('base_gguf_file', '')}` |"
        )
    return "\n".join(lines) + "\n"


def compare_standard(args: argparse.Namespace) -> list[dict[str, Any]]:
    LOGGER.info("Starting standard comparison workflow")
    inventory = load_full_inventory(args.inventory, comfy_limit=args.comfy_limit)
    repo_filter = set(args.repo or []) or None
    variant_filter = set(args.variant or []) or None
    entries = _standard_entries(inventory, repo_filter, variant_filter)
    if args.limit is not None:
        entries = entries[: args.limit]

    LOGGER.info("Standard comparison plan has %d entries", len(entries))
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "comparison_plan.json").write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    results_path = args.report_dir / "standard_results.json"
    existing = _load_existing_standard_results(results_path) if args.resume else {}
    results: list[dict[str, Any]] = []
    for index, entry in enumerate(entries, start=1):
        key = _standard_result_key(entry)
        previous = existing.get(key)
        if previous is not None and _can_resume_standard_result(previous, execute=args.execute):
            LOGGER.info(
                "Resuming %d/%d %s %s from existing status=%s",
                index,
                len(entries),
                entry["repo_id"],
                entry["variant"],
                previous.get("status"),
            )
            results.append(previous)
            continue
        LOGGER.info("Running %d/%d %s %s", index, len(entries), entry["repo_id"], entry["variant"])
        result = _run_standard_entry(entry, args)
        results.append(result)
        results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    ud_path = args.report_dir / "ud_policy_analysis.json"
    ud_results = json.loads(ud_path.read_text(encoding="utf-8")) if ud_path.is_file() else []
    _write_text_report(args.report_dir / "comparison_summary.md", _comparison_summary(results, ud_results))
    LOGGER.info("Wrote standard comparison reports to %s", args.report_dir)
    return results


QTYPE_RANK = {
    "F32": 32,
    "BF16": 16,
    "F16": 16,
    "Q8_0": 8,
    "Q6_K": 6,
    "Q5_1": 5.1,
    "Q5_0": 5.0,
    "Q5_K": 5.0,
    "Q5_K_M": 5.0,
    "Q5_K_S": 5.0,
    "Q4_1": 4.1,
    "Q4_0": 4.0,
    "Q4_K": 4.0,
    "Q4_K_M": 4.0,
    "Q4_K_S": 4.0,
    "Q3_K": 3.0,
    "Q3_K_L": 3.2,
    "Q3_K_M": 3.0,
    "Q3_K_S": 2.9,
    "Q2_K": 2.0,
}


def _tensor_class(name: str) -> str:
    lower = name.lower()
    if "norm" in lower:
        return "norm"
    if "modulation" in lower or "mod." in lower:
        return "modulation"
    if "embed" in lower or "embedding" in lower:
        return "embedding"
    if "final" in lower or "output" in lower or "proj_out" in lower or "conv_out" in lower:
        return "output"
    if any(token in lower for token in ("attn_v.weight", ".to_v.weight", ".v.weight", "v_proj.weight")):
        return "attention_value"
    if any(token in lower for token in ("qkv.weight", "attn_qkv.weight", "attention.qkv.weight")):
        return "fused_qkv"
    if any(token in lower for token in ("ffn_down", ".w2.weight", "ff.net.2.weight", "mlp.layer2.weight")):
        return "ffn_down_projection"
    if "attn" in lower:
        return "attention_other"
    if any(token in lower for token in ("mlp", "ffn", "feed_forward")):
        return "ffn_other"
    return "other"


def _change_direction(base_qtype: str, ud_qtype: str) -> str:
    if base_qtype == ud_qtype:
        return "same"
    base_rank = QTYPE_RANK.get(base_qtype)
    ud_rank = QTYPE_RANK.get(ud_qtype)
    if ud_qtype in FLOAT_QTYPE_VARIANTS:
        return "kept_unquantized"
    if base_rank is None or ud_rank is None:
        return "changed"
    if ud_rank > base_rank:
        return "promoted"
    if ud_rank < base_rank:
        return "demoted"
    return "changed_same_rank"


def _ud_entries(inventory: list[RepoInventory], repo_filter: set[str] | None, variant_filter: set[str] | None) -> list[tuple[RepoInventory, FileInfo, FileInfo | None]]:
    entries: list[tuple[RepoInventory, FileInfo, FileInfo | None]] = []
    for repo in inventory:
        if repo_filter and repo.repo_id not in repo_filter:
            continue
        for file in repo.gguf_files:
            variant = file.qtype_variant
            if not is_ud_analysis_variant(variant):
                continue
            if variant_filter and variant not in variant_filter:
                continue
            entries.append((repo, file, _matching_gguf_file(repo, base_qtype_variant(variant or ""))))
    return entries


def _analyze_ud_entry(repo: RepoInventory, ud_file: FileInfo, base_file: FileInfo | None) -> dict[str, Any]:
    LOGGER.info("Analyzing UD policy for %s %s", repo.repo_id, ud_file.qtype_variant)
    variant = ud_file.qtype_variant or ""
    item: dict[str, Any] = {
        "repo_id": repo.repo_id,
        "variant": variant,
        "base_variant": base_qtype_variant(variant),
        "ud_gguf_file": ud_file.path,
        "base_gguf_file": base_file.path if base_file else None,
    }
    if base_file is None:
        LOGGER.warning("Cannot analyze %s %s: missing base reference", repo.repo_id, variant)
        return {**item, "status": "missing_base_reference", "confidence": "insufficient_metadata"}

    ud_header = remote_gguf_header(repo.repo_id, ud_file.path)
    base_header = remote_gguf_header(repo.repo_id, base_file.path)
    ud_tensors = _tensor_map(ud_header)
    base_tensors = _tensor_map(base_header)
    shared = sorted(set(ud_tensors) & set(base_tensors))
    changes = []
    for name in shared:
        base_qtype = base_tensors[name]["qtype"]
        ud_qtype = ud_tensors[name]["qtype"]
        if base_qtype == ud_qtype:
            continue
        changes.append(
            {
                "name": name,
                "class": _tensor_class(name),
                "base_qtype": base_qtype,
                "ud_qtype": ud_qtype,
                "direction": _change_direction(base_qtype, ud_qtype),
                "shape": ud_tensors[name]["shape"],
            }
        )

    metadata = _metadata_simple(ud_header)
    unsloth_keys = [key for key in metadata if "unsloth" in key.lower() or "dynamic" in key.lower() or key.lower().startswith("ud.")]
    transition_counts = Counter(f"{change['base_qtype']}->{change['ud_qtype']}" for change in changes)
    class_counts = Counter(change["class"] for change in changes)
    direction_counts = Counter(change["direction"] for change in changes)
    confidence = "metadata_declared" if unsloth_keys else ("inferred_from_tensor_types" if changes else "insufficient_metadata")
    LOGGER.info(
        "UD analysis complete for %s %s: confidence=%s changed_tensors=%d",
        repo.repo_id,
        variant,
        confidence,
        len(changes),
    )

    return {
        **item,
        "status": "analyzed",
        "confidence": confidence,
        "metadata": metadata,
        "unsloth_metadata_keys": unsloth_keys,
        "tensor_count": len(ud_tensors),
        "base_tensor_count": len(base_tensors),
        "tensor_names_match": set(ud_tensors) == set(base_tensors),
        "missing_in_ud": sorted(set(base_tensors) - set(ud_tensors))[:50],
        "extra_in_ud": sorted(set(ud_tensors) - set(base_tensors))[:50],
        "changed_tensor_count": len(changes),
        "transition_counts": dict(sorted(transition_counts.items())),
        "class_counts": dict(sorted(class_counts.items())),
        "direction_counts": dict(sorted(direction_counts.items())),
        "changes": changes,
    }


def analyze_ud(args: argparse.Namespace) -> list[dict[str, Any]]:
    LOGGER.info("Starting UD analysis workflow")
    inventory = load_full_inventory(args.inventory, comfy_limit=args.comfy_limit)
    repo_filter = set(args.repo or []) or None
    variant_filter = set(args.variant or []) or None
    entries = _ud_entries(inventory, repo_filter, variant_filter)
    if args.limit is not None:
        entries = entries[: args.limit]

    LOGGER.info("UD analysis plan has %d entries", len(entries))
    args.report_dir.mkdir(parents=True, exist_ok=True)
    results = [_analyze_ud_entry(repo, ud_file, base_file) for repo, ud_file, base_file in entries]
    (args.report_dir / "ud_policy_analysis.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    standard_path = args.report_dir / "standard_results.json"
    standard_results = json.loads(standard_path.read_text(encoding="utf-8")) if standard_path.is_file() else []
    _write_text_report(args.report_dir / "comparison_summary.md", _comparison_summary(standard_results, results))
    LOGGER.info("Wrote UD analysis reports to %s", args.report_dir)
    return results


def _write_report(path: Path | None, data: Any) -> None:
    text = json.dumps(data, indent=2, sort_keys=True)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


def _add_log_level(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--log-level", choices=LOG_LEVELS, default="INFO", help="Logging verbosity for stderr output.")


def _configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level), format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory and compare Hub GGUF files against local libgguf output.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    inv = sub.add_parser("inventory", help="List GGUF repos, qtypes, and likely Comfy-Org source safetensors.")
    inv.add_argument("--no-city96", action="store_true", help="Do not include all city96 GGUF repos.")
    inv.add_argument("--no-unsloth", action="store_true", help="Do not include the built-in unsloth GGUF list.")
    inv.add_argument("--repo", action="append", default=[], help="Additional GGUF repo id.")
    inv.add_argument("--comfy-limit", type=int, default=200, help="Max Comfy-Org repos to scan.")
    inv.add_argument("--out", type=Path, help="Optional JSON report path.")
    _add_log_level(inv)

    keys = sub.add_parser("validate-keys", help="Compare tensor names between a local safetensors file and local GGUF.")
    keys.add_argument("--source", type=Path, required=True)
    keys.add_argument("--gguf", type=Path, required=True)
    keys.add_argument("--out", type=Path)
    _add_log_level(keys)

    remote_keys = sub.add_parser("validate-remote-keys", help="Compare tensor names from Hub file headers without full downloads.")
    remote_keys.add_argument("--source-repo", required=True)
    remote_keys.add_argument("--source-file", required=True)
    remote_keys.add_argument("--gguf-repo", required=True)
    remote_keys.add_argument("--gguf-file", required=True)
    remote_keys.add_argument("--out", type=Path)
    _add_log_level(remote_keys)

    cmp_parser = sub.add_parser("compare", help="Download one source/reference pair, convert, and compare hashes/keys.")
    cmp_parser.add_argument("--source-repo", required=True)
    cmp_parser.add_argument("--source-file", required=True)
    cmp_parser.add_argument("--gguf-repo", required=True)
    cmp_parser.add_argument("--gguf-file", required=True)
    cmp_parser.add_argument("--qtype")
    cmp_parser.add_argument("--policy", choices=("comfy", "uniform"), default="comfy")
    cmp_parser.add_argument("--cache-dir", type=Path, default=Path(".cache/hf"))
    cmp_parser.add_argument("--output-dir", type=Path, default=Path(".cache/gguf-compare"))
    cmp_parser.add_argument("--exe")
    cmp_parser.add_argument("--out", type=Path)
    _add_log_level(cmp_parser)

    standard = sub.add_parser("compare-standard", help="Plan, key-check, and optionally execute standard non-UD Q/K comparisons.")
    standard.add_argument("--inventory", type=Path, default=DEFAULT_FULL_INVENTORY)
    standard.add_argument("--repo", action="append", help="Restrict to a GGUF repo id. May be repeated.")
    standard.add_argument("--variant", action="append", choices=sorted(STANDARD_QTYPE_VARIANTS), help="Restrict to a qtype variant.")
    standard.add_argument("--limit", type=int, help="Limit number of comparison entries.")
    standard.add_argument("--execute", action="store_true", help="Download key-compatible pairs, convert, and compare SHA-256.")
    standard.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="Reuse completed entries from standard_results.json.")
    standard.add_argument("--policy", choices=("comfy", "uniform"), default="comfy")
    standard.add_argument("--cache-dir", type=Path, default=Path(".cache/hf"))
    standard.add_argument("--output-dir", type=Path, default=Path(".cache/gguf-compare"))
    standard.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    standard.add_argument("--comfy-limit", type=int, default=200)
    standard.add_argument("--exe")
    _add_log_level(standard)

    ud = sub.add_parser("analyze-ud", help="Analyze Unsloth Dynamic GGUF tensor qtype policies from metadata and tensor info.")
    ud.add_argument("--inventory", type=Path, default=DEFAULT_FULL_INVENTORY)
    ud.add_argument("--repo", action="append", help="Restrict to a GGUF repo id. May be repeated.")
    ud.add_argument("--variant", action="append", choices=sorted(UD_QTYPE_VARIANTS), help="Restrict to a UD qtype variant.")
    ud.add_argument("--limit", type=int, help="Limit number of UD entries.")
    ud.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    ud.add_argument("--comfy-limit", type=int, default=200)
    _add_log_level(ud)

    args = parser.parse_args()
    _configure_logging(args.log_level)
    if args.cmd == "inventory":
        repos = list(args.repo)
        if not args.no_unsloth:
            repos.extend(UNSLOTH_REPOS)
        inventory = build_inventory(not args.no_city96, repos, args.comfy_limit)
        _write_report(args.out, [asdict(item) for item in inventory])
    elif args.cmd == "validate-keys":
        _write_report(args.out, validate_keys(args.source, args.gguf))
    elif args.cmd == "validate-remote-keys":
        _write_report(args.out, validate_remote_keys(args.source_repo, args.source_file, args.gguf_repo, args.gguf_file))
    elif args.cmd == "compare":
        _write_report(args.out, compare_case(args))
    elif args.cmd == "compare-standard":
        _write_report(None, compare_standard(args))
    elif args.cmd == "analyze-ud":
        _write_report(None, analyze_ud(args))
    else:
        raise AssertionError(args.cmd)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
