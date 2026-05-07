from __future__ import annotations

from types import SimpleNamespace

import pytest

from libgguf import quantize
from libgguf._metadata import GGMLQuantizationType, LlamaFileType
from libgguf.quantize import (
    ModelAura,
    ModelHiDream,
    _dynamic_policy_qtype,
    _mixed_policy_qtype,
    _normalize_overrides,
    _override_qtype,
    _policy_allows_quant_shape,
    _prepare_conversion,
    parse_qtype,
    parse_tensor_qtype,
)


class _TensorSource:
    def __init__(self, meta: dict[str, tuple[tuple[int, ...], str]]) -> None:
        self._meta = meta

    def keys(self) -> tuple[str, ...]:
        return tuple(self._meta)

    def tensor_meta(self, key: str) -> tuple[tuple[int, ...], str]:
        return self._meta[key]

    def load_tensor(self, key: str) -> object:
        raise AssertionError("planning tests should not load tensor payloads")


class _FakeWriter:
    def __init__(self, path: object, arch: str) -> None:
        self.path = path
        self.arch = arch
        self.tensor_info: list[tuple[str, object]] = []

    def add_quantization_version(self, version: int) -> None:
        self.quantization_version = version

    def add_file_type(self, file_type: object) -> None:
        self.file_type = file_type

    def add_array(self, name: str, value: object) -> None:
        pass

    def add_tensor_info(
        self,
        name: str,
        tensor_shape: tuple[int, ...],
        tensor_dtype: object,
        tensor_nbytes: int,
        *,
        raw_dtype: object,
    ) -> None:
        self.tensor_info.append((name, raw_dtype))


def test_parse_qtype_normalizes_aliases_and_enum_like_names() -> None:
    assert parse_qtype("q4_k") == ("Q4_K_M", "Q4_K")
    assert parse_qtype("Q5_K_M") == ("Q5_K_M", "Q5_K")
    assert parse_qtype(SimpleNamespace(name="q3_k")) == ("Q3_K_M", "Q3_K")
    assert parse_qtype(LlamaFileType.MOSTLY_Q4_K_M) == ("Q4_K_M", "Q4_K")
    assert parse_qtype(GGMLQuantizationType.Q6_K) == ("Q6_K", "Q6_K")


def test_parse_tensor_qtype_accepts_storage_quant_and_file_types() -> None:
    assert parse_tensor_qtype("q4_k") == "Q4_K"
    assert parse_tensor_qtype("Q4_K_M") == "Q4_K"
    assert parse_tensor_qtype("IQ2_M") == "IQ2_S"
    assert parse_tensor_qtype(SimpleNamespace(name="mostly_iq3_m")) == "IQ3_S"
    assert parse_tensor_qtype(GGMLQuantizationType.F32) == "F32"
    assert parse_tensor_qtype("f16") == "F16"
    assert parse_tensor_qtype("BF16") == "BF16"


def test_parse_qtypes_raise_clear_value_errors_for_unsupported_values() -> None:
    with pytest.raises(ValueError, match="Unsupported direct quantization type: F16"):
        parse_qtype("F16")
    with pytest.raises(ValueError, match="Unsupported quantization type: 12"):
        parse_qtype(12)

    with pytest.raises(ValueError, match="Unsupported GGML tensor type: Q8_1"):
        parse_tensor_qtype("Q8_1")
    with pytest.raises(ValueError, match="Unsupported tensor type: 12"):
        parse_tensor_qtype(12)


def test_normalize_overrides_uppercases_qtypes_and_preserves_order() -> None:
    overrides = _normalize_overrides([("*.weight", "f16"), ("blocks.*", "q4_k_m"), ("*", "bf16")])

    assert overrides == [("*.weight", "F16"), ("blocks.*", "Q4_K_M"), ("*", "BF16")]


def test_override_qtype_uses_first_matching_pattern() -> None:
    overrides = _normalize_overrides([("blocks.*", "f16"), ("blocks.0.*", "q4_k")])

    assert _override_qtype("blocks.0.attn.weight", overrides) == "F16"


@pytest.mark.parametrize(
    ("override_qtype", "expected_qtype"),
    [
        ("F32", GGMLQuantizationType.F32),
        ("F16", GGMLQuantizationType.F16),
        ("BF16", GGMLQuantizationType.BF16),
    ],
)
def test_prepare_conversion_storage_override_forces_storage_and_disables_quantization(
    monkeypatch: pytest.MonkeyPatch,
    override_qtype: str,
    expected_qtype: GGMLQuantizationType,
) -> None:
    monkeypatch.setattr(
        quantize,
        "_require_gguf",
        lambda: SimpleNamespace(GGUFWriter=_FakeWriter, GGML_QUANT_VERSION=2),
    )
    key_map = {
        "caption_projection.0.linear.weight": "model.caption_projection.0.linear.weight",
        "double_stream_blocks.0.block.ff_i.shared_experts.w3.weight": (
            "model.double_stream_blocks.0.block.ff_i.shared_experts.w3.weight"
        ),
    }
    source = _TensorSource(
        {
            "caption_projection.0.linear.weight": ((256, 256), "F32"),
            "double_stream_blocks.0.block.ff_i.shared_experts.w3.weight": ((256, 256), "F32"),
        }
    )

    _, _, plans, tensor_counts, fallback_counts = _prepare_conversion(
        LlamaFileType.MOSTLY_Q4_K_M,
        "Q4_K_M",
        "Q4_K",
        key_map,
        source,
        policy="uniform",
        imatrix_data={},
        overrides=[("double_stream_blocks.*.w3.weight", override_qtype)],
        include=None,
        exclude=None,
    )

    plan_by_key = {plan.key: plan for plan in plans}
    forced_plan = plan_by_key["double_stream_blocks.0.block.ff_i.shared_experts.w3.weight"]
    assert forced_plan.target_qtype == expected_qtype
    assert not forced_plan.quantize
    assert tensor_counts[override_qtype] == 1
    assert fallback_counts == {}


def test_hidream_shared_expert_w2_uses_ffn_down_policy() -> None:
    key = "double_stream_blocks.0.block.ff_i.shared_experts.w2.weight"

    assert _mixed_policy_qtype("Q3_K_M", "Q3_K", key, ModelHiDream(), {"attention_value": 0, "ffn_down": 0}) == "Q4_K"
    assert _mixed_policy_qtype("Q4_0", "Q4_0", key, ModelHiDream(), {"attention_value": 0, "ffn_down": 0}) == "Q4_1"
    assert _mixed_policy_qtype("Q4_K_S", "Q4_K", key, ModelHiDream(), {"attention_value": 0, "ffn_down": 0}) == "Q5_K"
    assert _mixed_policy_qtype("Q4_K_M", "Q4_K", key, ModelHiDream(), {"attention_value": 0, "ffn_down": 0}) == "Q6_K"
    assert _mixed_policy_qtype("Q5_0", "Q5_0", key, ModelHiDream(), {"attention_value": 0, "ffn_down": 0}) == "Q5_1"
    assert _mixed_policy_qtype("Q5_K_M", "Q5_K", key, ModelHiDream(), {"attention_value": 0, "ffn_down": 0}) == "Q6_K"


def test_aura_q3_k_m_promotes_c_proj_without_changing_other_file_types() -> None:
    key = "double_layers.0.mlpC.c_proj.weight"

    assert _mixed_policy_qtype("Q3_K_M", "Q3_K", key, ModelAura(), {"attention_value": 0, "ffn_down": 0}) == "Q4_K"
    assert _mixed_policy_qtype("Q4_0", "Q4_0", key, ModelAura(), {"attention_value": 0, "ffn_down": 0}) == "Q4_0"


def test_dynamic_policy_promotes_ernie_style_projection_keys() -> None:
    assert _dynamic_policy_qtype("Q5_K_M", "Q5_K", "layers.10.self_attention.to_k.weight") == "Q6_K"
    assert _dynamic_policy_qtype("Q5_K_M", "Q5_K", "layers.10.self_attention.to_out.0.weight") == "Q6_K"
    assert _dynamic_policy_qtype("Q5_K_M", "Q6_K", "layers.10.self_attention.to_v.weight") == "Q8_0"
    assert _dynamic_policy_qtype("Q5_K_M", "Q5_K", "layers.10.mlp.linear_fc2.weight") == "Q6_K"


def test_dynamic_policy_uses_more_than_ernie_key_names() -> None:
    assert _dynamic_policy_qtype("Q4_K_M", "Q4_K", "transformer_blocks.5.attn.q_proj.weight") == "Q5_K"
    assert _dynamic_policy_qtype("Q4_K_M", "Q4_K", "double_blocks.3.img_attn.out_proj.weight") == "Q5_K"
    assert _dynamic_policy_qtype("Q4_K_M", "Q4_K", "blocks.7.mlp.down_proj.weight") == "Q6_K"
    assert _dynamic_policy_qtype("Q5_K_M", "Q5_K", "layers.0.mlp.gate_proj.weight") == "Q8_0"


def test_dynamic_policy_requires_numeric_block_context() -> None:
    assert _dynamic_policy_qtype("Q5_K_M", "Q5_K", "embedder.to_q.weight") == "Q5_K"
    assert _policy_allows_quant_shape("layers.0.self_attention.to_q.weight", (256, 256), ModelHiDream(), "dynamic")
