from __future__ import annotations

from libgguf.quantize import ModelAura, ModelHiDream, _dynamic_policy_qtype, _mixed_policy_qtype, _policy_allows_quant_shape


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
