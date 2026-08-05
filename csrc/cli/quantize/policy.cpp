#include "cli/quantize/policy.h"

#include "conversion/common.h"
#include "conversion/safetensors.h"

#include <libgguf/libgguf.h>

#include <algorithm>
#include <limits>
#include <map>
#include <set>

constexpr int64_t QUANTIZATION_THRESHOLD = 1024;
constexpr int64_t REARRANGE_THRESHOLD = 512;
constexpr size_t MAX_TENSOR_NAME_LENGTH = 127;
bool qtype_matches_source(native_source_dtype dtype, ggml_type qtype) {
  return (dtype == NATIVE_DTYPE_F32 && qtype == GGML_TYPE_F32) ||
         (dtype == NATIVE_DTYPE_F16 && qtype == GGML_TYPE_F16) ||
         (dtype == NATIVE_DTYPE_BF16 && qtype == GGML_TYPE_BF16);
}

std::string qtype_name(ggml_type qtype) {
  switch (qtype) {
  case GGML_TYPE_F32:
    return "F32";
  case GGML_TYPE_F16:
    return "F16";
  case GGML_TYPE_BF16:
    return "BF16";
  case GGML_TYPE_Q1_0:
    return "Q1_0";
  case GGML_TYPE_Q4_0:
    return "Q4_0";
  case GGML_TYPE_Q4_1:
    return "Q4_1";
  case GGML_TYPE_Q5_0:
    return "Q5_0";
  case GGML_TYPE_Q5_1:
    return "Q5_1";
  case GGML_TYPE_Q8_0:
    return "Q8_0";
  case GGML_TYPE_Q2_K:
    return "Q2_K";
  case GGML_TYPE_Q3_K:
    return "Q3_K";
  case GGML_TYPE_Q4_K:
    return "Q4_K";
  case GGML_TYPE_Q5_K:
    return "Q5_K";
  case GGML_TYPE_Q6_K:
    return "Q6_K";
  default:
    return "UNKNOWN";
  }
}

bool is_quantized_qtype(ggml_type qtype) {
  return qtype != GGML_TYPE_F32 && qtype != GGML_TYPE_F16 && qtype != GGML_TYPE_BF16;
}

uint64_t qtype_block_size(ggml_type qtype) {
  switch (qtype) {
  case GGML_TYPE_F32:
  case GGML_TYPE_F16:
  case GGML_TYPE_BF16:
    return 1;
  case GGML_TYPE_Q4_0:
  case GGML_TYPE_Q4_1:
  case GGML_TYPE_Q5_0:
  case GGML_TYPE_Q5_1:
  case GGML_TYPE_Q8_0:
    return 32;
  case GGML_TYPE_Q1_0:
    return 128;
  case GGML_TYPE_Q2_K:
  case GGML_TYPE_Q3_K:
  case GGML_TYPE_Q4_K:
  case GGML_TYPE_Q5_K:
  case GGML_TYPE_Q6_K:
    return 256;
  default:
    return 0;
  }
}

ggml_type qtype_from_name(const std::string &name) {
  if (name == "F32") return GGML_TYPE_F32;
  if (name == "F16") return GGML_TYPE_F16;
  if (name == "BF16") return GGML_TYPE_BF16;
  if (name == "Q1_0") return GGML_TYPE_Q1_0;
  if (name == "Q4_0") return GGML_TYPE_Q4_0;
  if (name == "Q4_1") return GGML_TYPE_Q4_1;
  if (name == "Q5_0") return GGML_TYPE_Q5_0;
  if (name == "Q5_1") return GGML_TYPE_Q5_1;
  if (name == "Q8_0") return GGML_TYPE_Q8_0;
  if (name == "Q2_K") return GGML_TYPE_Q2_K;
  if (name == "Q3_K") return GGML_TYPE_Q3_K;
  if (name == "Q4_K") return GGML_TYPE_Q4_K;
  if (name == "Q5_K") return GGML_TYPE_Q5_K;
  if (name == "Q6_K") return GGML_TYPE_Q6_K;
  fail("unsupported GGML tensor type: " + name);
}

uint32_t file_type_value(const std::string &file_type_name) {
  if (file_type_name == "Q4_0") return 2;
  if (file_type_name == "Q4_1") return 3;
  if (file_type_name == "Q8_0") return 7;
  if (file_type_name == "Q5_0") return 8;
  if (file_type_name == "Q5_1") return 9;
  if (file_type_name == "Q2_K") return 10;
  if (file_type_name == "Q3_K_S") return 11;
  if (file_type_name == "Q3_K_M") return 12;
  if (file_type_name == "Q3_K_L") return 13;
  if (file_type_name == "Q4_K_S") return 14;
  if (file_type_name == "Q4_K_M") return 15;
  if (file_type_name == "Q5_K_S") return 16;
  if (file_type_name == "Q5_K_M") return 17;
  if (file_type_name == "Q6_K") return 18;
  if (file_type_name == "Q2_K_S") return 21;
  if (file_type_name == "Q1_0") return 27;
  fail("unsupported output file type: " + file_type_name);
}

std::pair<std::string, std::string> parse_qtype(const std::string &input) {
  std::string file_type = upper_ascii(input);
  if (file_type == "Q8_K") {
    fail("Q8_K output is not supported: this repo has metadata for Q8_K but no output quantizer");
  }
  if (file_type.rfind("IQ", 0) == 0 || file_type.rfind("TQ", 0) == 0 || file_type == "MXFP4" || file_type == "NVFP4") {
    fail("non-Q/K quantization families are not supported by this native executable yet");
  }
  if (file_type == "Q3_K") file_type = "Q3_K_M";
  if (file_type == "Q4_K") file_type = "Q4_K_M";
  if (file_type == "Q5_K") file_type = "Q5_K_M";

  std::string tensor_qtype = file_type;
  if (file_type == "Q3_K_S" || file_type == "Q3_K_M" || file_type == "Q3_K_L") tensor_qtype = "Q3_K";
  if (file_type == "Q4_K_S" || file_type == "Q4_K_M") tensor_qtype = "Q4_K";
  if (file_type == "Q5_K_S" || file_type == "Q5_K_M") tensor_qtype = "Q5_K";
  if (file_type == "Q2_K_S") tensor_qtype = "Q2_K";

  if (!is_supported_quant_qtype_name(tensor_qtype)) {
    fail("unsupported direct quantization type: " + input);
  }
  (void)file_type_value(file_type);
  return {file_type, tensor_qtype};
}

uint64_t qtype_nbytes(const std::vector<int64_t> &shape, ggml_type qtype) {
  const uint64_t values = product_shape(shape);
  const uint64_t block_size = qtype_block_size(qtype);
  if (block_size == 0) {
    fail("unsupported qtype: " + qtype_name(qtype));
  }
  if (values % block_size != 0) {
    fail("Tensor with incompatible shape cannot be stored as " + qtype_name(qtype));
  }
  if (shape.empty()) {
    const size_t row_size = libgguf_row_size(qtype, 1);
    if (row_size == 0) {
      fail("unsupported qtype or scalar shape for " + qtype_name(qtype));
    }
    return (uint64_t)row_size;
  }
  const int64_t n_per_row = shape.back();
  const size_t row_size = libgguf_row_size(qtype, n_per_row);
  if (row_size == 0) {
    fail("unsupported qtype or row width for " + qtype_name(qtype));
  }
  return (values / (uint64_t)n_per_row) * (uint64_t)row_size;
}

bool fnmatch_case(const char *pattern, const char *text) {
  while (*pattern) {
    if (*pattern == '*') {
      while (*pattern == '*') {
        ++pattern;
      }
      if (!*pattern) {
        return true;
      }
      while (*text) {
        if (fnmatch_case(pattern, text)) {
          return true;
        }
        ++text;
      }
      return false;
    }
    if (*pattern == '?') {
      if (!*text) {
        return false;
      }
      ++pattern;
      ++text;
      continue;
    }
    if (*pattern != *text) {
      return false;
    }
    ++pattern;
    ++text;
  }
  return *text == '\0';
}

bool matches_any(const std::string &name, const std::vector<std::string> &patterns) {
  for (const std::string &pattern : patterns) {
    if (fnmatch_case(pattern.c_str(), name.c_str())) {
      return true;
    }
  }
  return false;
}

bool is_digits(const std::string &value) {
  if (value.empty()) {
    return false;
  }
  for (char ch : value) {
    if (ch < '0' || ch > '9') {
      return false;
    }
  }
  return true;
}

std::vector<std::string> split_dot(const std::string &value) {
  std::vector<std::string> parts;
  size_t start = 0;
  while (start <= value.size()) {
    size_t pos = value.find('.', start);
    if (pos == std::string::npos) {
      parts.push_back(value.substr(start));
      break;
    }
    parts.push_back(value.substr(start, pos - start));
    start = pos + 1;
  }
  return parts;
}

int dynamic_layer_index(const std::string &key) {
  static const std::set<std::string> block_names = {
      "blocks", "double_blocks", "double_layers", "double_stream_blocks", "input_blocks", "joint_blocks",
      "layers", "output_blocks", "single_blocks", "single_stream_blocks", "transformer_blocks"};
  const std::vector<std::string> parts = split_dot(key);
  for (size_t i = 0; i + 1 < parts.size(); ++i) {
    if (block_names.count(parts[i]) && is_digits(parts[i + 1])) {
      return std::stoi(parts[i + 1]);
    }
  }
  for (size_t i = 0; i + 1 < parts.size(); ++i) {
    if ((parts[i] == "context_refiner" || parts[i] == "noise_refiner") && is_digits(parts[i + 1])) {
      return std::stoi(parts[i + 1]);
    }
  }
  return -1;
}

std::string promote_k_qtype(const std::string &qtype, int steps = 1) {
  static const std::vector<std::string> order = {"Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0"};
  for (size_t i = 0; i < order.size(); ++i) {
    if (order[i] == qtype) {
      return order[std::min<size_t>(i + (size_t)steps, order.size() - 1)];
    }
  }
  return qtype;
}

bool contains_any(const std::string &name, const std::vector<std::string> &markers) {
  for (const std::string &marker : markers) {
    if (name.find(marker) != std::string::npos) {
      return true;
    }
  }
  return false;
}


std::vector<model_template> model_templates() {
  return {
      {"flux", false, {{"transformer_blocks.0.attn.norm_added_k.weight"}, {"double_blocks.0.img_attn.proj.weight"}}, {"transformer_blocks.0.attn.norm_added_k.weight"}, {}, {}},
      {"sd3", false, {{"transformer_blocks.0.attn.add_q_proj.weight"}, {"joint_blocks.0.x_block.attn.qkv.weight"}}, {"transformer_blocks.0.attn.add_q_proj.weight"}, {}, {}},
      {"aura", false, {{"double_layers.3.modX.1.weight"}, {"joint_transformer_blocks.3.ff_context.out_projection.weight"}}, {"joint_transformer_blocks.3.ff_context.out_projection.weight"}, {}, {}},
      {"hidream", false, {{"caption_projection.0.linear.weight", "double_stream_blocks.0.block.ff_i.shared_experts.w3.weight"}}, {}, {".ff_i.gate.weight", "img_emb.emb_pos"}, {}},
      {"cosmos", false, {{"blocks.0.mlp.layer1.weight", "blocks.0.adaln_modulation_cross_attn.1.weight"}}, {}, {"pos_embedder"}, {"_extra_state", "accum_"}},
      {"hyvid", false, {{"double_blocks.0.img_attn_proj.weight", "txt_in.individual_token_refiner.blocks.1.self_attn_qkv.weight"}}, {}, {}, {}},
      {"wan", false, {{"blocks.0.self_attn.norm_q.weight", "text_embedding.2.weight", "head.modulation"}}, {}, {".modulation"}, {}},
      {"ltxv", false, {{"adaln_single.emb.timestep_embedder.linear_2.weight", "transformer_blocks.27.scale_shift_table", "caption_projection.linear_2.weight"}}, {}, {"scale_shift_table"}, {}},
      {"sdxl", true, {{"down_blocks.0.downsamplers.0.conv.weight", "add_embedding.linear_1.weight"}, {"input_blocks.3.0.op.weight", "input_blocks.6.0.op.weight", "output_blocks.2.2.conv.weight", "output_blocks.5.2.conv.weight"}, {"label_emb.0.0.weight"}}, {}, {}, {}},
      {"sd1", true, {{"down_blocks.0.downsamplers.0.conv.weight"}, {"input_blocks.3.0.op.weight", "input_blocks.6.0.op.weight", "input_blocks.9.0.op.weight", "output_blocks.2.1.conv.weight", "output_blocks.5.2.conv.weight", "output_blocks.8.2.conv.weight"}}, {}, {}, {}},
      {"lumina2", false, {{"cap_embedder.1.weight", "context_refiner.0.attention.qkv.weight"}}, {}, {}, {}},
  };
}

model_template detect_arch(const std::vector<tensor_meta> &tensors) {
  std::set<std::string> keys;
  for (const tensor_meta &tensor : tensors) {
    keys.insert(tensor.key);
  }

  for (const model_template &model : model_templates()) {
    bool matched = false;
    for (const auto &match_list : model.keys_detect) {
      bool all = true;
      for (const std::string &key : match_list) {
        if (!keys.count(key)) {
          all = false;
          break;
        }
      }
      if (all) {
        matched = true;
        break;
      }
    }
    if (matched) {
      for (const std::string &key : model.keys_banned) {
        if (keys.count(key)) {
          fail("Model architecture not allowed for conversion; use the reference checkpoint key format");
        }
      }
      return model;
    }
  }
  fail("Unknown model architecture");
}

const std::vector<std::string> &arch_skip_patterns(const std::string &arch) {
  static const std::vector<std::string> empty;
  static const std::map<std::string, std::vector<std::string>> patterns = {
      {"flux", {"txt_in.*", "img_in.*", "time_in.*", "vector_in.*", "guidance_in.*", "final_layer.*"}},
      {"sd1", {"class_embedding.*", "time_embedding.*", "add_embedding.*", "time_embed.*", "label_emb.*", "conv_in.*", "conv_out.*", "input_blocks.0.0.weight", "out.2.weight"}},
      {"sdxl", {"class_embedding.*", "time_embedding.*", "add_embedding.*", "time_embed.*", "label_emb.*", "conv_in.*", "conv_out.*", "input_blocks.0.0.weight", "out.2.weight"}},
      {"sd3", {"final_layer.*", "time_text_embed.*", "context_embedder.*", "t_embedder.*", "y_embedder.*", "x_embedder.*", "proj_out.weight", "pos_embed"}},
      {"aura", {"t_embedder.*", "init_x_linear.*", "modF.1.weight", "cond_seq_linear.weight", "final_linear.weight", "positional_encoding", "register_tokens"}},
      {"ltxv", {"adaln_single.*", "caption_projection.*", "patchify_proj.*", "proj_out.*", "*scale_shift_table*"}},
      {"hyvid", {"txt_in.*", "img_in.*", "time_in.*", "vector_in.*", "guidance_in.*", "final_layer.*"}},
      {"wan", {"*modulation.*", "patch_embedding.*", "text_embedding.*", "time_projection.*", "time_embedding.*", "img_emb.*", "head.*"}},
      {"hidream", {"p_embedder.*", "t_embedder.*", "x_embedder.*", "final_layer.*", "*.ff_i.gate.weight", "caption_projection.*"}},
      {"cosmos", {"p_embedder.*", "t_embedder.*", "t_embedding_norm.*", "x_embedder.*", "pos_embedder.*", "final_layer.*"}},
      {"lumina2", {"t_embedder.*", "x_embedder.*", "final_layer.*", "cap_embedder.*", "context_refiner.*", "noise_refiner.*"}},
  };
  auto it = patterns.find(arch);
  return it == patterns.end() ? empty : it->second;
}

bool policy_allows_quant_shape(const std::string &key, const std::vector<int64_t> &shape, const model_template &model, const std::string &policy) {
  if (shape.size() != 2) {
    return false;
  }
  if (!ends_with(key, "weight")) {
    return false;
  }
  if (policy == "uniform") {
    return true;
  }
  if (matches_any(key, arch_skip_patterns(model.arch))) {
    return false;
  }
  return true;
}

std::string mixed_policy_qtype(const std::string &file_type, const std::string &base_qtype, const std::string &key, const model_template &model, std::map<std::string, int> &counters) {
  static const std::vector<std::string> attention_value = {
      "*attn_v.weight*", "*.to_v.weight*", "*.v.weight*", "*.attn.w1v.weight*", "*.attn.w2v.weight*", "*_attn.v_proj.weight*"};
  static const std::vector<std::string> fused_qkv = {"*attn_qkv.weight*", "*attn.qkv.weight*", "*attention.qkv.weight*"};
  static const std::vector<std::string> ffn_down = {
      "*ffn_down*", "*experts.*.w2.weight*", "*shared_experts.w2.weight*", "*.ffn.2.weight*", "*.ff.net.2.weight*", "*.mlp.layer2.weight*", "*.adaln_modulation_mlp.2.weight*", "*.feed_forward.w2.weight*"};

  std::string qtype = base_qtype;
  if (matches_any(key, attention_value)) {
    if (file_type == "Q2_K") qtype = "Q3_K";
    else if (file_type == "Q3_K_M") qtype = counters["attention_value"] < 2 ? "Q5_K" : "Q4_K";
    else if (file_type == "Q3_K_L") qtype = "Q5_K";
    else if (file_type == "Q4_K_M" || file_type == "Q5_K_M") qtype = "Q6_K";
    else if (file_type == "Q4_K_S" && counters["attention_value"] < 4) qtype = "Q5_K";
    counters["attention_value"] += 1;
  } else if (matches_any(key, fused_qkv)) {
    if (file_type == "Q3_K_M" || file_type == "Q3_K_L") qtype = "Q4_K";
    else if (file_type == "Q4_K_M") qtype = "Q5_K";
    else if (file_type == "Q5_K_M") qtype = "Q6_K";
  } else if (matches_any(key, ffn_down)) {
    if (file_type == "Q3_K_M") qtype = "Q4_K";
    else if (file_type == "Q3_K_L") qtype = "Q5_K";
    else if (file_type == "Q4_K_S") qtype = "Q5_K";
    else if (file_type == "Q4_K_M" || file_type == "Q5_K_M") qtype = "Q6_K";
    else if (file_type == "Q4_0") qtype = "Q4_1";
    else if (file_type == "Q5_0") qtype = "Q5_1";
    counters["ffn_down"] += 1;
  } else if (model.arch == "aura" && file_type == "Q3_K_M" && fnmatch_case("double_layers.*.mlp?.c_proj.weight", key.c_str())) {
    qtype = "Q4_K";
  }
  return qtype;
}

std::string dynamic_policy_qtype(const std::string &file_type, const std::string &comfy_qtype, const std::string &key, int max_layer) {
  static const std::vector<std::string> attention_value = {
      "*attn_v.weight*", "*.to_v.weight*", "*.v.weight*", "*.attn.w1v.weight*", "*.attn.w2v.weight*", "*_attn.v_proj.weight*"};
  static const std::vector<std::string> fused_qkv = {"*attn_qkv.weight*", "*attn.qkv.weight*", "*attention.qkv.weight*"};
  static const std::vector<std::string> attention_qko = {
      "*.to_q.weight", "*.q_proj.weight", "*.query_proj.weight", "*_attn.q_proj.weight", "*attn_q.weight",
      "*.to_k.weight", "*.k_proj.weight", "*.key_proj.weight", "*_attn.k_proj.weight", "*attn_k.weight",
      "*.to_out.0.weight", "*.to_out.weight", "*.o_proj.weight", "*.out_proj.weight", "*.proj_out.weight", "*_attn.proj.weight"};
  static const std::vector<std::string> ffn_up_down = {
      "*ffn_down*", "*experts.*.w2.weight*", "*shared_experts.w2.weight*", "*.ffn.2.weight*", "*.ff.net.2.weight*",
      "*.mlp.layer2.weight*", "*.adaln_modulation_mlp.2.weight*", "*.feed_forward.w2.weight*",
      "*.mlp.linear_fc2.weight", "*.linear_fc2.weight", "*.down_proj.weight", "*.c_proj.weight", "*.out_projection.weight",
      "*.mlp.up_proj.weight", "*.up_proj.weight", "*.w3.weight", "*.fc1.weight", "*.linear_fc1.weight", "*.mlp.layer1.weight"};
  static const std::vector<std::string> ffn_gate = {"*.mlp.gate_proj.weight", "*.gate_proj.weight", "*.w1.weight"};
  static const std::vector<std::string> ffn = {
      "*ffn_down*", "*experts.*.w2.weight*", "*shared_experts.w2.weight*", "*.ffn.2.weight*", "*.ff.net.2.weight*",
      "*.mlp.layer2.weight*", "*.adaln_modulation_mlp.2.weight*", "*.feed_forward.w2.weight*",
      "*.mlp.linear_fc2.weight", "*.linear_fc2.weight", "*.down_proj.weight", "*.c_proj.weight", "*.out_projection.weight",
      "*.mlp.up_proj.weight", "*.up_proj.weight", "*.w3.weight", "*.fc1.weight", "*.linear_fc1.weight", "*.mlp.layer1.weight",
      "*.mlp.gate_proj.weight", "*.gate_proj.weight", "*.w1.weight"};

  const int layer = dynamic_layer_index(key);
  if (layer < 0) {
    return comfy_qtype;
  }
  const bool early = layer <= 1;
  const bool final_tail = max_layer >= 0 && layer >= max_layer - 1;
  if (matches_any(key, attention_value)) {
    if (file_type == "Q2_K" || file_type == "Q3_K_M" || file_type == "Q3_K_L") return "Q6_K";
    if (file_type == "Q4_K_S" || file_type == "Q4_K_M" || file_type == "Q5_K_S" || file_type == "Q5_K_M") return "Q8_0";
    return promote_k_qtype(comfy_qtype);
  }
  if (matches_any(key, fused_qkv)) {
    return promote_k_qtype(comfy_qtype, early ? 2 : 1);
  }
  if (matches_any(key, attention_qko)) {
    std::string qtype = promote_k_qtype(comfy_qtype, early ? 2 : 1);
    if (matches_any(key, {"*.to_out.0.weight", "*.to_out.weight", "*.o_proj.weight", "*.out_proj.weight", "*.proj_out.weight", "*_attn.proj.weight"})) {
      if ((file_type == "Q2_K" || file_type == "Q3_K_M" || file_type == "Q4_K_M") && ((layer >= 15 && layer <= 24) || (layer >= 28 && !final_tail))) {
        qtype = promote_k_qtype(qtype);
      } else if (file_type == "Q5_K_M" && (layer == 3 || layer == 24 || (layer >= 31 && !final_tail))) {
        qtype = "Q8_0";
      }
    }
    return qtype;
  }
  if (matches_any(key, ffn)) {
    std::string qtype = promote_k_qtype(comfy_qtype, early && (file_type == "Q4_K_M" || file_type == "Q5_K_M") ? 2 : 1);
    if (matches_any(key, ffn_up_down)) {
      if ((file_type == "Q2_K" || file_type == "Q3_K_M" || file_type == "Q4_K_M") && (layer == 7 || layer == 21 || (layer >= 23 && !final_tail))) {
        qtype = promote_k_qtype(qtype);
      } else if (file_type == "Q5_K_M" && (layer == 7 || (layer >= 27 && !final_tail))) {
        qtype = "Q8_0";
      }
    }
    (void)ffn_gate;
    return qtype;
  }
  return comfy_qtype;
}

ggml_type base_storage_type_for_meta(const tensor_meta &tensor, const model_template &model) {
  const int64_t n_params = (int64_t)product_shape(tensor.shape);
  ggml_type data_qtype = tensor.dtype == "BF16" ? GGML_TYPE_BF16 : GGML_TYPE_F16;
  if (tensor.dtype == "F32" || tensor.dtype == "BF16") {
    if (tensor.shape.size() == 1 || n_params <= QUANTIZATION_THRESHOLD || contains_any(tensor.key, model.keys_hiprec)) {
      data_qtype = GGML_TYPE_F32;
    }
  }
  return data_qtype;
}

std::vector<int64_t> shape_fix_shape(const tensor_meta &tensor, const model_template &model, std::vector<kv_value> &kv) {
  std::vector<int64_t> shape = tensor.shape;
  const int64_t n_params = (int64_t)product_shape(shape);
  if (model.shape_fix && shape.size() > 1 && n_params >= REARRANGE_THRESHOLD && n_params % 256 == 0 && shape.back() % 256 != 0) {
    kv_value value;
    value.key = "comfy.gguf.orig_shape." + tensor.key;
    value.type = GGUF_TYPE_ARRAY;
    value.arr_i64 = shape;
    kv.push_back(std::move(value));
    return {n_params / 256, 256};
  }
  return shape;
}

std::string override_qtype(const std::string &key, const std::vector<std::pair<std::string, std::string>> &overrides) {
  for (const auto &item : overrides) {
    if (fnmatch_case(item.first.c_str(), key.c_str())) {
      return item.second;
    }
  }
  return "";
}

std::vector<tensor_plan> prepare_plans(
    const std::vector<tensor_meta> &tensors,
    const model_template &model,
    const std::string &file_type_name,
    const std::string &base_qtype_name,
    const cli_options &options,
    std::vector<kv_value> &kv,
    std::vector<tensor_info> &infos,
    conversion_result &result) {
  std::vector<tensor_plan> plans;
  std::map<std::string, int> counters = {{"attention_value", 0}, {"ffn_down", 0}};
  int max_dynamic_layer = -1;
  for (const tensor_meta &tensor : tensors) {
    max_dynamic_layer = std::max(max_dynamic_layer, dynamic_layer_index(tensor.key));
  }

  for (const tensor_meta &tensor : tensors) {
    if (contains_any(tensor.key, model.keys_ignore)) {
      continue;
    }
    if (tensor.key.size() > MAX_TENSOR_NAME_LENGTH) {
      fail("Can only handle tensor names up to 127 characters: " + tensor.key);
    }
    ggml_type storage_qtype = base_storage_type_for_meta(tensor, model);
    std::vector<int64_t> write_shape = shape_fix_shape(tensor, model, kv);
    bool quantize = policy_allows_quant_shape(tensor.key, write_shape, model, options.policy);
    if (matches_any(tensor.key, options.include) && write_shape.size() == 2) {
      quantize = true;
    }
    if (matches_any(tensor.key, options.exclude)) {
      quantize = false;
    }

    ggml_type target_qtype = storage_qtype;
    const std::string forced = override_qtype(tensor.key, options.tensor_overrides);
    if (!forced.empty()) {
      target_qtype = qtype_from_name(parse_tensor_qtype_name(forced));
      quantize = is_quantized_qtype(target_qtype);
    } else if (quantize) {
      std::string target_name = base_qtype_name;
      if (options.policy == "comfy" || options.policy == "dynamic") {
        target_name = mixed_policy_qtype(file_type_name, base_qtype_name, tensor.key, model, counters);
        if (options.policy == "dynamic") {
          target_name = dynamic_policy_qtype(file_type_name, target_name, tensor.key, max_dynamic_layer);
        }
      }
      target_qtype = qtype_from_name(target_name);
    }

    if (quantize && is_quantized_qtype(target_qtype)) {
      const uint64_t block_size = qtype_block_size(target_qtype);
      if (block_size == 0 || (uint64_t)write_shape.back() % block_size != 0) {
        result.fallback_counts[qtype_name(target_qtype)] += 1;
        target_qtype = GGML_TYPE_F16;
        quantize = false;
      }
    }

    tensor_plan plan;
    plan.key = tensor.key;
    plan.source_dtype = parse_source_dtype(tensor.dtype);
    plan.qtype = target_qtype;
    plan.source_shape = tensor.shape;
    plan.write_shape = write_shape;
    plan.data_begin = tensor.data_begin;
    plan.data_end = tensor.data_end;
    plan.n_values = product_shape(write_shape);
    plan.n_per_row = write_shape.empty() ? 1 : (uint64_t)write_shape.back();
    plan.n_rows = plan.n_values / plan.n_per_row;
    plan.expected_nbytes = qtype_nbytes(write_shape, target_qtype);

    tensor_info info;
    info.key = tensor.key;
    info.gguf_shape = write_shape;
    info.qtype = target_qtype;
    info.nbytes = plan.expected_nbytes;
    infos.push_back(info);
    plans.push_back(std::move(plan));
    result.tensor_type_counts[qtype_name(target_qtype)] += 1;
  }
  return plans;
}


