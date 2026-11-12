#include "conversion/safetensors.h"

#include "conversion/common.h"

#include <cstring>
#include <limits>
#include <utility>

size_t source_dtype_size(native_source_dtype dtype) {
  switch (dtype) {
  case NATIVE_DTYPE_F32:
    return sizeof(float);
  case NATIVE_DTYPE_F16:
  case NATIVE_DTYPE_BF16:
    return sizeof(uint16_t);
  }
  return 0;
}

native_source_dtype parse_source_dtype(const std::string &dtype) {
  if (dtype == "F32") {
    return NATIVE_DTYPE_F32;
  }
  if (dtype == "F16") {
    return NATIVE_DTYPE_F16;
  }
  if (dtype == "BF16") {
    return NATIVE_DTYPE_BF16;
  }
  fail("unsupported source dtype: " + dtype);
}

struct json_value {
  enum kind_t { NIL, STRING, NUMBER, ARRAY, OBJECT } kind = NIL;
  std::string str;
  uint64_t num = 0;
  std::vector<json_value> array;
  std::vector<std::pair<std::string, json_value>> object;
};

struct json_parser {
  const char *p;
  const char *end;

  void skip_ws() {
    while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) {
      ++p;
    }
  }

  json_value parse() {
    skip_ws();
    if (p >= end) {
      fail("invalid JSON: unexpected end");
    }
    if (*p == '"') {
      json_value v;
      v.kind = json_value::STRING;
      v.str = parse_string();
      return v;
    }
    if (*p == '{') {
      return parse_object();
    }
    if (*p == '[') {
      return parse_array();
    }
    if (*p >= '0' && *p <= '9') {
      json_value v;
      v.kind = json_value::NUMBER;
      v.num = parse_number();
      return v;
    }
    fail("invalid JSON: unsupported value");
  }

  std::string parse_string() {
    if (*p != '"') {
      fail("invalid JSON: expected string");
    }
    ++p;
    std::string out;
    while (p < end && *p != '"') {
      unsigned char ch = (unsigned char)*p++;
      if (ch == '\\') {
        if (p >= end) {
          fail("invalid JSON escape");
        }
        char esc = *p++;
        switch (esc) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        default:
          fail("invalid JSON escape");
        }
      } else {
        out.push_back((char)ch);
      }
    }
    if (p >= end || *p != '"') {
      fail("invalid JSON: unterminated string");
    }
    ++p;
    return out;
  }

  uint64_t parse_number() {
    uint64_t value = 0;
    if (p >= end || *p < '0' || *p > '9') {
      fail("invalid JSON: expected number");
    }
    while (p < end && *p >= '0' && *p <= '9') {
      const uint64_t digit = (uint64_t)(*p - '0');
      if (value > (std::numeric_limits<uint64_t>::max() - digit) / 10) {
        fail("invalid JSON: number is too large");
      }
      value = value * 10 + digit;
      ++p;
    }
    return value;
  }

  json_value parse_array() {
    json_value v;
    v.kind = json_value::ARRAY;
    ++p;
    skip_ws();
    if (p < end && *p == ']') {
      ++p;
      return v;
    }
    while (true) {
      v.array.push_back(parse());
      skip_ws();
      if (p >= end) {
        fail("invalid JSON array");
      }
      if (*p == ']') {
        ++p;
        return v;
      }
      if (*p != ',') {
        fail("invalid JSON array: expected comma");
      }
      ++p;
    }
  }

  json_value parse_object() {
    json_value v;
    v.kind = json_value::OBJECT;
    ++p;
    skip_ws();
    if (p < end && *p == '}') {
      ++p;
      return v;
    }
    while (true) {
      skip_ws();
      std::string key = parse_string();
      skip_ws();
      if (p >= end || *p != ':') {
        fail("invalid JSON object: expected colon");
      }
      ++p;
      json_value value = parse();
      v.object.push_back({key, std::move(value)});
      skip_ws();
      if (p >= end) {
        fail("invalid JSON object");
      }
      if (*p == '}') {
        ++p;
        return v;
      }
      if (*p != ',') {
        fail("invalid JSON object: expected comma");
      }
      ++p;
    }
  }
};

const json_value *object_get(const json_value &object, const std::string &key) {
  if (object.kind != json_value::OBJECT) {
    return nullptr;
  }
  for (const auto &item : object.object) {
    if (item.first == key) {
      return &item.second;
    }
  }
  return nullptr;
}

std::vector<tensor_meta> parse_safetensors_header(const input_file &input, uint64_t *data_start) {
  if (input.size < 8) {
    fail("invalid safetensors file");
  }
  uint64_t header_len = 0;
  std::string error;
  if (!input.read_at(0, &header_len, sizeof(header_len), &error)) {
    fail(error);
  }
  if (header_len > input.size - 8) {
    fail("invalid safetensors header length");
  }
  std::vector<uint8_t> header((size_t)header_len);
  if (!input.read_at(8, header.data(), header_len, &error)) {
    fail(error);
  }
  *data_start = 8 + header_len;
  json_parser parser{(const char *)header.data(), (const char *)(header.data() + header.size())};
  json_value root = parser.parse();
  if (root.kind != json_value::OBJECT) {
    fail("invalid safetensors header");
  }

  std::vector<tensor_meta> tensors;
  for (const auto &entry : root.object) {
    if (entry.first == "__metadata__") {
      continue;
    }
    const json_value &info = entry.second;
    const json_value *dtype = object_get(info, "dtype");
    const json_value *shape = object_get(info, "shape");
    const json_value *offsets = object_get(info, "data_offsets");
    if (!dtype || dtype->kind != json_value::STRING || !shape || shape->kind != json_value::ARRAY || !offsets || offsets->kind != json_value::ARRAY || offsets->array.size() != 2) {
      fail("invalid safetensors tensor entry for " + entry.first);
    }
    tensor_meta meta;
    meta.source_key = entry.first;
    meta.key = entry.first;
    meta.dtype = dtype->str;
    for (const json_value &dim : shape->array) {
      if (dim.kind != json_value::NUMBER || dim.num > (uint64_t)std::numeric_limits<int64_t>::max()) {
        fail("invalid safetensors shape for " + entry.first);
      }
      meta.shape.push_back((int64_t)dim.num);
    }
    if (offsets->array[0].kind != json_value::NUMBER || offsets->array[1].kind != json_value::NUMBER) {
      fail("invalid safetensors data_offsets for " + entry.first);
    }
    meta.data_begin = *data_start + offsets->array[0].num;
    meta.data_end = *data_start + offsets->array[1].num;
    const uint64_t values = product_shape(meta.shape);
    const native_source_dtype source_dtype = parse_source_dtype(meta.dtype);
    uint64_t expected_source_bytes = 0;
    if (!checked_mul_u64(values, (uint64_t)source_dtype_size(source_dtype), &expected_source_bytes)) {
      fail("source byte count is too large for " + entry.first);
    }
    if (meta.data_end < meta.data_begin || meta.data_end - meta.data_begin != expected_source_bytes) {
      fail("safetensors byte length does not match shape/dtype for " + entry.first);
    }
    if (meta.data_end > input.size) {
      fail("safetensors byte range exceeds file size for " + entry.first);
    }
    tensors.push_back(std::move(meta));
  }
  return tensors;
}

void strip_prefix(std::vector<tensor_meta> &tensors) {
  // Decoder checkpoints use model.* alongside an unprefixed lm_head. These
  // are original tensor names, not a diffusion wrapper to remove.
  for (const tensor_meta &tensor : tensors) {
    if (tensor.key == "model.layers.0.block_sparse_moe.gate.weight") {
      return;
    }
  }
  const std::vector<std::string> prefixes = {"model.diffusion_model.", "model."};
  for (const std::string &prefix : prefixes) {
    bool any = false;
    for (const tensor_meta &tensor : tensors) {
      if (starts_with(tensor.key, prefix)) {
        any = true;
        break;
      }
    }
    if (any) {
      std::vector<tensor_meta> stripped;
      for (tensor_meta tensor : tensors) {
        if (starts_with(tensor.key, prefix)) {
          tensor.key = tensor.key.substr(prefix.size());
          stripped.push_back(std::move(tensor));
        }
      }
      tensors = std::move(stripped);
      return;
    }
  }

  bool all_net = !tensors.empty();
  for (const tensor_meta &tensor : tensors) {
    if (!starts_with(tensor.key, "net.")) {
      all_net = false;
      break;
    }
  }
  if (all_net) {
    for (tensor_meta &tensor : tensors) {
      tensor.key = tensor.key.substr(4);
    }
  }
}
