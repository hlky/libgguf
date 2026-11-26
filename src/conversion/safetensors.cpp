#include "conversion/safetensors.h"

#include "conversion/common.h"

#include <cstring>
#include <limits>
#include <set>
#include <algorithm>
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
  unsigned depth = 0;

  void skip_ws() {
    while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) {
      ++p;
    }
  }

  json_value parse() {
    if (depth >= 64) fail("JSON nesting exceeds 64 levels");
    ++depth;
    struct depth_guard { unsigned& value; ~depth_guard() { --value; } } guard{depth};
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

  uint32_t unicode_unit() {
    uint32_t value=0;
    for (int i=0;i<4;++i) {
      if (p>=end) fail("truncated JSON unicode escape");
      const unsigned char c=*p++;
      const int digit=c>='0'&&c<='9'?c-'0':c>='a'&&c<='f'?c-'a'+10:c>='A'&&c<='F'?c-'A'+10:-1;
      if (digit<0) fail("invalid JSON unicode escape");
      value=(value<<4)|uint32_t(digit);
    }
    return value;
  }
  static bool valid_utf8(const std::string& value) {
    for (size_t i=0;i<value.size();) {
      const unsigned char first=value[i++];
      if(first<128)continue;
      int extra=first>=0xc2&&first<=0xdf?1:first>=0xe0&&first<=0xef?2:first>=0xf0&&first<=0xf4?3:-1;
      if(extra<0 || size_t(extra)>value.size()-i)return false;
      uint32_t code=first&((1u<<(6-extra))-1);
      for(int j=0;j<extra;++j){const unsigned char c=value[i++];if((c&0xc0)!=0x80)return false;code=(code<<6)|(c&63);}
      if((extra==1&&code<128)||(extra==2&&code<2048)||(extra==3&&code<65536)
          ||code>0x10ffff||(code>=0xd800&&code<=0xdfff))return false;
    }
    return true;
  }
  std::string parse_string() {
    if (p >= end || *p != '"') {
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
        case 'u': {
          uint32_t code=unicode_unit();
          if(code>=0xd800&&code<=0xdbff) {
            if(end-p<2 || p[0]!='\\' || p[1]!='u') fail("missing JSON low surrogate");
            p+=2;const auto low=unicode_unit();
            if(low<0xdc00||low>0xdfff)fail("invalid JSON low surrogate");
            code=0x10000+((code-0xd800)<<10)+(low-0xdc00);
          } else if(code>=0xdc00&&code<=0xdfff)fail("unpaired JSON low surrogate");
          if(code<0x80)out.push_back(char(code));
          else if(code<0x800){out.push_back(char(0xc0|(code>>6)));out.push_back(char(0x80|(code&63)));}
          else if(code<0x10000){out.push_back(char(0xe0|(code>>12)));out.push_back(char(0x80|((code>>6)&63)));out.push_back(char(0x80|(code&63)));}
          else {out.push_back(char(0xf0|(code>>18)));out.push_back(char(0x80|((code>>12)&63)));out.push_back(char(0x80|((code>>6)&63)));out.push_back(char(0x80|(code&63)));}
          break;
        }
        default:
          fail("invalid JSON escape");
        }
      } else {
        if (ch < 0x20) fail("invalid JSON: unescaped control character");
        out.push_back((char)ch);
      }
    }
    if (p >= end || *p != '"') {
      fail("invalid JSON: unterminated string");
    }
    ++p;
    if(!valid_utf8(out)) fail("invalid UTF-8 in JSON string");
    return out;
  }

  uint64_t parse_number() {
    uint64_t value = 0;
    if (p >= end || *p < '0' || *p > '9') {
      fail("invalid JSON: expected number");
    }
    if (*p == '0' && p + 1 < end && p[1] >= '0' && p[1] <= '9')
      fail("invalid JSON: leading zero in number");
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
    std::set<std::string> keys;
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
      if (!keys.insert(key).second) fail("invalid JSON: duplicate object key");
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

std::vector<tensor_meta> parse_safetensors_header(const input_file &input, uint64_t *data_start,
    std::map<std::string,std::string>* metadata) {
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
  parser.skip_ws();
  if (parser.p != parser.end) fail("invalid JSON: trailing content");
  if (root.kind != json_value::OBJECT) {
    fail("invalid safetensors header");
  }

  std::vector<tensor_meta> tensors;
  for (const auto &entry : root.object) {
    if (entry.first == "__metadata__") {
      if (entry.second.kind != json_value::OBJECT) fail("safetensors metadata must be an object");
      for (const auto& field : entry.second.object) {
        if (field.second.kind != json_value::STRING) fail("safetensors metadata values must be strings");
        if (metadata) metadata->emplace(field.first,field.second.str);
      }
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
    if (offsets->array[0].num > input.size - *data_start ||
        offsets->array[1].num > input.size - *data_start)
      fail("safetensors offset exceeds file data");
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
  std::vector<std::pair<uint64_t,uint64_t>> spans;
  for (const auto& tensor : tensors) if (tensor.data_begin != tensor.data_end)
    spans.emplace_back(tensor.data_begin,tensor.data_end);
  std::sort(spans.begin(),spans.end());
  for (size_t i=1;i<spans.size();++i) if (spans[i].first < spans[i-1].second)
    fail("safetensors tensor byte ranges overlap");
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
