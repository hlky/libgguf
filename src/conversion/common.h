#pragma once

#include <cstdint>
#include <string>
#include <vector>

[[noreturn]] void fail(const std::string & message);
bool ends_with(const std::string & value, const std::string & suffix);
bool starts_with(const std::string & value, const std::string & prefix);
std::string upper_ascii(std::string value);
bool checked_mul_u64(uint64_t a, uint64_t b, uint64_t * out);
uint64_t product_shape(const std::vector<int64_t> & shape);
