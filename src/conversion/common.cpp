#include "conversion/common.h"

#include <limits>
#include <stdexcept>

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error(message);
}

bool ends_with(const std::string & value, const std::string & suffix) {
    return value.size() >= suffix.size() && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool starts_with(const std::string & value, const std::string & prefix) {
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

std::string upper_ascii(std::string value) {
    for (char & ch : value) {
        if (ch >= 'a' && ch <= 'z') {
            ch = char(ch - 'a' + 'A');
        }
    }
    return value;
}

bool checked_mul_u64(uint64_t a, uint64_t b, uint64_t * out) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        return false;
    }
    *out = a * b;
    return true;
}

uint64_t product_shape(const std::vector<int64_t> & shape) {
    uint64_t total = 1;
    for (int64_t dim : shape) {
        if (dim <= 0) {
            fail("shape dimensions must be positive");
        }
        uint64_t next = 0;
        if (!checked_mul_u64(total, (uint64_t)dim, &next)) {
            fail("shape element count is too large");
        }
        total = next;
    }
    return total;
}
