#include "Random.h"
#include <climits>

std::random_device Random::rd;
std::mt19937_64 Random::gen(rd());
std::uniform_int_distribution<uint64_t> Random::dis(0, UINT64_MAX);

void Random::init() {
    gen.seed(rd());
}

Int Random::getRandom() {
    std::array<uint64_t, 4> buffer;
    fillRandomBuffer(buffer);
    return Int(buffer.data(), buffer.size(), false);
}

Int Random::getRandomRange(const Int& range) {
    Int result;
    do {
        result = getRandom() % range;
    } while (result.isZero()); // Ensure non-zero result
    return result;
}

void Random::getRandomBytes(unsigned char* buf, size_t count) {
    size_t filled = 0;
    while (filled < count) {
        uint64_t val = dis(gen);
        size_t toCopy = std::min(sizeof(val), count - filled);
        memcpy(buf + filled, &val, toCopy);
        filled += toCopy;
    }
}

void Random::fillRandomBuffer(std::array<uint64_t, 4>& buffer) {
    for (auto& item : buffer) {
        item = dis(gen);
    }
}
