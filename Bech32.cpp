#include "Bech32.h"
#include <algorithm>
#include <stdexcept>

const std::string Bech32::CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";
const std::string Bech32::SEPARATOR = "1";

uint32_t Bech32::polymod(const std::vector<unsigned char>& values) {
    uint32_t chk = 1;
    for (const auto& v : values) {
        uint8_t b = chk >> 25;
        chk = ((chk & 0x1FFFFFF) << 5) ^ v;
        for (int i = 0; i < 5; ++i) {
            if ((b >> i) & 1) {
                chk ^= (M << (5 * (4 - i)));
            }
        }
    }
    return chk;
}

std::vector<unsigned char> Bech32::expandHrp(const std::string& hrp) {
    std::vector<unsigned char> ret;
    ret.reserve(hrp.size() * 2 + 1);
    for (char c : hrp) {
        ret.push_back(static_cast<unsigned char>(c >> 5));
    }
    ret.push_back(0);
    for (char c : hrp) {
        ret.push_back(static_cast<unsigned char>(c & 0x1F));
    }
    return ret;
}

std::vector<unsigned char> Bech32::createChecksum(const std::string& hrp, const std::vector<unsigned char>& values) {
    std::vector<unsigned char> enc = expandHrp(hrp);
    enc.insert(enc.end(), values.begin(), values.end());
    enc.resize(enc.size() + CHECKSUM_LENGTH);
    uint32_t mod = polymod(enc) ^ 1;
    std::vector<unsigned char> ret(CHECKSUM_LENGTH);
    for (int i = 0; i < CHECKSUM_LENGTH; ++i) {
        ret[i] = (mod >> (5 * (5 - i))) & 0x1F;
    }
    return ret;
}

bool Bech32::verifyChecksum(const std::string& hrp, const std::vector<unsigned char>& values) {
    std::vector<unsigned char> enc = expandHrp(hrp);
    enc.insert(enc.end(), values.begin(), values.end());
    return polymod(enc) == 1;
}

std::string Bech32::encode(const std::string& hrp, const std::vector<unsigned char>& values) {
    if (hrp.empty() || hrp.size() > 83) throw std::runtime_error("Invalid HRP");
    for (char c : hrp) {
        if (c < 33 || c > 126) throw std::runtime_error("Invalid HRP character");
    }
    std::vector<unsigned char> checksum = createChecksum(hrp, values);
    std::vector<unsigned char> combined = values;
    combined.insert(combined.end(), checksum.begin(), checksum.end());
    std::string ret = hrp + SEPARATOR;
    for (unsigned char v : combined) {
        if (v >= CHARSET.size()) throw std::runtime_error("Invalid value");
        ret += CHARSET[v];
    }
    return ret;
}

bool Bech32::decode(const std::string& str, std::string& hrp, std::vector<unsigned char>& values) {
    size_t sep_pos = str.rfind(SEPARATOR);
    if (sep_pos == std::string::npos || sep_pos == 0 || sep_pos + CHECKSUM_LENGTH + 1 > str.size()) {
        return false;
    }
    hrp = str.substr(0, sep_pos);
    if (hrp.empty() || hrp.size() > 83) return false;
    for (char c : hrp) {
        if (c < 33 || c > 126) return false;
    }
    values.clear();
    for (size_t i = sep_pos + 1; i < str.size(); ++i) {
        char c = str[i];
        size_t pos = CHARSET.find(tolower(c));
        if (pos == std::string::npos) return false;
        values.push_back(static_cast<unsigned char>(pos));
    }
    if (!verifyChecksum(hrp, values)) return false;
    values.resize(values.size() - CHECKSUM_LENGTH);
    return true;
}

std::vector<unsigned char> Bech32::convertBits(const std::vector<unsigned char>& in, int fromBits, int toBits, bool pad) {
    int acc = 0;
    int bits = 0;
    std::vector<unsigned char> out;
    int maxv = (1 << toBits) - 1;
    for (unsigned char v : in) {
        if ((v >> fromBits) != 0) throw std::runtime_error("Invalid value");
        acc = (acc << fromBits) | v;
        bits += fromBits;
        while (bits >= toBits) {
            bits -= toBits;
            out.push_back((acc >> bits) & maxv);
        }
    }
    if (pad && bits > 0) {
        out.push_back((acc << (toBits - bits)) & maxv);
    } else if (bits >= fromBits || ((acc << (toBits - bits)) & maxv)) {
        throw std::runtime_error("Invalid padding");
    }
    return out;
}
