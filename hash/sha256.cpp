#include "sha256.h"
#include <cstring>
#include <immintrin.h>
#include <array>

namespace {

constexpr std::array<unsigned char, 64> PAD = {
    0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};

inline uint32_t bswap_32(uint32_t x) {
#ifdef _MSC_VER
    return _byteswap_ulong(x);
#else
    return __builtin_bswap32(x);
#endif
}

inline uint64_t bswap_64(uint64_t x) {
#ifdef _MSC_VER
    return _byteswap_uint64(x);
#else
    return __builtin_bswap64(x);
#endif
}

inline uint32_t rotr32(uint32_t x, uint8_t r) {
    return (x >> r) | (x << (32 - r));
}

#define S0(x) (rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22))
#define S1(x) (rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25))
#define s0(x) (rotr32(x, 7) ^ rotr32(x, 18) ^ (x >> 3))
#define s1(x) (rotr32(x, 17) ^ rotr32(x, 19) ^ (x >> 10))

#define Ch(x, y, z) ((z) ^ ((x) & ((y) ^ (z))))
#define Maj(x, y, z) (((x) & (y)) | ((z) & ((x) | (y))))

#define Round(a, b, c, d, e, f, g, h, k, w) do { \
    uint32_t t1 = (h) + S1(e) + Ch((e), (f), (g)) + (k) + (w); \
    uint32_t t2 = S0(a) + Maj((a), (b), (c)); \
    (d) += t1; \
    (h) = t1 + t2; \
} while (0)

void sha256_transform(uint32_t state[8], const uint8_t data[64]) {
    uint32_t a, b, c, d, e, f, g, h;
    uint32_t w[64];
    
    // Load and byteswap data
    for (int i = 0; i < 16; ++i) {
        w[i] = bswap_32(((const uint32_t*)data)[i]);
    }
    
    // Initialize working variables
    a = state[0]; b = state[1]; c = state[2]; d = state[3];
    e = state[4]; f = state[5]; g = state[6]; h = state[7];
    
    // Main compression loop
    for (int i = 0; i < 64; ++i) {
        if (i >= 16) {
            w[i] = s1(w[i-2]) + w[i-7] + s0(w[i-15]) + w[i-16];
        }
        
        static const uint32_t k[] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
            0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
            0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
            0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
            0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
            0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
            0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
            0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
        };
        
        Round(a, b, c, d, e, f, g, h, k[i], w[i]);
        
        // Rotate variables
        uint32_t temp = h;
        h = g; g = f; f = e; e = d;
        d = c; c = b; b = a; a = temp;
    }
    
    // Update state
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

class SHA256 {
public:
    SHA256() {
        reset();
    }
    
    void reset() {
        m_state = {
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
        };
        m_count = 0;
        m_buf.fill(0);
    }
    
    void update(const uint8_t* data, size_t len) {
        size_t index = m_count % 64;
        m_count += len;
        
        // Fill buffer
        if (index) {
            size_t fill = std::min(len, 64 - index);
            memcpy(&m_buf[index], data, fill);
            
            if (index + fill == 64) {
                sha256_transform(m_state.data(), m_buf.data());
            }
            
            data += fill;
            len -= fill;
        }
        
        // Process full blocks
        while (len >= 64) {
            sha256_transform(m_state.data(), data);
            data += 64;
            len -= 64;
        }
        
        // Store remaining bytes
        if (len) {
            memcpy(m_buf.data(), data, len);
        }
    }
    
    void finalize(uint8_t digest[32]) {
        static constexpr uint8_t padding[64] = {0x80};
        
        // Save length
        uint64_t bits = m_count * 8;
        uint8_t length[8];
        for (int i = 0; i < 8; ++i) {
            length[i] = (bits >> (56 - i * 8)) & 0xff;
        }
        
        // Pad to 56 mod 64
        size_t index = m_count % 64;
        size_t padLen = (index < 56) ? (56 - index) : (120 - index);
        update(padding, padLen);
        
        // Append length
        update(length, 8);
        
        // Store digest
        for (int i = 0; i < 8; ++i) {
            ((uint32_t*)digest)[i] = bswap_32(m_state[i]);
        }
    }

private:
    std::array<uint32_t, 8> m_state;
    std::array<uint8_t, 64> m_buf;
    uint64_t m_count;
};

} // namespace

void sha256(uint8_t *input, int length, uint8_t *digest) {
    SHA256 ctx;
    ctx.update(input, length);
    ctx.finalize(digest);
}

void sha256_33(uint8_t *input, uint8_t *digest) {
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    
    uint8_t buffer[64];
    memcpy(buffer, input, 33);
    memset(buffer + 33, 0, 64 - 33);
    buffer[33] = 0x80;
    *reinterpret_cast<uint64_t*>(buffer + 56) = bswap_64(33 * 8);
    
    sha256_transform(state, buffer);
    
    for (int i = 0; i < 8; ++i) {
        ((uint32_t*)digest)[i] = bswap_32(state[i]);
    }
}

void sha256_65(uint8_t *input, uint8_t *digest) {
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    
    uint8_t buffer[128];
    memcpy(buffer, input, 65);
    memset(buffer + 65, 0, 128 - 65);
    buffer[65] = 0x80;
    *reinterpret_cast<uint64_t*>(buffer + 120) = bswap_64(65 * 8);
    
    sha256_transform(state, buffer);
    sha256_transform(state, buffer + 64);
    
    for (int i = 0; i < 8; ++i) {
        ((uint32_t*)digest)[i] = bswap_32(state[i]);
    }
}

void sha256_checksum(uint8_t *input, int length, uint8_t *checksum) {
    uint32_t state[8];
    uint8_t buffer[64];
    
    memcpy(buffer, input, length);
    buffer[length] = 0x80;
    if (length < 56) {
        memset(buffer + length + 1, 0, 55 - length);
    }
    *reinterpret_cast<uint64_t*>(buffer + 56) = bswap_64(length * 8);
    
    sha256_transform(state, buffer);
    *reinterpret_cast<uint32_t*>(checksum) = bswap_32(state[0]);
}

std::string sha256_hex(const unsigned char *digest) {
    static const char hex_chars[] = "0123456789abcdef";
    std::string result;
    result.reserve(64);
    
    for (int i = 0; i < 32; ++i) {
        result += hex_chars[(digest[i] >> 4) & 0xf];
        result += hex_chars[digest[i] & 0xf];
    }
    
    return result;
}
