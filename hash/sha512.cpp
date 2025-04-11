#include "sha512.h"
#include <cstring>
#include <immintrin.h>
#include <array>

namespace {

constexpr std::array<uint64_t, 80> K = {
    0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL,
    0xe9b5dba58189dbbcULL, 0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL,
    0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL, 0xd807aa98a3030242ULL,
    0x12835b0145706fbeULL, 0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL,
    0xc19bf174cf692694ULL, 0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL,
    0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL, 0x2de92c6f592b0275ULL,
    0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL, 0xb00327c898fb213fULL,
    0xbf597fc7beef0ee4ULL, 0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL,
    0x06ca6351e003826fULL, 0x142929670a0e6e70ULL, 0x27b70a8546d22ffcULL,
    0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL,
    0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL,
    0x92722c851482353bULL, 0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL,
    0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL, 0xd192e819d6ef5218ULL,
    0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, 0x2748774cdf8eeb99ULL,
    0x34b0bcb5e19b48a8ULL, 0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL,
    0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL, 0x748f82ee5defb2fcULL,
    0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL,
    0xc67178f2e372532bULL, 0xca273eceea26619cULL, 0xd186b8c721c0c207ULL,
    0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL, 0x06f067aa72176fbaULL,
    0x0a637dc5a2c898a6ULL, 0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
    0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL,
    0x431d67c49c100d4cULL, 0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL,
    0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL
};

inline uint64_t bswap_64(uint64_t x) {
#ifdef _MSC_VER
    return _byteswap_uint64(x);
#else
    return __builtin_bswap64(x);
#endif
}

inline uint64_t rotr64(uint64_t x, uint8_t n) {
    return (x >> n) | (x << (64 - n));
}

inline uint64_t S0(uint64_t x) { return rotr64(x, 28) ^ rotr64(x, 34) ^ rotr64(x, 39); }
inline uint64_t S1(uint64_t x) { return rotr64(x, 14) ^ rotr64(x, 18) ^ rotr64(x, 41); }
inline uint64_t G0(uint64_t x) { return rotr64(x, 1) ^ rotr64(x, 8) ^ (x >> 7); }
inline uint64_t G1(uint64_t x) { return rotr64(x, 19) ^ rotr64(x, 61) ^ (x >> 6); }

#define ROUND(i, a, b, c, d, e, f, g, h) do { \
    uint64_t t1 = h + S1(e) + Ch(e, f, g) + K[i] + W[i]; \
    d += t1; \
    h = t1 + S0(a) + Maj(a, b, c); \
} while(0)

inline uint64_t Ch(uint64_t x, uint64_t y, uint64_t z) { return z ^ (x & (y ^ z)); }
inline uint64_t Maj(uint64_t x, uint64_t y, uint64_t z) { return (x & y) | (z & (x | y)); }

void sha512_transform(uint64_t state[8], const uint8_t data[128]) {
    uint64_t W[80];
    uint64_t a, b, c, d, e, f, g, h;
    
    // Load and byteswap data
    for (int i = 0; i < 16; i++) {
        W[i] = bswap_64(((const uint64_t*)data)[i]);
    }
    
    // Message schedule
    for (int i = 16; i < 80; i++) {
        W[i] = G1(W[i-2]) + W[i-7] + G0(W[i-15]) + W[i-16];
    }
    
    // Initialize working variables
    a = state[0]; b = state[1]; c = state[2]; d = state[3];
    e = state[4]; f = state[5]; g = state[6]; h = state[7];
    
    // Compression function main loop
    for (int i = 0; i < 80; i++) {
        ROUND(i, a, b, c, d, e, f, g, h);
        
        // Rotate variables
        uint64_t temp = h;
        h = g; g = f; f = e; e = d;
        d = c; c = b; b = a; a = temp;
    }
    
    // Update state
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

class SHA512 {
public:
    SHA512() { reset(); }
    
    void reset() {
        m_state = {
            0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
            0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
            0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
            0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
        };
        m_count = 0;
        m_buf.fill(0);
        m_bufSize = 0;
    }
    
    void update(const uint8_t* data, size_t len) {
        m_count += len;
        
        // Process any buffered data
        if (m_bufSize > 0) {
            size_t fill = std::min(len, 128 - m_bufSize);
            memcpy(&m_buf[m_bufSize], data, fill);
            m_bufSize += fill;
            data += fill;
            len -= fill;
            
            if (m_bufSize == 128) {
                sha512_transform(m_state.data(), m_buf.data());
                m_bufSize = 0;
            }
        }
        
        // Process full blocks
        while (len >= 128) {
            sha512_transform(m_state.data(), data);
            data += 128;
            len -= 128;
        }
        
        // Store remaining data
        if (len > 0) {
            memcpy(m_buf.data(), data, len);
            m_bufSize = len;
        }
    }
    
    void finalize(uint8_t digest[64]) {
        // Pad the message
        m_buf[m_bufSize++] = 0x80;
        
        // If not enough space for length (128 bits)
        if (m_bufSize > 112) {
            memset(&m_buf[m_bufSize], 0, 128 - m_bufSize);
            sha512_transform(m_state.data(), m_buf.data());
            m_bufSize = 0;
        }
        
        // Pad with zeros
        memset(&m_buf[m_bufSize], 0, 112 - m_bufSize);
        
        // Store length in bits (big-endian)
        uint64_t bitCount = m_count * 8;
        m_buf[112] = (bitCount >> 56) & 0xff;
        m_buf[113] = (bitCount >> 48) & 0xff;
        m_buf[114] = (bitCount >> 40) & 0xff;
        m_buf[115] = (bitCount >> 32) & 0xff;
        m_buf[116] = (bitCount >> 24) & 0xff;
        m_buf[117] = (bitCount >> 16) & 0xff;
        m_buf[118] = (bitCount >> 8) & 0xff;
        m_buf[119] = bitCount & 0xff;
        
        // Final transform
        sha512_transform(m_state.data(), m_buf.data());
        
        // Store digest (big-endian)
        for (int i = 0; i < 8; i++) {
            ((uint64_t*)digest)[i] = bswap_64(m_state[i]);
        }
    }

private:
    std::array<uint64_t, 8> m_state;
    std::array<uint8_t, 128> m_buf;
    size_t m_bufSize;
    uint64_t m_count;
};

constexpr uint8_t IPAD = 0x36;
constexpr uint8_t OPAD = 0x5c;

} // namespace

void sha512(uint8_t *input, size_t length, uint8_t *digest) {
    SHA512 ctx;
    ctx.update(input, length);
    ctx.finalize(digest);
}

void hmac_sha512(const uint8_t *key, size_t key_length, 
                const uint8_t *message, size_t message_length, 
                uint8_t *digest) {
    uint8_t k[SHA512_BLOCK_SIZE] = {0};
    
    // Normalize key
    if (key_length > SHA512_BLOCK_SIZE) {
        sha512(key, key_length, k);
    } else {
        memcpy(k, key, key_length);
    }
    
    // Prepare inner and outer padding
    uint8_t ipad[SHA512_BLOCK_SIZE];
    uint8_t opad[SHA512_BLOCK_SIZE];
    
    for (size_t i = 0; i < SHA512_BLOCK_SIZE; i++) {
        ipad[i] = k[i] ^ IPAD;
        opad[i] = k[i] ^ OPAD;
    }
    
    // Inner hash
    uint8_t ihash[SHA512_HASH_LENGTH];
    {
        SHA512 ctx;
        ctx.update(ipad, SHA512_BLOCK_SIZE);
        ctx.update(message, message_length);
        ctx.finalize(ihash);
    }
    
    // Outer hash
    SHA512 ctx;
    ctx.update(opad, SHA512_BLOCK_SIZE);
    ctx.update(ihash, SHA512_HASH_LENGTH);
    ctx.finalize(digest);
}

void pbkdf2_hmac_sha512(uint8_t *out, size_t outlen,
                       const uint8_t *passwd, size_t passlen,
                       const uint8_t *salt, size_t saltlen,
                       uint64_t iter) {
    uint8_t key[SHA512_BLOCK_SIZE] = {0};
    
    // Normalize password
    if (passlen > SHA512_BLOCK_SIZE) {
        sha512(passwd, passlen, key);
    } else {
        memcpy(key, passwd, passlen);
    }
    
    uint8_t U[SHA512_HASH_LENGTH];
    uint8_t T[SHA512_HASH_LENGTH];
    
    for (uint32_t i = 1; outlen > 0; i++) {
        // Prepare salt + block number
        uint8_t salt_block[SHA512_BLOCK_SIZE + 4];
        size_t salt_block_len = 0;
        
        if (saltlen > 0) {
            memcpy(salt_block, salt, saltlen);
            salt_block_len = saltlen;
        }
        
        uint32_t be_i = bswap_64(i);
        memcpy(salt_block + salt_block_len, &be_i, 4);
        salt_block_len += 4;
        
        // First iteration
        hmac_sha512(key, SHA512_BLOCK_SIZE, salt_block, salt_block_len, U);
        memcpy(T, U, SHA512_HASH_LENGTH);
        
        // Subsequent iterations
        for (uint64_t j = 1; j < iter; j++) {
            hmac_sha512(key, SHA512_BLOCK_SIZE, U, SHA512_HASH_LENGTH, U);
            for (size_t k = 0; k < SHA512_HASH_LENGTH; k++) {
                T[k] ^= U[k];
            }
        }
        
        // Copy output
        size_t copy_len = std::min(outlen, SHA512_HASH_LENGTH);
        memcpy(out, T, copy_len);
        out += copy_len;
        outlen -= copy_len;
    }
}

std::string sha512_hex(const uint8_t *digest) {
    static const char hex[] = "0123456789abcdef";
    std::string result;
    result.reserve(128);
    
    for (size_t i = 0; i < 64; i++) {
        result += hex[digest[i] >> 4];
        result += hex[digest[i] & 0xf];
    }
    
    return result;
}
