#ifndef SHA512_H
#define SHA512_H

#include <cstdint>
#include <string>
#include <array>

constexpr size_t SHA512_BLOCK_SIZE = 128;
constexpr size_t SHA512_HASH_LENGTH = 64;

void sha512(uint8_t *input, size_t length, uint8_t *digest);
void pbkdf2_hmac_sha512(uint8_t *out, size_t outlen, const uint8_t *passwd, 
                       size_t passlen, const uint8_t *salt, size_t saltlen, 
                       uint64_t iter);
void hmac_sha512(const uint8_t *key, size_t key_length, 
                const uint8_t *message, size_t message_length, 
                uint8_t *digest);

std::string sha512_hex(const uint8_t *digest);

#endif // SHA512_H
