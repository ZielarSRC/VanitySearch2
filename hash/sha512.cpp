/*
 * This file is part of the VanitySearch distribution (https://github.com/JeanLucPons/VanitySearch).
 * Copyright (c) 2019 Jean Luc PONS.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include <string.h>

#include <cstddef>
#include <cstdint>

#include "sha512.h"

#define BSWAP
#define SHA512_BLOCK_SIZE 128
#define SHA512_HASH_LENGTH 64
#define MIN(x, y) (x < y) ? x : y;
#define MAX(x, y) (x > y) ? x : y;

/// Internal SHA-512 implementation.
namespace _sha512 {

static const uint64_t K[80] = {
    0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL, 0x3956c25bf348b538ULL,
    0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL, 0xd807aa98a3030242ULL, 0x12835b0145706fbeULL,
    0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL, 0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL,
    0xc19bf174cf692694ULL, 0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL, 0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL,
    0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL, 0x983e5152ee66dfabULL,
    0xa831c66d2db43210ULL, 0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL, 0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL,
    0x06ca6351e003826fULL, 0x142929670a0e6e70ULL, 0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL,
    0x53380d139d95b3dfULL, 0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL, 0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL, 0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL, 0xd192e819d6ef5218ULL,
    0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL, 0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL,
    0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL, 0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL, 0x5b9cca4f7763e373ULL,
    0x682e6ff3d6b2b8a3ULL, 0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL, 0xc67178f2e372532BULL, 0xca273eceea26619cULL,
    0xd186b8c721c0c207ULL, 0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL, 0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL,
    0x113f9804bef90daeULL, 0x1b710b35131c471bULL, 0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL,
    0x431d67c49c100d4cULL, 0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL, 0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL};

inline void T1(const uint64_t &e, const uint64_t &f, const uint64_t &g, const uint64_t &h, const uint64_t &k,
               const uint64_t &w, uint64_t &temp) {
  temp = h + (ROR64(e, 14) ^ ROR64(e, 18) ^ ROR64(e, 41)) + ((e & f) ^ (~e & g)) + k + w;
}

inline void T2(const uint64_t &a, const uint64_t &b, const uint64_t &c, uint64_t &temp) {
  temp = (ROR64(a, 28) ^ ROR64(a, 34) ^ ROR64(a, 39)) + ((a & b) ^ (a & c) ^ (b & c));
}

void transform(const unsigned char *message, uint64_t *w, uint64_t *digest) {
  uint64_t a = digest[0];
  uint64_t b = digest[1];
  uint64_t c = digest[2];
  uint64_t d = digest[3];
  uint64_t e = digest[4];
  uint64_t f = digest[5];
  uint64_t g = digest[6];
  uint64_t h = digest[7];
  uint64_t t1, t2, tmp;

  memcpy(w, message, SHA512_BLOCK_SIZE);

#ifdef BSWAP
  for (int i = 0; i < 16; i++) w[i] = SWAP64(w[i]);
#endif

  for (int i = 16; i < 80; i++) w[i] = SIGMA_1(w[i - 2]) + w[i - 7] + SIGMA_0(w[i - 15]) + w[i - 16];

  for (int i = 0; i < 80; ++i) {
#ifdef UNROLLED
#define UNROLLED64
#endif
#ifdef UNROLLED64
    t1 = h + SIGMA1(e) + Ch(e, f, g) + K[i] + w[i];
    t2 = SIGMA0(a) + Maj(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
#else
    T1(e, f, g, h, K[i], w[i], t1);
    T2(a, b, c, t2);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
#endif
  }

  digest[0] += a;
  digest[1] += b;
  digest[2] += c;
  digest[3] += d;
  digest[4] += e;
  digest[5] += f;
  digest[6] += g;
  digest[7] += h;
}

void sha512(const unsigned char *message, int len, unsigned char *digest) {
  static const uint64_t SHA512_hInit[] = {0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL,
                                          0xa54ff53a5f1d36f1ULL, 0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
                                          0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL};

  uint8_t msg[SHA512_BLOCK_SIZE];
  uint64_t digestTmp[8];

  for (int i = 0; i < 8; i++) digestTmp[i] = SHA512_hInit[i];

  while (len >= SHA512_BLOCK_SIZE) {
    transform(message, (uint64_t *)msg, digestTmp);
    message += SHA512_BLOCK_SIZE;
    len -= SHA512_BLOCK_SIZE;
  }

  memcpy(msg, message, len);
  msg[len++] = 0x80;
  if (len > 112) {
    memset(msg + len, 0x00, SHA512_BLOCK_SIZE - len);
    transform(msg, (uint64_t *)msg, digestTmp);
    len = 0;
  }

  memset(msg + len, 0x00, 120 - len);
  msg[127] = 8 * len;
  transform(msg, (uint64_t *)msg, digestTmp);

#ifdef BSWAP
  for (int i = 0; i < 8; i++) digestTmp[i] = SWAP64(digestTmp[i]);
#endif

  memcpy(digest, digestTmp, SHA512_HASH_LENGTH);
}

void hmac_sha512(const unsigned char *key, int keylen, const unsigned char *msg, int msglen, unsigned char *digest) {
  unsigned char kopad[128 + 64];
  unsigned char kipad[128 + 64];
  unsigned char ihash[64];

  if (keylen > 128) {
    sha512((unsigned char *)key, keylen, ihash);
    keylen = 64;
    key = ihash;
  }

  memset(kopad, 0x5c, 128);
  memset(kipad, 0x36, 128);

  for (int i = 0; i < keylen; i++) {
    kopad[i] ^= key[i];
    kipad[i] ^= key[i];
  }

  memcpy(kipad + 128, msg, msglen);

  sha512(kipad, 128 + msglen, ihash);
  memcpy(kopad + 128, ihash, 64);
  sha512(kopad, 192, digest);
}

void pbkdf2_hmac_sha512(unsigned char *out, size_t outlen, const unsigned char *p, size_t plen, const unsigned char *s,
                        size_t slen, uint64_t iter) {
  // Check size
  if (outlen > 0xFFFFFFFF) {
    return;
  }

  unsigned char ihash[64];
  unsigned char ihash2[64];
  uint8_t iobuf[4];
  size_t j;

  memset(ihash, 0, sizeof(ihash));
  memset(ihash2, 0, sizeof(ihash2));

  hmac_sha512(p, (int)plen, s, (int)slen, ihash);

  for (uint32_t i = 0, blocks = (uint32_t)ceil((double)outlen / (double)64); i < blocks; i++) {
    iobuf[0] = (i + 1 >> 24) & 0xff;
    iobuf[1] = (i + 1 >> 16) & 0xff;
    iobuf[2] = (i + 1 >> 8) & 0xff;
    iobuf[3] = (i + 1 >> 0) & 0xff;

    memcpy(ihash2, s, slen);
    memcpy(ihash2 + slen, iobuf, 4);
    hmac_sha512(p, (int)plen, ihash2, (int)slen + 4, ihash2);

    for (int j = 0; j < 64; j++) ihash[j] ^= ihash2[j];

    for (uint64_t k = 1; k < iter; k++) {
      memcpy(ihash2, ihash, 64);
      hmac_sha512(p, (int)plen, ihash2, 64, ihash2);
      for (int j = 0; j < 64; j++) ihash[j] ^= ihash2[j];
    }

    for (j = 0; j < 64 && 64 * i + j < outlen; j++) out[64 * i + j] = ihash[j];
  }
}

}  // namespace _sha512

void sha512(unsigned char *input, int length, unsigned char *digest) { _sha512::sha512(input, length, digest); }

void hmac_sha512(unsigned char *key, int key_length, unsigned char *message, int message_length,
                 unsigned char *digest) {
  _sha512::hmac_sha512(key, key_length, message, message_length, digest);
}

void pbkdf2_hmac_sha512(uint8_t *out, size_t outlen, const uint8_t *passwd, size_t passlen, const uint8_t *salt,
                        size_t saltlen, uint64_t iter) {
  _sha512::pbkdf2_hmac_sha512(out, outlen, passwd, passlen, salt, saltlen, iter);
}

std::string sha512_hex(unsigned char *digest) {
  char buf[2 * 64 + 1];
  buf[2 * 64] = 0;
  for (int i = 0; i < 64; i++) sprintf(buf + i * 2, "%02x", digest[i]);
  return std::string(buf);
}
