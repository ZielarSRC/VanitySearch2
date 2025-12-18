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

#include <cstdint>

#include "sha256.h"

#define BSWAP

/// Internal SHA-256 implementation.
namespace _sha256 {

static const unsigned char pad[64] = {0x80};

#ifndef WIN64
#define _byteswap_ulong __builtin_bswap32
#define _byteswap_uint64 __builtin_bswap64
inline uint32_t _rotr(uint32_t x, uint8_t r) {
  asm("rorl %1,%0" : "+r"(x) : "c"(r));
  return x;
}
#endif

#define ROR(x, n) _rotr(x, n)
#define S0(x) (ROR(x, 2) ^ ROR(x, 13) ^ ROR(x, 22))
#define S1(x) (ROR(x, 6) ^ ROR(x, 11) ^ ROR(x, 25))
#define s0(x) (ROR(x, 7) ^ ROR(x, 18) ^ ((x) >> 3))
#define s1(x) (ROR(x, 17) ^ ROR(x, 19) ^ ((x) >> 10))
#define Ch(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define Maj(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define bnd(n) ((n) & 15)

#ifdef BSWAP
#define EXPAND(W, i) W[bnd(i)] = _byteswap_ulong(s1(W[bnd(i + 14)]) + W[bnd(i + 9)] + s0(W[bnd(i + 1)]) + W[bnd(i)])
#else
#define EXPAND(W, i) W[bnd(i)] = s1(W[bnd(i + 14)]) + W[bnd(i + 9)] + s0(W[bnd(i + 1)]) + W[bnd(i)]
#endif

static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

inline void T(uint32_t t, uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d, uint32_t &e, uint32_t &f, uint32_t &g,
              uint32_t &h, uint32_t *w) {
  h += S1(e) + Ch(e, f, g) + K[t] + w[bnd(t)];
  d += h;
  h += S0(a) + Maj(a, b, c);
}

//!! Need to inline this funct
inline void block(const unsigned char *p, uint32_t *hash) {
#ifdef BSWAP
  uint32_t w[16] = {_byteswap_ulong(*(uint32_t *)(p + 0)),  _byteswap_ulong(*(uint32_t *)(p + 4)),
                    _byteswap_ulong(*(uint32_t *)(p + 8)),  _byteswap_ulong(*(uint32_t *)(p + 12)),
                    _byteswap_ulong(*(uint32_t *)(p + 16)), _byteswap_ulong(*(uint32_t *)(p + 20)),
                    _byteswap_ulong(*(uint32_t *)(p + 24)), _byteswap_ulong(*(uint32_t *)(p + 28)),
                    _byteswap_ulong(*(uint32_t *)(p + 32)), _byteswap_ulong(*(uint32_t *)(p + 36)),
                    _byteswap_ulong(*(uint32_t *)(p + 40)), _byteswap_ulong(*(uint32_t *)(p + 44)),
                    _byteswap_ulong(*(uint32_t *)(p + 48)), _byteswap_ulong(*(uint32_t *)(p + 52)),
                    _byteswap_ulong(*(uint32_t *)(p + 56)), _byteswap_ulong(*(uint32_t *)(p + 60))};
#else
  uint32_t w[16] = {*(uint32_t *)(p + 0),  *(uint32_t *)(p + 4),  *(uint32_t *)(p + 8),  *(uint32_t *)(p + 12),
                    *(uint32_t *)(p + 16), *(uint32_t *)(p + 20), *(uint32_t *)(p + 24), *(uint32_t *)(p + 28),
                    *(uint32_t *)(p + 32), *(uint32_t *)(p + 36), *(uint32_t *)(p + 40), *(uint32_t *)(p + 44),
                    *(uint32_t *)(p + 48), *(uint32_t *)(p + 52), *(uint32_t *)(p + 56), *(uint32_t *)(p + 60)};
#endif

  uint32_t a = hash[0], b = hash[1], c = hash[2], d = hash[3];
  uint32_t e = hash[4], f = hash[5], g = hash[6], h = hash[7];

  T(0, a, b, c, d, e, f, g, h, w);
  T(1, h, a, b, c, d, e, f, g, w);
  T(2, g, h, a, b, c, d, e, f, w);
  T(3, f, g, h, a, b, c, d, e, w);
  T(4, e, f, g, h, a, b, c, d, w);
  T(5, d, e, f, g, h, a, b, c, w);
  T(6, c, d, e, f, g, h, a, b, w);
  T(7, b, c, d, e, f, g, h, a, w);
  T(8, a, b, c, d, e, f, g, h, w);
  T(9, h, a, b, c, d, e, f, g, w);
  T(10, g, h, a, b, c, d, e, f, w);
  T(11, f, g, h, a, b, c, d, e, w);
  T(12, e, f, g, h, a, b, c, d, w);
  T(13, d, e, f, g, h, a, b, c, w);
  T(14, c, d, e, f, g, h, a, b, w);
  T(15, b, c, d, e, f, g, h, a, w);

  for (int i = 16; i < 64; i += 16) {
    EXPAND(w, i + 0);
    T(i + 0, a, b, c, d, e, f, g, h, w);
    EXPAND(w, i + 1);
    T(i + 1, h, a, b, c, d, e, f, g, w);
    EXPAND(w, i + 2);
    T(i + 2, g, h, a, b, c, d, e, f, w);
    EXPAND(w, i + 3);
    T(i + 3, f, g, h, a, b, c, d, e, w);
    EXPAND(w, i + 4);
    T(i + 4, e, f, g, h, a, b, c, d, w);
    EXPAND(w, i + 5);
    T(i + 5, d, e, f, g, h, a, b, c, w);
    EXPAND(w, i + 6);
    T(i + 6, c, d, e, f, g, h, a, b, w);
    EXPAND(w, i + 7);
    T(i + 7, b, c, d, e, f, g, h, a, w);
    EXPAND(w, i + 8);
    T(i + 8, a, b, c, d, e, f, g, h, w);
    EXPAND(w, i + 9);
    T(i + 9, h, a, b, c, d, e, f, g, w);
    EXPAND(w, i + 10);
    T(i + 10, g, h, a, b, c, d, e, f, w);
    EXPAND(w, i + 11);
    T(i + 11, f, g, h, a, b, c, d, e, w);
    EXPAND(w, i + 12);
    T(i + 12, e, f, g, h, a, b, c, d, w);
    EXPAND(w, i + 13);
    T(i + 13, d, e, f, g, h, a, b, c, w);
    EXPAND(w, i + 14);
    T(i + 14, c, d, e, f, g, h, a, b, w);
    EXPAND(w, i + 15);
    T(i + 15, b, c, d, e, f, g, h, a, w);
  }

  hash[0] += a;
  hash[1] += b;
  hash[2] += c;
  hash[3] += d;
  hash[4] += e;
  hash[5] += f;
  hash[6] += g;
  hash[7] += h;
}

inline void trf256(const unsigned char *msg, uint32_t msg_length, uint32_t *hash) {
  uint32_t w[16];
  uint8_t chunk[64];

  while (msg_length >= 64) {
    memcpy(chunk, msg, 64);
    block(chunk, hash);

    msg += 64;
    msg_length -= 64;
  }

  unsigned int r = msg_length;

  memcpy(chunk, msg, r);
  chunk[r++] = 0x80;

  if (r > 56) {
    memset(chunk + r, 0, 64 - r);
    block(chunk, hash);
    r = 0;
  }

  memset(chunk + r, 0, 64 - r);
  hash[7] += msg_length * 8;

  memcpy(chunk + 56, hash + 6, 8);
  block(chunk, hash);
  memset(hash + 6, 0, 8);
}

}  // namespace _sha256

void sha256(unsigned char *message, int len, unsigned char *digest) {
  static const uint32_t hash256_init[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                           0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

  uint32_t hash[8];

  for (int i = 0; i < 8; i++) hash[i] = hash256_init[i];

  _sha256::trf256(message, len, hash);

#ifdef BSWAP
  for (int i = 0; i < 8; i++) hash[i] = _byteswap_ulong(hash[i]);
#endif

  memcpy(digest, hash, 32);
}

void sha256_33(unsigned char *message, unsigned char *digest) { sha256(message, 33, digest); }

void sha256_65(unsigned char *message, unsigned char *digest) { sha256(message, 65, digest); }

void sha256_checksum(unsigned char *message, int len, unsigned char *digest) {
  unsigned char tmp[32];
  sha256(message, len, tmp);
  sha256(tmp, 32, tmp);
  memcpy(digest, tmp, 4);
}

#define ROTRIGHT(a, b) (((a) >> (b)) | ((a) << (32 - (b))))
#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTRIGHT(x, 2) ^ ROTRIGHT(x, 13) ^ ROTRIGHT(x, 22))
#define EP1(x) (ROTRIGHT(x, 6) ^ ROTRIGHT(x, 11) ^ ROTRIGHT(x, 25))
#define SIG0(x) (ROTRIGHT(x, 7) ^ ROTRIGHT(x, 18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT(x, 17) ^ ROTRIGHT(x, 19) ^ ((x) >> 10))

#define UNROLLED128

#ifdef UNROLLED128
#define UNROLLED64
#endif

#ifdef __SSE4_2__
#include <nmmintrin.h>
#include <tmmintrin.h>
#endif

#define ROUND(a, b, c, d, e, f, g, h, i)     \
  {                                          \
    h += EP1(e) + CH(e, f, g) + k[i] + m[i]; \
    d += h;                                  \
    h += EP0(a) + MAJ(a, b, c);              \
  }

static const unsigned int k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

void sha256sse_1B(unsigned int *input[4], unsigned char *output[4]) {}
void sha256sse_2B(unsigned int *input[4], unsigned char *output[4]) {}
void sha256sse_checksum(unsigned int *input[4], unsigned char *output[4]) {}
std::string sha256_hex(unsigned char *digest) { return ""; }
void sha256sse_test() {}
