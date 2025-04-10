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

// ---------------------------------------------------------------------------------
// Base58 - Optimized for CUDA
// ---------------------------------------------------------------------------------

__device__ __constant__ char pszBase58[59] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

__device__ __constant__ int8_t b58digits_map[256] = {
  -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
  -1, 0, 1, 2, 3, 4, 5, 6,  7, 8,-1,-1,-1,-1,-1,-1,
  -1, 9,10,11,12,13,14,15, 16,-1,17,18,19,20,21,-1,
  22,23,24,25,26,27,28,29, 30,31,32,-1,-1,-1,-1,-1,
  -1,33,34,35,36,37,38,39, 40,41,42,43,-1,44,45,46,
  47,48,49,50,51,52,53,54, 55,56,57,-1,-1,-1,-1,-1,
};

__device__ __forceinline__ void _GetAddress(int type, uint32_t *hash, char *b58Add) {
  uint32_t addBytes[16] = {0};
  uint32_t s[8];
  unsigned char A[25];
  unsigned char digits[128] = {0};
  int retPos = 0;

  // Set address type prefix
  A[0] = (type == P2PKH) ? 0x00 : 0x05;
  
  // Copy hash (20 bytes) to A[1..20]
  #pragma unroll 5
  for (int i = 0; i < 5; i++) {
    ((uint32_t *)(A + 1))[i] = hash[i];
  }

  // Prepare SHA256 input
  addBytes[0] = __byte_perm(hash[0], A[0], 0x4012);
  addBytes[1] = __byte_perm(hash[0], hash[1], 0x3456);
  addBytes[2] = __byte_perm(hash[1], hash[2], 0x3456);
  addBytes[3] = __byte_perm(hash[2], hash[3], 0x3456);
  addBytes[4] = __byte_perm(hash[3], hash[4], 0x3456);
  addBytes[5] = __byte_perm(hash[4], 0x80, 0x3456);
  addBytes[15] = 0xA8;

  // Compute first SHA256 round
  SHA256Initialize(s);
  SHA256Transform(s, addBytes);

  // Prepare second SHA256 round
  #pragma unroll 8
  for (int i = 0; i < 8; i++) {
    addBytes[i] = s[i];
  }
  addBytes[8] = 0x80000000;
  addBytes[15] = 0x100;

  // Compute second SHA256 round
  SHA256Initialize(s);
  SHA256Transform(s, addBytes);

  // Store checksum (last 4 bytes of hash)
  A[21] = ((uint8_t *)s)[3];
  A[22] = ((uint8_t *)s)[2];
  A[23] = ((uint8_t *)s)[1];
  A[24] = ((uint8_t *)s)[0];

  // Skip leading zeroes and add '1's to output
  unsigned char *addPtr = A;
  while (addPtr[0] == 0) {
    b58Add[retPos++] = '1';
    addPtr++;
  }
  int length = 25 - (addPtr - A);

  // Base58 conversion
  int digitslen = 1;
  for (int i = 0; i < length; i++) {
    uint32_t carry = addPtr[i];
    #pragma unroll 4
    for (int j = 0; j < digitslen; j++) {
      carry += (uint32_t)(digits[j]) << 8;
      digits[j] = (unsigned char)(carry % 58);
      carry /= 58;
    }
    while (carry > 0) {
      digits[digitslen++] = (unsigned char)(carry % 58);
      carry /= 58;
    }
  }

  // Reverse and store Base58 result
  #pragma unroll 4
  for (int i = 0; i < digitslen; i++) {
    b58Add[retPos++] = pszBase58[digits[digitslen - 1 - i]];
  }
  b58Add[retPos] = '\0';
}
