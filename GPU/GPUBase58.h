/*
 * This file is part of the VanitySearch distribution (https://github.com/JeanLucPons/VanitySearch).
 * Copyright (c) 2019 Jean Luc PONS.
 * Copyright (c) 2025 Modernized by Zielar
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

#pragma once

#include <cstdint>
#include <array>

namespace Base58GPU {

// Constants in constant memory for faster access
__constant__ constexpr char BASE58_ALPHABET[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

__constant__ constexpr int8_t BASE58_DIGITS_MAP[256] = {
    -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
    -1, 0, 1, 2, 3, 4, 5, 6,  7, 8,-1,-1,-1,-1,-1,-1,
    -1, 9,10,11,12,13,14,15, 16,-1,17,18,19,20,21,-1,
    22,23,24,25,26,27,28,29, 30,31,32,-1,-1,-1,-1,-1,
    -1,33,34,35,36,37,38,39, 40,41,42,43,-1,44,45,46,
    47,48,49,50,51,52,53,54, 55,56,57,-1,-1,-1,-1,-1,
};

enum class AddressType : uint8_t {
    P2PKH = 0x00,
    P2SH  = 0x05
};

__device__ void compute_checksum(const uint32_t* hash, uint8_t version_byte, uint8_t* checksum) {
    // Use modern CUDA intrinsics for byte manipulation
    uint32_t message[16] = {
        __byte_perm(hash[0], version_byte, 0x4012),
        __byte_perm(hash[0], hash[1], 0x3456),
        __byte_perm(hash[1], hash[2], 0x3456),
        __byte_perm(hash[2], hash[3], 0x3456),
        __byte_perm(hash[3], hash[4], 0x3456),
        __byte_perm(hash[4], 0x80, 0x3456),
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0xA8
    };

    uint32_t s[8];
    SHA256Initialize(s);
    SHA256Transform(s, message);

    // Second SHA256 round
    message[0] = s[0]; message[1] = s[1]; message[2] = s[2]; message[3] = s[3];
    message[4] = s[4]; message[5] = s[5]; message[6] = s[6]; message[7] = s[7];
    message[8] = 0x80000000;
    message[9] = 0; message[10] = 0; message[11] = 0;
    message[12] = 0; message[13] = 0; message[14] = 0; message[15] = 0x100;

    SHA256Initialize(s);
    SHA256Transform(s, message);

    // Store checksum in big-endian order
    checksum[0] = static_cast<uint8_t>(s[3] >> 24);
    checksum[1] = static_cast<uint8_t>(s[3] >> 16);
    checksum[2] = static_cast<uint8_t>(s[3] >> 8);
    checksum[3] = static_cast<uint8_t>(s[3]);
}

__device__ void encode_base58(const uint8_t* input, int length, char* output) {
    // Skip leading zeros
    int zero_count = 0;
    while (zero_count < length && input[zero_count] == 0) {
        output[zero_count] = '1';
        zero_count++;
    }

    // Allocate digits buffer on stack
    uint8_t digits[128] = {0};
    int digits_length = 1;

    // Process remaining bytes
    for (int i = zero_count; i < length; i++) {
        uint32_t carry = input[i];
        
        for (int j = 0; j < digits_length; j++) {
            carry += static_cast<uint32_t>(digits[j]) << 8;
            digits[j] = carry % 58;
            carry /= 58;
        }

        while (carry > 0) {
            digits[digits_length++] = carry % 58;
            carry /= 58;
        }
    }

    // Convert digits to Base58 characters
    for (int i = 0; i < digits_length; i++) {
        output[zero_count + i] = BASE58_ALPHABET[digits[digits_length - 1 - i]];
    }

    output[zero_count + digits_length] = '\0';
}

__device__ void get_address(AddressType type, const uint32_t* hash, char* b58_address) {
    // Use std::array for better type safety
    std::array<uint8_t, 25> address_bytes{};
    
    // Set version byte
    address_bytes[0] = static_cast<uint8_t>(type);
    
    // Copy hash (20 bytes)
    #pragma unroll 5
    for (int i = 0; i < 5; i++) {
        reinterpret_cast<uint32_t*>(address_bytes.data() + 1)[i] = hash[i];
    }

    // Compute checksum
    compute_checksum(hash, address_bytes[0], address_bytes.data() + 21);
    
    // Encode to Base58
    encode_base58(address_bytes.data(), address_bytes.size(), b58_address);
}

} // namespace Base58GPU
