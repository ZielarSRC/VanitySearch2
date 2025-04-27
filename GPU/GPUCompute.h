/*
 * This file is part of the VanitySearch distribution (https://github.com/JeanLucPons/VanitySearch).
 * Copyright (c) 2019 Jean Luc PONS.
 * Copyright (c) 2025 Refactored by Zielar
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

namespace GPUCompute {

// Constants
__constant__ constexpr uint32_t ITEM_SIZE32 = 8;
__constant__ constexpr uint32_t GRP_SIZE = 128;
__constant__ constexpr uint32_t HSIZE = GRP_SIZE / 2 - 1;
__constant__ constexpr uint32_t STEP_SIZE = 1 << 20;

// Address types
enum class AddressType : uint8_t {
    P2PKH = 0x00,
    P2SH  = 0x05
};

// Search modes
enum class SearchMode : uint32_t {
    COMPRESSED = 0,
    UNCOMPRESSED = 1,
    BOTH = 2
};

// Helper functions
__device__ inline void store_result(uint32_t* out, uint32_t* count, uint32_t max_found,
                                   uint32_t tid, uint32_t h[5], int32_t incr, 
                                   int32_t endo, int32_t mode, AddressType type) {
    uint32_t pos = atomicAdd(count, 1);
    if (pos < max_found) {
        out[pos*ITEM_SIZE32 + 1] = tid;
        out[pos*ITEM_SIZE32 + 2] = (uint32_t)(incr << 16) | (uint32_t)(mode << 15) | (uint32_t)(endo);
        out[pos*ITEM_SIZE32 + 3] = h[0];
        out[pos*ITEM_SIZE32 + 4] = h[1];
        out[pos*ITEM_SIZE32 + 5] = h[2];
        out[pos*ITEM_SIZE32 + 6] = h[3];
        out[pos*ITEM_SIZE32 + 7] = h[4];
    }
}

// Main check function
__device__ void check_address(uint32_t* hash, int32_t incr, int32_t endo, int32_t mode,
                             uint32_t* prefix_table, uint32_t* lookup32, 
                             uint32_t max_found, uint32_t* out, AddressType type) {
    
    uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    char address[48];

    if (prefix_table == nullptr) {
        // Direct pattern matching mode
        _GetAddress(type, hash, address);
        if (_Match(address, (char*)lookup32)) {
            store_result(out, &out[0], max_found, tid, hash, incr, endo, mode, type);
        }
    } else {
        // Two-level lookup table mode
        uint32_t prefix = *(uint32_t*)hash;
        uint32_t hit = prefix_table[prefix];

        if (hit) {
            if (lookup32) {
                // Binary search in secondary table
                uint32_t start = lookup32[prefix];
                uint32_t end = start + hit - 1;
                
                while (start <= end) {
                    uint32_t mid = (start + end) / 2;
                    uint32_t val = lookup32[mid];
                    
                    if (hash[0] < val) {
                        end = mid - 1;
                    } else if (hash[0] == val) {
                        store_result(out, &out[0], max_found, tid, hash, incr, endo, mode, type);
                        return;
                    } else {
                        start = mid + 1;
                    }
                }
            } else {
                store_result(out, &out[0], max_found, tid, hash, incr, endo, mode, type);
            }
        }
    }
}

// Compressed key processing
__device__ void process_compressed_keys(uint64_t* px, uint8_t is_odd, int32_t incr,
                                       uint32_t* prefix_table, uint32_t* lookup32,
                                       uint32_t max_found, uint32_t* out, AddressType type) {
    
    uint32_t h[5];
    uint64_t pe1x[4], pe2x[4];

    _GetHash160Comp(px, is_odd, (uint8_t*)h);
    check_address(h, incr, 0, true, prefix_table, lookup32, max_found, out, type);
    
    _ModMult(pe1x, px, _beta);
    _GetHash160Comp(pe1x, is_odd, (uint8_t*)h);
    check_address(h, incr, 1, true, prefix_table, lookup32, max_found, out, type);
    
    _ModMult(pe2x, px, _beta2);
    _GetHash160Comp(pe2x, is_odd, (uint8_t*)h);
    check_address(h, incr, 2, true, prefix_table, lookup32, max_found, out, type);

    // Process opposite parity
    _GetHash160Comp(px, !is_odd, (uint8_t*)h);
    check_address(h, -incr, 0, true, prefix_table, lookup32, max_found, out, type);
    
    _GetHash160Comp(pe1x, !is_odd, (uint8_t*)h);
    check_address(h, -incr, 1, true, prefix_table, lookup32, max_found, out, type);
    
    _GetHash160Comp(pe2x, !is_odd, (uint8_t*)h);
    check_address(h, -incr, 2, true, prefix_table, lookup32, max_found, out, type);
}

// Uncompressed key processing
__device__ void process_uncompressed_keys(uint64_t* px, uint64_t* py, int32_t incr,
                                         uint32_t* prefix_table, uint32_t* lookup32,
                                         uint32_t max_found, uint32_t* out, AddressType type) {
    
    uint32_t h[5];
    uint64_t pe1x[4], pe2x[4], pyn[4];

    _GetHash160(px, py, (uint8_t*)h);
    check_address(h, incr, 0, false, prefix_table, lookup32, max_found, out, type);
    
    _ModMult(pe1x, px, _beta);
    _GetHash160(pe1x, py, (uint8_t*)h);
    check_address(h, incr, 1, false, prefix_table, lookup32, max_found, out, type);
    
    _ModMult(pe2x, px, _beta2);
    _GetHash160(pe2x, py, (uint8_t*)h);
    check_address(h, incr, 2, false, prefix_table, lookup32, max_found, out, type);

    // Process negative Y
    ModNeg256(pyn, py);
    
    _GetHash160(px, pyn, (uint8_t*)h);
    check_address(h, -incr, 0, false, prefix_table, lookup32, max_found, out, type);
    
    _GetHash160(pe1x, pyn, (uint8_t*)h);
    check_address(h, -incr, 1, false, prefix_table, lookup32, max_found, out, type);
    
    _GetHash160(pe2x, pyn, (uint8_t*)h);
    check_address(h, -incr, 2, false, prefix_table, lookup32, max_found, out, type);
}

// Main computation kernel
__device__ void compute_keys(SearchMode mode, uint64_t* startx, uint64_t* starty,
                            uint32_t* prefix_table, uint32_t* lookup32,
                            uint32_t max_found, uint32_t* out, AddressType type) {
    
    // Thread-local storage
    uint64_t dx[GRP_SIZE/2+1][4];
    uint64_t px[4], py[4], pyn[4];
    uint64_t sx[4], sy[4], dy[4];
    uint64_t _s[4], _p2[4];
    char pattern[48];

    // Load starting point
    __syncthreads();
    Load256A(sx, startx);
    Load256A(sy, starty);
    Load256(px, sx);
    Load256(py, sy);

    if (prefix_table == nullptr) {
        memcpy(pattern, lookup32, 48);
        lookup32 = (uint32_t*)pattern;
    }

    for (uint32_t j = 0; j < STEP_SIZE / GRP_SIZE; j++) {
        // Prepare delta x values
        for (uint32_t i = 0; i < HSIZE; i++)
            ModSub256(dx[i], Gx[i], sx);
        ModSub256(dx[HSIZE], Gx[HSIZE], sx);
        ModSub256(dx[HSIZE+1], _2Gnx, sx);

        // Compute modular inverses
        _ModInvGrouped(dx);

        // Process center point
        if (type == AddressType::P2PKH) {
            process_compressed_keys(px, py[0] & 1, j*GRP_SIZE + (GRP_SIZE/2),
                                  prefix_table, lookup32, max_found, out, type);
        } else {
            process_uncompressed_keys(px, py, j*GRP_SIZE + (GRP_SIZE/2),
                                     prefix_table, lookup32, max_found, out, type);
        }

        ModNeg256(pyn, py);

        // Process group points
        for (uint32_t i = 0; i < HSIZE; i++) {
            // Positive increment
            Load256(px, sx);
            Load256(py, sy);
            ModSub256(dy, Gy[i], py);
            _ModMult(_s, dy, dx[i]);
            _ModSqr(_p2, _s);
            ModSub256(px, _p2, px);
            ModSub256(px, Gx[i]);

            if (type == AddressType::P2PKH) {
                process_compressed_keys(px, 1, j*GRP_SIZE + (GRP_SIZE/2 + (i+1)),
                                      prefix_table, lookup32, max_found, out, type);
            } else {
                ModSub256(py, Gx[i], px);
                _ModMult(py, _s);
                ModSub256(py, Gy[i]);
                process_uncompressed_keys(px, py, j*GRP_SIZE + (GRP_SIZE/2 + (i+1)),
                                        prefix_table, lookup32, max_found, out, type);
            }

            // Negative increment
            Load256(px, sx);
            ModSub256(dy, pyn, Gy[i]);
            _ModMult(_s, dy, dx[i]);
            _ModSqr(_p2, _s);
            ModSub256(px, _p2, px);
            ModSub256(px, Gx[i]);

            if (type == AddressType::P2PKH) {
                process_compressed_keys(px, 0, j*GRP_SIZE + (GRP_SIZE/2 - (i+1)),
                                      prefix_table, lookup32, max_found, out, type);
            } else {
                ModSub256(py, px, Gx[i]);
                _ModMult(py, _s);
                ModSub256(py, Gy[i], py);
                process_uncompressed_keys(px, py, j*GRP_SIZE + (GRP_SIZE/2 - (i+1)),
                                        prefix_table, lookup32, max_found, out, type);
            }
        }

        // Update starting point for next group
        Load256(px, sx);
        Load256(py, sy);
        ModSub256(dy, _2Gny, py);
        _ModMult(_s, dy, dx[HSIZE+1]);
        _ModSqr(_p2, _s);
        ModSub256(px, _p2, px);
        ModSub256(px, _2Gnx);
        ModSub256(py, _2Gnx, px);
        _ModMult(py, _s);
        ModSub256(py, _2Gny);
    }

    // Store final position
    __syncthreads();
    Store256A(startx, px);
    Store256A(starty, py);
}

} // namespace GPUCompute
