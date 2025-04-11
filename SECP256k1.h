/*
 * Modernized version of SECP256k1 implementation for VanitySearch
 * Optimized for contemporary CPUs with SIMD and multi-core support
 * Copyright (c) 2023 Modern Crypto Solutions
 */

#ifndef SECP256K1_H
#define SECP256K1_H

#include "Point.h"
#include <string>
#include <vector>
#include <array>
#include <atomic>
#include <immintrin.h>

// Address types with enum class for better type safety
enum class AddressType : uint8_t {
    P2PKH = 0,
    P2SH = 1,
    BECH32 = 2
};

class Secp256K1 {
public:
    Secp256K1();
    ~Secp256K1() = default;

    // Initialize with precomputed tables
    void Init();

    // Key operations
    Point ComputePublicKey(const Int& privKey) const;
    Point NextKey(const Point& key) const;
    
    // Address generation
    std::string GetAddress(AddressType type, bool compressed, const Point& pubKey) const;
    std::string GetAddress(AddressType type, bool compressed, const uint8_t hash160[20]) const;
    std::vector<std::string> GetAddressBatch(AddressType type, bool compressed, 
                                           const uint8_t* h1, const uint8_t* h2, 
                                           const uint8_t* h3, const uint8_t* h4) const;
    std::string GetPrivateAddress(bool compressed, const Int& privKey) const;
    
    // Conversion helpers
    std::string GetPublicKeyHex(bool compressed, const Point& p) const;
    Point ParsePublicKeyHex(const std::string& str, bool& isCompressed) const;

    // Validation
    bool CheckPublicKey(const Point& p) const;
    bool ValidateAddress(const std::string& address) const;

    // Key decoding
    static Int DecodePrivateKey(const char* key, bool* compressed);

    // Point operations optimized with const references
    Point Add(const Point& p1, const Point& p2) const;
    Point Add2(const Point& p1, const Point& p2) const;
    Point AddDirect(const Point& p1, const Point& p2) const;
    Point Double(const Point& p) const;
    Point DoubleDirect(const Point& p) const;

    // Constants
    const Point& GetGenerator() const { return G; }
    const Int& GetOrder() const { return order; }

    // Batch processing with SIMD
    void GetHash160Batch(AddressType type, bool compressed,
                        const Point& k0, const Point& k1, 
                        const Point& k2, const Point& k3,
                        uint8_t* h0, uint8_t* h1, 
                        uint8_t* h2, uint8_t* h3) const;

private:
    // Helper methods
    uint8_t GetByte(const std::string& str, int idx) const;
    Int GetY(const Int& x, bool isEven) const;

    // Generator table with cache alignment
    alignas(64) std::array<Point, 256*32> GTable;
    
    // Curve constants
    Point G;          // Generator point
    Int order;        // Curve order
};

#endif // SECP256K1_H
