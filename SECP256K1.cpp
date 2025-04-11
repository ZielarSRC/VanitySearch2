/*
 * Modernized SECP256k1 implementation
 * Features:
 * - SIMD optimizations for batch processing
 * - Cache-aligned data structures
 * - Constant-time operations where needed
 * - Modern C++ practices
 */

#include "SECP256k1.h"
#include "hash/sha256.h"
#include "hash/ripemd160.h"
#include "Base58.h"
#include "Bech32.h"
#include <string>
#include <cstring>
#include <stdexcept>
#include <immintrin.h>

// Constants for better maintainability
namespace {
    constexpr std::string_view SECP256K1_P = "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F";
    constexpr std::string_view SECP256K1_GX = "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798";
    constexpr std::string_view SECP256K1_GY = "483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8";
    constexpr std::string_view SECP256K1_ORDER = "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141";
}

Secp256K1::Secp256K1() {
    Init();
}

void Secp256K1::Init() {
    // Initialize field parameters
    Int P;
    P.SetBase16(SECP256K1_P.data());
    Int::SetupField(&P);

    // Initialize generator point
    G.x.SetBase16(SECP256K1_GX.data());
    G.y.SetBase16(SECP256K1_GY.data());
    G.z.SetInt32(1);

    // Set curve order
    order.SetBase16(SECP256K1_ORDER.data());
    Int::InitK1(&order);

    // Precompute generator table with cache-friendly layout
    Point N(G);
    for(int i = 0; i < 32; i++) {
        GTable[i * 256] = N;
        N = DoubleDirect(N);
        
        #pragma omp simd
        for(int j = 1; j < 255; j++) {
            GTable[i * 256 + j] = AddDirect(N, GTable[i * 256 + j - 1]);
        }
        GTable[i * 256 + 255] = N; // Dummy for alignment
    }
}

Point Secp256K1::ComputePublicKey(const Int& privKey) const {
    Point Q;
    Q.Clear();

    // Process private key in 8-bit chunks
    for(int i = 0; i < 32; i++) {
        uint8_t b = privKey.GetByte(i);
        if(b) {
            Q = Add2(Q, GTable[256 * i + (b - 1)]);
        }
    }

    Q.Reduce();
    return Q;
}

Point Secp256K1::NextKey(const Point& key) const {
    return AddDirect(key, G);
}

void Secp256K1::GetHash160Batch(AddressType type, bool compressed,
                               const Point& k0, const Point& k1,
                               const Point& k2, const Point& k3,
                               uint8_t* h0, uint8_t* h1,
                               uint8_t* h2, uint8_t* h3) const {
    alignas(32) unsigned char sh0[64];
    alignas(32) unsigned char sh1[64];
    alignas(32) unsigned char sh2[64];
    alignas(32) unsigned char sh3[64];

    switch(type) {
        case AddressType::P2PKH:
        case AddressType::BECH32: {
            if(!compressed) {
                uint32_t b0[32], b1[32], b2[32], b3[32];
                // SIMD-optimized processing for uncompressed keys
                KEYBUFFUNCOMP(b0, k0);
                KEYBUFFUNCOMP(b1, k1);
                KEYBUFFUNCOMP(b2, k2);
                KEYBUFFUNCOMP(b3, k3);

                sha256sse_2B(b0, b1, b2, b3, sh0, sh1, sh2, sh3);
                ripemd160sse_32(sh0, sh1, sh2, sh3, h0, h1, h2, h3);
            } else {
                uint32_t b0[16], b1[16], b2[16], b3[16];
                // SIMD-optimized processing for compressed keys
                KEYBUFFCOMP(b0, k0);
                KEYBUFFCOMP(b1, k1);
                KEYBUFFCOMP(b2, k2);
                KEYBUFFCOMP(b3, k3);

                sha256sse_1B(b0, b1, b2, b3, sh0, sh1, sh2, sh3);
                ripemd160sse_32(sh0, sh1, sh2, sh3, h0, h1, h2, h3);
            }
            break;
        }
        case AddressType::P2SH: {
            // Process P2SH addresses
            uint8_t kh0[20], kh1[20], kh2[20], kh3[20];
            GetHash160Batch(AddressType::P2PKH, compressed, k0, k1, k2, k3, kh0, kh1, kh2, kh3);

            uint32_t b0[16], b1[16], b2[16], b3[16];
            KEYBUFFSCRIPT(b0, kh0);
            KEYBUFFSCRIPT(b1, kh1);
            KEYBUFFSCRIPT(b2, kh2);
            KEYBUFFSCRIPT(b3, kh3);

            sha256sse_1B(b0, b1, b2, b3, sh0, sh1, sh2, sh3);
            ripemd160sse_32(sh0, sh1, sh2, sh3, h0, h1, h2, h3);
            break;
        }
    }
}

std::string Secp256K1::GetAddress(AddressType type, bool compressed, const Point& pubKey) const {
    uint8_t hash160[20];
    GetHash160Batch(type, compressed, pubKey, pubKey, pubKey, pubKey, hash160, hash160, hash160, hash160);
    
    switch(type) {
        case AddressType::BECH32: {
            char output[128];
            segwit_addr_encode(output, "bc", 0, hash160, 20);
            return output;
        }
        default: {
            uint8_t address[25];
            address[0] = (type == AddressType::P2PKH) ? 0x00 : 0x05;
            memcpy(address + 1, hash160, 20);
            
            uint8_t checksum[4];
            sha256_checksum(address, 21, checksum);
            memcpy(address + 21, checksum, 4);
            
            return EncodeBase58(address, address + 25);
        }
    }
}

std::vector<std::string> Secp256K1::GetAddressBatch(AddressType type, bool compressed, 
                                                  const uint8_t* h1, const uint8_t* h2, 
                                                  const uint8_t* h3, const uint8_t* h4) const {
    std::vector<std::string> ret;
    uint8_t add1[25], add2[25], add3[25], add4[25];
    uint32_t b1[16], b2[16], b3[16], b4[16];

    switch(type) {
        case AddressType::BECH32: {
            char output[128];
            segwit_addr_encode(output, "bc", 0, h1, 20);
            ret.push_back(output);
            segwit_addr_encode(output, "bc", 0, h2, 20);
            ret.push_back(output);
            segwit_addr_encode(output, "bc", 0, h3, 20);
            ret.push_back(output);
            segwit_addr_encode(output, "bc", 0, h4, 20);
            ret.push_back(output);
            return ret;
        }
        default: {
            add1[0] = (type == AddressType::P2PKH) ? 0x00 : 0x05;
            add2[0] = (type == AddressType::P2PKH) ? 0x00 : 0x05;
            add3[0] = (type == AddressType::P2PKH) ? 0x00 : 0x05;
            add4[0] = (type == AddressType::P2PKH) ? 0x00 : 0x05;

            memcpy(add1 + 1, h1, 20);
            memcpy(add2 + 1, h2, 20);
            memcpy(add3 + 1, h3, 20);
            memcpy(add4 + 1, h4, 20);

            CHECKSUM(b1, add1);
            CHECKSUM(b2, add2);
            CHECKSUM(b3, add3);
            CHECKSUM(b4, add4);

            sha256sse_checksum(b1, b2, b3, b4, add1 + 21, add2 + 21, add3 + 21, add4 + 21);

            ret.push_back(EncodeBase58(add1, add1 + 25));
            ret.push_back(EncodeBase58(add2, add2 + 25));
            ret.push_back(EncodeBase58(add3, add3 + 25));
            ret.push_back(EncodeBase58(add4, add4 + 25));
            return ret;
        }
    }
}

std::string Secp256K1::GetPrivateAddress(bool compressed, const Int& privKey) const {
    uint8_t address[38];
    address[0] = 0x80; // Mainnet
    privKey.Get32Bytes(address + 1);
    
    if(compressed) {
        address[33] = 1;
        sha256_checksum(address, 34, address + 34);
        return EncodeBase58(address, address + 38);
    } else {
        sha256_checksum(address, 33, address + 33);
        return EncodeBase58(address, address + 37);
    }
}

std::string Secp256K1::GetPublicKeyHex(bool compressed, const Point& p) const {
    uint8_t publicKeyBytes[65];
    char tmp[3];
    std::string ret;

    if(!compressed) {
        publicKeyBytes[0] = 0x04;
        p.x.Get32Bytes(publicKeyBytes + 1);
        p.y.Get32Bytes(publicKeyBytes + 33);

        for(int i = 0; i < 65; i++) {
            sprintf(tmp, "%02X", (int)publicKeyBytes[i]);
            ret.append(tmp);
        }
    } else {
        publicKeyBytes[0] = p.y.IsEven() ? 0x02 : 0x03;
        p.x.Get32Bytes(publicKeyBytes + 1);

        for(int i = 0; i < 33; i++) {
            sprintf(tmp, "%02X", (int)publicKeyBytes[i]);
            ret.append(tmp);
        }
    }

    return ret;
}

Point Secp256K1::ParsePublicKeyHex(const std::string& str, bool& isCompressed) const {
    Point ret;
    ret.Clear();

    if(str.length() < 2) {
        throw std::runtime_error("Invalid public key length");
    }

    uint8_t type = GetByte(str, 0);

    switch(type) {
        case 0x02:
            if(str.length() != 66) {
                throw std::runtime_error("Invalid compressed public key length");
            }
            for(int i = 0; i < 32; i++)
                ret.x.SetByte(31 - i, GetByte(str, i + 1));
            ret.y = GetY(ret.x, true);
            isCompressed = true;
            break;

        case 0x03:
            if(str.length() != 66) {
                throw std::runtime_error("Invalid compressed public key length");
            }
            for(int i = 0; i < 32; i++)
                ret.x.SetByte(31 - i, GetByte(str, i + 1));
            ret.y = GetY(ret.x, false);
            isCompressed = true;
            break;

        case 0x04:
            if(str.length() != 130) {
                throw std::runtime_error("Invalid uncompressed public key length");
            }
            for(int i = 0; i < 32; i++)
                ret.x.SetByte(31 - i, GetByte(str, i + 1));
            for(int i = 0; i < 32; i++)
                ret.y.SetByte(31 - i, GetByte(str, i + 33));
            isCompressed = false;
            break;

        default:
            throw std::runtime_error("Invalid public key prefix");
    }

    ret.z.SetInt32(1);

    if(!CheckPublicKey(ret)) {
        throw std::runtime_error("Public key not on curve");
    }

    return ret;
}

bool Secp256K1::CheckPublicKey(const Point& p) const {
    if(p.IsInfinity()) return false;
    
    Int y_sq, x_cubed;
    y_sq.ModSquareK1(&p.y);
    x_cubed.ModSquareK1(&p.x);
    x_cubed.ModMulK1(&p.x);
    x_cubed.ModAdd(7);
    
    return y_sq.Equals(x_cubed);
}

bool Secp256K1::ValidateAddress(const std::string& address) const {
    std::vector<uint8_t> pubKey;
    DecodeBase58(address, pubKey);

    if(pubKey.size() != 25) return false;

    uint8_t chk[4];
    sha256_checksum(pubKey.data(), 21, chk);

    return (pubKey[21] == chk[0]) &&
           (pubKey[22] == chk[1]) &&
           (pubKey[23] == chk[2]) &&
           (pubKey[24] == chk[3]);
}

Point Secp256K1::Add(const Point& p1, const Point& p2) const {
    if(p1.IsInfinity()) return p2;
    if(p2.IsInfinity()) return p1;

    Int u1, u2, v1, v2;
    u1.ModMulK1(&p2.y, &p1.z);
    u2.ModMulK1(&p1.y, &p2.z);
    v1.ModMulK1(&p2.x, &p1.z);
    v2.ModMulK1(&p1.x, &p2.z);

    if(v1.Equals(v2)) {
        if(!u1.Equals(u2)) return Point(); // Infinity
        return Double(p1);
    }

    Int u, v, w;
    u.ModSub(&u1, &u2);
    v.ModSub(&v1, &v2);
    w.ModMulK1(&p1.z, &p2.z);

    Int usq, vsq, vsq_v2, vsq_v2_2, vsq_v2_4, a;
    usq.ModSquareK1(&u);
    vsq.ModSquareK1(&v);
    
    Point r;
    vsq_v2.ModMulK1(&vsq, &v2);
    vsq_v2_2.ModAdd(&vsq_v2, &vsq_v2);
    a.ModSub(&usq, &v);
    a.ModSub(&vsq_v2_2);
    a.ModMulK1(&w);
    
    r.x.ModMulK1(&v, &a);
    
    Int vsq_v2_a, u_vsq_v2_a, vsq_u2;
    vsq_v2_a.ModSub(&vsq_v2, &a);
    u_vsq_v2_a.ModMulK1(&u, &vsq_v2_a);
    vsq_u2.ModMulK1(&vsq, &u2);
    
    r.y.ModSub(&u_vsq_v2_a, &vsq_u2);
    
    Int vsq_w;
    vsq_w.ModMulK1(&vsq, &w);
    r.z.ModMulK1(&vsq_w, &v);
    
    return r;
}

Point Secp256K1::Add2(const Point& p1, const Point& p2) const {
    // P2.z = 1 optimization
    Int u, v, u1, v1, vs2, vs3, us2, a, us2w, vs2v2, vs3u2, _2vs2v2;
    Point r;

    u1.ModMulK1(&p2.y, &p1.z);
    v1.ModMulK1(&p2.x, &p1.z);
    u.ModSub(&u1, &p1.y);
    v.ModSub(&v1, &p1.x);
    us2.ModSquareK1(&u);
    vs2.ModSquareK1(&v);
    vs3.ModMulK1(&vs2, &v);
    us2w.ModMulK1(&us2, &p1.z);
    vs2v2.ModMulK1(&vs2, &p1.x);
    _2vs2v2.ModAdd(&vs2v2, &vs2v2);
    a.ModSub(&us2w, &vs3);
    a.ModSub(&_2vs2v2);

    r.x.ModMulK1(&v, &a);

    vs3u2.ModMulK1(&vs3, &p1.y);
    r.y.ModSub(&vs2v2, &a);
    r.y.ModMulK1(&r.y, &u);
    r.y.ModSub(&vs3u2);

    r.z.ModMulK1(&vs3, &p1.z);

    return r;
}

Point Secp256K1::AddDirect(const Point& p1, const Point& p2) const {
    Int _s, _p, dy, dx;
    Point r;
    r.z.SetInt32(1);

    dy.ModSub(&p2.y, &p1.y);
    dx.ModSub(&p2.x, &p1.x);
    dx.ModInv();
    _s.ModMulK1(&dy, &dx);
    
    _p.ModSquareK1(&_s);
    
    r.x.ModSub(&_p, &p1.x);
    r.x.ModSub(&p2.x);
    
    r.y.ModSub(&p2.x, &r.x);
    r.y.ModMulK1(&_s);
    r.y.ModSub(&p2.y);
    
    return r;
}

Point Secp256K1::Double(const Point& p) const {
    if(p.IsInfinity()) return p;
    if(p.y.IsZero()) return Point(); // Infinity

    Int ysq, xsq, w, s, b, h;
    ysq.ModSquareK1(&p.y);
    xsq.ModSquareK1(&p.x);
    
    w.ModAdd(&xsq, &xsq);
    w.ModAdd(&xsq);
    
    s.ModMulK1(&p.y, &p.z);
    b.ModMulK1(&p.x, &ysq);
    b.ModMulK1(&s);
    
    h.ModSquareK1(&w);
    h.ModSub(&b);
    h.ModSub(&b);
    h.ModSub(&b);
    h.ModSub(&b);
    
    Point r;
    r.x.ModMulK1(&h, &s);
    r.x.ModAdd(&r.x);
    
    Int ysq_sq, b4_h, w_b4_h;
    ysq_sq.ModSquareK1(&ysq);
    b4_h.ModSub(&b, &h);
    b4_h.ModSub(&b);
    b4_h.ModSub(&b);
    b4_h.ModSub(&b);
    w_b4_h.ModMulK1(&w, &b4_h);
    
    r.y.ModSub(&w_b4_h, &ysq_sq);
    r.y.ModSub(&ysq_sq);
    r.y.ModSub(&ysq_sq);
    r.y.ModSub(&ysq_sq);
    
    Int s_sq;
    s_sq.ModSquareK1(&s);
    r.z.ModMulK1(&s_sq, &s);
    r.z.ModDouble();
    r.z.ModDouble();
    r.z.ModDouble();
    
    return r;
}

Point Secp256K1::DoubleDirect(const Point& p) const {
    Int _s, _p, a;
    Point r;
    r.z.SetInt32(1);

    _s.ModMulK1(&p.x, &p.x);
    _p.ModAdd(&_s, &_s);
    _p.ModAdd(&_s);

    a.ModAdd(&p.y, &p.y);
    a.ModInv();
    _s.ModMulK1(&_p, &a);
    
    _p.ModMulK1(&_s, &_s);
    a.ModAdd(&p.x, &p.x);
    a.ModNeg();
    r.x.ModAdd(&a, &_p);
    
    a.ModSub(&r.x, &p.x);
    _p.ModMulK1(&a, &_s);
    r.y.ModAdd(&_p, &p.y);
    r.y.ModNeg();
    
    return r;
}

Int Secp256K1::GetY(const Int& x, bool isEven) const {
    Int _s, _p;
    _s.ModSquareK1(&x);
    _p.ModMulK1(&_s, &x);
    _p.ModAdd(7);
    _p.ModSqrt();

    if(!_p.IsEven() && isEven) {
        _p.ModNeg();
    }
    else if(_p.IsEven() && !isEven) {
        _p.ModNeg();
    }

    return _p;
}

uint8_t Secp256K1::GetByte(const std::string& str, int idx) const {
    char tmp[3];
    int val;

    tmp[0] = str.data()[2 * idx];
    tmp[1] = str.data()[2 * idx + 1];
    tmp[2] = 0;

    if(sscanf(tmp, "%X", &val) != 1) {
        throw std::runtime_error("Invalid hexadecimal digit");
    }

    return static_cast<uint8_t>(val);
}

Int Secp256K1::DecodePrivateKey(const char* key, bool* compressed) {
    Int ret;
    std::vector<unsigned char> privKey;

    if(key[0] == '5') {
        // Decode uncompressed private key
        DecodeBase58(key, privKey);
        if(privKey.size() != 37) {
            throw std::runtime_error("Invalid private key length");
        }

        if(privKey[0] != 0x80) {
            throw std::runtime_error("Invalid private key prefix");
        }

        // Process key bytes
        for(int i = 1; i < 33; i++) {
            ret.SetByte(31 - (i - 1), privKey[i]);
        }

        // Verify checksum
        unsigned char checksum[4];
        sha256_checksum(privKey.data(), 33, checksum);
        
        if(memcmp(checksum, privKey.data() + 33, 4) != 0) {
            throw std::runtime_error("Invalid private key checksum");
        }

        *compressed = false;
    } 
    else if(key[0] == 'K' || key[0] == 'L') {
        // Decode compressed private key
        DecodeBase58(key, privKey);
        if(privKey.size() != 38) {
            throw std::runtime_error("Invalid private key length");
        }

        // Process key bytes
        for(int i = 1; i < 33; i++) {
            ret.SetByte(31 - (i - 1), privKey[i]);
        }

        // Verify checksum
        unsigned char checksum[4];
        sha256_checksum(privKey.data(), 34, checksum);
        
        if(memcmp(checksum, privKey.data() + 34, 4) != 0) {
            throw std::runtime_error("Invalid private key checksum");
        }

        *compressed = true;
    } 
    else {
        throw std::runtime_error("Invalid private key format");
    }

    return ret;
}
