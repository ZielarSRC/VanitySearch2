#include "SECP256k1.h"
#include "hash/sha256.h"
#include "hash/ripemd160.h"
#include "Base58.h"
#include "Bech32.h"
#include <string>
#include <stdexcept>
#include <charconv>
#include <algorithm>
#include <array>
#include <vector>
#include <iostream>

namespace {

constexpr std::string_view P = "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F";
constexpr std::string_view Gx = "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798";
constexpr std::string_view Gy = "483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8";
constexpr std::string_view Order = "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141";

void PrintResult(bool ok) {
    std::cout << (ok ? "OK\n" : "Failed!\n");
}

void CheckAddress(const Secp256K1* secp, const std::string& address, const std::string& privKeyStr) {
    bool isCompressed;
    AddressType type;

    Int privKey = secp->DecodePrivateKey(privKeyStr.c_str(), &isCompressed);
    Point pub = secp->ComputePublicKey(&privKey);

    switch (address[0]) {
        case '1': type = P2PKH; break;
        case '3': type = P2SH; break;
        case 'b': case 'B': type = BECH32; break;
        default:
            std::cout << "Failed!\n" << address << " Address format not supported\n";
            return;
    }

    std::string calcAddress = secp->GetAddress(type, isCompressed, pub);
    std::cout << "Address: " << address << " " << (address == calcAddress ? "OK!\n" : "Failed!\n");
}

} // namespace

Secp256K1::Secp256K1() {
    Init();
}

void Secp256K1::Init() {
    Int p;
    p.SetBase16(P.data());
    Int::SetupField(&p);

    G.x.SetBase16(Gx.data());
    G.y.SetBase16(Gy.data());
    G.z.SetInt32(1);
    order.SetBase16(Order.data());
    Int::InitK1(&order);

    Point N(G);
    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 255; j++) {
            GTable[i * 256 + j] = N;
            N = DoubleDirect(N);
        }
        GTable[i * 256 + 255] = N;
    }
}

void Secp256K1::Check() {
    std::cout << "Check Generator: ";
    bool ok = std::all_of(GTable.begin(), GTable.end(), [this](const Point& p) { return EC(p); });
    PrintResult(ok);

    std::cout << "Check Double: ";
    Point R1 = Double(G);
    R1.Reduce();
    PrintResult(EC(R1));

    std::cout << "Check Add: ";
    Point R2 = Add(G, R1);
    Point R3 = Add(R1, R2);
    R3.Reduce();
    PrintResult(EC(R3));

    std::cout << "Check GenKey: ";
    Int privKey;
    privKey.SetBase16("46b9e861b63d3509c88b7817275a30d22d62c8cd8fa6486ddee35ef0d8e0495f");
    Point pub = ComputePublicKey(&privKey);
    
    Point expectedPubKey;
    expectedPubKey.x.SetBase16("2500e7f3fbddf2842903f544ddc87494ce95029ace4e257d54ba77f2bc1f3a88");
    expectedPubKey.y.SetBase16("37a9461c4f1c57fecc499753381e772a128a5820a924a2fa05162eb662987a9f");
    expectedPubKey.z.SetInt32(1);

    PrintResult(pub.equals(expectedPubKey));

    CheckAddress(this, "15t3Nt1zyMETkHbjJTTshxLnqPzQvAtdCe", "5HqoeNmaz17FwZRqn7kCBP1FyJKSe4tt42XZB7426EJ2MVWDeqk");
    CheckAddress(this, "1BoatSLRHtKNngkdXEeobR76b53LETtpyT", "5J4XJRyLVgzbXEgh8VNi4qovLzxRftzMd8a18KkdXv4EqAwX3tS");
    CheckAddress(this, "1Test6BNjSJC5qwYXsjwKVLvz7DpfLehy", "5HytzR8p5hp8Cfd8jsVFnwMNXMsEW1sssFxMQYqEUjGZN72iLJ2");
    CheckAddress(this, "16S5PAsGZ8VFM1CRGGLqm37XHrp46f6CTn", "KxMUSkFhEzt2eJHscv2vNSTnnV2cgAXgL4WDQBTx7Ubd9TZmACAz");
    CheckAddress(this, "1Tst2RwMxZn9cYY5mQhCdJic3JJrK7Fq7", "L1vamTpSeK9CgynRpSJZeqvUXf6dJa25sfjb2uvtnhj65R5TymgF");
    CheckAddress(this, "3CyQYcByvcWK8BkYJabBS82yDLNWt6rWSx", "KxMUSkFhEzt2eJHscv2vNSTnnV2cgAXgL4WDQBTx7Ubd9TZmACAz");
    CheckAddress(this, "31to1KQe67YjoDfYnwFJThsGeQcFhVDM5Q", "KxV2Tx5jeeqLHZ1V9ufNv1doTZBZuAc5eY24e6b27GTkDhYwVad7");
    CheckAddress(this, "bc1q6tqytpg06uhmtnhn9s4f35gkt8yya5a24dptmn", "L2wAVD273GwAxGuEDHvrCqPfuWg5wWLZWy6H3hjsmhCvNVuCERAQ");

    Point pubTest;
    pubTest.x.SetBase16("75249c39f38baa6bf20ab472191292349426dc3652382cdc45f65695946653dc");
    pubTest.y.SetBase16("978b2659122fe1df1be132167f27b74e5d4a2f3ecbbbd0b3fbcc2f4983518674");
    std::cout << "Check Calc PubKey (full) " << GetAddress(P2PKH, false, pubTest) << ": ";
    PrintResult(EC(pubTest));

    pubTest.x.SetBase16("c931af9f331b7a9eb2737667880dacb91428906fbffad0173819a873172d21c4");
    pubTest.y = GetY(pubTest.x, false);
    std::cout << "Check Calc PubKey (even) " << GetAddress(P2SH, true, pubTest) << ": ";
    PrintResult(EC(pubTest));

    pubTest.x.SetBase16("3bf3d80f868fa33c6353012cb427e98b080452f19b5c1149ea2acfe4b7599739");
    pubTest.y = GetY(pubTest.x, false);
    std::cout << "Check Calc PubKey (odd) " << GetAddress(P2PKH, true, pubTest) << ": ";
    PrintResult(EC(pubTest));
}

Point Secp256K1::ComputePublicKey(const Int* privKey) {
    Point Q;
    Q.Clear();

    for (int i = 0; i < 32; i++) {
        uint8_t b = privKey->GetByte(i);
        if (b) {
            Q = GTable[256 * i + (b - 1)];
            for (i++; i < 32; i++) {
                b = privKey->GetByte(i);
                if (b) Q = Add2(Q, GTable[256 * i + (b - 1)]);
            }
            break;
        }
    }

    Q.Reduce();
    return Q;
}

Point Secp256K1::NextKey(const Point& key) {
    return AddDirect(key, G);
}

uint8_t Secp256K1::GetByte(const std::string& str, size_t idx) const {
    if (idx * 2 + 1 >= str.length()) {
        throw std::runtime_error("Invalid public key hex string");
    }

    uint8_t val;
    auto result = std::from_chars(&str[idx * 2], &str[idx * 2 + 2], val, 16);
    if (result.ec != std::errc()) {
        throw std::runtime_error("Invalid hexadecimal digit in public key");
    }

    return val;
}

Point Secp256K1::ParsePublicKeyHex(const std::string& str, bool& isCompressed) const {
    if (str.length() < 2) {
        throw std::runtime_error("Invalid public key length");
    }

    Point ret;
    uint8_t type = GetByte(str, 0);

    switch (type) {
        case 0x02:
            if (str.length() != 66) throw std::runtime_error("Invalid compressed public key length");
            for (int i = 0; i < 32; i++)
                ret.x.SetByte(31 - i, GetByte(str, i + 1));
            ret.y = GetY(ret.x, true);
            isCompressed = true;
            break;

        case 0x03:
            if (str.length() != 66) throw std::runtime_error("Invalid compressed public key length");
            for (int i = 0; i < 32; i++)
                ret.x.SetByte(31 - i, GetByte(str, i + 1));
            ret.y = GetY(ret.x, false);
            isCompressed = true;
            break;

        case 0x04:
            if (str.length() != 130) throw std::runtime_error("Invalid uncompressed public key length");
            for (int i = 0; i < 32; i++)
                ret.x.SetByte(31 - i, GetByte(str, i + 1));
            for (int i = 0; i < 32; i++)
                ret.y.SetByte(31 - i, GetByte(str, i + 33));
            isCompressed = false;
            break;

        default:
            throw std::runtime_error("Invalid public key prefix");
    }

    ret.z.SetInt32(1);

    if (!EC(ret)) {
        throw std::runtime_error("Public key does not lie on the elliptic curve");
    }

    return ret;
}

std::string Secp256K1::GetPublicKeyHex(bool compressed, const Point& pubKey) const {
    std::array<uint8_t, 65> publicKeyBytes;
    std::string ret;

    if (!compressed) {
        publicKeyBytes[0] = 0x04;
        pubKey.x.Get32Bytes(publicKeyBytes.data() + 1);
        pubKey.y.Get32Bytes(publicKeyBytes.data() + 33);
        
        ret.reserve(130);
        for (int i = 0; i < 65; i++) {
            char buf[3];
            snprintf(buf, sizeof(buf), "%02X", publicKeyBytes[i]);
            ret += buf;
        }
    } else {
        publicKeyBytes[0] = pubKey.y.IsEven() ? 0x02 : 0x03;
        pubKey.x.Get32Bytes(publicKeyBytes.data() + 1);
        
        ret.reserve(66);
        for (int i = 0; i < 33; i++) {
            char buf[3];
            snprintf(buf, sizeof(buf), "%02X", publicKeyBytes[i]);
            ret += buf;
        }
    }

    return ret;
}

void Secp256K1::GetHash160(AddressType type, bool compressed, const Point& pubKey, uint8_t* hash) const {
    std::array<uint8_t, 65> publicKeyBytes;
    std::array<uint8_t, 32> shapk;

    switch (type) {
        case P2PKH:
        case BECH32:
            if (!compressed) {
                publicKeyBytes[0] = 0x04;
                pubKey.x.Get32Bytes(publicKeyBytes.data() + 1);
                pubKey.y.Get32Bytes(publicKeyBytes.data() + 33);
                sha256_65(publicKeyBytes.data(), shapk.data());
            } else {
                publicKeyBytes[0] = pubKey.y.IsEven() ? 0x02 : 0x03;
                pubKey.x.Get32Bytes(publicKeyBytes.data() + 1);
                sha256_33(publicKeyBytes.data(), shapk.data());
            }
            ripemd160_32(shapk.data(), hash);
            break;

        case P2SH: {
            std::array<uint8_t, 22> script;
            script[0] = 0x00;  // OP_0
            script[1] = 0x14;  // PUSH 20 bytes
            GetHash160(P2PKH, compressed, pubKey, script.data() + 2);
            sha256(script.data(), 22, shapk.data());
            ripemd160_32(shapk.data(), hash);
            break;
        }
    }
}

std::string Secp256K1::GetPrivAddress(bool compressed, const Int& privKey) const {
    std::array<uint8_t, 38> address;
    address[0] = 0x80; // Mainnet
    privKey.Get32Bytes(address.data() + 1);

    if (compressed) {
        address[33] = 1;
        sha256_checksum(address.data(), 34, address.data() + 34);
        return EncodeBase58(address.data(), address.data() + 38);
    } else {
        sha256_checksum(address.data(), 33, address.data() + 33);
        return EncodeBase58(address.data(), address.data() + 37);
    }
}

std::string Secp256K1::GetAddress(AddressType type, bool compressed, const uint8_t* hash160) const {
    std::array<uint8_t, 25> address;

    switch (type) {
        case P2PKH:
            address[0] = 0x00;
            break;
        case P2SH:
            address[0] = 0x05;
            break;
        case BECH32: {
            char output[128];
            segwit_addr_encode(output, "bc", 0, hash160, 20);
            return std::string(output);
        }
    }

    std::copy_n(hash160, 20, address.data() + 1);
    sha256_checksum(address.data(), 21, address.data() + 21);
    return EncodeBase58(address.data(), address.data() + 25);
}

std::string Secp256K1::GetAddress(AddressType type, bool compressed, const Point& pubKey) const {
    switch (type) {
        case BECH32:
            if (!compressed) {
                return "BECH32: Only compressed key";
            }
            [[fallthrough]];
        case P2PKH:
        case P2SH: {
            std::array<uint8_t, 25> address;
            address[0] = (type == P2PKH) ? 0x00 : 0x05;
            GetHash160(type, compressed, pubKey, address.data() + 1);
            sha256_checksum(address.data(), 21, address.data() + 21);
            return EncodeBase58(address.data(), address.data() + 25);
        }
    }
    return "";
}

bool Secp256K1::CheckPudAddress(const std::string& address) const {
    std::vector<unsigned char> pubKey;
    DecodeBase58(address, pubKey);

    if (pubKey.size() != 25) return false;

    uint8_t chk[4];
    sha256_checksum(pubKey.data(), 21, chk);

    return std::equal(chk, chk + 4, pubKey.begin() + 21);
}

Point Secp256K1::AddDirect(const Point& p1, const Point& p2) const {
    Int dy, dx, _s, _p;
    Point r;
    r.z.SetInt32(1);

    dy.ModSub(&p2.y, &p1.y);
    dx.ModSub(&p2.x, &p1.x);
    dx.ModInv();
    _s.ModMulK1(&dy, &dx);     // s = (p2.y-p1.y)*inverse(p2.x-p1.x);

    _p.ModSquareK1(&_s);       // _p = pow2(s)

    r.x.ModSub(&_p, &p1.x);
    r.x.ModSub(&p2.x);         // rx = pow2(s) - p1.x - p2.x;

    r.y.ModSub(&p2.x, &r.x);
    r.y.ModMulK1(&_s);
    r.y.ModSub(&p2.y);         // ry = - p2.y - s*(ret.x-p2.x);

    return r;
}

Point Secp256K1::Add2(const Point& p1, const Point& p2) const {
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

Point Secp256K1::Add(const Point& p1, const Point& p2) const {
    Int u, v, u1, u2, v1, v2, vs2, vs3, us2, w, a, us2w, vs2v2, vs3u2, _2vs2v2;
    Point r;

    u1.ModMulK1(&p2.y, &p1.z);
    u2.ModMulK1(&p1.y, &p2.z);
    v1.ModMulK1(&p2.x, &p1.z);
    v2.ModMulK1(&p1.x, &p2.z);
    u.ModSub(&u1, &u2);
    v.ModSub(&v1, &v2);
    w.ModMulK1(&p1.z, &p2.z);
    us2.ModSquareK1(&u);
    vs2.ModSquareK1(&v);
    vs3.ModMulK1(&vs2, &v);
    us2w.ModMulK1(&us2, &w);
    vs2v2.ModMulK1(&vs2, &v2);
    _2vs2v2.ModAdd(&vs2v2, &vs2v2);
    a.ModSub(&us2w, &vs3);
    a.ModSub(&_2vs2v2);

    r.x.ModMulK1(&v, &a);

    vs3u2.ModMulK1(&vs3, &u2);
    r.y.ModSub(&vs2v2, &a);
    r.y.ModMulK1(&r.y, &u);
    r.y.ModSub(&vs3u2);

    r.z.ModMulK1(&vs3, &w);

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
    _s.ModMulK1(&_p, &a);     // s = (3*pow2(p.x))*inverse(2*p.y);

    _p.ModMulK1(&_s, &_s);
    a.ModAdd(&p.x, &p.x);
    a.ModNeg();
    r.x.ModAdd(&a, &_p);      // rx = pow2(s) + neg(2*p.x);

    a.ModSub(&r.x, &p.x);
    _p.ModMulK1(&a, &_s);
    r.y.ModAdd(&_p, &p.y);
    r.y.ModNeg();             // ry = neg(p.y + s*(ret.x+neg(p.x)));

    return r;
}

Point Secp256K1::Double(const Point& p) const {
    Int z2, x2, _3x2, w, s, s2, b, _8b, _8y2s2, y2, h;
    Point r;

    z2.ModSquareK1(&p.z);
    z2.SetInt32(0); // a=0
    x2.ModSquareK1(&p.x);
    _3x2.ModAdd(&x2, &x2);
    _3x2.ModAdd(&x2);
    w.ModAdd(&z2, &_3x2);
    s.ModMulK1(&p.y, &p.z);
    b.ModMulK1(&p.y, &s);
    b.ModMulK1(&p.x);
    h.ModSquareK1(&w);
    _8b.ModAdd(&b, &b);
    _8b.ModDouble();
    _8b.ModDouble();
    h.ModSub(&_8b);

    r.x.ModMulK1(&h, &s);
    r.x.ModAdd(&r.x);

    s2.ModSquareK1(&s);
    y2.ModSquareK1(&p.y);
    _8y2s2.ModMulK1(&y2, &s2);
    _8y2s2.ModDouble();
    _8y2s2.ModDouble();
    _8y2s2.ModDouble();

    r.y.ModAdd(&b, &b);
    r.y.ModAdd(&r.y, &r.y);
    r.y.ModSub(&h);
    r.y.ModMulK1(&w);
    r.y.ModSub(&_8y2s2);

    r.z.ModMulK1(&s2, &s);
    r.z.ModDouble();
    r.z.ModDouble();
    r.z.ModDouble();

    return r;
}

Int Secp256K1::GetY(const Int& x, bool isEven) const {
    Int _s, _p;

    _s.ModSquareK1(&x);
    _p.ModMulK1(&_s, &x);
    _p.ModAdd(7);
    _p.ModSqrt();

    if (_p.IsEven() != isEven) {
        _p.ModNeg();
    }

    return _p;
}

bool Secp256K1::EC(const Point& p) const {
    Int _s, _p;

    _s.ModSquareK1(&p.x);
    _p.ModMulK1(&_s, &p.x);
    _p.ModAdd(7);
    _s.ModMulK1(&p.y, &p.y);
    _s.ModSub(&_p);

    return _s.IsZero();
}
