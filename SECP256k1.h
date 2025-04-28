#ifndef SECP256K1_H
#define SECP256K1_H

#include "Point.h"
#include <string>
#include <vector>
#include <array>
#include <span>

enum AddressType {
    P2PKH = 0,
    P2SH = 1,
    BECH32 = 2
};

class Secp256K1 {
public:
    Secp256K1();
    ~Secp256K1() = default;

    void Init();
    Point ComputePublicKey(const Int* privKey);
    Point NextKey(const Point& key);
    void Check();
    bool EC(const Point& p) const;

    void GetHash160(AddressType type, bool compressed,
                   const Point& k0, const Point& k1, const Point& k2, const Point& k3,
                   uint8_t* h0, uint8_t* h1, uint8_t* h2, uint8_t* h3) const;

    void GetHash160(AddressType type, bool compressed, const Point& pubKey, uint8_t* hash) const;

    std::string GetAddress(AddressType type, bool compressed, const Point& pubKey) const;
    std::string GetAddress(AddressType type, bool compressed, const uint8_t* hash160) const;
    std::vector<std::string> GetAddress(AddressType type, bool compressed, 
                                      const uint8_t* h1, const uint8_t* h2, 
                                      const uint8_t* h3, const uint8_t* h4) const;
    std::string GetPrivAddress(bool compressed, const Int& privKey) const;
    std::string GetPublicKeyHex(bool compressed, const Point& p) const;
    Point ParsePublicKeyHex(const std::string& str, bool& isCompressed) const;

    bool CheckPudAddress(const std::string& address) const;

    static Int DecodePrivateKey(const char* key, bool* compressed);

    Point Add(const Point& p1, const Point& p2) const;
    Point Add2(const Point& p1, const Point& p2) const;
    Point AddDirect(const Point& p1, const Point& p2) const;
    Point Double(const Point& p) const;
    Point DoubleDirect(const Point& p) const;

    Point G;                 // Generator
    Int order;               // Curve order

private:
    uint8_t GetByte(const std::string& str, size_t idx) const;
    Int GetY(const Int& x, bool isEven) const;

    std::array<Point, 256*32> GTable;  // Generator table
};

#endif // SECP256K1_H
