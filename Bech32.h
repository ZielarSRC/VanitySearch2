#pragma once
#include <string>
#include <vector>

class Bech32 {
public:
    static const std::string CHARSET;
    static const std::string SEPARATOR;
    static const uint32_t M = 0x2BC830A3;
    static const int CHECKSUM_LENGTH = 6;
    
    static std::string encode(const std::string& hrp, const std::vector<unsigned char>& values);
    static bool decode(const std::string& str, std::string& hrp, std::vector<unsigned char>& values);
    static bool verifyChecksum(const std::string& hrp, const std::vector<unsigned char>& values);
    static std::vector<unsigned char> createChecksum(const std::string& hrp, const std::vector<unsigned char>& values);
    
private:
    static uint32_t polymod(const std::vector<unsigned char>& values);
    static std::vector<unsigned char> expandHrp(const std::string& hrp);
    static std::vector<unsigned char> convertBits(const std::vector<unsigned char>& in, int fromBits, int toBits, bool pad);
};
