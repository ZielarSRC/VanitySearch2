#pragma once
#include "Int.h"
#include <random>
#include <array>

class Random {
public:
    static void init();
    static Int getRandom();
    static Int getRandomRange(const Int& range);
    static void getRandomBytes(unsigned char* buf, size_t count);
    
private:
    static std::random_device rd;
    static std::mt19937_64 gen;
    static std::uniform_int_distribution<uint64_t> dis;
    
    static void fillRandomBuffer(std::array<uint64_t, 4>& buffer);
};
