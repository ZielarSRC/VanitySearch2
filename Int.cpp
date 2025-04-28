#include "Int.h"
#include <algorithm>
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <random>
#include <climits>

// Constructors
Int::Int() {
    Init();
    SetZero();
}

Int::Int(int64_t value) {
    Init();
    InitFromInt64(value);
}

Int::Int(const std::string &value) {
    Init();
    InitFromString(value);
}

Int::Int(const Int &value) {
    Init();
    Set(value);
}

// Initialization
void Int::Init() {
    bits.clear();
    sign = false;
    base = 10;
}

void Int::InitFromInt64(int64_t value) {
    if (value < 0) {
        sign = true;
        value = -value;
    } else {
        sign = false;
    }
    
    bits.resize(2);
    bits[0] = (uint32_t)(value & 0xFFFFFFFF);
    bits[1] = (uint32_t)(value >> 32);
    Contract();
}

void Int::InitFromString(const std::string &value) {
    if (value.empty()) {
        SetZero();
        return;
    }

    size_t start = 0;
    if (value[0] == '-') {
        sign = true;
        start = 1;
    } else {
        sign = false;
    }

    SetZero();
    Int power(1);

    for (size_t i = value.length(); i > start; ) {
        --i;
        uint32_t digit = 0;
        char c = value[i];

        if (c >= '0' && c <= '9') {
            digit = c - '0';
        } else if (c >= 'A' && c <= 'F') {
            digit = 10 + c - 'A';
        } else if (c >= 'a' && c <= 'f') {
            digit = 10 + c - 'a';
        } else {
            throw std::runtime_error("Invalid character in number string");
        }

        if (digit >= base) {
            throw std::runtime_error("Digit exceeds current base");
        }

        *this += power * Int(digit);
        if (i > start) {
            power *= Int(base);
        }
    }
}

// Assignment
Int& Int::operator=(const Int &value) {
    Set(value);
    return *this;
}

Int& Int::operator=(int64_t value) {
    Set(value);
    return *this;
}

void Int::Set(const Int &value) {
    bits = value.bits;
    sign = value.sign;
    base = value.base;
}

void Int::Set(int64_t value) {
    InitFromInt64(value);
}

void Int::Set(const std::string &value) {
    InitFromString(value);
}

void Int::SetZero() {
    bits.clear();
    bits.push_back(0);
    sign = false;
}

void Int::SetOne() {
    bits.clear();
    bits.push_back(1);
    sign = false;
}

void Int::SetInt32(uint32_t value) {
    bits.clear();
    bits.push_back(value);
    sign = false;
}

void Int::SetInt64(uint64_t value) {
    bits.clear();
    bits.push_back((uint32_t)(value & 0xFFFFFFFF));
    bits.push_back((uint32_t)(value >> 32));
    sign = false;
    Contract();
}

// Comparison
int Int::Compare(const Int &other) const {
    if (sign != other.sign) {
        return sign ? -1 : 1;
    }

    if (bits.size() != other.bits.size()) {
        return (bits.size() > other.bits.size()) ? (sign ? -1 : 1) : (sign ? 1 : -1);
    }

    for (int i = (int)bits.size() - 1; i >= 0; --i) {
        if (bits[i] != other.bits[i]) {
            return (bits[i] > other.bits[i]) ? (sign ? -1 : 1) : (sign ? 1 : -1);
        }
    }

    return 0;
}

bool Int::operator==(const Int &other) const {
    return Compare(other) == 0;
}

bool Int::operator!=(const Int &other) const {
    return Compare(other) != 0;
}

bool Int::operator>=(const Int &other) const {
    return Compare(other) >= 0;
}

bool Int::operator<=(const Int &other) const {
    return Compare(other) <= 0;
}

bool Int::operator>(const Int &other) const {
    return Compare(other) > 0;
}

bool Int::operator<(const Int &other) const {
    return Compare(other) < 0;
}

// Arithmetic operations
Int Int::operator+(const Int &other) const {
    Int result(*this);
    result.Add(other);
    return result;
}

Int Int::operator-(const Int &other) const {
    Int result(*this);
    result.Sub(other);
    return result;
}

Int Int::operator*(const Int &other) const {
    Int result;
    result.Mult(*this, other);
    return result;
}

Int Int::operator/(const Int &other) const {
    Int result;
    result.Div(*this, other, NULL);
    return result;
}

Int Int::operator%(const Int &other) const {
    Int result;
    result.Mod(*this, other);
    return result;
}

Int Int::operator-() const {
    Int result(*this);
    result.Negate();
    return result;
}

// Bitwise operations
Int Int::operator&(const Int &other) const {
    Int result(*this);
    result.And(other);
    return result;
}

Int Int::operator|(const Int &other) const {
    Int result(*this);
    result.Or(other);
    return result;
}

Int Int::operator^(const Int &other) const {
    Int result(*this);
    result.Xor(other);
    return result;
}

Int Int::operator~() const {
    Int result(*this);
    result.Not();
    return result;
}

Int Int::operator>>(uint32_t shift) const {
    Int result(*this);
    result.ShiftR(shift);
    return result;
}

Int Int::operator<<(uint32_t shift) const {
    Int result(*this);
    result.ShiftL(shift);
    return result;
}

// Assignment operators
Int& Int::operator+=(const Int &other) {
    Add(other);
    return *this;
}

Int& Int::operator-=(const Int &other) {
    Sub(other);
    return *this;
}

Int& Int::operator*=(const Int &other) {
    Mult(other);
    return *this;
}

Int& Int::operator/=(const Int &other) {
    Div(other, NULL);
    return *this;
}

Int& Int::operator%=(const Int &other) {
    Mod(other);
    return *this;
}

Int& Int::operator&=(const Int &other) {
    And(other);
    return *this;
}

Int& Int::operator|=(const Int &other) {
    Or(other);
    return *this;
}

Int& Int::operator^=(const Int &other) {
    Xor(other);
    return *this;
}

Int& Int::operator>>=(uint32_t shift) {
    ShiftR(shift);
    return *this;
}

Int& Int::operator<<=(uint32_t shift) {
    ShiftL(shift);
    return *this;
}

// Increment/decrement
Int& Int::operator++() {
    Inc();
    return *this;
}

Int Int::operator++(int) {
    Int temp(*this);
    Inc();
    return temp;
}

Int& Int::operator--() {
    Dec();
    return *this;
}

Int Int::operator--(int) {
    Int temp(*this);
    Dec();
    return temp;
}

// Addition with carry
void Int::Add(const Int &a, uint32_t c) {
    uint64_t carry = c;
    Extend(MAX(bits.size(), a.bits.size()) + 1);

    for (size_t i = 0; i < bits.size(); i++) {
        carry += (uint64_t)bits[i] + (i < a.bits.size() ? a.bits[i] : 0);
        bits[i] = (uint32_t)(carry & 0xFFFFFFFF);
        carry >>= 32;
    }

    Contract();
}

// Subtraction with borrow
void Int::Sub(const Int &a, uint32_t c) {
    uint64_t borrow = c;
    Extend(MAX(bits.size(), a.bits.size()));

    for (size_t i = 0; i < bits.size(); i++) {
        uint64_t temp = (uint64_t)bits[i] - (i < a.bits.size() ? a.bits[i] : 0) - borrow;
        borrow = (temp >> 63) & 1;
        bits[i] = (uint32_t)(temp & 0xFFFFFFFF);
    }

    Contract();
}

// Multiplication by single word
void Int::Mult(const Int &a, uint32_t b) {
    uint64_t carry = 0;
    Extend(a.bits.size() + 1);

    for (size_t i = 0; i < a.bits.size(); i++) {
        carry += (uint64_t)a.bits[i] * b;
        bits[i] = (uint32_t)(carry & 0xFFFFFFFF);
        carry >>= 32;
    }

    if (carry) {
        bits[a.bits.size()] = (uint32_t)carry;
    }

    Contract();
}

// Division with remainder
void Int::Div(const Int &a, Int *mod) {
    if (a.IsZero()) {
        throw std::runtime_error("Division by zero");
    }

    Int divisor(a);
    Int remainder;
    SetZero();
    remainder.bits = bits;
    remainder.sign = sign;

    divisor.sign = false;
    remainder.sign = false;

    int shift = remainder.GetBitLength() - divisor.GetBitLength();
    if (shift > 0) {
        divisor <<= shift;
    }

    while (shift >= 0) {
        if (remainder >= divisor) {
            remainder -= divisor;
            SetBit(shift);
        }
        divisor >>= 1;
        shift--;
    }

    if (mod) {
        *mod = remainder;
        mod->sign = sign ^ a.sign;
    }

    sign = sign ^ a.sign;
    Contract();
}

// Modular arithmetic
void Int::Mod(const Int &a, const Int &m) {
    Int q;
    Div(a, m, &q);
    *this = q;
}

// Modular multiplication
Int Int::ModMul(Int a, Int m) const {
    Int result;
    result.Mult(*this, a);
    result.Mod(m);
    return result;
}

// Modular exponentiation
Int Int::ModPow(Int exp, Int m) const {
    if (m.IsOne()) {
        return Int(0);
    }

    Int result(1);
    Int base(*this);
    base.Mod(m);

    while (!exp.IsZero()) {
        if (exp.IsOdd()) {
            result = result.ModMul(base, m);
        }
        exp >>= 1;
        base = base.ModMul(base, m);
    }

    return result;
}

// Modular inverse
Int Int::ModInv(Int m) const {
    Int g, x, y;
    g.ExtendedGCD(*this, m, &x, &y);
    if (g != Int(1)) {
        throw std::runtime_error("No modular inverse exists");
    }
    return x.Mod(m);
}

// Extended GCD
void Int::ExtendedGCD(const Int &a, const Int &b, Int *x, Int *y) {
    if (a.IsZero()) {
        *x = Int(0);
        *y = Int(1);
        return;
    }

    Int x1, y1;
    ExtendedGCD(b % a, a, &x1, &y1);
    *x = y1 - (b / a) * x1;
    *y = x1;
}

// Bit manipulation
void Int::ShiftLeft(uint32_t shift) {
    if (IsZero()) return;

    uint32_t wordShift = shift / 32;
    uint32_t bitShift = shift % 32;

    Extend(bits.size() + wordShift + 1);

    // Shift by whole words first
    if (wordShift > 0) {
        for (int i = (int)bits.size() - 1; i >= (int)wordShift; i--) {
            bits[i] = bits[i - wordShift];
        }
        for (uint32_t i = 0; i < wordShift; i++) {
            bits[i] = 0;
        }
    }

    // Shift by remaining bits
    if (bitShift > 0) {
        uint32_t carry = 0;
        for (size_t i = 0; i < bits.size(); i++) {
            uint64_t temp = (uint64_t)bits[i] << bitShift | carry;
            bits[i] = (uint32_t)(temp & 0xFFFFFFFF);
            carry = (uint32_t)(temp >> 32);
        }
        if (carry) {
            bits.push_back(carry);
        }
    }

    Contract();
}

void Int::ShiftRight(uint32_t shift) {
    if (IsZero()) return;

    uint32_t wordShift = shift / 32;
    uint32_t bitShift = shift % 32;

    // Shift by whole words first
    if (wordShift > 0) {
        if (wordShift >= bits.size()) {
            SetZero();
            return;
        }
        for (size_t i = 0; i < bits.size() - wordShift; i++) {
            bits[i] = bits[i + wordShift];
        }
        for (size_t i = bits.size() - wordShift; i < bits.size(); i++) {
            bits[i] = 0;
        }
    }

    // Shift by remaining bits
    if (bitShift > 0) {
        uint32_t carry = 0;
        for (int i = (int)bits.size() - 1; i >= 0; i--) {
            uint32_t temp = bits[i];
            bits[i] = (temp >> bitShift) | carry;
            carry = temp << (32 - bitShift);
        }
    }

    Contract();
}

// Bit access
uint32_t Int::GetBit(uint32_t n) const {
    uint32_t word = n / 32;
    uint32_t bit = n % 32;
    if (word >= bits.size()) return 0;
    return (bits[word] >> bit) & 1;
}

void Int::SetBit(uint32_t n, uint32_t val) {
    uint32_t word = n / 32;
    uint32_t bit = n % 32;
    Extend(word + 1);
    if (val) {
        bits[word] |= (1 << bit);
    } else {
        bits[word] &= ~(1 << bit);
    }
    Contract();
}

void Int::ClearBit(uint32_t n) {
    SetBit(n, 0);
}

// Bit length
int Int::GetBitLength() const {
    if (IsZero()) return 0;
    
    int bitsInLastWord = 0;
    uint32_t lastWord = bits.back();
    while (lastWord) {
        lastWord >>= 1;
        bitsInLastWord++;
    }
    
    return (int)((bits.size() - 1) * 32 + bitsInLastWord);
}

// String conversion
std::string Int::GetBase10() const {
    if (IsZero()) return "0";
    
    std::string result;
    Int temp(*this);
    temp.sign = false;
    
    while (!temp.IsZero()) {
        Int remainder;
        temp.Div(Int(10), &remainder);
        result.push_back((char)('0' + remainder.bits[0]));
    }
    
    if (sign) {
        result.push_back('-');
    }
    
    std::reverse(result.begin(), result.end());
    return result;
}

std::string Int::GetBase16() const {
    if (IsZero()) return "0";
    
    static const char* digits = "0123456789ABCDEF";
    std::string result;
    
    for (int i = (int)bits.size() - 1; i >= 0; i--) {
        uint32_t word = bits[i];
        for (int j = 28; j >= 0; j -= 4) {
            uint32_t nibble = (word >> j) & 0xF;
            if (!result.empty() || nibble != 0) {
                result.push_back(digits[nibble]);
            }
        }
    }
    
    if (result.empty()) {
        result = "0";
    }
    
    if (sign) {
        result = "-" + result;
    }
    
    return result;
}

// Random number generation
void Int::Random(int nbits) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dist;

    bits.clear();
    int words = (nbits + 31) / 32;
    for (int i = 0; i < words; i++) {
        bits.push_back(dist(gen));
    }

    // Clear excess bits
    int excessBits = words * 32 - nbits;
    if (excessBits > 0) {
        bits.back() >>= excessBits;
    }

    sign = false;
    Contract();
}

// Bitwise AND
void Int::And(const Int &a) {
    size_t maxSize = MIN(bits.size(), a.bits.size());
    for (size_t i = 0; i < maxSize; i++) {
        bits[i] &= a.bits[i];
    }
    for (size_t i = maxSize; i < bits.size(); i++) {
        bits[i] = 0;
    }
    Contract();
}

// Bitwise OR
void Int::Or(const Int &a) {
    Extend(a.bits.size());
    for (size_t i = 0; i < a.bits.size(); i++) {
        bits[i] |= a.bits[i];
    }
}

// Bitwise XOR
void Int::Xor(const Int &a) {
    Extend(a.bits.size());
    for (size_t i = 0; i < a.bits.size(); i++) {
        bits[i] ^= a.bits[i];
    }
    Contract();
}

// Bitwise NOT
void Int::Not() {
    for (size_t i = 0; i < bits.size(); i++) {
        bits[i] = ~bits[i];
    }
    Contract();
}

// Left Shift
void Int::ShiftL(uint32_t shift) {
    if (IsZero()) return;
    uint32_t wordShift = shift / 32;
    uint32_t bitShift = shift % 32;

    if (wordShift > 0) {
        bits.insert(bits.begin(), wordShift, 0);
    }

    if (bitShift > 0) {
        uint32_t carry = 0;
        for (size_t i = wordShift; i < bits.size(); i++) {
            uint64_t temp = (uint64_t)bits[i] << bitShift | carry;
            bits[i] = (uint32_t)(temp & 0xFFFFFFFF);
            carry = (uint32_t)(temp >> 32);
        }
        if (carry) {
            bits.push_back(carry);
        }
    }
}

// Right Shift
void Int::ShiftR(uint32_t shift) {
    if (IsZero()) return;
    uint32_t wordShift = shift / 32;
    uint32_t bitShift = shift % 32;

    if (wordShift >= bits.size()) {
        SetZero();
        return;
    }

    if (wordShift > 0) {
        bits.erase(bits.begin(), bits.begin() + wordShift);
    }

    if (bitShift > 0) {
        uint32_t carry = 0;
        for (int i = (int)bits.size() - 1; i >= 0; i--) {
            uint32_t temp = bits[i];
            bits[i] = (temp >> bitShift) | carry;
            carry = temp << (32 - bitShift);
        }
    }
    Contract();
}

// Modular Inverse (Extended Euclidean Algorithm)
void Int::InvMod(const Int &a, const Int &m) {
    Int m0 = m;
    Int y(0), x(1);
    Int q, t;
    Int a_temp = a;

    if (m == Int(1)) {
        SetZero();
        return;
    }

    while (a_temp > Int(1)) {
        q = a_temp / m0;
        t = m0;
        m0 = a_temp % m0;
        a_temp = t;
        t = y;
        y = x - q * y;
        x = t;
    }

    if (x < Int(0)) {
        x += m;
    }

    *this = x;
}

// Montgomery Multiplication
void Int::MontgomeryMult(Int &a, Int &b, const Int &m, uint32_t mDash) {
    Int T;
    uint64_t carry;
    uint32_t ui;

    T.SetZero();
    T.Extend(m.bits.size() + 1);

    for (size_t i = 0; i < m.bits.size(); i++) {
        ui = (uint32_t)((T.bits[0] + a.bits[i] * b.bits[0]) * mDash);
        carry = 0;

        // Compute T = T + a_i * b + ui * m
        for (size_t j = 0; j < m.bits.size(); j++) {
            uint64_t t = (uint64_t)T.bits[j] + 
                        (uint64_t)a.bits[i] * (j < b.bits.size() ? b.bits[j] : 0) + 
                        (uint64_t)ui * (j < m.bits.size() ? m.bits[j] : 0) + 
                        carry;
            T.bits[j] = (uint32_t)(t & 0xFFFFFFFF);
            carry = t >> 32;
        }

        // Propagate carry
        for (size_t j = m.bits.size(); j < T.bits.size() && carry > 0; j++) {
            uint64_t t = (uint64_t)T.bits[j] + carry;
            T.bits[j] = (uint32_t)(t & 0xFFFFFFFF);
            carry = t >> 32;
        }
    }

    // Final reduction
    if (T >= m) {
        T -= m;
    }

    *this = T;
}

// Barrett Reduction
void Int::BarrettMod(const Int &a, const Int &m, const Int &mu) {
    Int q1, q2, q3, r1, r2;
    size_t k = m.bits.size();

    // q1 = floor(x / b^{k-1})
    q1.bits.assign(a.bits.begin() + (k - 1), a.bits.end());

    // q2 = q1 * mu
    q2.Mult(q1, mu);

    // q3 = floor(q2 / b^{k+1})
    if (q2.bits.size() > (k + 1)) {
        q3.bits.assign(q2.bits.begin() + (k + 1), q2.bits.end());
    }

    // r1 = x mod b^{k+1}
    r1.bits.assign(a.bits.begin(), a.bits.begin() + MIN(k + 1, a.bits.size()));

    // r2 = (q3 * m) mod b^{k+1}
    r2.Mult(q3, m);
    if (r2.bits.size() > (k + 1)) {
        r2.bits.resize(k + 1);
    }

    // r = r1 - r2
    if (r1 < r2) {
        Int tmp;
        tmp.bits.resize(k + 1, 0);
        tmp.bits[k + 1] = 1;
        r1 += tmp;
    }
    r1 -= r2;

    // Final reduction
    while (r1 >= m) {
        r1 -= m;
    }

    *this = r1;
}

// Get hexadecimal string
std::string Int::GetBase16() const {
    if (IsZero()) return "0";

    static const char* digits = "0123456789ABCDEF";
    std::string result;

    for (int i = (int)bits.size() - 1; i >= 0; i--) {
        uint32_t word = bits[i];
        for (int j = 28; j >= 0; j -= 4) {
            uint32_t nibble = (word >> j) & 0xF;
            if (!result.empty() || nibble != 0) {
                result.push_back(digits[nibble]);
            }
        }
    }

    if (result.empty()) {
        result = "0";
    }

    if (sign) {
        result = "-" + result;
    }

    return result;
}

// Get binary string
std::string Int::GetBase2() const {
    if (IsZero()) return "0";

    std::string result;
    bool leadingZeros = true;

    for (int i = (int)bits.size() - 1; i >= 0; i--) {
        uint32_t word = bits[i];
        for (int j = 31; j >= 0; j--) {
            uint32_t bit = (word >> j) & 1;
            if (bit) leadingZeros = false;
            if (!leadingZeros) {
                result.push_back(bit ? '1' : '0');
            }
        }
    }

    if (result.empty()) {
        result = "0";
    }

    if (sign) {
        result = "-" + result;
    }

    return result;
}

// Get decimal string
std::string Int::GetBase10() const {
    if (IsZero()) return "0";

    std::string result;
    Int temp(*this);
    temp.sign = false;

    while (!temp.IsZero()) {
        Int remainder;
        temp.Div(Int(10), &remainder);
        result.push_back((char)('0' + remainder.bits[0]));
    }

    if (sign) {
        result.push_back('-');
    }

    std::reverse(result.begin(), result.end());
    return result;
}

// Get block string (for debugging)
std::string Int::GetBlockStr() const {
    std::string result;
    for (size_t i = 0; i < bits.size(); i++) {
        char buf[16];
        snprintf(buf, sizeof(buf), "%08X", bits[i]);
        if (i > 0) result += " ";
        result += buf;
    }
    if (sign) result = "-" + result;
    return result;
}

// Get string representation in specified base
std::string Int::GetStr(uint32_t base) const {
    switch (base) {
        case 2:  return GetBase2();
        case 10: return GetBase10();
        case 16: return GetBase16();
        default: throw std::runtime_error("Unsupported base");
    }
}

// Set from bytes (little-endian)
void Int::Set32Bytes(const unsigned char *bytes) {
    bits.resize(8);
    for (int i = 0; i < 8; i++) {
        bits[i] = 
            (uint32_t)bytes[i*4]        | 
            (uint32_t)bytes[i*4+1] << 8 | 
            (uint32_t)bytes[i*4+2] << 16 | 
            (uint32_t)bytes[i*4+3] << 24;
    }
    Contract();
}

// Get bytes (little-endian)
void Int::Get32Bytes(unsigned char *bytes) const {
    for (int i = 0; i < 8; i++) {
        uint32_t word = i < (int)bits.size() ? bits[i] : 0;
        bytes[i*4]   = (word) & 0xFF;
        bytes[i*4+1] = (word >> 8) & 0xFF;
        bytes[i*4+2] = (word >> 16) & 0xFF;
        bytes[i*4+3] = (word >> 24) & 0xFF;
    }
}

// Similar implementations for 64, 128, 256, 512 bytes
void Int::Set64Bytes(const unsigned char *bytes) {
    bits.resize(16);
    for (int i = 0; i < 16; i++) {
        bits[i] = 
            (uint32_t)bytes[i*4]        | 
            (uint32_t)bytes[i*4+1] << 8 | 
            (uint32_t)bytes[i*4+2] << 16 | 
            (uint32_t)bytes[i*4+3] << 24;
    }
    Contract();
}

void Int::Get64Bytes(unsigned char *bytes) const {
    for (int i = 0; i < 16; i++) {
        uint32_t word = i < (int)bits.size() ? bits[i] : 0;
        bytes[i*4]   = (word) & 0xFF;
        bytes[i*4+1] = (word >> 8) & 0xFF;
        bytes[i*4+2] = (word >> 16) & 0xFF;
        bytes[i*4+3] = (word >> 24) & 0xFF;
    }
}

// 128 bytes (1024 bits)
void Int::Set128Bytes(const unsigned char *bytes) {
    bits.resize(32);
    for (int i = 0; i < 32; i++) {
        bits[i] = 
            (uint32_t)bytes[i*4]        | 
            (uint32_t)bytes[i*4+1] << 8 | 
            (uint32_t)bytes[i*4+2] << 16 | 
            (uint32_t)bytes[i*4+3] << 24;
    }
    Contract();
}

void Int::Get128Bytes(unsigned char *bytes) const {
    for (int i = 0; i < 32; i++) {
        uint32_t word = i < (int)bits.size() ? bits[i] : 0;
        bytes[i*4]   = (word) & 0xFF;
        bytes[i*4+1] = (word >> 8) & 0xFF;
        bytes[i*4+2] = (word >> 16) & 0xFF;
        bytes[i*4+3] = (word >> 24) & 0xFF;
    }
}

// 256 bytes (2048 bits)
void Int::Set256Bytes(const unsigned char *bytes) {
    bits.resize(64);
    for (int i = 0; i < 64; i++) {
        bits[i] = 
            (uint32_t)bytes[i*4]        | 
            (uint32_t)bytes[i*4+1] << 8 | 
            (uint32_t)bytes[i*4+2] << 16 | 
            (uint32_t)bytes[i*4+3] << 24;
    }
    Contract();
}

void Int::Get256Bytes(unsigned char *bytes) const {
    for (int i = 0; i < 64; i++) {
        uint32_t word = i < (int)bits.size() ? bits[i] : 0;
        bytes[i*4]   = (word) & 0xFF;
        bytes[i*4+1] = (word >> 8) & 0xFF;
        bytes[i*4+2] = (word >> 16) & 0xFF;
        bytes[i*4+3] = (word >> 24) & 0xFF;
    }
}

// 512 bytes (4096 bits)
void Int::Set512Bytes(const unsigned char *bytes) {
    bits.resize(128);
    for (int i = 0; i < 128; i++) {
        bits[i] = 
            (uint32_t)bytes[i*4]        | 
            (uint32_t)bytes[i*4+1] << 8 | 
            (uint32_t)bytes[i*4+2] << 16 | 
            (uint32_t)bytes[i*4+3] << 24;
    }
    Contract();
}

void Int::Get512Bytes(unsigned char *bytes) const {
    for (int i = 0; i < 128; i++) {
        uint32_t word = i < (int)bits.size() ? bits[i] : 0;
        bytes[i*4]   = (word) & 0xFF;
        bytes[i*4+1] = (word >> 8) & 0xFF;
        bytes[i*4+2] = (word >> 16) & 0xFF;
        bytes[i*4+3] = (word >> 24) & 0xFF;
    }
}
// Random number generation within range
void Int::Rand(Int min, Int max) {
    Int range = max - min + Int(1);
    Random(range.GetBitLength());
    *this = min + (*this % range);
}

// Check if number is even
bool Int::IsEven() const {
    return bits.empty() || (bits[0] & 1) == 0;
}

// Check if number is odd
bool Int::IsOdd() const {
    return !IsEven();
}

// Get number size in words
int Int::GetSize() const {
    return (int)bits.size();
}

// Exponentiation
void Int::Pow(const Int &a, const Int &b) {
    SetOne();
    Int base(a);
    Int exp(b);

    while (!exp.IsZero()) {
        if (exp.IsOdd()) {
            *this *= base;
        }
        base *= base;
        exp >>= 1;
    }
}

// Modular exponentiation
void Int::PowMod(const Int &a, const Int &b, const Int &m) {
    SetOne();
    Int base(a % m);
    Int exp(b);

    while (!exp.IsZero()) {
        if (exp.IsOdd()) {
            *this = (*this * base) % m;
        }
        base = (base * base) % m;
        exp >>= 1;
    }
}

// Increment
void Int::Inc() {
    Add(Int(1));
}

// Decrement
void Int::Dec() {
    Sub(Int(1));
}

// Negate
void Int::Negate() {
    if (!IsZero()) {
        sign = !sign;
    }
}

// Absolute value
void Int::Abs() {
    sign = false;
}

// Check if zero
bool Int::IsZero() const {
    return bits.size() == 1 && bits[0] == 0;
}

// Check if one
bool Int::IsOne() const {
    return !sign && bits.size() == 1 && bits[0] == 1;
}

// Check if negative
bool Int::IsNegative() const {
    return sign && !IsZero();
}

// Static methods
Int Int::Add(Int a, Int b) {
    Int result(a);
    result.Add(b);
    return result;
}

Int Int::Sub(Int a, Int b) {
    Int result(a);
    result.Sub(b);
    return result;
}

Int Int::Mul(Int a, Int b) {
    Int result;
    result.Mult(a, b);
    return result;
}

Int Int::Div(Int a, Int b) {
    Int result;
    result.Div(a, b, NULL);
    return result;
}

Int Int::Mod(Int a, Int b) {
    Int result;
    result.Mod(a, b);
    return result;
}

Int Int::And(Int a, Int b) {
    Int result(a);
    result.And(b);
    return result;
}

Int Int::Or(Int a, Int b) {
    Int result(a);
    result.Or(b);
    return result;
}

Int Int::Xor(Int a, Int b) {
    Int result(a);
    result.Xor(b);
    return result;
}

Int Int::Not(Int a) {
    Int result(a);
    result.Not();
    return result;
}

Int Int::Rand(int nbit) {
    Int result;
    result.Random(nbit);
    return result;
}

Int Int::Rand(Int min, Int max) {
    Int range = max - min + Int(1);
    Int result;
    result.Random(range.GetBitLength());
    result = min + (result % range);
    return result;
}

Int Int::InvMod(Int a, Int m) {
    return a.ModInv(m);
}

Int Int::Pow(Int a, Int b) {
    Int result(1);
    while (!b.IsZero()) {
        if (b.IsOdd()) {
            result *= a;
        }
        a *= a;
        b >>= 1;
    }
    return result;
}

Int Int::PowMod(Int a, Int b, Int m) {
    return a.ModPow(b, m);
}
