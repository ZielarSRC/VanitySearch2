#include "IntMod.h"
#include "Random.h"
#include <stdexcept>
#include <sstream>

// Konstruktory
IntMod::IntMod(const Int& value, const Int& modulus) 
    : val(value), mod(modulus) {
    if (mod.isZero()) throw std::invalid_argument("Modulus cannot be zero");
    reduce();
    initMontgomery();
}

IntMod::IntMod(const Int& value) : val(value), mod(0) {}

// Inicjalizacja Montgomery'ego
void IntMod::initMontgomery() const {
    if (montgomeryReady || mod.isZero()) return;
    
    int k = mod.getBitLength();
    montgomeryR = Int::one().leftShift(k);
    montgomeryR2 = (montgomeryR * montgomeryR) % mod;
    
    // Oblicz inv64 dla redukcji
    Int::precomputeMontgomery(mod);
    montgomeryReady = true;
}

// Operacje podstawowe
void IntMod::setModulus(const Int& modulus) {
    if (modulus.isZero()) throw std::invalid_argument("Modulus cannot be zero");
    mod = modulus;
    montgomeryReady = false;
    reduce();
    initMontgomery();
}

void IntMod::checkModulus() const {
    if (mod.isZero()) throw std::runtime_error("Modulus not initialized");
}

void IntMod::checkModulusMatch(const IntMod& other) const {
    if (mod != other.mod) throw std::runtime_error("Modulus mismatch");
}

void IntMod::reduce() {
    if (!mod.isZero()) {
        val %= mod;
        if (val.isNegative()) val += mod;
    }
}

// Operatory przypisania
IntMod& IntMod::operator=(const Int& value) {
    val = value;
    reduce();
    return *this;
}

IntMod& IntMod::operator+=(const IntMod& other) {
    checkModulus();
    checkModulusMatch(other);
    val += other.val;
    if (val >= mod) val -= mod;
    return *this;
}

IntMod& IntMod::operator-=(const IntMod& other) {
    checkModulus();
    checkModulusMatch(other);
    val -= other.val;
    if (val.isNegative()) val += mod;
    return *this;
}

IntMod& IntMod::operator*=(const IntMod& other) {
    checkModulus();
    checkModulusMatch(other);
    if (montgomeryReady) {
        val.montgomeryMul(other.val, mod, montgomeryR2);
    } else {
        val *= other.val;
        val %= mod;
    }
    return *this;
}

IntMod& IntMod::operator/=(const IntMod& other) {
    return *this *= other.inverse();
}

// Operatory arytmetyczne
IntMod IntMod::operator+(const IntMod& other) const {
    IntMod result(*this);
    result += other;
    return result;
}

IntMod IntMod::operator-(const IntMod& other) const {
    IntMod result(*this);
    result -= other;
    return result;
}

IntMod IntMod::operator*(const IntMod& other) const {
    IntMod result(*this);
    result *= other;
    return result;
}

IntMod IntMod::operator/(const IntMod& other) const {
    IntMod result(*this);
    result /= other;
    return result;
}

IntMod IntMod::operator-() const {
    checkModulus();
    return IntMod(mod - val, mod);
}

// Operatory porównania
bool IntMod::operator==(const IntMod& other) const {
    checkModulus();
    checkModulusMatch(other);
    return val == other.val;
}

bool IntMod::operator!=(const IntMod& other) const {
    return !(*this == other);
}

// Funkcje specjalne
IntMod IntMod::inverse() const {
    checkModulus();
    if (val.isZero()) throw std::runtime_error("Division by zero");
    return IntMod(val.modInverse(mod), mod);
}

IntMod IntMod::pow(const Int& exponent) const {
    checkModulus();
    if (exponent.isNegative()) return inverse().pow(-exponent);
    
    IntMod result(Int::one(), mod);
    IntMod base(*this);
    Int exp(exponent);
    
    while (!exp.isZero()) {
        if (exp.isOdd()) result *= base;
        base *= base;
        exp >>= 1;
    }
    return result;
}

IntMod IntMod::sqrt() const {
    checkModulus();
    if (val.isZero() || val.isOne()) return *this;
    
    // Tonelli-Shanks
    if (mod % Int(4) == Int(3)) {
        return pow((mod + Int(1)) / Int(4));
    }
    
    Int Q = mod - Int::one();
    int S = 0;
    while (Q.isEven()) {
        Q >>= 1;
        S++;
    }
    
    Int z = Int::two();
    while (z.pow((mod - Int::one()) / Int(2)) % mod == Int::one()) {
        z += Int::one();
    }
    
    Int c = z.pow(Q);
    Int x = val.pow((Q + Int::one()) / Int(2));
    Int t = val.pow(Q);
    int M = S;
    
    while (t != Int::one()) {
        Int tt = t;
        int i = 0;
        while (tt != Int::one() && i < M) {
            tt = tt.pow(Int::two()) % mod;
            i++;
        }
        
        Int b = c.pow(Int::one().leftShift(M - i - 1));
        x = (x * b) % mod;
        t = (t * b * b) % mod;
        c = (b * b) % mod;
        M = i;
    }
    
    return IntMod(x, mod);
}

void IntMod::randomize() {
    checkModulus();
    val = Int::rand(mod);
}

std::string IntMod::toString() const {
    std::ostringstream oss;
    oss << val << " (mod " << mod << ")";
    return oss.str();
}

// Operatory globalne
IntMod operator+(const Int& lhs, const IntMod& rhs) {
    return IntMod(lhs, rhs.getModulus()) + rhs;
}

IntMod operator-(const Int& lhs, const IntMod& rhs) {
    return IntMod(lhs, rhs.getModulus()) - rhs;
}

IntMod operator*(const Int& lhs, const IntMod& rhs) {
    return IntMod(lhs, rhs.getModulus()) * rhs;
}

IntMod operator/(const Int& lhs, const IntMod& rhs) {
    return IntMod(lhs, rhs.getModulus()) / rhs;
}
