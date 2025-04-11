#pragma once
#include "Int.h"
#include <array>

class IntMod {
public:
    // Konstruktory
    IntMod() = default;
    explicit IntMod(const Int& value, const Int& modulus);
    explicit IntMod(const Int& value);

    // Funkcje zarządzające modulusem
    void setModulus(const Int& modulus);
    const Int& getModulus() const { return mod; }
    const Int& getValue() const { return val; }

    // Operatory przypisania
    IntMod& operator=(const Int& value);
    IntMod& operator+=(const IntMod& other);
    IntMod& operator-=(const IntMod& other);
    IntMod& operator*=(const IntMod& other);
    IntMod& operator/=(const IntMod& other);

    // Operatory arytmetyczne
    IntMod operator+(const IntMod& other) const;
    IntMod operator-(const IntMod& other) const;
    IntMod operator*(const IntMod& other) const;
    IntMod operator/(const IntMod& other) const;
    IntMod operator-() const;

    // Operatory porównania
    bool operator==(const IntMod& other) const;
    bool operator!=(const IntMod& other) const;

    // Funkcje specjalne
    IntMod inverse() const;
    IntMod pow(const Int& exponent) const;
    IntMod sqrt() const;
    void randomize();

    // Metody pomocnicze
    bool isZero() const { return val.isZero(); }
    bool isOne() const { return val == Int::one(); }
    std::string toString() const;

private:
    Int val;
    Int mod;
    mutable Int montgomeryR;  // Cache dla Montgomery'ego
    mutable Int montgomeryR2; // R^2 mod N
    mutable bool montgomeryReady = false;

    void checkModulus() const;
    void checkModulusMatch(const IntMod& other) const;
    void reduce();
    void initMontgomery() const;
    void montgomeryReduce(Int& x) const;
};

// Operatory globalne
IntMod operator+(const Int& lhs, const IntMod& rhs);
IntMod operator*(const Int& lhs, const IntMod& rhs);
