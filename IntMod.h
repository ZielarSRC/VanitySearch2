#pragma once
#include "Int.h"
#include <array>
#include <stdexcept>

class IntMod {
public:
    IntMod() = default;
    explicit IntMod(const Int& value, const Int& modulus);
    explicit IntMod(const Int& value); // modulus must be set later
    
    void setModulus(const Int& modulus);
    const Int& getModulus() const { return mod; }
    const Int& getValue() const { return val; }
    
    IntMod& operator=(const Int& value);
    IntMod& operator+=(const IntMod& other);
    IntMod& operator-=(const IntMod& other);
    IntMod& operator*=(const IntMod& other);
    IntMod& operator/=(const IntMod& other);
    IntMod& operator%=(const IntMod& other);
    IntMod& operator+=(const Int& value);
    IntMod& operator-=(const Int& value);
    IntMod& operator*=(const Int& value);
    IntMod& operator/=(const Int& value);
    IntMod& operator%=(const Int& value);
    
    IntMod operator+(const IntMod& other) const;
    IntMod operator-(const IntMod& other) const;
    IntMod operator*(const IntMod& other) const;
    IntMod operator/(const IntMod& other) const;
    IntMod operator%(const IntMod& other) const;
    IntMod operator+(const Int& value) const;
    IntMod operator-(const Int& value) const;
    IntMod operator*(const Int& value) const;
    IntMod operator/(const Int& value) const;
    IntMod operator%(const Int& value) const;
    
    IntMod operator-() const;
    
    bool operator==(const IntMod& other) const;
    bool operator!=(const IntMod& other) const;
    bool operator<(const IntMod& other) const;
    bool operator<=(const IntMod& other) const;
    bool operator>(const IntMod& other) const;
    bool operator>=(const IntMod& other) const;
    
    IntMod inverse() const;
    IntMod pow(const Int& exponent) const;
    IntMod sqrt() const;
    
    void randomize();
    bool isZero() const { return val.isZero(); }
    bool isOne() const { return val.isOne(); }
    
    std::string toString() const;

private:
    Int val;
    Int mod;
    
    void checkModulus() const;
    void checkModulusMatch(const IntMod& other) const;
    void reduce();
    void normalize();
};

// Non-member functions
IntMod operator+(const Int& lhs, const IntMod& rhs);
IntMod operator-(const Int& lhs, const IntMod& rhs);
IntMod operator*(const Int& lhs, const IntMod& rhs);
IntMod operator/(const Int& lhs, const IntMod& rhs);