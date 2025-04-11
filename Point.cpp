#include "Point.h"
#include "SECP256k1.h"
#include <stdexcept>
#include <sstream>

const Int Point::P = SECP256k1::P;
const Int Point::A = SECP256k1::A;
const Int Point::B = SECP256k1::B;
const Point Point::G(SECP256k1::Gx, SECP256k1::Gy);

Point::Point() : x(0), y(0), compressed(true) {}

Point::Point(const Int& x, const Int& y, bool compressed) 
    : x(x), y(y), compressed(compressed) {
    checkOnCurve();
}

Point::Point(const std::array<unsigned char, COMPRESSED_SIZE>& data) {
    if (!parse(data.data(), data.size())) {
        throw std::runtime_error("Invalid compressed point data");
    }
}

Point::Point(const std::array<unsigned char, UNCOMPRESSED_SIZE>& data) {
    if (!parse(data.data(), data.size())) {
        throw std::runtime_error("Invalid uncompressed point data");
    }
}

bool Point::parse(const unsigned char* data, size_t size) {
    if (size == COMPRESSED_SIZE && (data[0] == COMPRESSED_EVEN || data[0] == COMPRESSED_ODD)) {
        compressed = true;
        x.set(data + 1, 32);
        
        Int x3 = x;
        x3.modMul(x3, P);
        x3.modMul(x, P);
        
        Int y2 = x3 + A * x + B;
        y = y2.modSqrt(P);
        
        if ((y.isOdd() && data[0] == COMPRESSED_ODD) || (!y.isOdd() && data[0] == COMPRESSED_EVEN)) {
            return isValid();
        }
        y = P - y;
        return isValid();
    }
    else if (size == UNCOMPRESSED_SIZE && data[0] == UNCOMPRESSED) {
        compressed = false;
        x.set(data + 1, 32);
        y.set(data + 33, 32);
        return isValid();
    }
    return false;
}

void Point::serialize(std::array<unsigned char, COMPRESSED_SIZE>& out) const {
    if (isInfinity()) {
        throw std::runtime_error("Cannot serialize infinity point");
    }
    
    out[0] = y.isOdd() ? COMPRESSED_ODD : COMPRESSED_EVEN;
    x.get(out.data() + 1, 32);
}

void Point::serialize(std::array<unsigned char, UNCOMPRESSED_SIZE>& out) const {
    if (isInfinity()) {
        throw std::runtime_error("Cannot serialize infinity point");
    }
    
    out[0] = UNCOMPRESSED;
    x.get(out.data() + 1, 32);
    y.get(out.data() + 33, 32);
}

std::vector<unsigned char> Point::serialize(bool forceUncompressed) const {
    if (forceUncompressed || !compressed) {
        std::array<unsigned char, UNCOMPRESSED_SIZE> data;
        serialize(data);
        return std::vector<unsigned char>(data.begin(), data.end());
    } else {
        std::array<unsigned char, COMPRESSED_SIZE> data;
        serialize(data);
        return std::vector<unsigned char>(data.begin(), data.end());
    }
}

bool Point::isInfinity() const {
    return x.isZero() && y.isZero();
}

bool Point::isValid() const {
    if (isInfinity()) return true;
    
    Int y2 = y;
    y2.modMul(y2, P);
    
    Int x3 = x;
    x3.modMul(x3, P);
    x3.modMul(x, P);
    
    Int rhs = x3 + A * x + B;
    rhs.mod(P);
    
    return y2.equals(rhs);
}

bool Point::operator==(const Point& other) const {
    if (isInfinity() && other.isInfinity()) return true;
    return x == other.x && y == other.y;
}

bool Point::operator!=(const Point& other) const {
    return !(*this == other);
}

Point Point::operator+(const Point& other) const {
    Point result(*this);
    result += other;
    return result;
}

Point& Point::operator+=(const Point& other) {
    if (isInfinity()) {
        *this = other;
    } else if (other.isInfinity()) {
        // pozostaje bez zmian
    } else if (x == other.x) {
        if (y == other.y) {
            doublePoint();
        } else {
            *this = infinity();
        }
    } else {
        addPoint(other);
    }
    return *this;
}

Point Point::operator*(const Int& scalar) const {
    Point result(*this);
    result *= scalar;
    return result;
}

Point& Point::operator*=(const Int& scalar) {
    multiply(scalar);
    return *this;
}

Point Point::infinity() {
    return Point();
}

Point Point::generator() {
    return G;
}

void Point::doublePoint() {
    if (isInfinity()) return;
    
    Int slope = x;
    slope.modMul(slope, P);
    slope *= 3;
    slope += A;
    slope.modMul(Int(2) * y, P);
    
    Int x3 = slope;
    x3.modMul(x3, P);
    x3 -= Int(2) * x;
    x3.mod(P);
    
    Int y3 = x - x3;
    y3.modMul(slope, P);
    y3 -= y;
    y3.mod(P);
    
    x = x3;
    y = y3;
}

void Point::addPoint(const Point& other) {
    if (other.isInfinity()) return;
    if (isInfinity()) {
        *this = other;
        return;
    }
    
    Int slope = y - other.y;
    slope.modMul(x - other.x, P);
    
    Int x3 = slope;
    x3.modMul(x3, P);
    x3 -= x + other.x;
    x3.mod(P);
    
    Int y3 = x - x3;
    y3.modMul(slope, P);
    y3 -= y;
    y3.mod(P);
    
    x = x3;
    y = y3;
}

void Point::multiply(const Int& scalar) {
    Point result = infinity();
    Point addend = *this;
    
    Int k = scalar;
    while (!k.isZero()) {
        if (k.isOdd()) {
            result += addend;
        }
        addend.doublePoint();
        k >>= 1;
    }
    
    *this = result;
}

void Point::checkOnCurve() const {
    if (!isInfinity() && !isValid()) {
        throw std::runtime_error("Point is not on the curve");
    }
}
