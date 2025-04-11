#pragma once
#include "Int.h"
#include <array>
#include <functional>
#include <stdexcept>

class Point {
public:
    static constexpr size_t COMPRESSED_SIZE = 33;
    static constexpr size_t UNCOMPRESSED_SIZE = 65;
    static constexpr uint8_t COMPRESSED_EVEN = 0x02;
    static constexpr uint8_t COMPRESSED_ODD = 0x03;
    static constexpr uint8_t UNCOMPRESSED = 0x04;
    
    Point();
    explicit Point(const Int& x, const Int& y, bool compressed = false);
    explicit Point(const std::array<unsigned char, COMPRESSED_SIZE>& data);
    explicit Point(const std::array<unsigned char, UNCOMPRESSED_SIZE>& data);
    
    bool parse(const unsigned char* data, size_t size);
    void serialize(std::array<unsigned char, COMPRESSED_SIZE>& out) const;
    void serialize(std::array<unsigned char, UNCOMPRESSED_SIZE>& out) const;
    std::vector<unsigned char> serialize(bool forceUncompressed = false) const;
    
    bool isInfinity() const;
    bool isValid() const;
    bool operator==(const Point& other) const;
    bool operator!=(const Point& other) const;
    Point operator+(const Point& other) const;
    Point& operator+=(const Point& other);
    Point operator*(const Int& scalar) const;
    Point& operator*=(const Int& scalar);
    
    Int getX() const { return x; }
    Int getY() const { return y; }
    bool isCompressed() const { return compressed; }
    void setCompressed(bool val) { compressed = val; }

    static Point infinity();
    static Point generator();

private:
    Int x;
    Int y;
    bool compressed = false;
    
    void doublePoint();
    void addPoint(const Point& other);
    void multiply(const Int& scalar);
    void checkOnCurve() const;
    
    static const Int P;
    static const Int A;
    static const Int B;
    static const Point G;
};
