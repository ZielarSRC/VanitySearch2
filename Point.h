#ifndef POINT_H
#define POINT_H

#include "Int.h"
#include <string>

class Point {
public:
    // Coordinates
    Int x;
    Int y;
    Int z;
    
    // Constructors
    Point() = default;
    Point(const Int& x, const Int& y, const Int& z);
    
    // Core functionality
    void Clear();
    bool isZero() const;
    void Reduce();
    bool equals(const Point& p) const;
    
    // String conversion
    std::string toString() const;
    
    // Point operations
    Point Neg() const;
    
    // Serialization
    void Set(const Point& p);
    
    // Validation
    bool IsValid() const;
};

#endif // POINT_H
