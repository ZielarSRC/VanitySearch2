#include "Point.h"
#include "Int.h"
#include <sstream>

Point::Point(const Int& x, const Int& y, const Int& z) 
    : x(x), y(y), z(z) {}

void Point::Clear() {
    x.SetInt32(0);
    y.SetInt32(0);
    z.SetInt32(0);
}

bool Point::isZero() const {
    return x.IsZero() && y.IsZero() && z.IsZero();
}

void Point::Reduce() {
    if (z.IsZero()) return;
    
    Int zInv = z;
    zInv.ModInv();
    
    Int zInv2;
    zInv2.ModSquare(zInv);
    
    x.ModMul(x, zInv2);
    y.ModMul(y, zInv2);
    z.SetInt32(1);
}

bool Point::equals(const Point& p) const {
    Point p1 = *this;
    Point p2 = p;
    p1.Reduce();
    p2.Reduce();
    return p1.x.IsEqual(p2.x) && p1.y.IsEqual(p2.y);
}

std::string Point::toString() const {
    std::ostringstream oss;
    oss << "(" << x.GetBase16() << ", " << y.GetBase16() << ", " << z.GetBase16() << ")";
    return oss.str();
}

Point Point::Neg() const {
    Point r;
    r.x = x;
    r.y = y;
    r.y.ModNeg();
    r.z = z;
    return r;
}

void Point::Set(const Point& p) {
    x.Set(&p.x);
    y.Set(&p.y);
    z.Set(&p.z);
}

bool Point::IsValid() const {
    if (z.IsZero()) return false;
    
    Point reduced = *this;
    reduced.Reduce();
    
    Int y2;
    y2.ModSquare(reduced.y);
    
    Int x3;
    x3.ModSquare(reduced.x);
    x3.ModMul(x3, reduced.x);
    x3.ModAdd(7);
    
    return y2.IsEqual(x3);
}
