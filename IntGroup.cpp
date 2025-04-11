/*
 * Modernized IntGroup implementation
 * Features:
 * - Memory-safe operations using std::vector
 * - Cache-friendly data layout
 * - Const-correctness
 * - Bounds checking in debug builds
 */

#include "IntGroup.h"
#include <stdexcept>
#include <algorithm>

IntGroup::IntGroup(size_t size) 
    : size_(size), 
      ints_(size), 
      subp_(size) {
    if(size == 0) {
        throw std::invalid_argument("Group size cannot be zero");
    }
}

void IntGroup::Set(const std::vector<Int>& points) {
    if(points.size() != size_) {
        throw std::invalid_argument("Input size mismatch");
    }
    std::copy(points.begin(), points.end(), ints_.begin());
}

void IntGroup::Set(Int* points) {
    if(points == nullptr) {
        throw std::invalid_argument("Null pointer provided");
    }
    std::copy(points, points + size_, ints_.begin());
}

void IntGroup::ModInv() {
    if(size_ == 0) return;

    // Compute partial products
    subp_[0] = ints_[0];
    for(size_t i = 1; i < size_; ++i) {
        subp_[i].ModMulK1(&subp_[i-1], &ints_[i]);
    }

    // Compute the inverse of the last partial product
    Int inverse = subp_.back();
    inverse.ModInv();

    // Compute individual inverses using the partial products
    for(size_t i = size_ - 1; i > 0; --i) {
        Int newValue;
        newValue.ModMulK1(&subp_[i-1], &inverse);
        inverse.ModMulK1(&ints_[i]);
        ints_[i] = std::move(newValue);
    }
    ints_[0] = std::move(inverse);
}

Int& IntGroup::operator[](size_t index) {
    if(index >= size_) {
        throw std::out_of_range("Index out of range");
    }
    return ints_[index];
}

const Int& IntGroup::operator[](size_t index) const {
    if(index >= size_) {
        throw std::out_of_range("Index out of range");
    }
    return ints_[index];
}
