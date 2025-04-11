/*
 * Modernized IntGroup implementation for cryptographic operations
 * Optimized for contemporary CPUs with cache-aware data structures
 * Copyright (c) 2023 Modern Crypto Solutions
 */

#ifndef INT_GROUP_H
#define INT_GROUP_H

#include "Int.h"
#include <vector>
#include <memory>

class IntGroup {
public:
    // Constructor with size initialization
    explicit IntGroup(size_t size);
    
    // Destructor
    ~IntGroup() = default;
    
    // Set the group values
    void Set(const std::vector<Int>& points);
    void Set(Int* points);
    
    // Compute modular inversions for the entire group
    void ModInv();
    
    // Get group size
    size_t GetSize() const { return size_; }
    
    // Access operators
    Int& operator[](size_t index);
    const Int& operator[](size_t index) const;

private:
    std::vector<Int> ints_;      // Main group elements
    std::vector<Int> subp_;      // Partial products for computation
    size_t size_;                // Group size
};

#endif // INT_GROUP_H
