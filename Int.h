#ifndef INT_H
#define INT_H

#include <string>
#include <cstdint>

class Int {
public:
    // Constants
    static constexpr int MAX_BITS = 256;
    static constexpr int WORD_SIZE = 64;
    static constexpr int WORD_COUNT = MAX_BITS / WORD_SIZE;

    // Core data
    uint64_t bits64[WORD_COUNT] = {0};

    // Constructors
    Int() = default;
    explicit Int(uint32_t value);
    
    // Core functionality
    void Set(const Int& a);
    void Set(uint32_t a);
    void SetBase10(const std::string& s);
    void SetBase16(const std::string& s);
    void SetBaseN(int n, const char* value);
    void SetByte(int index, uint8_t value);
    
    // Arithmetic operations
    void Add(const Int& a);
    void Add(uint32_t a);
    void Sub(const Int& a);
    void Sub(uint32_t a);
    void Mult(const Int& a);
    void Mult(uint32_t a);
    void Div(const Int& a);
    void Mod(const Int& a);
    
    // Modular arithmetic
    void ModMul(const Int& a, const Int& mod);
    void ModSquare(const Int& mod);
    void ModInv();
    void ModExp(Int& e, const Int& mod);
    
    // Bit operations
    void ShiftL(uint32_t bits);
    void ShiftR(uint32_t bits);
    void And(const Int& a);
    void Or(const Int& a);
    void Xor(const Int& a);
    void Not();
    
    // Comparison
    bool IsGreater(const Int& a) const;
    bool IsLower(const Int& a) const;
    bool IsEqual(const Int& a) const;
    bool IsZero() const;
    bool IsOne() const;
    bool IsOdd() const;
    
    // Utility
    std::string GetBase16() const;
    std::string GetBase10() const;
    int GetBitLength() const;
    int GetSize() const;
    uint8_t GetByte(int index) const;
    
    // Special functions
    void Rand(int nbits);
    void Rand(const Int& randMax);
    void Check() const;
    
    // Static methods
    static void SetupField(const Int* p);
    static void InitK1(const Int* order);
};

#endif // INT_H
