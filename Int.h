#pragma once
#include "Random.h"
#include <string>
#include <inttypes.h>
#include <array>
#include <immintrin.h>

#define BISIZE 256
#define NB64BLOCK 5
#define NB32BLOCK 10

class Int {
public:
    // Constructors
    Int();
    explicit Int(int64_t i64);
    explicit Int(const Int& a);
    
    // Memory operations
    void CLEAR();
    void CLEARFF();
    
    // Assignment
    void Set(const Int* a);
    
    // Arithmetic operations
    void Add(const Int* a);
    void Add(uint64_t a);
    void AddOne();
    void Add(const Int* a, const Int* b);
    void Sub(const Int* a);
    void Sub(const Int* a, const Int* b);
    void Sub(uint64_t a);
    void SubOne();
    void Mult(const Int* a);
    void Mult(uint64_t a);
    void IMult(int64_t a);
    void Mult(const Int* a, uint64_t b);
    void IMult(const Int* a, int64_t b);
    void Mult(const Int* a, const Int* b);
    void Div(const Int* a, Int* mod = nullptr);
    void MultModN(const Int* a, const Int* b, const Int* n);
    void Neg();
    void Abs();
    
    // Bit operations
    void ShiftR(uint32_t n);
    void ShiftR32Bit();
    void ShiftR64Bit();
    void ShiftL(uint32_t n);
    void ShiftL32Bit();
    void ShiftL64Bit();
    
    // Comparison
    bool IsGreater(const Int* a) const;
    bool IsLower(const Int* a) const;
    bool IsGreaterOrEqual(const Int* a) const;
    bool IsLowerOrEqual(const Int* a) const;
    bool IsEqual(const Int* a) const;
    bool IsZero() const;
    bool IsOne() const;
    bool IsStrictPositive() const;
    bool IsPositive() const;
    bool IsNegative() const;
    bool IsEven() const;
    bool IsOdd() const;
    
    // Getters/Setters
    void SetInt32(uint32_t value);
    uint32_t GetInt32() const;
    unsigned char GetByte(int n) const;
    void Set32Bytes(const unsigned char* bytes);
    void Get32Bytes(unsigned char* buff) const;
    void SetByte(int n, unsigned char byte);
    void SetDWord(int n, uint32_t b);
    void SetQWord(int n, uint64_t b);
    void Rand(int nbit);
    void MaskByte(int n);
    
    // Conversions
    void SetBase10(const char* value);
    void SetBase16(const char* value);
    void SetBaseN(int n, const char* charset, const char* value);
    std::string GetBase2() const;
    std::string GetBase10() const;
    std::string GetBase16() const;
    std::string GetBaseN(int n, const char* charset) const;
    std::string GetBlockStr() const;
    std::string GetC64Str(int nbDigit) const;
    
    // Bit operations
    int GetBit(uint32_t n) const;
    int GetBitLength() const;
    int GetSize() const;
    int GetLowestBit() const;
    
    // Modular arithmetic
    static void SetupField(const Int* n, Int* R = nullptr, Int* R2 = nullptr, Int* R3 = nullptr, Int* R4 = nullptr);
    static const Int* GetR();
    static const Int* GetR2();
    static const Int* GetR3();
    static const Int* GetR4();
    static const Int* GetFieldCharacteristic();
    
    void GCD(const Int* a);
    void Mod(const Int* n);
    void ModInv();
    void MontgomeryMult(const Int* a, const Int* b);
    void MontgomeryMult(const Int* a);
    void ModAdd(const Int* a);
    void ModAdd(const Int* a, const Int* b);
    void ModAdd(uint64_t a);
    void ModSub(const Int* a);
    void ModSub(const Int* a, const Int* b);
    void ModSub(uint64_t a);
    void ModMul(const Int* a, const Int* b);
    void ModMul(const Int* a);
    void ModSquare(const Int* a);
    void ModCube(const Int* a);
    void ModDouble();
    void ModExp(const Int* e);
    void ModNeg();
    void ModSqrt();
    bool HasSqrt() const;
    
    // SecpK1 specific
    static void InitK1(Int* order);
    void ModMulK1(const Int* a, const Int* b);
    void ModMulK1(const Int* a);
    void ModMulK1order(const Int* a);
    void ModSquareK1(const Int* a);
    void ModAddK1order(const Int* a, const Int* b);
    
    // Testing
    static void Check();
    
    // Data
    union {
        uint32_t bits[NB32BLOCK];
        uint64_t bits64[NB64BLOCK];
        __m256i  m256[NB64BLOCK / 4];
    };

private:
    void ShiftL32BitAndSub(const Int* a, int n);
    uint64_t AddC(const Int* a);
    void AddAndShift(const Int* a, const Int* b, uint64_t cH);
    void Mult(const Int* a, uint32_t b);
    
    // Montgomery helpers
    static Int _R;
    static Int _R2;
    static Int _R3;
    static Int _R4;
    static Int _P;
    static uint64_t _mp;
    
    void MontgomeryReduce();
    void MontgomeryReduceFull();
};

// Intrinsic helpers
namespace intrin {
    static inline uint64_t umul128(uint64_t a, uint64_t b, uint64_t* hi) {
        return _umul128(a, b, hi);
    }
    
    static inline uint64_t shiftright128(uint64_t a, uint64_t b, unsigned char n) {
        return __shiftright128(a, b, n);
    }
    
    static inline uint64_t shiftleft128(uint64_t a, uint64_t b, unsigned char n) {
        return __shiftleft128(a, b, n);
    }
    
    static inline unsigned char addcarry_u64(unsigned char c, uint64_t a, uint64_t b, uint64_t* out) {
        return _addcarry_u64(c, a, b, out);
    }
    
    static inline unsigned char subborrow_u64(unsigned char c, uint64_t a, uint64_t b, uint64_t* out) {
        return _subborrow_u64(c, a, b, out);
    }
    
    static inline uint64_t byteswap_uint64(uint64_t x) {
        return _byteswap_uint64(x);
    }
}
