#ifndef SECP256K1_H
#define SECP256K1_H

#include <cstdint>
#include <vector>
#include <array>

class SECP256k1 {
public:
    struct uint256_t {
        std::array<uint32_t, 8> data;

        uint256_t() : data{0} {}
        uint256_t(uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3,
                  uint32_t d4, uint32_t d5, uint32_t d6, uint32_t d7) : 
            data{d0, d1, d2, d3, d4, d5, d6, d7} {}
            
        bool operator<(const uint256_t& other) const {
            for(int i = 7; i >= 0; --i) {
                if(data[i] != other.data[i]) 
                    return data[i] < other.data[i];
            }
            return false;
        }
    };

    static constexpr size_t PRECOMP_TABLE_SIZE = 256;
    static constexpr size_t KEY_BATCH_SIZE = 1024;

    // Inicjalizacja krzywej
    static void Init();
    
    // Operacje podstawowe
    static void Multiply(const uint256_t& k, uint256_t& outX, uint256_t& outY);
    static void Add(const uint256_t& p1x, const uint256_t& p1y,
                   const uint256_t& p2x, const uint256_t& p2y,
                   uint256_t& outX, uint256_t& outY);
                   
    // Operacje batchowe
    static void BatchMultiply(const std::vector<uint256_t>& keys,
                             std::vector<uint256_t>& outPoints);

    // Wsparcie GPU
    #ifdef USE_CUDA
    static void InitCuda();
    static void CudaMultiply(const uint256_t* keys, uint256_t* results, size_t count);
    #endif

private:
    // Stałe krzywej
    static const uint256_t P;
    static const uint256_t Gx;
    static const uint256_t Gy;
    static const uint256_t Beta;
    
    // Prekomputacje
    static std::array<uint256_t, PRECOMP_TABLE_SIZE> precompTable;
    static std::array<uint256_t, PRECOMP_TABLE_SIZE> precompTableY;
    
    // Funkcje pomocnicze
    static void ToJacobian(const uint256_t& x, const uint256_t& y, uint256_t& xj, uint256_t& yj, uint256_t& zj);
    static void FromJacobian(const uint256_t& xj, const uint256_t& yj, const uint256_t& zj, uint256_t& x, uint256_t& y);
    static void PointDoubleJacobian(uint256_t& x, uint256_t& y, uint256_t& z);
    static void PointAddJacobian(const uint256_t& x1, const uint256_t& y1, const uint256_t& z1,
                                const uint256_t& x2, const uint256_t& y2, const uint256_t& z2,
                                uint256_t& outX, uint256_t& outY, uint256_t& outZ);
    static void ModReduce(uint256_t& val);
    static void ModAdd(uint256_t& a, const uint256_t& b);
    static void ModSub(uint256_t& a, const uint256_t& b);
    static void ModMul(const uint256_t& a, const uint256_t& b, uint256_t& result);
    static void ModInverse(const uint256_t& a, uint256_t& result);
};

#endif
