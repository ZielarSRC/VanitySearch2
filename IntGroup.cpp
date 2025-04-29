#include "IntGroup.h"
#include <stdexcept>

// Constructors
IntGroup::IntGroup() {
    Init(0);
}

IntGroup::IntGroup(int size) {
    Init(size);
}

IntGroup::IntGroup(const IntGroup &other) {
    Init(other.groupSize);
    Set(other);
}

// Destructor
IntGroup::~IntGroup() {
    Free();
}

// Initialization
void IntGroup::Init(int size) {
    groupSize = size;
    if (size > 0) {
        ints.resize(size);
    }
}

void IntGroup::Free() {
    ints.clear();
    groupSize = 0;
}

// Management
void IntGroup::Create(int size) {
    Free();
    Init(size);
}

void IntGroup::Set(const IntGroup &other) {
    if (groupSize != other.groupSize) {
        throw std::runtime_error("IntGroup size mismatch in Set");
    }
    for (int i = 0; i < groupSize; i++) {
        ints[i].Set(other.ints[i]);
    }
}

void IntGroup::Clear() {
    for (auto &i : ints) {
        i.SetZero();
    }
}

// Operations
void IntGroup::ModularReduce(Int *mod) {
    for (auto &i : ints) {
        i.Mod(*mod);
    }
}

void IntGroup::ModularAdd(Int *mod) {
    for (auto &i : ints) {
        i.Add(*mod);
    }
}

void IntGroup::Normalize() {
    for (auto &i : ints) {
        i.Normalize();
    }
}

void IntGroup::ShiftL(uint32_t bits) {
    for (auto &i : ints) {
        i.ShiftL(bits);
    }
}

void IntGroup::ShiftR(uint32_t bits) {
    for (auto &i : ints) {
        i.ShiftR(bits);
    }
}

// Data access
int IntGroup::GetSize() const {
    return groupSize;
}

Int *IntGroup::GetInt(int index) {
    CheckIndex(index);
    return &ints[index];
}

const Int *IntGroup::GetInt(int index) const {
    CheckIndex(index);
    return &ints[index];
}

// Private methods
void IntGroup::CheckIndex(int index) const {
    if (index < 0 || index >= groupSize) {
        throw std::out_of_range("IntGroup index out of range");
    }
}

// Static operations
void IntGroup::ModInv(IntGroup *in, Int *mod) {
    for (int i = 0; i < in->groupSize; i++) {
        in->ints[i].ModInv(*mod);
    }
}

void IntGroup::ModAdd(IntGroup *in, Int *mod) {
    for (int i = 0; i < in->groupSize; i++) {
        in->ints[i].Mod(*mod);
    }
}

void IntGroup::ModMul(IntGroup *in, Int *mod) {
    for (int i = 0; i < in->groupSize; i++) {
        in->ints[i].ModMul(in->ints[i], *mod);
    }
}

void IntGroup::ModK1(IntGroup *in) {
    // secp256k1 curve prime: 2^256 - 2^32 - 977
    const Int p("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F", 16);
    
    for (int i = 0; i < in->groupSize; i++) {
        Int &num = in->ints[i];
        if (num >= p) {
            num.Mod(p);
        }
        // Dodatkowa optymalizacja dla specyfiki K1
        if (num.bits.size() >= 8) {  // Jeśli liczba > 256 bitów
            num.Mod(p);
        }
    }
}

void IntGroup::ModK2(IntGroup *in) {
    // secp256k1 curve order: 2^256 - 432420386565659656852420866394968145599
    const Int n("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141", 16);
    
    for (int i = 0; i < in->groupSize; i++) {
        Int &num = in->ints[i];
        if (num >= n) {
            // Specjalizowana redukcja dla kluczy
            Int t(num);
            t.ShiftR(256);
            t.Mult(Int("14551231950B75FC4402DA1732FC9BEBF", 16));
            num.Add(t);
            
            if (num >= n) {
                num.Sub(n);
            }
        }
    }
}

void IntGroup::ModK3(IntGroup *in) {
    // Custom modular reduction for special cases (np. dla 2^256 - 189)
    const Int k3("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF43", 16);
    
    for (int i = 0; i < in->groupSize; i++) {
        Int &num = in->ints[i];
        if (num >= k3) {
            // Optymalizacja: 2^256 ≡ 189 mod k3
            while (num.bits.size() > 8) {  // Dopóki liczba > 256 bitów
                Int high;
                high.bits.assign(num.bits.begin() + 8, num.bits.end());
                high.Mult(Int(189));
                num.bits.resize(8);
                num.Add(high);
            }
            
            if (num >= k3) {
                num.Sub(k3);
            }
        }
    }
}
