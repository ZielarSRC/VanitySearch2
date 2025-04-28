#ifndef __INTGROUP_H__
#define __INTGROUP_H__

#include "Int.h"
#include <vector>

class IntGroup {

public:

    // Constructors
    IntGroup();
    IntGroup(int size);
    IntGroup(const IntGroup &other);

    // Destructor
    ~IntGroup();

    // Management
    void Create(int size);
    void Set(const IntGroup &other);
    void Clear();

    // Operations
    void ModularReduce(Int *mod);
    void ModularAdd(Int *mod);
    void Normalize();
    void ShiftL(uint32_t bits);
    void ShiftR(uint32_t bits);

    // Data access
    int GetSize() const;
    Int *GetInt(int index);
    const Int *GetInt(int index) const;

    // Static operations
    static void ModInv(IntGroup *in, Int *mod);
    static void ModAdd(IntGroup *in, Int *mod);
    static void ModMul(IntGroup *in, Int *mod);
    static void ModK1(IntGroup *in);
    static void ModK2(IntGroup *in);
    static void ModK3(IntGroup *in);

private:

    std::vector<Int> ints;
    int groupSize;

    void Init(int size);
    void Free();
    void CheckIndex(int index) const;

};

#endif // __INTGROUP_H__
