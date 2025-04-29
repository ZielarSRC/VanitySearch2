#include "Int.h"
#include <emmintrin.h>
#include <string.h>

#define MAX(x,y) (((x)>(y))?(x):(y))
#define MIN(x,y) (((x)<(y))?(x):(y))

// Global Montgomery constants
static Int _P;        // Field characteristic
static Int _R;        // Montgomery R
static Int _R2;       // Montgomery R^2
static Int _R3;       // Montgomery R^3
static Int _R4;       // Montgomery R^4
static int32_t Msize;  // Montgomery size
static uint32_t MM32;  // 32-bit inverse
static uint64_t MM64;  // 64-bit inverse
#define MSK62 0x3FFFFFFFFFFFFFFF

// Secp256k1 constants
static Int _R2o;      // R^2 for SecpK1 order
static uint64_t MM64o = 0x4B0DFF665588B13FULL; // Inverse of SecpK1 order
static Int* _O;       // SecpK1 order pointer

extern Int _ONE;      // Global constant 1

// Modular addition
void Int::ModAdd(Int* a) {
    Int p;
    Add(a);
    p.Sub(this, &_P);
    if (p.IsPositive())
        Set(&p);
}

void Int::ModAdd(Int* a, Int* b) {
    Int p;
    Add(a, b);
    p.Sub(this, &_P);
    if (p.IsPositive())
        Set(&p);
}

void Int::ModDouble() {
    Int p;
    Add(this);
    p.Sub(this, &_P);
    if (p.IsPositive())
        Set(&p);
}

void Int::ModAdd(uint64_t a) {
    Int p;
    Add(a);
    p.Sub(this, &_P);
    if (p.IsPositive())
        Set(&p);
}

// Modular subtraction
void Int::ModSub(Int* a) {
    Sub(a);
    if (IsNegative())
        Add(&_P);
}

void Int::ModSub(uint64_t a) {
    Sub(a);
    if (IsNegative())
        Add(&_P);
}

void Int::ModSub(Int* a, Int* b) {
    Sub(a, b);
    if (IsNegative())
        Add(&_P);
}

// Modular negation
void Int::ModNeg() {
    Neg();
    Add(&_P);
}

// Modular inversion table
static const int64_t INV256[] = {
    -0LL,-1LL,-0LL,-235LL,-0LL,-141LL,-0LL,-183LL,-0LL,-57LL,-0LL,-227LL,-0LL,-133LL,-0LL,-239LL,
    -0LL,-241LL,-0LL,-91LL,-0LL,-253LL,-0LL,-167LL,-0LL,-41LL,-0LL,-83LL,-0LL,-245LL,-0LL,-223LL,
    -0LL,-225LL,-0LL,-203LL,-0LL,-109LL,-0LL,-151LL,-0LL,-25LL,-0LL,-195LL,-0LL,-101LL,-0LL,-207LL,
    -0LL,-209LL,-0LL,-59LL,-0LL,-221LL,-0LL,-135LL,-0LL,-9LL,-0LL,-51LL,-0LL,-213LL,-0LL,-191LL,
    -0LL,-193LL,-0LL,-171LL,-0LL,-77LL,-0LL,-119LL,-0LL,-249LL,-0LL,-163LL,-0LL,-69LL,-0LL,-175LL,
    -0LL,-177LL,-0LL,-27LL,-0LL,-189LL,-0LL,-103LL,-0LL,-233LL,-0LL,-19LL,-0LL,-181LL,-0LL,-159LL,
    -0LL,-161LL,-0LL,-139LL,-0LL,-45LL,-0LL,-87LL,-0LL,-217LL,-0LL,-131LL,-0LL,-37LL,-0LL,-143LL,
    -0LL,-145LL,-0LL,-251LL,-0LL,-157LL,-0LL,-71LL,-0LL,-201LL,-0LL,-243LL,-0LL,-149LL,-0LL,-127LL,
    -0LL,-129LL,-0LL,-107LL,-0LL,-13LL,-0LL,-55LL,-0LL,-185LL,-0LL,-99LL,-0LL,-5LL,-0LL,-111LL,
    -0LL,-113LL,-0LL,-219LL,-0LL,-125LL,-0LL,-39LL,-0LL,-169LL,-0LL,-211LL,-0LL,-117LL,-0LL,-95LL,
    -0LL,-97LL,-0LL,-75LL,-0LL,-237LL,-0LL,-23LL,-0LL,-153LL,-0LL,-67LL,-0LL,-229LL,-0LL,-79LL,
    -0LL,-81LL,-0LL,-187LL,-0LL,-93LL,-0LL,-7LL,-0LL,-137LL,-0LL,-179LL,-0LL,-85LL,-0LL,-63LL,
    -0LL,-65LL,-0LL,-43LL,-0LL,-205LL,-0LL,-247LL,-0LL,-121LL,-0LL,-35LL,-0LL,-197LL,-0LL,-47LL,
    -0LL,-49LL,-0LL,-155LL,-0LL,-61LL,-0LL,-231LL,-0LL,-105LL,-0LL,-147LL,-0LL,-53LL,-0LL,-31LL,
    -0LL,-33LL,-0LL,-11LL,-0LL,-173LL,-0LL,-215LL,-0LL,-89LL,-0LL,-3LL,-0LL,-165LL,-0LL,-15LL,
    -0LL,-17LL,-0LL,-123LL,-0LL,-29LL,-0LL,-199LL,-0LL,-73LL,-0LL,-115LL,-0LL,-21LL,-0LL,-255LL
};

// Division step for modular inversion
void Int::DivStep62(Int* u, Int* v, int64_t* eta, int* pos, 
                   int64_t* uu, int64_t* uv, int64_t* vu, int64_t* vv) {
    // Implementation using delayed right shift 62 bits
    uint64_t uh, vh;
    uint64_t w, x;
    unsigned char c = 0;

    // Extract MSB words
    while (*pos >= 1 && (u->bits64[*pos] | v->bits64[*pos]) == 0) (*pos)--;
    
    if (*pos == 0) {
        uh = u->bits64[0];
        vh = v->bits64[0];
    } else {
        uint64_t s = LZC(u->bits64[*pos] | v->bits64[*pos]);
        if (s == 0) {
            uh = u->bits64[*pos];
            vh = v->bits64[*pos];
        } else {
            uh = __shiftleft128(u->bits64[*pos-1], u->bits64[*pos], (uint8_t)s);
            vh = __shiftleft128(v->bits64[*pos-1], v->bits64[*pos], (uint8_t)s);
        }
    }

    int bitCount = 62;
    __m128i _u, _v, _t;
    
    #ifdef WIN64
    _u.m128i_u64[0] = 1;
    _u.m128i_u64[1] = 0;
    _v.m128i_u64[0] = 0;
    _v.m128i_u64[1] = 1;
    #else
    ((int64_t*)&_u)[0] = 1;
    ((int64_t*)&_u)[1] = 0;
    ((int64_t*)&_v)[0] = 0;
    ((int64_t*)&_v)[1] = 1;
    #endif

    uint64_t u0 = u->bits64[0];
    uint64_t v0 = v->bits64[0];

    while (true) {
        uint64_t zeros = TZC(v0 | (1ULL << bitCount));
        vh >>= zeros;
        v0 >>= zeros;
        _u = _mm_slli_epi64(_u, (int)zeros);
        bitCount -= (int)zeros;

        if (bitCount <= 0) break;

        if (vh < uh) {
            SWAP(w, uh, vh);
            SWAP(x, u0, v0);
            SWAP(_t, _u, _v);
        }

        vh -= uh;
        v0 -= u0;
        _v = _mm_sub_epi64(_v, _u);
    }

    #ifdef WIN64
    *uu = _u.m128i_u64[0];
    *uv = _u.m128i_u64[1];
    *vu = _v.m128i_u64[0];
    *vv = _v.m128i_u64[1];
    #else
    *uu = ((int64_t*)&_u)[0];
    *uv = ((int64_t*)&_u)[1];
    *vu = ((int64_t*)&_v)[0];
    *vv = ((int64_t*)&_v)[1];
    #endif
}

// Modular inversion
void Int::ModInv() {
    Int u(&_P);
    Int v(this);
    Int r((int64_t)0);
    Int s((int64_t)1);

    // Delayed right shift 62bits implementation
    Int r0_P, s0_P;
    int64_t eta = -1;
    int64_t uu, uv, vu, vv;
    uint64_t carryS, carryR;
    int pos = NB64BLOCK - 1;
    
    while (pos >= 1 && (u.bits64[pos] | v.bits64[pos]) == 0) pos--;

    while (!v.IsZero()) {
        DivStep62(&u, &v, &eta, &pos, &uu, &uv, &vu, &vv);

        // Update BigInt variables
        MatrixVecMul(&u, &v, uu, uv, vu, vv);

        // Make u,v positive
        if (u.IsNegative()) {
            u.Neg();
            uu = -uu;
            uv = -uv;
        }
        if (v.IsNegative()) {
            v.Neg();
            vu = -vu;
            vv = -vv;
        }

        MatrixVecMul(&r, &s, uu, uv, vu, vv, &carryR, &carryS);

        // Compute multiple of P
        uint64_t r0 = (r.bits64[0] * MM64) & MSK62;
        uint64_t s0 = (s.bits64[0] * MM64) & MSK62;
        r0_P.Mult(&_P, r0);
        s0_P.Mult(&_P, s0);
        carryR = r.AddCh(&r0_P, carryR);
        carryS = s.AddCh(&s0_P, carryS);

        // Right shift by 62 bits
        shiftR(62, u.bits64);
        shiftR(62, v.bits64);
        shiftR(62, r.bits64, carryR);
        shiftR(62, s.bits64, carryS);
    }

    if (u.IsNegative()) {
        u.Neg();
        r.Neg();
    }

    if (!u.IsOne()) {
        CLEAR();
        return;
    }

    while (r.IsNegative()) r.Add(&_P);
    while (r.IsGreaterOrEqual(&_P)) r.Sub(&_P);

    Set(&r);
}

// Modular exponentiation
void Int::ModExp(Int* e) {
    Int base(this);
    SetInt32(1);
    uint32_t nbBit = e->GetBitLength();
    
    for (int i = 0; i < (int)nbBit; i++) {
        if (e->GetBit(i))
            ModMul(&base);
        base.ModMul(&base);
    }
}

// Montgomery multiplication
void Int::MontgomeryMult(Int* a) {
    Int t, pr, p;
    uint64_t ML, c;

    imm_umul(a->bits64, bits64[0], pr.bits64);
    ML = pr.bits64[0] * MM64;
    imm_umul(_P.bits64, ML, p.bits64);
    c = pr.AddC(&p);
    memcpy(t.bits64, pr.bits64 + 1, 8 * (NB64BLOCK - 1));
    t.bits64[NB64BLOCK - 1] = c;

    for (int i = 1; i < Msize; i++) {
        imm_umul(a->bits64, bits64[i], pr.bits64);
        ML = (pr.bits64[0] + t.bits64[0]) * MM64;
        imm_umul(_P.bits64, ML, p.bits64);
        c = pr.AddC(&p);
        t.AddAndShift(&t, &pr, c);
    }

    p.Sub(&t, &_P);
    if (p.IsPositive())
        Set(&p);
    else
        Set(&t);
}

// Modular square root
void Int::ModSqrt() {
    if (_P.IsEven()) {
        CLEAR();
        return;
    }

    if (!HasSqrt()) {
        CLEAR();
        return;
    }

    if ((_P.bits64[0] & 3) == 3) {
        Int e(&_P);
        e.AddOne();
        e.ShiftR(2);
        ModExp(&e);
    } else if ((_P.bits64[0] & 3) == 1) {
        // Tonelli-Shanks implementation
        int nbBit = _P.GetBitLength();
        uint64_t e = 0;
        Int S(&_P);
        S.SubOne();
        
        while (S.IsEven()) {
            S.ShiftR(1);
            e++;
        }

        // Find smallest non-residue
        Int q((uint64_t)1);
        do {
            q.AddOne();
        } while (q.HasSqrt());

        Int c(&q);
        c.ModExp(&S);

        Int t(this);
        t.ModExp(&S);

        Int r(this);
        Int ex(&S);
        ex.AddOne();
        ex.ShiftR(1);
        r.ModExp(&ex);

        uint64_t M = e;
        while (!t.IsOne()) {
            Int t2(&t);
            uint64_t i = 0;
            while (!t2.IsOne()) {
                t2.ModSquare(&t2);
                i++;
            }

            Int b(&c);
            for (uint64_t j = 0; j < M - i - 1; j++)
                b.ModSquare(&b);
            M = i;
            c.ModSquare(&b);
            t.ModMul(&t, &c);
            r.ModMul(&r, &b);
        }

        Set(&r);
    }
}

// Secp256k1 specific implementations
void Int::ModMulK1(Int* a, Int* b) {
    uint64_t ah, al;
    uint64_t t[NB64BLOCK];
    uint64_t r512[8] = {0};

    // 256x256 multiplication
    imm_umul(a->bits64, b->bits64[0], r512);
    imm_umul(a->bits64, b->bits64[1], t);
    unsigned char c = _addcarry_u64(0, r512[1], t[0], r512 + 1);
    c = _addcarry_u64(c, r512[2], t[1], r512 + 2);
    c = _addcarry_u64(c, r512[3], t[2], r512 + 3);
    c = _addcarry_u64(c, r512[4], t[3], r512 + 4);
    c = _addcarry_u64(c, r512[5], t[4], r512 + 5);

    imm_umul(a->bits64, b->bits64[2], t);
    c = _addcarry_u64(0, r512[2], t[0], r512 + 2);
    c = _addcarry_u64(c, r512[3], t[1], r512 + 3);
    c = _addcarry_u64(c, r512[4], t[2], r512 + 4);
    c = _addcarry_u64(c, r512[5], t[3], r512 + 5);
    c = _addcarry_u64(c, r512[6], t[4], r512 + 6);

    imm_umul(a->bits64, b->bits64[3], t);
    c = _addcarry_u64(0, r512[3], t[0], r512 + 3);
    c = _addcarry_u64(c, r512[4], t[1], r512 + 4);
    c = _addcarry_u64(c, r512[5], t[2], r512 + 5);
    c = _addcarry_u64(c, r512[6], t[3], r512 + 6);
    c = _addcarry_u64(c, r512[7], t[4], r512 + 7);

    // Reduction
    imm_umul(r512 + 4, 0x1000003D1ULL, t);
    c = _addcarry_u64(0, r512[0], t[0], r512 + 0);
    c = _addcarry_u64(c, r512[1], t[1], r512 + 1);
    c = _addcarry_u64(c, r512[2], t[2], r512 + 2);
    c = _addcarry_u64(c, r512[3], t[3], r512 + 3);

    al = _umul128(t[4] + c, 0x1000003D1ULL, &ah);
    c = _addcarry_u64(0, r512[0], al, bits64 + 0);
    c = _addcarry_u64(c, r512[1], ah, bits64 + 1);
    c = _addcarry_u64(c, r512[2], 0ULL, bits64 + 2);
    c = _addcarry_u64(c, r512[3], 0ULL, bits64 + 3);
    bits64[4] = 0;
}

// Initialize Secp256k1 parameters
void Int::InitK1(Int* order) {
    _O = order;
    _R2o.SetBase16("9D671CD581C69BC5E697F5E45BCD07C6741496C20E7CF878896CF21467D7D140");
}

// Secp256k1 order operations
void Int::ModAddK1order(Int* a, Int* b) {
    Add(a, b);
    Sub(_O);
    if (IsNegative())
        Add(_O);
}

void Int::ModSubK1order(Int* a) {
    Sub(a);
    if (IsNegative())
        Add(_O);
}

void Int::ModNegK1order() {
    Neg();
    Add(_O);
}

// Setup Montgomery field
void Int::SetupField(Int* n, Int* R, Int* R2, Int* R3, Int* R4) {
    int nSize = n->GetSize();
    
    // Compute inverse
    {
        int64_t x, t;
        x = t = (int64_t)n->bits64[0];
        x = x * (2 - t * x);
        x = x * (2 - t * x);
        x = x * (2 - t * x);
        x = x * (2 - t * x);
        x = x * (2 - t * x);
        MM64 = (uint64_t)(-x);
        MM32 = (uint32_t)MM64;
    }
    
    _P.Set(n);
    Msize = nSize / 2;

    // Compute R powers
    Int Ri;
    Ri.MontgomeryMult(&_ONE, &_ONE);
    _R.Set(&Ri);
    _R2.MontgomeryMult(&Ri, &_ONE);
    _R3.MontgomeryMult(&Ri, &Ri);
    _R4.MontgomeryMult(&_R3, &_ONE);

    _R.ModInv();
    _R2.ModInv();
    _R3.ModInv();
    _R4.ModInv();

    if (R) R->Set(&_R);
    if (R2) R2->Set(&_R2);
    if (R3) R3->Set(&_R3);
    if (R4) R4->Set(&_R4);
}

// Accessors for Montgomery constants
Int* Int::GetFieldCharacteristic() { return &_P; }
Int* Int::GetR() { return &_R; }
Int* Int::GetR2() { return &_R2; }
Int* Int::GetR3() { return &_R3; }
Int* Int::GetR4() { return &_R4; }
