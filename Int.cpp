/*
 * This file is part of the VanitySearch distribution (https://github.com/JeanLucPons/VanitySearch).
 * Copyright (c) 2019 Jean Luc PONS.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include "Int.h"
#include "IntGroup.h"
#include <string.h>
#include <immintrin.h>
#include "Timer.h"

#define MAX(x,y) (((x)>(y))?(x):(y))
#define MIN(x,y) (((x)<(y))?(x):(y))

Int _ONE(1);

Int::Int() {
    CLEAR();
}

Int::Int(Int *a) {
    if(a) Set(a);
    else CLEAR();
}

Int::Int(int64_t i64) {
    if (i64 < 0) {
        CLEARFF();
    } else {
        CLEAR();
    }
    bits64[0] = i64;
}

void Int::CLEAR() {
    memset(bits64, 0, NB64BLOCK * 8);
}

void Int::CLEARFF() {
    memset(bits64, 0xFF, NB64BLOCK * 8);
}

void Int::Set(Int *a) {
    memcpy(bits64, a->bits64, NB64BLOCK * 8);
}

void Int::Add(Int *a) {
    unsigned char c = 0;
    for (int i = 0; i < NB64BLOCK; i++)
        c = _addcarry_u64(c, bits64[i], a->bits64[i], bits64 + i);
}

void Int::Add(uint64_t a) {
    unsigned char c = _addcarry_u64(0, bits64[0], a, bits64);
    for (int i = 1; i < NB64BLOCK; i++)
        c = _addcarry_u64(c, bits64[i], 0, bits64 + i);
}

void Int::AddOne() {
    unsigned char c = _addcarry_u64(0, bits64[0], 1, bits64);
    for (int i = 1; i < NB64BLOCK && c; i++)
        c = _addcarry_u64(c, bits64[i], 0, bits64 + i);
}

void Int::Add(Int *a, Int *b) {
    unsigned char c = 0;
    for (int i = 0; i < NB64BLOCK; i++)
        c = _addcarry_u64(c, a->bits64[i], b->bits64[i], bits64 + i);
}

bool Int::IsGreater(Int *a) {
    for (int i = NB64BLOCK - 1; i >= 0; i--) {
        if (a->bits64[i] != bits64[i])
            return bits64[i] > a->bits64[i];
    }
    return false;
}

bool Int::IsLower(Int *a) {
    for (int i = NB64BLOCK - 1; i >= 0; i--) {
        if (a->bits64[i] != bits64[i])
            return bits64[i] < a->bits64[i];
    }
    return false;
}

bool Int::IsGreaterOrEqual(Int *a) {
    return !IsLower(a);
}

bool Int::IsLowerOrEqual(Int *a) {
    return !IsGreater(a);
}

bool Int::IsEqual(Int *a) {
    for (int i = 0; i < NB64BLOCK; i++)
        if (bits64[i] != a->bits64[i])
            return false;
    return true;
}

bool Int::IsOne() {
    return IsEqual(&_ONE);
}

bool Int::IsZero() {
    for (int i = 0; i < NB64BLOCK; i++)
        if (bits64[i] != 0)
            return false;
    return true;
}

void Int::SetInt32(uint32_t value) {
    CLEAR();
    bits[0] = value;
}

uint32_t Int::GetInt32() {
    return bits[0];
}

unsigned char Int::GetByte(int n) {
    return ((unsigned char *)bits)[n];
}

void Int::Set32Bytes(unsigned char *bytes) {
    CLEAR();
    uint64_t *ptr = (uint64_t *)bytes;
    bits64[3] = _byteswap_uint64(ptr[0]);
    bits64[2] = _byteswap_uint64(ptr[1]);
    bits64[1] = _byteswap_uint64(ptr[2]);
    bits64[0] = _byteswap_uint64(ptr[3]);
}

void Int::Get32Bytes(unsigned char *buff) {
    uint64_t *ptr = (uint64_t *)buff;
    ptr[3] = _byteswap_uint64(bits64[0]);
    ptr[2] = _byteswap_uint64(bits64[1]);
    ptr[1] = _byteswap_uint64(bits64[2]);
    ptr[0] = _byteswap_uint64(bits64[3]);
}

void Int::SetByte(int n, unsigned char byte) {
    ((unsigned char *)bits)[n] = byte;
}

void Int::SetDWord(int n, uint32_t b) {
    bits[n] = b;
}

void Int::SetQWord(int n, uint64_t b) {
    bits64[n] = b;
}

void Int::Sub(Int *a) {
    unsigned char c = 0;
    for (int i = 0; i < NB64BLOCK; i++)
        c = _subborrow_u64(c, bits64[i], a->bits64[i], bits64 + i);
}

void Int::Sub(Int *a, Int *b) {
    unsigned char c = 0;
    for (int i = 0; i < NB64BLOCK; i++)
        c = _subborrow_u64(c, a->bits64[i], b->bits64[i], bits64 + i);
}

void Int::Sub(uint64_t a) {
    unsigned char c = _subborrow_u64(0, bits64[0], a, bits64);
    for (int i = 1; i < NB64BLOCK; i++)
        c = _subborrow_u64(c, bits64[i], 0, bits64 + i);
}

void Int::SubOne() {
    unsigned char c = _subborrow_u64(0, bits64[0], 1, bits64);
    for (int i = 1; i < NB64BLOCK && c; i++)
        c = _subborrow_u64(c, bits64[i], 0, bits64 + i);
}

bool Int::IsPositive() {
    return (int64_t)(bits64[NB64BLOCK - 1]) >= 0;
}

bool Int::IsNegative() {
    return (int64_t)(bits64[NB64BLOCK - 1]) < 0;
}

bool Int::IsStrictPositive() {
    return IsPositive() && !IsZero();
}

bool Int::IsEven() {
    return (bits[0] & 0x1) == 0;
}

bool Int::IsOdd() {
    return (bits[0] & 0x1) == 1;
}

void Int::Neg() {
    unsigned char c = 1;
    for (int i = 0; i < NB64BLOCK; i++) {
        uint64_t tmp = ~bits64[i];
        c = _addcarry_u64(c, tmp, 0, bits64 + i);
    }
}

void Int::ShiftL32Bit() {
    for (int i = NB32BLOCK - 1; i > 0; i--)
        bits[i] = bits[i - 1];
    bits[0] = 0;
}

void Int::ShiftL64Bit() {
    for (int i = NB64BLOCK - 1; i > 0; i--)
        bits64[i] = bits64[i - 1];
    bits64[0] = 0;
}

void Int::ShiftL(uint32_t n) {
    if (n < 64) {
        shiftL((unsigned char)n, bits64);
    } else {
        uint32_t nb64 = n / 64;
        uint32_t nb = n % 64;
        for (uint32_t i = 0; i < nb64; i++)
            ShiftL64Bit();
        shiftL((unsigned char)nb, bits64);
    }
}

void Int::ShiftR32Bit() {
    for (int i = 0; i < NB32BLOCK - 1; i++)
        bits[i] = bits[i + 1];
    bits[NB32BLOCK - 1] = ((int32_t)bits[NB32BLOCK - 2]) < 0 ? 0xFFFFFFFF : 0;
}

void Int::ShiftR64Bit() {
    for (int i = 0; i < NB64BLOCK - 1; i++)
        bits64[i] = bits64[i + 1];
    bits64[NB64BLOCK - 1] = ((int64_t)bits64[NB64BLOCK - 2]) < 0 ? 0xFFFFFFFFFFFFFFFF : 0;
}

void Int::ShiftR(uint32_t n) {
    if (n < 64) {
        shiftR((unsigned char)n, bits64);
    } else {
        uint32_t nb64 = n / 64;
        uint32_t nb = n % 64;
        for (uint32_t i = 0; i < nb64; i++)
            ShiftR64Bit();
        shiftR((unsigned char)nb, bits64);
    }
}

void Int::Mult(Int *a) {
    Int b(this);
    Mult(a, &b);
}

void Int::Mult(Int *a, Int *b) {
    uint64_t t[NB64BLOCK * 2] = {0};

    for (int i = 0; i < NB64BLOCK; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < NB64BLOCK; j++) {
            uint64_t hi, lo;
            lo = _umul128(a->bits64[i], b->bits64[j], &hi);
            
            uint64_t sum = t[i + j] + lo + carry;
            carry = hi + (sum < lo ? 1 : 0);
            t[i + j] = sum;
        }
        t[i + NB64BLOCK] = carry;
    }

    memcpy(bits64, t, NB64BLOCK * 8);
}

void Int::Div(Int *a, Int *b, Int *mod) {
    if (a->IsGreater(b)) {
        if (mod) {
            mod->Set(a);
            SetZero();
        } else {
            Set(a);
        }
        return;
    }

    if (a->IsZero()) {
        SetZero();
        if (mod) mod->SetZero();
        return;
    }

    if (a->IsEqual(b)) {
        if (mod) mod->SetZero();
        SetOne();
        return;
    }

    if (b->IsOneWord()) {
        uint64_t divWord = b->bits64[0];
        if (divWord == 1) {
            if (mod) mod->SetZero();
            Set(a);
            return;
        }
        
        uint64_t carry = 0;
        Int tmp(*a);
        
        for (int i = NB64BLOCK - 1; i >= 0; i--) {
            uint64_t d = (carry << 64) | tmp.bits64[i];
            tmp.bits64[i] = d / divWord;
            carry = d % divWord;
        }
        
        if (mod) mod->Set(carry);
        Set(&tmp);
        return;
    }

    Int u(*a);
    Int v(*b);
    Int q;
    q.SetZero();

    uint64_t d = (uint64_t)1 << (64 - v.GetBitLength() % 64);
    u.Mult(&u, d);
    v.Mult(&v, d);

    int m = u.GetLength();
    int n = v.GetLength();

    for (int j = m - n; j >= 0; j--) {
        uint64_t qhat = u.bits64[j + n] * _BASE + u.bits64[j + n - 1];
        uint64_t rhat = qhat % v.bits64[n - 1];
        qhat /= v.bits64[n - 1];

        while (qhat >= _BASE || 
               (n > 1 && qhat * v.bits64[n - 2] > _BASE * rhat + u.bits64[j + n - 2])) {
            qhat--;
            rhat += v.bits64[n - 1];
            if (rhat >= _BASE) break;
        }

        uint64_t carry = 0;
        uint64_t borrow = 0;
        for (int i = 0; i < n; i++) {
            uint64_t p = qhat * v.bits64[i] + carry;
            carry = p >> 64;
            p &= 0xFFFFFFFFFFFFFFFF;
            uint64_t sub = u.bits64[j + i] - p - borrow;
            borrow = (u.bits64[j + i] < p) || (sub > u.bits64[j + i]) ? 1 : 0;
            u.bits64[j + i] = sub;
        }
        uint64_t sub = u.bits64[j + n] - carry - borrow;
        borrow = (u.bits64[j + n] < carry) || (sub > u.bits64[j + n]) ? 1 : 0;
        u.bits64[j + n] = sub;

        if (borrow != 0) {
            qhat--;
            carry = 0;
            for (int i = 0; i < n; i++) {
                uint64_t sum = u.bits64[j + i] + v.bits64[i] + carry;
                carry = sum >> 64;
                u.bits64[j + i] = sum & 0xFFFFFFFFFFFFFFFF;
            }
            u.bits64[j + n] += carry;
        }

        if (j < NB64BLOCK)
            q.bits64[j] = qhat;
    }

    if (mod) {
        u.ShiftR(1);
        mod->Set(&u);
    }

    Set(&q);
}

void Int::Mod(Int *n) {
    Int r;
    Div(n, &r);
    Set(&r);
}

void Int::ModMul(Int *a, Int *b, Int *n) {
    Int p;
    p.Mult(a, b);
    p.Mod(n);
    Set(&p);
}

void Int::ModSquare(Int *n) {
    Int p;
    p.Mult(this, this);
    p.Mod(n);
    Set(&p);
}

void Int::ModPow(Int *e, Int *n) {
    Int base(*this);
    SetOne();

    for (int i = NB64BLOCK - 1; i >= 0; i--) {
        uint64_t mask = (uint64_t)1 << 63;
        for (int j = 0; j < 64; j++) {
            ModSquare(n);
            if (e->bits64[i] & mask)
                ModMul(&base, n);
            mask >>= 1;
        }
    }
}

void Int::ModInv() {
    Int m(*this);
    SetOne();
    Int a = m;
    Int b = MODULO;
    Int u(*this);
    Int v;
    v.SetZero();

    while (!a.IsZero()) {
        Int q, r;
        b.Div(&a, &q, &r);
        b.Set(&a);
        a.Set(&r);

        Int tmp = u;
        u.Set(&v);
        v.Mult(&q, &v);
        v.Sub(&tmp, &v);
    }

    if (b.IsOne()) {
        if (u.IsNegative())
            u.Add(&MODULO);
        Set(&u);
    } else {
        SetZero();
    }
}

int Int::GetBitLength() {
    int bitLength = 0;
    int i = NB64BLOCK - 1;

    while (i >= 0 && bits64[i] == 0)
        i--;

    if (i >= 0) {
        uint64_t mask = (uint64_t)1 << 63;
        while (mask > 0 && (bits64[i] & mask) == 0) {
            mask >>= 1;
            bitLength++;
        }
        bitLength = 64 * (i + 1) - bitLength;
    }

    return bitLength;
}

int Int::GetLength() {
    int length = NB64BLOCK;
    while (length > 0 && bits64[length - 1] == 0)
        length--;
    return length;
}

void Int::Rand(int nbits) {
    SetZero();
    for (int i = 0; i < (nbits + 63) / 64 && i < NB64BLOCK; i++)
        bits64[i] = rndl();
    
    int shift = nbits % 64;
    if (shift != 0)
        bits64[(nbits - 1) / 64] &= (((uint64_t)1 << shift) - 1);
}

void Int::SetBase10(char *value) {
    CLEAR();
    Int pw(1);
    Int c;
    int lgth = (int)strlen(value);
    for (int i = lgth - 1; i >= 0; i--) {
        uint32_t id = (uint32_t)(value[i] - '0');
        c.Set(&pw);
        c.Mult(id);
        Add(&c);
        pw.Mult(10);
    }
}

std::string Int::GetBase10() {
    std::string ret;
    Int N(this);
    bool isNegative = N.IsNegative();
    if (isNegative) N.Neg();

    if (N.IsZero()) return "0";

    Int zero;
    zero.SetZero();
    Int ten;
    ten.SetInt32(10);
    Int digit;

    while (!N.IsEqual(&zero)) {
        N.Div(&ten, &digit);
        ret.insert(0, 1, '0' + digit.GetInt32());
    }

    if (isNegative)
        ret.insert(0, 1, '-');

    return ret;
}

std::string Int::GetBase16() {
    const char *hex = "0123456789ABCDEF";
    std::string ret;
    bool found = false;

    for (int i = NB64BLOCK - 1; i >= 0; i--) {
        for (int j = 15; j >= 0; j--) {
            uint8_t b = (bits64[i] >> (j * 4)) & 0xF;
            if (!found && b != 0) found = true;
            if (found) ret.push_back(hex[b]);
        }
    }

    return ret.empty() ? "0" : ret;
}

int Int::GetBit(uint32_t n) {
    uint32_t byte = n >> 5;
    uint32_t bit = n & 31;
    return (bits[byte] >> bit) & 1;
}

void Int::Check() {
    Int a, b, c, d, e, R;

    a.SetBase10("4743256844168384767987");
    b.SetBase10("1679314142928575978367");
    if (strcmp(a.GetBase10().c_str(), "4743256844168384767987") != 0) {
        printf(" GetBase10() failed ! %s!=4743256844168384767987\n", a.GetBase10().c_str());
    }
    if (strcmp(b.GetBase10().c_str(), "1679314142928575978367") != 0) {
        printf(" GetBase10() failed ! %s!=1679314142928575978367\n", b.GetBase10().c_str());
        return;
    }

    printf("GetBase10() Results OK\n");

    double t0 = Timer::get_tick();
    for (int i = 0; i < 10000; i++) c.Add(&a, &b);
    double t1 = Timer::get_tick();

    if (c.GetBase10() == "6422570987096960746354") {
        printf("Add() Results OK : ");
        Timer::printResult("Add", 10000, t0, t1);
    } else {
        printf("Add() Results Wrong\nR=%s\nT=6422570987096960746354\n", c.GetBase10().c_str());
        return;
    }

    a.SetBase10("3890902718436931151119442452387018319292503094706912504064239834754167");
    b.SetBase10("474325684416838476798716793141429285759783676422570987096960746354");
    e.SetBase10("1845555094921934741640873731771879197054909502699192730283220486240724687661257894226660948002650341240452881231721004292250660431557118");

    t0 = Timer::get_tick();
    for (int i = 0; i < 10000; i++) c.Mult(&a, &b);
    t1 = Timer::get_tick();

    if (c.IsEqual(&e)) {
        printf("Mult() Results OK : ");
        Timer::printResult("Mult", 10000, t0, t1);
    } else {
        printf("Mult() Results Wrong\nR=%s\nT=%s\n",e.GetBase10().c_str(), c.GetBase10().c_str());
        return;
    }
}
