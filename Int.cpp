/*
 * This file is part of the VanitySearch distribution (https://github.com/JeanLucPons/VanitySearch).
 * Copyright (c) 2019 Jean Luc PONS.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of th#include "Int.h"
#include "IntGroup.h"
#include <string.h>
#include <emmintrin.h>
#include "Timer.h"

#define MAX(x,y) (((x)>(y))?(x):(y))
#define MIN(x,y) (((x)<(y))?(x):(y))

Int _ONE(1);

// ------------------------------------------------

Int::Int() {
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

// ------------------------------------------------

void Int::CLEAR() {
  memset(bits64,0, NB64BLOCK*8);
}

void Int::CLEARFF() {
  memset(bits64, 0xFF, NB64BLOCK * 8);
}

// ------------------------------------------------

void Int::Set(Int *a) {
  for (int i = 0; i<NB64BLOCK; i++)
  	bits64[i] = a->bits64[i];
}

// ------------------------------------------------

void Int::Add(Int *a) {
  unsigned char c = 0;
  c = _addcarry_u64(c, bits64[0], a->bits64[0], bits64 +0);
  c = _addcarry_u64(c, bits64[1], a->bits64[1], bits64 +1);
  c = _addcarry_u64(c, bits64[2], a->bits64[2], bits64 +2);
  c = _addcarry_u64(c, bits64[3], a->bits64[3], bits64 +3);
  c = _addcarry_u64(c, bits64[4], a->bits64[4], bits64 +4);
#if NB64BLOCK > 5
  c = _addcarry_u64(c, bits64[5], a->bits64[5], bits64 +5);
  c = _addcarry_u64(c, bits64[6], a->bits64[6], bits64 +6);
  c = _addcarry_u64(c, bits64[7], a->bits64[7], bits64 +7);
  c = _addcarry_u64(c, bits64[8], a->bits64[8], bits64 +8);
#endif
}

// ------------------------------------------------

void Int::Add(uint64_t a) {
	unsigned char c = 0;
	c = _addcarry_u64(c, bits64[0], a, bits64 + 0);
	c = _addcarry_u64(c, bits64[1], 0, bits64 + 1);
	c = _addcarry_u64(c, bits64[2], 0, bits64 + 2);
	c = _addcarry_u64(c, bits64[3], 0, bits64 + 3);
	c = _addcarry_u64(c, bits64[4], 0, bits64 + 4);
#if NB64BLOCK > 5
	c = _addcarry_u64(c, bits64[5], 0, bits64 + 5);
	c = _addcarry_u64(c, bits64[6], 0, bits64 + 6);
	c = _addcarry_u64(c, bits64[7], 0, bits64 + 7);
	c = _addcarry_u64(c, bits64[8], 0, bits64 + 8);
#endif
}

// ------------------------------------------------
void Int::AddOne() {
  unsigned char c = 0;
  c = _addcarry_u64(c, bits64[0],1, bits64 +0);
  c = _addcarry_u64(c, bits64[1],0, bits64 +1);
  c = _addcarry_u64(c, bits64[2],0, bits64 +2);
  c = _addcarry_u64(c, bits64[3],0, bits64 +3);
  c = _addcarry_u64(c, bits64[4],0, bits64 +4);
#if NB64BLOCK > 5
  c = _addcarry_u64(c, bits64[5],0, bits64 +5);
  c = _addcarry_u64(c, bits64[6],0, bits64 +6);
  c = _addcarry_u64(c, bits64[7],0, bits64 +7);
  c = _addcarry_u64(c, bits64[8],0, bits64 +8);
#endif
}

// ------------------------------------------------

void Int::Add(Int *a,Int *b) {
  unsigned char c = 0;
  c = _addcarry_u64(c, b->bits64[0], a->bits64[0], bits64 +0);
  c = _addcarry_u64(c, b->bits64[1], a->bits64[1], bits64 +1);
  c = _addcarry_u64(c, b->bits64[2], a->bits64[2], bits64 +2);
  c = _addcarry_u64(c, b->bits64[3], a->bits64[3], bits64 +3);
  c = _addcarry_u64(c, b->bits64[4], a->bits64[4], bits64 +4);
#if NB64BLOCK > 5
  c = _addcarry_u64(c, b->bits64[5], a->bits64[5], bits64 +5);
  c = _addcarry_u64(c, b->bits64[6], a->bits64[6], bits64 +6);
  c = _addcarry_u64(c, b->bits64[7], a->bits64[7], bits64 +7);
  c = _addcarry_u64(c, b->bits64[8], a->bits64[8], bits64 +8);
#endif
}

// ------------------------------------------------

bool Int::IsGreater(Int *a) {
  int i;
  for(i=NB64BLOCK-1;i>=0;) {
    if( a->bits64[i]!= bits64[i] )
		break;
    i--;
  }
  if(i>=0) {
    return bits64[i]>a->bits64[i];
  } else {
    return false;
  }
}

// ------------------------------------------------

bool Int::IsLower(Int *a) {
  int i;
  for (i = NB64BLOCK - 1; i >= 0;) {
    if (a->bits64[i] != bits64[i])
      break;
    i--;
  }
  if (i >= 0) {
    return bits64[i]<a->bits64[i];
  } else {
    return false;
  }
}

// ------------------------------------------------

bool Int::IsGreaterOrEqual(Int *a) {
  Int p;
  p.Sub(this,a);
  return p.IsPositive();
}

// ------------------------------------------------

bool Int::IsLowerOrEqual(Int *a) {
  int i = NB64BLOCK - 1;
  while (i >= 0) {
    if (a->bits64[i] != bits64[i])
      break;
    i--;
  }
  if (i >= 0) {
    return bits64[i]<a->bits64[i];
  } else {
    return true;
  }
}

bool Int::IsEqual(Int *a) {
#if NB64BLOCK > 5
  return (bits64[8] == a->bits64[8]) &&
  (bits64[7] == a->bits64[7]) &&
  (bits64[6] == a->bits64[6]) &&
  (bits64[5] == a->bits64[5]) &&
#endif
  (bits64[4] == a->bits64[4]) &&
  (bits64[3] == a->bits64[3]) &&
  (bits64[2] == a->bits64[2]) &&
  (bits64[1] == a->bits64[1]) &&
  (bits64[0] == a->bits64[0]);
}

bool Int::IsOne() {
  return IsEqual(&_ONE);
}

bool Int::IsZero() {
#if NB64BLOCK > 5
  return (bits64[8] | bits64[7] | bits64[6] | bits64[5] | bits64[4] | bits64[3] | bits64[2] | bits64[1] | bits64[0]) == 0;
#else
  return (bits64[4] | bits64[3] | bits64[2] | bits64[1] | bits64[0]) == 0;
#endif
}

// ------------------------------------------------

void Int::SetInt32(uint32_t value) {
  CLEAR();
  bits[0]=value;
}

// ------------------------------------------------

uint32_t Int::GetInt32() {
  return bits[0];
}

// ------------------------------------------------

unsigned char Int::GetByte(int n) {
  unsigned char *bbPtr = (unsigned char *)bits;
  return bbPtr[n];
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

// ------------------------------------------------

void Int::SetByte(int n,unsigned char byte) {
	unsigned char *bbPtr = (unsigned char *)bits;
	bbPtr[n] = byte;
}

// ------------------------------------------------

void Int::SetDWord(int n,uint32_t b) {
  bits[n] = b;
}

// ------------------------------------------------

void Int::SetQWord(int n, uint64_t b) {
	bits64[n] = b;
}

// ------------------------------------------------

void Int::Sub(Int *a) {
  unsigned char c = 0;
  c = _subborrow_u64(c, bits64[0], a->bits64[0], bits64 +0);
  c = _subborrow_u64(c, bits64[1], a->bits64[1], bits64 +1);
  c = _subborrow_u64(c, bits64[2], a->bits64[2], bits64 +2);
  c = _subborrow_u64(c, bits64[3], a->bits64[3], bits64 +3);
  c = _subborrow_u64(c, bits64[4], a->bits64[4], bits64 +4);
#if NB64BLOCK > 5
  c = _subborrow_u64(c, bits64[5], a->bits64[5], bits64 +5);
  c = _subborrow_u64(c, bits64[6], a->bits64[6], bits64 +6);
  c = _subborrow_u64(c, bits64[7], a->bits64[7], bits64 +7);
  c = _subborrow_u64(c, bits64[8], a->bits64[8], bits64 +8);
#endif
}

// ------------------------------------------------

void Int::Sub(Int *a,Int *b) {
  unsigned char c = 0;
  c = _subborrow_u64(c, a->bits64[0], b->bits64[0], bits64 + 0);
  c = _subborrow_u64(c, a->bits64[1], b->bits64[1], bits64 + 1);
  c = _subborrow_u64(c, a->bits64[2], b->bits64[2], bits64 + 2);
  c = _subborrow_u64(c, a->bits64[3], b->bits64[3], bits64 + 3);
  c = _subborrow_u64(c, a->bits64[4], b->bits64[4], bits64 + 4);
#if NB64BLOCK > 5
  c = _subborrow_u64(c, a->bits64[5], b->bits64[5], bits64 + 5);
  c = _subborrow_u64(c, a->bits64[6], b->bits64[6], bits64 + 6);
  c = _subborrow_u64(c, a->bits64[7], b->bits64[7], bits64 + 7);
  c = _subborrow_u64(c, a->bits64[8], b->bits64[8], bits64 + 8);
#endif
}

void Int::Sub(uint64_t a) {
  unsigned char c = 0;
  c = _subborrow_u64(c, bits64[0], a, bits64 + 0);
  c = _subborrow_u64(c, bits64[1], 0, bits64 + 1);
  c = _subborrow_u64(c, bits64[2], 0, bits64 + 2);
  c = _subborrow_u64(c, bits64[3], 0, bits64 + 3);
  c = _subborrow_u64(c, bits64[4], 0, bits64 + 4);
#if NB64BLOCK > 5
  c = _subborrow_u64(c, bits64[5], 0, bits64 + 5);
  c = _subborrow_u64(c, bits64[6], 0, bits64 + 6);
  c = _subborrow_u64(c, bits64[7], 0, bits64 + 7);
  c = _subborrow_u64(c, bits64[8], 0, bits64 + 8);
#endif
}

void Int::SubOne() {
  unsigned char c = 0;
  c = _subborrow_u64(c, bits64[0], 1, bits64 + 0);
  c = _subborrow_u64(c, bits64[1], 0, bits64 + 1);
  c = _subborrow_u64(c, bits64[2], 0, bits64 + 2);
  c = _subborrow_u64(c, bits64[3], 0, bits64 + 3);
  c = _subborrow_u64(c, bits64[4], 0, bits64 + 4);
#if NB64BLOCK > 5
  c = _subborrow_u64(c, bits64[5], 0, bits64 + 5);
  c = _subborrow_u64(c, bits64[6], 0, bits64 + 6);
  c = _subborrow_u64(c, bits64[7], 0, bits64 + 7);
  c = _subborrow_u64(c, bits64[8], 0, bits64 + 8);
#endif
}

// ------------------------------------------------

bool Int::IsPositive() {
  return (int64_t)(bits64[NB64BLOCK - 1])>=0;
}

// ------------------------------------------------

bool Int::IsNegative() {
  return (int64_t)(bits64[NB64BLOCK - 1])<0;
}

// ------------------------------------------------

bool Int::IsStrictPositive() {
  if( IsPositive() )
	  return !IsZero();
  else
	  return false;
}

// ------------------------------------------------

bool Int::IsEven() {
  return (bits[0] & 0x1) == 0;
}

// ------------------------------------------------

bool Int::IsOdd() {
  return (bits[0] & 0x1) == 1;
}

// ------------------------------------------------

void Int::Neg() {
	volatile unsigned char c=0;
	c = _subborrow_u64(c, 0, bits64[0], bits64 + 0);
	c = _subborrow_u64(c, 0, bits64[1], bits64 + 1);
	c = _subborrow_u64(c, 0, bits64[2], bits64 + 2);
	c = _subborrow_u64(c, 0, bits64[3], bits64 + 3);
	c = _subborrow_u64(c, 0, bits64[4], bits64 + 4);
#if NB64BLOCK > 5
	c = _subborrow_u64(c, 0, bits64[5], bits64 + 5);
	c = _subborrow_u64(c, 0, bits64[6], bits64 + 6);
	c = _subborrow_u64(c, 0, bits64[7], bits64 + 7);
	c = _subborrow_u64(c, 0, bits64[8], bits64 + 8);
#endif
}

// ------------------------------------------------

void Int::ShiftL32Bit() {
  for(int i=NB32BLOCK-1;i>0;i--) {
    bits[i]=bits[i-1];
  }
  bits[0]=0;
}

// ------------------------------------------------

void Int::ShiftL64Bit() {
	for (int i = NB64BLOCK-1 ; i>0; i--) {
		bits64[i] = bits64[i - 1];
	}
	bits64[0] = 0;
}

// ------------------------------------------------

void Int::ShiftL32BitAndSub(Int *a,int n) {
  Int b;
  int i=NB32BLOCK-1;
  for(;i>=n;i--)
    b.bits[i] = ~a->bits[i-n];
  for(;i>=0;i--)
    b.bits[i] = 0xFFFFFFFF;
  Add(&b);
  AddOne();
}

// ------------------------------------------------

void Int::ShiftL(uint32_t n) {    
  if( n<64 ) {
	shiftL((unsigned char)n, bits64);
  } else {
    uint32_t nb64 = n/64;
    uint32_t nb   = n%64;
    for(uint32_t i=0;i<nb64;i++) ShiftL64Bit();
	  shiftL((unsigned char)nb, bits64);
  }  
}

// ------------------------------------------------

void Int::ShiftR32Bit() {
  for(int i=0;i<NB32BLOCK-1;i++) {
    bits[i]=bits[i+1];
  }
  if(((int32_t)bits[NB32BLOCK-2])<0)
    bits[NB32BLOCK-1] = 0xFFFFFFFF;
  else
    bits[NB32BLOCK-1]=0;
}

// ------------------------------------------------

void Int::ShiftR64Bit() {
	for (int i = 0; i<NB64BLOCK - 1; i++) {
		bits64[i] = bits64[i + 1];
	}
	if (((int64_t)bits64[NB64BLOCK - 2])<0)
		bits64[NB64BLOCK - 1] = 0xFFFFFFFFFFFFFFFF;
	else
		bits64[NB64BLOCK - 1] = 0;
}

// ------------------------------------------------

void Int::ShiftR(uint32_t n) {    
  if( n<64 ) {
    shiftR((unsigned char)n, bits64);
  } else {
    uint32_t nb64 = n/64;
    uint32_t nb   = n%64;
    for(uint32_t i=0;i<nb64;i++) ShiftR64Bit();
	  shiftR((unsigned char)nb, bits64);
  }  
}

// ------------------------------------------------

void Int::Mult(Int *a) {
  Int b(this);
  Mult(a,&b);
}

// ------------------------------------------------

void Int::IMult(int64_t a) {
	if (a < 0LL) {
		a = -a;
		Neg();
	}
	imm_mul(bits64, a, bits64);
}

// ------------------------------------------------

void Int::Mult(uint64_t a) {
	imm_mul(bits64, a, bits64);
}

// ------------------------------------------------

void Int::IMult(Int *a, int64_t b) {  
  Set(a);
  if (b < 0LL) {
	Neg();
	b = -b;
  }
  imm_mul(bits64, b, bits64);
}

// ------------------------------------------------

void Int::Mult(Int *a, uint64_t b) {
  imm_mul(a->bits64, b, bits64);
}

// ------------------------------------------------

void Int::Mult(Int *a,Int *b) {  
  unsigned char c = 0;
  uint64_t h;
  uint64_t pr = 0;
  uint64_t carryh = 0;
  uint64_t carryl = 0;

  bits64[0] = _umul128(a->bits64[0], b->bits64[0], &pr);

  for (int i = 1; i < NB64BLOCK; i++) {
    for (int j = 0; j <= i; j++) {
      c = _addcarry_u64(c, _umul128(a->bits64[j], b->bits64[i - j], &h), pr, &pr);
      c = _addcarry_u64(c, carryl, h, &carryl);
      c = _addcarry_u64(c, carryh, 0, &carryh);
    }
    bits64[i] = pr;
    pr = carryl;
    carryl = carryh;
    carryh = 0;
  }
}

// ------------------------------------------------

void Int::Mult(Int *a,uint32_t b) {
  imm_mul(a->bits64, (uint64_t)b, bits64);
}

// ------------------------------------------------

static uint32_t bitLength(uint32_t dw) {  
  uint32_t mask = 0x80000000;
  uint32_t b=0;
  while(b<32 && (mask & dw)==0) {
    b++;
    mask >>= 1;
  }
  return b;
}

// ------------------------------------------------

int Int::GetBitLength() {
  Int t(this);
  if(IsNegative())
	  t.Neg();
  int i=NB32BLOCK-1;
  while(i>=0 && t.bits[i]==0) i--;
  if(i<0) return 0;
  return (32-bitLength(t.bits[i])) + i*32;
}

// ------------------------------------------------

int Int::GetSize() {
  int i=NB32BLOCK-1;
  while(i>0 && bits[i]==0) i--;
  return i+1;
}

// ------------------------------------------------

void Int::MultModN(Int *a,Int *b,Int *n) {
  Int r;
  Mult(a,b);
  Div(n,&r);
  Set(&r);
}

// ------------------------------------------------

void Int::Mod(Int *n) {
  Int r;
  Div(n,&r);
  Set(&r);
}

// ------------------------------------------------

int Int::GetLowestBit() {
  // Assume this!=0
  int b=0;
  while(GetBit(b)==0) b++;
  return b;
}

// ------------------------------------------------

void Int::MaskByte(int n) {
  for (int i = n; i < NB32BLOCK; i++)
	  bits[i] = 0;
}

// ------------------------------------------------

void Int::Abs() {
  if (IsNegative())
    Neg();
}

// ------------------------------------------------

void Int::Rand(int nbit) {
	CLEAR();
	uint32_t nb = nbit/32;
	uint32_t leftBit = nbit%32;
	uint32_t mask = 1;
	mask = (mask << leftBit) - 1;
	uint32_t i=0;
	for(;i<nb;i++)
		bits[i]=rndl();
	bits[i]=rndl()&mask;
}

// ------------------------------------------------

void Int::Div(Int *a,Int *mod) {
  if(a->IsGreater(this)) {
    if(mod) mod->Set(this);
    CLEAR();
    return;
  }

  if(a->IsZero()) {
    printf("Divide by 0!\n");
    return;
  }

  if(IsEqual(a)) {
    if(mod) mod->CLEAR();
    Set(&_ONE);
    return;
  }

  //Division algorithm D (Knuth section 4.3.1)
  Int rem(this);
  Int d(a);
  Int dq;
  CLEAR();

  // Size
  uint32_t dSize = d.GetSize();
  uint32_t tSize = rem.GetSize();
  uint32_t qSize = tSize - dSize + 1;

  // D1 normalize the divisor
  uint32_t shift = bitLength(d.bits[dSize-1]);
  if (shift > 0) {
    d.ShiftL(shift);
    rem.ShiftL(shift);
  }

  uint32_t  _dh    = d.bits[dSize-1];
  uint64_t  dhLong = _dh;
  uint32_t  _dl    = (dSize>1)?d.bits[dSize-2]:0;
  int sb = tSize-1;
        
  // D2 Initialize j
  for(int j=0; j<(int)qSize; j++) {

    // D3 Estimate qhat
    uint32_t qhat = 0;
    uint32_t qrem = 0;
    int skipCorrection = false;
    uint32_t nh = rem.bits[sb-j+1];
    uint32_t nm = rem.bits[sb-j];

    if (nh == _dh) {
      qhat = ~0;
      qrem = nh + nm;
      skipCorrection = qrem < nh;
    } else {
      uint64_t nChunk = ((uint64_t)nh << 32) | (uint64_t)nm;
      qhat = (uint32_t) (nChunk / dhLong);
      qrem = (uint32_t) (nChunk % dhLong);
    }

    if (qhat == 0)
      continue;

    if (!skipCorrection) { 
      // Correct qhat
      uint64_t nl = (uint64_t)rem.bits[sb-j-1];
      uint64_t rs = ((uint64_t)qrem << 32) | nl;
      uint64_t estProduct = (uint64_t)_dl * (uint64_t)(qhat);

      if (estProduct>rs) {
        qhat--;
        qrem = (uint32_t)(qrem + (uint32_t)dhLong);
        if ((uint64_t)qrem >= dhLong) {
          estProduct = (uint64_t)_dl * (uint64_t)(qhat);
          rs = ((uint64_t)qrem << 32) | nl;
          if(estProduct>rs)
            qhat--;
        }
      }
    }

    // D4 Multiply and subtract    
    dq.Mult(&d,qhat);
    rem.ShiftL32BitAndSub(&dq,qSize-j-1);
    if( rem.IsNegative() ) {
      // Overflow
      rem.Add(&d);
      qhat--;
    }

    bits[qSize-j-1] = qhat;
 }

 if( mod ) {
   // Unnormalize remainder
   rem.ShiftR(shift);
   mod->Set(&rem);
 }
}

// ------------------------------------------------

void Int::GCD(Int *a) {
    uint32_t k;
    uint32_t b;

    Int U(this);
    Int V(a);
    Int T;

    if(U.IsZero()) {
      Set(&V);
      return;
    }

    if(V.IsZero()) {
      Set(&U);
      return;
    }

    if(U.IsNegative()) U.Neg();
    if(V.IsNegative()) V.Neg();

    k = 0;
    while (U.GetBit(k)==0 && V.GetBit(k)==0)
      k++;
    U.ShiftR(k);
    V.ShiftR(k);
    if (U.GetBit(0)==1) { 
      T.Set(&V);
      T.Neg();
    } else {
      T.Set(&U);
    }

    do {
      if( T.IsNegative() ) {
        T.Neg();
        b=0;while(T.GetBit(b)==0) b++;
        T.ShiftR(b);
        V.Set(&T);
        T.Set(&U);
      } else {
        b=0;while(T.GetBit(b)==0) b++;
        T.ShiftR(b);
        U.Set(&T);
      }
      T.Sub(&V);
    } while (!T.IsZero());

    // Store gcd
    Set(&U);
    ShiftL(k); 
}

// ------------------------------------------------

void Int::SetBase10(char *value) {  
  CLEAR();
  Int pw(1);
  Int c;
  int lgth = (int)strlen(value);
  for(int i=lgth-1;i>=0;i--) {
    uint32_t id = (uint32_t)(value[i]-'0');
    c.Set(&pw);
    c.Mult(id);
    Add(&c);
    pw.Mult(10);
  }
}

// ------------------------------------------------

void  Int::SetBase16(char *value) {  
  SetBaseN(16,"0123456789ABCDEF",value);
}

// ------------------------------------------------

std::string Int::GetBase10() {
  return GetBaseN(10,"0123456789");
}

// ------------------------------------------------

std::string Int::GetBase16() {
  return GetBaseN(16,"0123456789ABCDEF");
}

// ------------------------------------------------

std::string Int::GetBlockStr() {
	char tmp[256];
	char bStr[256];
	tmp[0] = 0;
	for (int i = NB32BLOCK-3; i>=0 ; i--) {
	  sprintf(bStr, "%08X", bits[i]);
	  strcat(tmp, bStr);
	  if(i!=0) strcat(tmp, " ");
	}
	return std::string(tmp);
}

// ------------------------------------------------

std::string Int::GetC64Str(int nbDigit) {
  char tmp[256];
  char bStr[256];
  tmp[0] = '{';
  tmp[1] = 0;
  for (int i = 0; i< nbDigit; i++) {
    if (bits64[i] != 0) {
#ifdef WIN64
      sprintf(bStr, "0x%016I64XULL", bits64[i]);
#else
      sprintf(bStr, "0x%" PRIx64  "ULL", bits64[i]);
#endif
    } else {
      sprintf(bStr, "0ULL");
    }
    strcat(tmp, bStr);
    if (i != nbDigit -1) strcat(tmp, ",");
  }
  strcat(tmp,"}");
  return std::string(tmp);
}

// ------------------------------------------------

void  Int::SetBaseN(int n,char *charset,char *value) {
  CLEAR();
  Int pw((uint32_t)1);
  Int nb((int32_t)n);
  Int c;

  int lgth = (int)strlen(value);
  for(int i=lgth-1;i>=0;i--) {
    char *p = strchr(charset,toupper(value[i]));
    if(!p) {
      printf("Invalid charset !!\n");
      return;
    }
    int id = (int)(p-charset);
    c.SetInt32(id);
    c.Mult(&pw);
    Add(&c);
    pw.Mult(&nb);
  }
}

// ------------------------------------------------

std::string Int::GetBaseN(int n,char *charset) {
  std::string ret;
  Int N(this);
  int isNegative = N.IsNegative();
  if (isNegative) N.Neg();

  unsigned char digits[1024];
  memset(digits, 0, sizeof(digits));

  int digitslen = 1;
  for (int i = 0; i < NB64BLOCK * 8; i++) {
    unsigned int carry = N.GetByte(NB64BLOCK*8 - i - 1);
    for (int j = 0; j < digitslen; j++) {
      carry += (unsigned int)(digits[j]) << 8;
      digits[j] = (unsigned char)(carry % n);
      carry /= n;
    }
    while (carry > 0) {
      digits[digitslen++] = (unsigned char)(carry % n);
      carry /= n;
    }
  }

  if (isNegative)
    ret.push_back('-');

  for (int i = 0; i < digitslen; i++)
    ret.push_back(charset[digits[digitslen - 1 - i]]);

  if (ret.length() == 0)
    ret.push_back('0');

  return ret;
}

// ------------------------------------------------

int Int::GetBit(uint32_t n) {
  uint32_t byte = n>>5;
  uint32_t bit  = n&31;
  uint32_t mask = 1 << bit;
  return (bits[byte] & mask)!=0;
}

// ------------------------------------------------

std::string Int::GetBase2() {
  char ret[1024];
  int k=0;

  for(int i=0;i<NB32BLOCK-1;i++) {
    unsigned int mask=0x80000000;
    for(int j=0;j<32;j++) {
      if(bits[i]&mask) ret[k]='1';
      else             ret[k]='0';
      k++;
      mask=mask>>1;
    }
  }
  ret[k]=0;

  return std::string(ret);
}

// ------------------------------------------------

void Int::Check() {
  double t0;
  double t1;
  double tTotal;
  int   i;
  bool ok;

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

  // Add -------------------------------------------------------------------------------------------
  t0 = Timer::get_tick();
  for (i = 0; i < 10000; i++) c.Add(&a, &b);
  t1 = Timer::get_tick();

  if (c.GetBase10() == "6422570987096960746354") {
    printf("Add() Results OK : ");
    Timer::printResult("Add", 10000, t0, t1);
  } else {
    printf("Add() Results Wrong\nR=%s\nT=6422570987096960746354\n", c.GetBase10().c_str());
    return;
  }

  // Mult -------------------------------------------------------------------------------------------
  a.SetBase10("3890902718436931151119442452387018319292503094706912504064239834754167");
  b.SetBase10("474325684416838476798716793141429285759783676422570987096960746354");
  e.SetBase10("1845555094921934741640873731771879197054909502699192730283220486240724687661257894226660948002650341240452881231721004292250660431557118");

  t0 = Timer::get_tick();
  for (i = 0; i < 10000; i++) c.Mult(&a, &b);
  t1 = Timer::get_tick();

  if (c.IsEqual(&e)) {
    printf("Mult() Results OK : ");
    Timer::printResult("Mult", 10000, t0, t1);
  } else {
    printf("Mult() Results Wrong\nR=%s\nT=%s\n",e.GetBase10().c_str(), c.GetBase10().c_str());
    return;
  }
 
  // Div -------------------------------------------------------------------------------------------

void Int::Div(const Int* a, const Int* b, Int* mod) {

    if (a->IsGreater(b)) {
        if (mod) {
            mod->Set(a);
            SetZero();
            return;
        } else {
            Set(a);
            return;
        }
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

    // Handle single-word divisor
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

    // Use Knuth's algorithm D for multi-word division
    int m = a->GetLength();
    int n = b->GetLength();
    
    if (n == 0) {
        // Division by zero
        SetZero();
        if (mod) mod->SetZero();
        return;
    }

    if (m < n) {
        if (mod) mod->Set(a);
        SetZero();
        return;
    }

    Int u(*a);
    Int v(*b);
    Int q;
    q.SetZero();

    // Normalize
    uint64_t d = (uint64_t)1 << (64 - v.GetBitLength() % 64);
    u.Mult(&u, d);
    v.Mult(&v, d);

    m = u.GetLength();
    n = v.GetLength();

    for (int j = m - n; j >= 0; j--) {
        // Estimate qhat
        uint64_t qhat = u.bits64[j + n] * _BASE + u.bits64[j + n - 1];
        uint64_t rhat = qhat % v.bits64[n - 1];
        qhat /= v.bits64[n - 1];

        while (qhat >= _BASE || 
               (n > 1 && qhat * v.bits64[n - 2] > _BASE * rhat + u.bits64[j + n - 2])) {
            qhat--;
            rhat += v.bits64[n - 1];
            if (rhat >= _BASE) break;
        }

        // Multiply and subtract
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

        // Test remainder
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

    // Denormalize remainder
    if (mod) {
        u.ShiftR(1);
        mod->Set(&u);
    }

    Set(&q);
}

void Int::Divide(Int* a, Int* b, Int* mod) {
    Div(a, b, mod);
}

// -----------------------------------------------------------------------------

// ModInv -----------------------------------------------------------------------

void Int::ModInv() {
    Int m(*this);
    SetOne();
    Int a = m;
    Int b = MODULO;
    Int u(*this);
    Int v;
    v.SetZero();

    while (!a.IsZero()) {
        Int q;
        Int r;
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

// -----------------------------------------------------------------------------

// ModK1 -----------------------------------------------------------------------

void Int::ModK1() {
    Int p(this);
    Int m(this);
    Int r;
    r.SetZero();

    p.Sub(&m, &r);
    if (r.IsNegative()) {
        Set(&p);
    } else {
        Set(&r);
    }
}

// -----------------------------------------------------------------------------

// ModMulK1 --------------------------------------------------------------------

void Int::ModMulK1(Int* a) {
    Int p;
    p.Mult(this, a);
    p.ModK1();
    Set(&p);
}

// -----------------------------------------------------------------------------

// ModSquareK1 -----------------------------------------------------------------

void Int::ModSquareK1() {
    Int p;
    p.Mult(this, this);
    p.ModK1();
    Set(&p);
}

// -----------------------------------------------------------------------------

// Mod -------------------------------------------------------------------------

void Int::Mod(Int* a) {
    if (IsGreaterOrEqual(a)) {
        Int r;
        Div(a, &r);
        Set(&r);
    }
}

// -----------------------------------------------------------------------------

// ModMul ----------------------------------------------------------------------

void Int::ModMul(Int* a, Int* b) {
    Int p;
    p.Mult(a, b);
    p.Mod(&MODULO);
    Set(&p);
}

// -----------------------------------------------------------------------------

// ModSquare -------------------------------------------------------------------

void Int::ModSquare() {
    Int p;
    p.Mult(this, this);
    p.Mod(&MODULO);
    Set(&p);
}

// -----------------------------------------------------------------------------

// ModPow ----------------------------------------------------------------------

void Int::ModPow(Int* e) {
    Int base(*this);
    SetOne();

    for (int i = NB64BLOCK - 1; i >= 0; i--) {
        uint64_t mask = (uint64_t)1 << 63;
        for (int j = 0; j < 64; j++) {
            ModSquare();
            if (e->bits64[i] & mask) {
                ModMul(&base);
            }
            mask >>= 1;
        }
    }
}

// -----------------------------------------------------------------------------

// ModPow2 ---------------------------------------------------------------------

void Int::ModPow2(Int* e, uint32_t pow2) {
    Int base(*this);
    SetOne();

    for (int i = NB64BLOCK - 1; i >= 0; i--) {
        uint64_t mask = (uint64_t)1 << 63;
        for (int j = 0; j < 64; j++) {
            ModSquare();
            if (e->bits64[i] & mask) {
                ModMul(&base);
            }
            mask >>= 1;
        }
    }

    // Apply pow2 modulo
    if (pow2 > 0) {
        Int p2;
        p2.SetInt32(1);
        p2.ShiftL(pow2);
        Mod(&p2);
    }
}

// -----------------------------------------------------------------------------

// Sqrt ------------------------------------------------------------------------

void Int::Sqrt() {
    if (IsNegative() || IsZero()) {
        SetZero();
        return;
    }

    Int x;
    x.Set(this);
    Int y;
    y.SetInt32(1);
    y.ShiftR(1);
    y.Add(&x);

    while (y.IsLower(&x)) {
        x.Set(&y);
        y.Set(this);
        y.Div(&x, &y);
        y.Add(&x);
        y.ShiftR(1);
    }

    Set(&x);
}

// -----------------------------------------------------------------------------

// GetBitLength ----------------------------------------------------------------

int Int::GetBitLength() {
    int bitLength = 0;
    int i = NB64BLOCK - 1;

    // Skip leading zeros
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

// -----------------------------------------------------------------------------

// GetLength -------------------------------------------------------------------

int Int::GetLength() {
    int length = NB64BLOCK;
    while (length > 0 && bits64[length - 1] == 0)
        length--;
    return length;
}

// -----------------------------------------------------------------------------

// IsProbablePrime -------------------------------------------------------------

bool Int::IsProbablePrime() {
    // Handle small primes
    if (IsNegative())
        return false;

    if (GetBitLength() <= 64) {
        uint64_t n = GetInt64();
        if (n < 2) return false;
        if (n == 2 || n == 3) return true;
        if (n % 2 == 0) return false;

        // Check against small primes
        static const uint64_t smallPrimes[] = {
            3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97
        };
        
        for (uint64_t p : smallPrimes) {
            if (n == p) return true;
            if (n % p == 0) return false;
        }
    }

    // Miller-Rabin test
    Int d(*this);
    d.SubOne();
    Int nMinusOne(d);

    int s = 0;
    while (d.IsEven()) {
        d.ShiftR(1);
        s++;
    }

    static const int k = 5; // Number of iterations
    static const Int smallPrimes[] = {
        Int(2), Int(3), Int(5), Int(7), Int(11)
    };

    for (int i = 0; i < k; i++) {
        Int a = smallPrimes[i];
        if (a.IsGreaterOrEqual(this))
            break;

        Int x;
        x.ModPow(&a, &d, this);

        if (x.IsOne() || x.IsEqual(&nMinusOne))
            continue;

        bool probablePrime = false;
        for (int j = 0; j < s - 1; j++) {
            x.ModSquare();
            if (x.IsOne())
                return false;
            if (x.IsEqual(&nMinusOne)) {
                probablePrime = true;
                break;
            }
        }

        if (!probablePrime)
            return false;
    }

    return true;
}

// -----------------------------------------------------------------------------

// NextPrime -------------------------------------------------------------------

void Int::NextPrime() {
    if (IsNegative())
        SetZero();

    if (IsEven())
        AddOne();

    while (!IsProbablePrime()) {
        AddOne();
        AddOne();
    }
}

// -----------------------------------------------------------------------------

// Rand ------------------------------------------------------------------------

void Int::Rand(int nbits) {
    SetZero();
    for (int i = 0; i < (nbits + 63) / 64 && i < NB64BLOCK; i++) {
        bits64[i] = rndl();
    }
    int shift = nbits % 64;
    if (shift != 0) {
        bits64[(nbits - 1) / 64] &= (((uint64_t)1 << shift) - 1);
    }
}

// -----------------------------------------------------------------------------

// Rand ------------------------------------------------------------------------

void Int::Rand(Int* randMax) {
    int b = randMax->GetBitLength();
    while (true) {
        Rand(b);
        if (IsLower(randMax))
            break;
    }
}

// -----------------------------------------------------------------------------

// RandMod ---------------------------------------------------------------------

void Int::RandMod(Int* mod) {
    Rand(mod);
    Mod(mod);
}

// -----------------------------------------------------------------------------

// Add -------------------------------------------------------------------------

void Int::Add(Int* a) {
    uint64_t carry = 0;
    for (int i = 0; i < NB64BLOCK; i++) {
        uint64_t sum = bits64[i] + a->bits64[i] + carry;
        carry = (sum < bits64[i]) || (carry != 0 && sum <= bits64[i]) ? 1 : 0;
        bits64[i] = sum;
    }
}

// -----------------------------------------------------------------------------

// AddOne ----------------------------------------------------------------------

void Int::AddOne() {
    for (int i = 0; i < NB64BLOCK; i++) {
        bits64[i]++;
        if (bits64[i] != 0)
            break;
    }
}

// -----------------------------------------------------------------------------

// Sub -------------------------------------------------------------------------

void Int::Sub(Int* a) {
    uint64_t borrow = 0;
    for (int i = 0; i < NB64BLOCK; i++) {
        uint64_t diff = bits64[i] - a->bits64[i] - borrow;
        borrow = (bits64[i] < a->bits64[i]) || (borrow != 0 && diff >= bits64[i]) ? 1 : 0;
        bits64[i] = diff;
    }
}

// -----------------------------------------------------------------------------

// SubOne ----------------------------------------------------------------------

void Int::SubOne() {
    for (int i = 0; i < NB64BLOCK; i++) {
        uint64_t old = bits64[i];
        bits64[i]--;
        if (old != 0)
            break;
    }
}

// -----------------------------------------------------------------------------

// Neg -------------------------------------------------------------------------

void Int::Neg() {
    for (int i = 0; i < NB64BLOCK; i++) {
        bits64[i] = ~bits64[i];
    }
    AddOne();
}

// -----------------------------------------------------------------------------

// ShiftL ----------------------------------------------------------------------

void Int::ShiftL(uint32_t shift) {
    if (shift == 0) return;

    int wordShift = shift / 64;
    int bitShift = shift % 64;

    if (bitShift == 0) {
        for (int i = NB64BLOCK - 1; i >= wordShift; i--) {
            bits64[i] = bits64[i - wordShift];
        }
    } else {
        for (int i = NB64BLOCK - 1; i > wordShift; i--) {
            bits64[i] = (bits64[i - wordShift] << bitShift) | 
                        (bits64[i - wordShift - 1] >> (64 - bitShift));
        }
        bits64[wordShift] = bits64[0] << bitShift;
    }

    for (int i = 0; i < wordShift; i++) {
        bits64[i] = 0;
    }
}

// -----------------------------------------------------------------------------

// ShiftR ----------------------------------------------------------------------

void Int::ShiftR(uint32_t shift) {
    if (shift == 0) return;

    int wordShift = shift / 64;
    int bitShift = shift % 64;

    if (bitShift == 0) {
        for (int i = 0; i < NB64BLOCK - wordShift; i++) {
            bits64[i] = bits64[i + wordShift];
        }
    } else {
        for (int i = 0; i < NB64BLOCK - wordShift - 1; i++) {
            bits64[i] = (bits64[i + wordShift] >> bitShift) | 
                        (bits64[i + wordShift + 1] << (64 - bitShift));
        }
        bits64[NB64BLOCK - wordShift - 1] = bits64[NB64BLOCK - 1] >> bitShift;
    }

    for (int i = NB64BLOCK - wordShift; i < NB64BLOCK; i++) {
        bits64[i] = 0;
    }
}

// -----------------------------------------------------------------------------

// Mult ------------------------------------------------------------------------

void Int::Mult(Int* a, Int* b) {
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

    for (int i = 0; i < NB64BLOCK; i++) {
        bits64[i] = t[i];
    }
}

// -----------------------------------------------------------------------------

// ModInvFast ------------------------------------------------------------------

void Int::ModInvFast() {
    Int u(*this);
    Int v = MODULO;
    Int b(1);
    Int c(0);

    while (!u.IsOne() && !v.IsOne()) {
        while (u.IsEven()) {
            u.ShiftR(1);
            if (b.IsEven()) {
                b.ShiftR(1);
            } else {
                b.Add(&MODULO);
                b.ShiftR(1);
            }
        }
        
        while (v.IsEven()) {
            v.ShiftR(1);
            if (c.IsEven()) {
                c.ShiftR(1);
            } else {
                c.Add(&MODULO);
                c.ShiftR(1);
            }
        }
        
        if (u.IsGreaterOrEqual(&v)) {
            u.Sub(&v);
            b.Sub(&c);
        } else {
            v.Sub(&u);
            c.Sub(&b);
        }
    }

    if (u.IsOne()) {
        while (b.IsNegative())
            b.Add(&MODULO);
        Set(&b);
    } else {
        while (c.IsNegative())
            c.Add(&MODULO);
        Set(&c);
    }
}

// -----------------------------------------------------------------------------

// ModMulFast ------------------------------------------------------------------

void Int::ModMulFast(Int* a, Int* b) {
    Int p;
    p.Mult(a, b);
    p.ModFast();
    Set(&p);
}

// -----------------------------------------------------------------------------

// ModSquareFast ---------------------------------------------------------------

void Int::ModSquareFast() {
    Int p;
    p.Mult(this, this);
    p.ModFast();
    Set(&p);
}

// -----------------------------------------------------------------------------

// ModFast ---------------------------------------------------------------------

void Int::ModFast() {
    if (IsGreaterOrEqual(&MODULO)) {
        Int r;
        Div(&MODULO, &r);
        Set(&r);
    }
}

// -----------------------------------------------------------------------------

// ModPowFast ------------------------------------------------------------------

void Int::ModPowFast(Int* e) {
    Int base(*this);
    SetOne();

    for (int i = NB64BLOCK - 1; i >= 0; i--) {
        uint64_t mask = (uint64_t)1 << 63;
        for (int j = 0; j < 64; j++) {
            ModSquareFast();
            if (e->bits64[i] & mask) {
                ModMulFast(&base);
            }
            mask >>= 1;
        }
    }
}

// -----------------------------------------------------------------------------

// ModPow2Fast -----------------------------------------------------------------

void Int::ModPow2Fast(Int* e, uint32_t pow2) {
    Int base(*this);
    SetOne();

    for (int i = NB64BLOCK - 1; i >= 0; i--) {
        uint64_t mask = (uint64_t)1 << 63;
        for (int j = 0; j < 64; j++) {
            ModSquareFast();
            if (e->bits64[i] & mask) {
                ModMulFast(&base);
            }
            mask >>= 1;
        }
    }

    // Apply pow2 modulo
    if (pow2 > 0) {
        Int p2;
        p2.SetInt32(1);
        p2.ShiftL(pow2);
        ModFast(&p2);
    }
}

// -----------------------------------------------------------------------------

// SetBaseN --------------------------------------------------------------------

void Int::SetBaseN(const char* value, int base) {
    SetZero();
    Int power(1);
    Int digit;
    Int baseInt;
    baseInt.SetInt32(base);

    const char* ptr = value;
    while (*ptr) ptr++; // Find end of string
    ptr--;

    while (ptr >= value) {
        char c = *ptr;
        if (c >= '0' && c <= '9') {
            digit.SetInt32(c - '0');
        } else if (c >= 'A' && c <= 'Z') {
            digit.SetInt32(c - 'A' + 10);
        } else if (c >= 'a' && c <= 'z') {
            digit.SetInt32(c - 'a' + 10);
        } else {
            digit.SetZero();
        }

        Int tmp;
        tmp.Mult(&digit, &power);
        Add(&tmp);

        if (ptr > value) {
            tmp.Mult(&power, &baseInt);
            power.Set(&tmp);
        }
        ptr--;
    }
}

// -----------------------------------------------------------------------------

// ToString --------------------------------------------------------------------

std::string Int::ToString(int base) {
    if (base < 2 || base > 36) return "0";

    if (IsZero()) return "0";

    Int tmp(*this);
    std::string result;
    Int baseInt;
    baseInt.SetInt32(base);
    Int digit;

    while (!tmp.IsZero()) {
        tmp.Div(&baseInt, &digit);
        uint32_t d = digit.GetInt32();
        char c = (d < 10) ? ('0' + d) : ('A' + d - 10);
        result = c + result;
    }

    return result;
}

// -----------------------------------------------------------------------------
