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

__device__ __noinline__ void CheckPoint(uint32_t *_h, int32_t incr, int32_t endo, int32_t mode, 
                                       prefix_t *prefix, uint32_t *lookup32, uint32_t maxFound, 
                                       uint32_t *out, int type) {

  uint32_t tid = (blockIdx.x*blockDim.x) + threadIdx.x;
  
  if (prefix == NULL) {
    char add[48];
    _GetAddress(type, _h, add);
    if (_Match(add, (char *)lookup32)) {
      uint32_t pos = atomicAdd(out, 1);
      if (pos < maxFound) {
        out[pos*ITEM_SIZE32 + 1] = tid;
        out[pos*ITEM_SIZE32 + 2] = (uint32_t)(incr << 16) | (uint32_t)(mode << 15) | (uint32_t)(endo);
        out[pos*ITEM_SIZE32 + 3] = _h[0];
        out[pos*ITEM_SIZE32 + 4] = _h[1];
        out[pos*ITEM_SIZE32 + 5] = _h[2];
        out[pos*ITEM_SIZE32 + 6] = _h[3];
        out[pos*ITEM_SIZE32 + 7] = _h[4];
      }
    }
  } else {
    prefix_t pr0 = *(prefix_t *)(_h);
    prefix_t hit = prefix[pr0];

    if (hit) {
      uint32_t off = lookup32[pr0];
      prefixl_t l32 = _h[0];
      
      uint32_t st = off;
      uint32_t ed = off + hit - 1;
      while (st <= ed) {
        uint32_t mi = (st + ed) / 2;
        prefixl_t lmi = lookup32[mi];
        if (l32 < lmi) {
          ed = mi - 1;
        } else if (l32 == lmi) {
          uint32_t pos = atomicAdd(out, 1);
          if (pos < maxFound) {
            out[pos*ITEM_SIZE32 + 1] = tid;
            out[pos*ITEM_SIZE32 + 2] = (uint32_t)(incr << 16) | (uint32_t)(mode << 15) | (uint32_t)(endo);
            out[pos*ITEM_SIZE32 + 3] = _h[0];
            out[pos*ITEM_SIZE32 + 4] = _h[1];
            out[pos*ITEM_SIZE32 + 5] = _h[2];
            out[pos*ITEM_SIZE32 + 6] = _h[3];
            out[pos*ITEM_SIZE32 + 7] = _h[4];
          }
          break;
        } else {
          st = mi + 1;
        }
      }
    }
  }
}

#define CHECK_POINT(_h,incr,endo,mode)  CheckPoint(_h,incr,endo,mode,prefix,lookup32,maxFound,out,P2PKH)
#define CHECK_POINT_P2SH(_h,incr,endo,mode)  CheckPoint(_h,incr,endo,mode,prefix,lookup32,maxFound,out,P2SH)

__device__ __noinline__ void CheckHashCompressed(prefix_t *prefix, uint64_t *px, uint8_t isOdd, 
                                                int32_t incr, uint32_t *lookup32, uint32_t maxFound, 
                                                uint32_t *out, bool isP2SH) {
  uint32_t h[5];
  
  if (isP2SH) {
    _GetHash160P2SHComp(px, isOdd, (uint8_t *)h);
    CHECK_POINT_P2SH(h, incr, 0, true);
  } else {
    _GetHash160Comp(px, isOdd, (uint8_t *)h);
    CHECK_POINT(h, incr, 0, true);
  }
}

__device__ __noinline__ void CheckHashUncompressed(prefix_t *prefix, uint64_t *px, uint64_t *py, 
                                                  int32_t incr, uint32_t *lookup32, uint32_t maxFound, 
                                                  uint32_t *out, bool isP2SH) {
  uint32_t h[5];
  
  if (isP2SH) {
    _GetHash160P2SHUncomp(px, py, (uint8_t *)h);
    CHECK_POINT_P2SH(h, incr, 0, false);
  } else {
    _GetHash160(px, py, (uint8_t *)h);
    CHECK_POINT(h, incr, 0, false);
  }
}

__device__ __noinline__ void CheckHash(uint32_t mode, prefix_t *prefix, uint64_t *px, uint64_t *py, 
                                      int32_t incr, uint32_t *lookup32, uint32_t maxFound, 
                                      uint32_t *out, bool isP2SH) {
  switch (mode) {
    case SEARCH_COMPRESSED:
      CheckHashCompressed(prefix, px, (uint8_t)(py[0] & 1), incr, lookup32, maxFound, out, isP2SH);
      break;
    case SEARCH_UNCOMPRESSED:
      CheckHashUncompressed(prefix, px, py, incr, lookup32, maxFound, out, isP2SH);
      break;
    case SEARCH_BOTH:
      CheckHashCompressed(prefix, px, (uint8_t)(py[0] & 1), incr, lookup32, maxFound, out, isP2SH);
      CheckHashUncompressed(prefix, px, py, incr, lookup32, maxFound, out, isP2SH);
      break;
  }
}

#define CHECK_PREFIX(incr) CheckHash(mode, sPrefix, px, py, j*GRP_SIZE + (incr), lookup32, maxFound, out, false)
#define CHECK_PREFIX_P2SH(incr) CheckHash(mode, sPrefix, px, py, j*GRP_SIZE + (incr), lookup32, maxFound, out, true)

__device__ void ComputeKeys(uint32_t mode, uint64_t *startx, uint64_t *starty, 
                           prefix_t *sPrefix, uint32_t *lookup32, uint32_t maxFound, 
                           uint32_t *out, bool isP2SH) {

  uint64_t dx[GRP_SIZE/2+1][4];
  uint64_t px[4];
  uint64_t py[4];
  uint64_t pyn[4];
  uint64_t sx[4];
  uint64_t sy[4];
  uint64_t dy[4];
  uint64_t _s[4];
  uint64_t _p2[4];
  char pattern[48];

  Load256A(sx, startx);
  Load256A(sy, starty);
  Load256(px, sx);
  Load256(py, sy);

  if (sPrefix == NULL) {
    memcpy(pattern, lookup32, 48);
    lookup32 = (uint32_t *)pattern;
  }

  for (uint32_t j = 0; j < STEP_SIZE / GRP_SIZE; j++) {
    uint32_t i;
    for (i = 0; i < HSIZE; i++)
      ModSub256(dx[i], Gx[i], sx);
    ModSub256(dx[i], Gx[i], sx);
    ModSub256(dx[i+1], _2Gnx, sx);

    _ModInvGrouped(dx);

    if (isP2SH) {
      CHECK_PREFIX_P2SH(GRP_SIZE / 2);
    } else {
      CHECK_PREFIX(GRP_SIZE / 2);
    }

    ModNeg256(pyn, py);

    for (i = 0; i < HSIZE; i++) {
      Load256(px, sx);
      Load256(py, sy);
      ModSub256(dy, Gy[i], py);

      _ModMult(_s, dy, dx[i]);
      _ModSqr(_p2, _s);
      ModSub256(px, _p2, px);
      ModSub256(px, Gx[i]);

      ModSub256(py, Gx[i], px);
      _ModMult(py, _s);
      ModSub256(py, Gy[i]);

      if (isP2SH) {
        CHECK_PREFIX_P2SH(GRP_SIZE / 2 + (i + 1));
      } else {
        CHECK_PREFIX(GRP_SIZE / 2 + (i + 1));
      }

      Load256(px, sx);
      ModSub256(dy, pyn, Gy[i]);

      _ModMult(_s, dy, dx[i]);
      _ModSqr(_p2, _s);
      ModSub256(px, _p2, px);
      ModSub256(px, Gx[i]);

      ModSub256(py, Gx[i], px);
      _ModMult(py, _s);
      ModAdd256(py, Gy[i]);

      if (isP2SH) {
        CHECK_PREFIX_P2SH(GRP_SIZE / 2 - (i + 1));
      } else {
        CHECK_PREFIX(GRP_SIZE / 2 - (i + 1));
      }
    }

    Load256(px, sx);
    Load256(py, sy);
    ModNeg256(dy, Gy[i]);
    ModSub256(dy, py);

    _ModMult(_s, dy, dx[i]);
    _ModSqr(_p2, _s);
    ModSub256(px, _p2, px);
    ModSub256(px, Gx[i]);

    if (isP2SH) {
      CHECK_PREFIX_P2SH(0);
    } else {
      CHECK_PREFIX(0);
    }

    i++;

    Load256(px, sx);
    Load256(py, sy);
    ModSub256(dy, _2Gny, py);

    _ModMult(_s, dy, dx[i]);
    _ModSqr(_p2, _s);
    ModSub256(px, _p2, px);
    ModSub256(px, _2Gnx);

    ModSub256(py, _2Gnx, px);
    _ModMult(py, _s);
    ModSub256(py, _2Gny);
  }

  __syncthreads();
  Store256A(startx, px);
  Store256A(starty, py);
}

__device__ void ComputeKeysP2PKH(uint32_t mode, uint64_t *startx, uint64_t *starty, 
                                prefix_t *sPrefix, uint32_t *lookup32, uint32_t maxFound, 
                                uint32_t *out) {
  ComputeKeys(mode, startx, starty, sPrefix, lookup32, maxFound, out, false);
}

__device__ void ComputeKeysP2SH(uint32_t mode, uint64_t *startx, uint64_t *starty, 
                               prefix_t *sPrefix, uint32_t *lookup32, uint32_t maxFound, 
                               uint32_t *out) {
  ComputeKeys(mode, startx, starty, sPrefix, lookup32, maxFound, out, true);
}

__device__ void ComputeKeysComp(uint64_t *startx, uint64_t *starty, prefix_t *sPrefix, 
                               uint32_t *lookup32, uint32_t maxFound, uint32_t *out) {

  uint64_t dx[GRP_SIZE/2+1][4];
  uint64_t px[4];
  uint64_t py[4];
  uint64_t pyn[4];
  uint64_t sx[4];
  uint64_t sy[4];
  uint64_t dy[4];
  uint64_t _s[4];
  uint64_t _p2[4];
  uint32_t h1[5], h2[5];
  uint64_t pe1x[4], pe2x[4];

  Load256A(sx, startx);
  Load256A(sy, starty);
  Load256(px, sx);
  Load256(py, sy);

  for (uint32_t j = 0; j < STEP_SIZE / GRP_SIZE; j++) {
    uint32_t i;
    for (i = 0; i < HSIZE; i++)
      ModSub256(dx[i], Gx[i], sx);
    ModSub256(dx[i], Gx[i], sx);
    ModSub256(dx[i+1], _2Gnx, sx);

    _ModInvGrouped(dx);

    CHECK_P2PKH_POINT(j*GRP_SIZE + (GRP_SIZE/2));

    ModNeg256(pyn, py);

    for (i = 0; i < HSIZE; i++) {
      Load256(px, sx);
      Load256(py, sy);
      ModSub256(dy, Gy[i], py);

      _ModMult(_s, dy, dx[i]);
      _ModSqr(_p2, _s);
      ModSub256(px, _p2, px);
      ModSub256(px, Gx[i]);

      __syncthreads();
      CHECK_P2PKH_POINT(j*GRP_SIZE + (GRP_SIZE/2 + (i + 1)));

      Load256(px, sx);
      ModSub256(dy, pyn, Gy[i]);

      _ModMult(_s, dy, dx[i]);
      _ModSqr(_p2, _s);
      ModSub256(px, _p2, px);
      ModSub256(px, Gx[i]);

      __syncthreads();
      CHECK_P2PKH_POINT(j*GRP_SIZE + (GRP_SIZE/2 - (i + 1)));
    }

    Load256(px, sx);
    Load256(py, sy);
    ModNeg256(dy, Gy[i]);
    ModSub256(dy, py);

    _ModMult(_s, dy, dx[i]);
    _ModSqr(_p2, _s);
    ModSub256(px, _p2, px);
    ModSub256(px, Gx[i]);

    CHECK_P2PKH_POINT(j*GRP_SIZE + (0));

    i++;

    Load256(px, sx);
    Load256(py, sy);
    ModSub256(dy, _2Gny, py);

    _ModMult(_s, dy, dx[i]);
    _ModSqr(_p2, _s);
    ModSub256(px, _p2, px);
    ModSub256(px, _2Gnx);

    ModSub256(py, _2Gnx, px);
    _ModMult(py, _s);
    ModSub256(py, _2Gny);
  }

  __syncthreads();
  Store256A(startx, px);
  Store256A(starty, py);
}
