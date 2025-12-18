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

#ifndef WIN64
#include <stdio.h>
#include <unistd.h>
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "../Timer.h"
#include "../hash/ripemd160.h"
#include "../hash/sha256.h"
#include "GPUBase58.h"
#include "GPUCompute.h"
#include "GPUEngine.h"
#include "GPUGroup.h"
#include "GPUHash.h"
#include "GPUMath.h"
#include "GPUWildcard.h"

// ---------------------------------------------------------------------------------------

__global__ void comp_keys(uint32_t mode, prefix_t *prefix, uint32_t *lookup32, uint64_t *keys, uint32_t maxFound,
                          uint32_t *found) {
  int xPtr = (blockIdx.x * blockDim.x) * 8;
  int yPtr = xPtr + 4 * blockDim.x;
  ComputeKeys(mode, keys + xPtr, keys + yPtr, prefix, lookup32, maxFound, found);
}

__global__ void comp_keys_p2sh(uint32_t mode, prefix_t *prefix, uint32_t *lookup32, uint64_t *keys, uint32_t maxFound,
                               uint32_t *found) {
  int xPtr = (blockIdx.x * blockDim.x) * 8;
  int yPtr = xPtr + 4 * blockDim.x;
  ComputeKeysP2SH(mode, keys + xPtr, keys + yPtr, prefix, lookup32, maxFound, found);
}

__global__ void comp_keys_comp(prefix_t *prefix, uint32_t *lookup32, uint64_t *keys, uint32_t maxFound,
                               uint32_t *found) {
  int xPtr = (blockIdx.x * blockDim.x) * 8;
  int yPtr = xPtr + 4 * blockDim.x;
  ComputeKeysComp(keys + xPtr, keys + yPtr, prefix, lookup32, maxFound, found);
}

__global__ void comp_keys_pattern(uint32_t mode, prefix_t *pattern, uint64_t *keys, uint32_t maxFound,
                                  uint32_t *found) {
  int xPtr = (blockIdx.x * blockDim.x) * 8;
  int yPtr = xPtr + 4 * blockDim.x;
  ComputeKeys(mode, keys + xPtr, keys + yPtr, NULL, (uint32_t *)pattern, maxFound, found);
}

__global__ void comp_keys_p2sh_pattern(uint32_t mode, prefix_t *pattern, uint64_t *keys, uint32_t maxFound,
                                       uint32_t *found) {
  int xPtr = (blockIdx.x * blockDim.x) * 8;
  int yPtr = xPtr + 4 * blockDim.x;
  ComputeKeysP2SH(mode, keys + xPtr, keys + yPtr, NULL, (uint32_t *)pattern, maxFound, found);
}

// #define FULLCHECK
#ifdef FULLCHECK

// ---------------------------------------------------------------------------------------

__global__ void chekc_mult(uint64_t *a, uint64_t *b, uint64_t *r) {
  _ModMult(r, a, b);
  r[4] = 0;
}

// ---------------------------------------------------------------------------------------

__global__ void chekc_hash160(uint64_t *x, uint64_t *y, uint32_t *h) {
  _GetHash160(x, y, (uint8_t *)h);
  _GetHash160Comp(x, y, (uint8_t *)(h + 5));
}

// ---------------------------------------------------------------------------------------

__global__ void get_endianness(uint32_t *endian) {
  uint32_t a = 0x01020304;
  uint8_t fb = *(uint8_t *)(&a);
  *endian = (fb == 0x04);
}

#endif  // FULLCHECK

// ---------------------------------------------------------------------------------------

using namespace std;

std::string toHex(unsigned char *data, int length) {
  string ret;
  char tmp[3];
  for (int i = 0; i < length; i++) {
    if (i && i % 4 == 0) ret.append(" ");
    sprintf(tmp, "%02x", (int)data[i]);
    ret.append(tmp);
  }
  return ret;
}

int _ConvertSMVer2Cores(int major, int minor) {
  // Defines for GPU Architecture types (using the SM version to determine
  // the # of cores per SM
  typedef struct {
    int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
    // and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {{0x30, 192}, {0x32, 192}, {0x35, 192}, {0x37, 192}, {0x50, 128}, {0x52, 128},
                                     {0x53, 128}, {0x60, 64},  {0x61, 128}, {0x62, 128}, {0x70, 64},  {0x72, 64},
                                     {0x75, 64},  {0x80, 64},  {0x86, 128}, {0x87, 128}, {0x89, 128}, {0x90, 128},
                                     {0xa0, 128}, {0xa1, 128}, {0xa3, 128}, {0xb0, 128}, {0xc0, 128}, {0xc1, 128},
                                     {0xd0, 256}, {0xd1, 256}, {0xd2, 256}, {-1, -1}};

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  return 0;
}

GPUEngine::GPUEngine(int nbThreadGroup, int nbThreadPerGroup, int gpuId, uint32_t maxFound, bool rekey) {
  // Initialise CUDA
  this->rekey = rekey;
  this->nbThreadPerGroup = nbThreadPerGroup;
  initialised = false;
  cudaError_t err;

  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (error_id != cudaSuccess) {
    printf("GPUEngine: CudaGetDeviceCount %s %d\n", cudaGetErrorString(error_id), error_id);
    return;
  }

  // This function call returns 0 if there are no CUDA capable devices.
  if (deviceCount == 0) {
    printf("GPUEngine: There are no available device(s) that support CUDA\n");
    return;
  }

  err = cudaSetDevice(gpuId);
  if (err != cudaSuccess) {
    printf("GPUEngine: %s\n", cudaGetErrorString(err));
    return;
  }

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, gpuId);

  if (nbThreadGroup == -1) {
    int groupMultiplier = 8;
    if (deviceProp.major >= 9) groupMultiplier = 16;
    if (deviceProp.major >= 11) groupMultiplier = 20;
    if (deviceProp.major >= 13) groupMultiplier = 24;
    nbThreadGroup = deviceProp.multiProcessorCount * groupMultiplier;
  }

  this->nbThread = nbThreadGroup * nbThreadPerGroup;
  this->maxFound = maxFound;
  this->outputSize = (maxFound * ITEM_SIZE + 4);

  char tmp[512];
  sprintf(tmp, "GPU #%d %s (%dx%d cores) Grid(%dx%d)", gpuId, deviceProp.name, deviceProp.multiProcessorCount,
          _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor), nbThread / nbThreadPerGroup, nbThreadPerGroup);
  deviceName = std::string(tmp);

  // Prefer L1 (We do not use __shared__ at all)
  err = cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  if (err != cudaSuccess) {
    printf("GPUEngine: %s\n", cudaGetErrorString(err));
    return;
  }

  size_t stackSize = 49152;
  err = cudaDeviceSetLimit(cudaLimitStackSize, stackSize);
  if (err != cudaSuccess) {
    printf("GPUEngine: %s\n", cudaGetErrorString(err));
    return;
  }

  /*
  size_t heapSize = ;
  err = cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapSize);
  if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
    exit(0);
  }

  size_t size;
  cudaDeviceGetLimit(&size, cudaLimitStackSize);
  printf("Stack Size %lld\n", size);
  cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize);
  printf("Heap Size %lld\n", size);
  */

  // Allocate memory
  err = cudaMalloc((void **)&inputPrefix, _64K * 2);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate prefix memory: %s\n", cudaGetErrorString(err));
    return;
  }
  err = cudaHostAlloc(&inputPrefixPinned, _64K * 2, cudaHostAllocWriteCombined | cudaHostAllocMapped);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate prefix pinned memory: %s\n", cudaGetErrorString(err));
    return;
  }
  err = cudaMalloc((void **)&inputKey, nbThread * 32 * 2);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate input memory: %s\n", cudaGetErrorString(err));
    return;
  }
  err = cudaHostAlloc(&inputKeyPinned, nbThread * 32 * 2, cudaHostAllocWriteCombined | cudaHostAllocMapped);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate input pinned memory: %s\n", cudaGetErrorString(err));
    return;
  }
  err = cudaMalloc((void **)&outputPrefix, outputSize);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate output memory: %s\n", cudaGetErrorString(err));
    return;
  }
  err = cudaHostAlloc(&outputPrefixPinned, outputSize, cudaHostAllocMapped);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate output pinned memory: %s\n", cudaGetErrorString(err));
    return;
  }

  searchMode = SEARCH_COMPRESSED;
  searchType = P2PKH;
  initialised = true;
  pattern = "";
  hasPattern = false;
  inputPrefixLookUp = NULL;
}

int GPUEngine::GetGroupSize() { return GRP_SIZE; }

void GPUEngine::PrintCudaInfo() {
  cudaError_t err;

  const char *sComputeMode[] = {
      "Multiple host threads", "Only one host thread", "No host thread", "Multiple process threads", "Unknown", NULL};

  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (error_id != cudaSuccess) {
    printf("GPUEngine: CudaGetDeviceCount %s\n", cudaGetErrorString(error_id));
    return;
  }

  // This function call returns 0 if there are no CUDA capable devices.
  if (deviceCount == 0) {
    printf("GPUEngine: There are no available device(s) that support CUDA\n");
    return;
  }

  for (int i = 0; i < deviceCount; i++) {
    err = cudaSetDevice(i);
    if (err != cudaSuccess) {
      printf("GPUEngine: cudaSetDevice(%d) %s\n", i, cudaGetErrorString(err));
      return;
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, i);
    printf("GPU #%d %s (%dx%d cores) (Cap %d.%d) (%.1f MB) (%s)\n", i, deviceProp.name, deviceProp.multiProcessorCount,
           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor), deviceProp.major, deviceProp.minor,
           (double)deviceProp.totalGlobalMem / 1048576.0, sComputeMode[deviceProp.computeMode]);
  }
}

std::vector<int> GPUEngine::GetAutoGpuIds(int limit) {
  std::vector<int> ids;
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (error_id != cudaSuccess || deviceCount == 0) return ids;

  int maxId = (limit < deviceCount) ? limit : deviceCount;
  for (int i = 0; i < maxId; i++) ids.push_back(i);

  return ids;
}

std::pair<int, int> GPUEngine::GetAutoGridSize(int gpuId) {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, gpuId);

  int groupMultiplier = 8;
  if (deviceProp.major >= 9) groupMultiplier = 16;
  if (deviceProp.major >= 11) groupMultiplier = 20;
  if (deviceProp.major >= 13) groupMultiplier = 24;

  int gridX = deviceProp.multiProcessorCount * groupMultiplier;
  int blockDim = (deviceProp.major >= 11) ? 256 : 128;

  // Keep block size within device limits
  if (blockDim > deviceProp.maxThreadsPerBlock) blockDim = deviceProp.maxThreadsPerBlock;

  return std::make_pair(gridX, blockDim);
}

GPUEngine::~GPUEngine() {
  cudaFree(inputKey);
  cudaFree(inputPrefix);
  if (inputPrefixLookUp) cudaFree(inputPrefixLookUp);
  cudaFreeHost(outputPrefixPinned);
  cudaFree(outputPrefix);
}

int GPUEngine::GetNbThread() { return nbThread; }

void GPUEngine::SetSearchMode(int searchMode) { this->searchMode = searchMode; }

void GPUEngine::SetSearchType(int searchType) { this->searchType = searchType; }

void GPUEngine::SetPrefix(std::vector<prefix_t> prefixes) {
  memset(inputPrefixPinned, 0, _64K * 2);
  for (int i = 0; i < (int)prefixes.size(); i++) inputPrefixPinned[prefixes[i]] = 1;

  // Fill device memory
  cudaMemcpy(inputPrefix, inputPrefixPinned, _64K * 2, cudaMemcpyHostToDevice);

  // We do not need the input pinned memory anymore
  cudaFreeHost(inputPrefixPinned);
  inputPrefixPinned = NULL;
  lostWarning = false;

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("GPUEngine: SetPrefix: %s\n", cudaGetErrorString(err));
  }
}

void GPUEngine::SetPattern(const char *pattern) {
  strcpy((char *)inputPrefixPinned, pattern);

  // Fill device memory
  cudaMemcpy(inputPrefix, inputPrefixPinned, _64K * 2, cudaMemcpyHostToDevice);

  // We do not need the input pinned memory anymore
  cudaFreeHost(inputPrefixPinned);
  inputPrefixPinned = NULL;
  lostWarning = false;

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("GPUEngine: SetPattern: %s\n", cudaGetErrorString(err));
  }
}

void GPUEngine::SetPrefix(std::vector<LPREFIX> prefixes, uint32_t totalPrefix) {
  // Allocate memory for the second level of lookup tables
  cudaError_t err = cudaMalloc((void **)&inputPrefixLookUp, (_64K + totalPrefix) * 4);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate prefix lookup memory: %s\n", cudaGetErrorString(err));
    return;
  }
  err = cudaHostAlloc(&inputPrefixLookUpPinned, (_64K + totalPrefix) * 4,
                      cudaHostAllocWriteCombined | cudaHostAllocMapped);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate prefix lookup pinned memory: %s\n", cudaGetErrorString(err));
    return;
  }

  uint32_t offset = _64K;
  memset(inputPrefixPinned, 0, _64K * 2);
  memset(inputPrefixLookUpPinned, 0, _64K * 4);
  for (int i = 0; i < (int)prefixes.size(); i++) {
    int nbLPrefix = (int)prefixes[i].lPrefixes.size();
    inputPrefixPinned[prefixes[i].sPrefix] = (uint16_t)nbLPrefix;
    inputPrefixLookUpPinned[prefixes[i].sPrefix] = offset;
    for (int j = 0; j < nbLPrefix; j++) {
      inputPrefixLookUpPinned[offset++] = prefixes[i].lPrefixes[j];
    }
  }

  if (offset != (_64K + totalPrefix)) {
    printf("GPUEngine: Wrong totalPrefix %d!=%d!\n", offset - _64K, totalPrefix);
    return;
  }

  // Fill device memory
  cudaMemcpy(inputPrefix, inputPrefixPinned, _64K * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(inputPrefixLookUp, inputPrefixLookUpPinned, (_64K + totalPrefix) * 4, cudaMemcpyHostToDevice);

  // We do not need the input pinned memory anymore
  cudaFreeHost(inputPrefixPinned);
  inputPrefixPinned = NULL;
  cudaFreeHost(inputPrefixLookUpPinned);
  inputPrefixLookUpPinned = NULL;
  lostWarning = false;

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("GPUEngine: SetPrefix (large): %s\n", cudaGetErrorString(err));
  }
}

bool GPUEngine::callKernel() {
  // Reset nbFound
  cudaMemset(outputPrefix, 0, 4);

  // Call the kernel (Perform STEP_SIZE keys per thread)
  if (searchType == P2SH) {
    if (hasPattern) {
      comp_keys_p2sh_pattern<<<nbThread / nbThreadPerGroup, nbThreadPerGroup>>>(searchMode, inputPrefix, inputKey,
                                                                                maxFound, outputPrefix);
    } else {
      comp_keys_p2sh<<<nbThread / nbThreadPerGroup, nbThreadPerGroup>>>(searchMode, inputPrefix, inputPrefixLookUp,
                                                                        inputKey, maxFound, outputPrefix);
    }

  } else {
    // P2PKH or BECH32
    if (hasPattern) {
      if (searchType == BECH32) {
        // TODO
        printf("GPUEngine: (TODO) BECH32 not yet supported with wildard\n");
        return false;
      }
      comp_keys_pattern<<<nbThread / nbThreadPerGroup, nbThreadPerGroup>>>(searchMode, inputPrefix, inputKey, maxFound,
                                                                           outputPrefix);
    } else {
      if (searchMode == SEARCH_COMPRESSED) {
        comp_keys_comp<<<nbThread / nbThreadPerGroup, nbThreadPerGroup>>>(inputPrefix, inputPrefixLookUp, inputKey,
                                                                          maxFound, outputPrefix);
      } else {
        comp_keys<<<nbThread / nbThreadPerGroup, nbThreadPerGroup>>>(searchMode, inputPrefix, inputPrefixLookUp,
                                                                     inputKey, maxFound, outputPrefix);
      }
    }
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("GPUEngine: Kernel: %s\n", cudaGetErrorString(err));
    return false;
  }
  return true;
}

bool GPUEngine::SetKeys(Point *p) {
  // Sets the starting keys for each thread
  // p must contains nbThread public keys
  for (int i = 0; i < nbThread; i += nbThreadPerGroup) {
    for (int j = 0; j < nbThreadPerGroup; j++) {
      inputKeyPinned[8 * i + j + 0 * nbThreadPerGroup] = p[i + j].x.bits64[0];
      inputKeyPinned[8 * i + j + 1 * nbThreadPerGroup] = p[i + j].x.bits64[1];
      inputKeyPinned[8 * i + j + 2 * nbThreadPerGroup] = p[i + j].x.bits64[2];
      inputKeyPinned[8 * i + j + 3 * nbThreadPerGroup] = p[i + j].x.bits64[3];

      inputKeyPinned[8 * i + j + 4 * nbThreadPerGroup] = p[i + j].y.bits64[0];
      inputKeyPinned[8 * i + j + 5 * nbThreadPerGroup] = p[i + j].y.bits64[1];
      inputKeyPinned[8 * i + j + 6 * nbThreadPerGroup] = p[i + j].y.bits64[2];
      inputKeyPinned[8 * i + j + 7 * nbThreadPerGroup] = p[i + j].y.bits64[3];
    }
  }

  // Fill device memory
  cudaMemcpy(inputKey, inputKeyPinned, nbThread * 64 * sizeof(uint32_t), cudaMemcpyHostToDevice);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("GPUEngine: SetKeys: %s\n", cudaGetErrorString(err));
    return false;
  }

  return true;
}

void GPUEngine::SetSearchMode(int searchMode) { this->searchMode = searchMode; }

void GPUEngine::SetSearchType(int searchType) { this->searchType = searchType; }

bool GPUEngine::CheckHash(uint8_t *h, std::vector<ITEM> &found, int tid, int incr, int endo, int *ok) {
  bool done = false;

  if (littleEndian) {
    if (h[19] != 0) {
      *ok = (h[19] == (uint8_t)(h[18] >> 4));
      done = true;
    } else if (h[18] != 0) {
      *ok = (h[18] == (uint8_t)(h[17] >> 4));
      done = true;
    } else if (h[17] != 0) {
      *ok = (h[17] == (uint8_t)(h[16] >> 4));
      done = true;
    } else if (h[16] != 0) {
      *ok = (h[16] == (uint8_t)(h[15] >> 4));
      done = true;
    }
  } else {
    if (h[0] != 0) {
      *ok = (h[0] == (uint8_t)(h[1] >> 4));
      done = true;
    } else if (h[1] != 0) {
      *ok = (h[1] == (uint8_t)(h[2] >> 4));
      done = true;
    } else if (h[2] != 0) {
      *ok = (h[2] == (uint8_t)(h[3] >> 4));
      done = true;
    } else if (h[3] != 0) {
      *ok = (h[3] == (uint8_t)(h[4] >> 4));
      done = true;
    }
  }

  if (done) {
    ITEM it;
    it.thId = tid;
    it.incr = incr;
    it.endo = endo;
    it.hash = h;
    it.mode = false;
    found.push_back(it);
  }

  return done;
}

bool GPUEngine::Launch(std::vector<ITEM> &prefixFound, bool spinWait) {
  if (!initialised) return false;

  // Launch kernel
  if (!callKernel()) return false;

  // Wait that the kernel finish
  if (spinWait) {
    while (cudaEventQuery(NULL) == cudaErrorNotReady) {
    }
  } else {
    cudaDeviceSynchronize();
  }

  // Retrieve results
  if (cudaMemcpy(outputPrefixPinned, outputPrefix, outputSize, cudaMemcpyDeviceToHost) != cudaSuccess) {
    printf("GPUEngine: cudaMemcpy %s\n", cudaGetErrorString(cudaGetLastError()));
    return false;
  }

  // Number of prefix found
  int nbFound = outputPrefixPinned[0];
  if (nbFound > (int)maxFound) {
    nbFound = maxFound;
    if (!lostWarning) {
      printf("GPUEngine: Warning too many prefixes found in one kernel call, some are lost...\n");
      lostWarning = true;
    }
  }

  // Check hash
  littleEndian = ((outputPrefixPinned[1] & 0x1) == 0);
  for (int i = 0; i < nbFound; i++) {
    int tid = outputPrefixPinned[(i * ITEM_SIZE32) + 2];
    int incr = outputPrefixPinned[(i * ITEM_SIZE32) + 3];
    int endo = (int)(outputPrefixPinned[(i * ITEM_SIZE32) + 4]);
    uint8_t *h = (uint8_t *)(outputPrefixPinned + i * ITEM_SIZE32 + 5);

    // Little endian
    if (littleEndian) {
      if (!CheckHash(h, prefixFound, tid, incr, endo, &outputPrefixPinned[(i * ITEM_SIZE32) + 1])) {
        printf("GPUEngine: Unknown little endian:  %02X %02X %02X %02X\n", h[19], h[18], h[17], h[16]);
        return false;
      }
    } else {
      if (!CheckHash(h, prefixFound, tid, incr, endo, &outputPrefixPinned[(i * ITEM_SIZE32) + 1])) {
        printf("GPUEngine: Unknown big endian: %02X %02X %02X %02X\n", h[0], h[1], h[2], h[3]);
        return false;
      }
    }
  }

  return true;
}

bool GPUEngine::Check(Secp256K1 *secp) {
  uint8_t h[21];
  int nbCheck = 100;
  int nbWork = nbThread * 8;
  int ok;
  int wrong = 0;

  printf("Check GPU (search mode %s)...\n", searchModes[searchMode]);

  // Trivial pattern
  if (searchMode != SEARCH_BOTH) {
    // Prefix AAAA or aaa
    memset(inputPrefixPinned, 0, _64K * 2);
    inputPrefixPinned[0xAAAA] = 2;
    inputPrefixPinned[0xaaaa] = 2;
    cudaMemcpy(inputPrefix, inputPrefixPinned, _64K * 2, cudaMemcpyHostToDevice);
    cudaFreeHost(inputPrefixPinned);
    inputPrefixPinned = NULL;

    memset(outputPrefixPinned, 0, outputSize);
    cudaMemcpy(outputPrefix, outputPrefixPinned, 4, cudaMemcpyHostToDevice);

    // Dummy input key
    for (int i = 0; i < nbThread; i++) {
      inputKeyPinned[0 * nbThread + i] = 0;
      inputKeyPinned[1 * nbThread + i] = 0;
      inputKeyPinned[2 * nbThread + i] = 0;
      inputKeyPinned[3 * nbThread + i] = 0;
      inputKeyPinned[4 * nbThread + i] = 0;
      inputKeyPinned[5 * nbThread + i] = 0;
      inputKeyPinned[6 * nbThread + i] = 0;
      inputKeyPinned[7 * nbThread + i] = 0;
    }
    cudaMemcpy(inputKey, inputKeyPinned, nbThread * 64 * sizeof(uint32_t), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    // Launch kernel
    if (!callKernel()) return false;

    cudaMemcpy(outputPrefixPinned, outputPrefix, outputSize, cudaMemcpyDeviceToHost);

    // Check result
    int nbFound = outputPrefixPinned[0];
    printf("Nb found %d\n", nbFound);
    for (int i = 0; i < nbFound; i++) {
      int tid = outputPrefixPinned[(i * ITEM_SIZE32) + 2];
      int incr = outputPrefixPinned[(i * ITEM_SIZE32) + 3];
      int endo = (int)(outputPrefixPinned[(i * ITEM_SIZE32) + 4]);
      uint8_t *h = (uint8_t *)(outputPrefixPinned + i * ITEM_SIZE32 + 5);

      Int k0((uint64_t)tid + incr);
      k0.Add((uint64_t)(nbThread / 2));

      if (searchMode == SEARCH_COMPRESSED)
        sha256Ripemd160(h, &secp->ComputePublicKey(&k0).GetCompKey()[0], 33);
      else
        sha256Ripemd160(h, &secp->ComputePublicKey(&k0).GetKey()[0], 65);

      CheckHash(h, checkRes, tid, incr, endo, &ok);

      if (ok != outputPrefixPinned[(i * ITEM_SIZE32) + 1]) {
        printf("Error in kernel output\n");
        return false;
      }
    }

    cudaFreeHost(outputPrefixPinned);
    return true;
  }

  // Create dummy input
  for (int i = 0; i < nbThread; i++) {
    Int k;
    k.SetInt32(i);

    Point p0 = secp->ComputePublicKey(&k);
    Point p1 = secp->ComputePublicKey(&secp->lambda);
    Point p2 = secp->ComputePublicKey(&secp->_lambda);
    p1 = secp->AddDirect(p1, p0);
    p2 = secp->AddDirect(p2, p0);

    inputKeyPinned[0 * nbThread + i] = p0.x.bits64[0];
    inputKeyPinned[1 * nbThread + i] = p0.x.bits64[1];
    inputKeyPinned[2 * nbThread + i] = p0.x.bits64[2];
    inputKeyPinned[3 * nbThread + i] = p0.x.bits64[3];

    inputKeyPinned[4 * nbThread + i] = p0.y.bits64[0];
    inputKeyPinned[5 * nbThread + i] = p0.y.bits64[1];
    inputKeyPinned[6 * nbThread + i] = p0.y.bits64[2];
    inputKeyPinned[7 * nbThread + i] = p0.y.bits64[3];

    inputKeyPinned[8 * nbThread + i] = p1.x.bits64[0];
    inputKeyPinned[9 * nbThread + i] = p1.x.bits64[1];
    inputKeyPinned[10 * nbThread + i] = p1.x.bits64[2];
    inputKeyPinned[11 * nbThread + i] = p1.x.bits64[3];

    inputKeyPinned[12 * nbThread + i] = p1.y.bits64[0];
    inputKeyPinned[13 * nbThread + i] = p1.y.bits64[1];
    inputKeyPinned[14 * nbThread + i] = p1.y.bits64[2];
    inputKeyPinned[15 * nbThread + i] = p1.y.bits64[3];

    inputKeyPinned[16 * nbThread + i] = p2.x.bits64[0];
    inputKeyPinned[17 * nbThread + i] = p2.x.bits64[1];
    inputKeyPinned[18 * nbThread + i] = p2.x.bits64[2];
    inputKeyPinned[19 * nbThread + i] = p2.x.bits64[3];

    inputKeyPinned[20 * nbThread + i] = p2.y.bits64[0];
    inputKeyPinned[21 * nbThread + i] = p2.y.bits64[1];
    inputKeyPinned[22 * nbThread + i] = p2.y.bits64[2];
    inputKeyPinned[23 * nbThread + i] = p2.y.bits64[3];
  }

  memset(outputPrefixPinned, 0, outputSize);
  cudaMemcpy(outputPrefix, outputPrefixPinned, 4, cudaMemcpyHostToDevice);

  cudaMemcpy(inputKey, inputKeyPinned, nbThread * 64 * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(inputPrefix, inputPrefixPinned, _64K * 2, cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();

  // Launch kernel and check hash
  wrong = 0;
  for (int i = 0; i < nbCheck; i++) {
    if (!callKernel()) return false;

    cudaMemcpy(outputPrefixPinned, outputPrefix, outputSize, cudaMemcpyDeviceToHost);

    // Number of prefix found
    int nbFound = outputPrefixPinned[0];

    if (nbFound > (int)maxFound) {
      nbFound = maxFound;
      printf("GPUEngine: Warning too many prefixes found in one kernel call, some are lost...\n");
    }

    for (int j = 0; j < nbFound; j++) {
      int tid = outputPrefixPinned[(j * ITEM_SIZE32) + 2];
      int incr = outputPrefixPinned[(j * ITEM_SIZE32) + 3];
      int endo = (int)(outputPrefixPinned[(j * ITEM_SIZE32) + 4]);
      uint8_t *h0 = (uint8_t *)(outputPrefixPinned + j * ITEM_SIZE32 + 5);

      memset(h, 0, sizeof(h));
      if (searchMode == SEARCH_COMPRESSED)
        sha256Ripemd160(h, &secp->ComputePublicKey(&secp->Add(secp->_2Gn[incr], tid)).GetCompKey()[0], 33);
      else
        sha256Ripemd160(h, &secp->ComputePublicKey(&secp->Add(secp->_2Gn[incr], tid)).GetKey()[0], 65);

      int ok = 0;

      if (littleEndian) {
        if (h[19] != 0) {
          ok = (h[19] == (uint8_t)(h[18] >> 4));
        } else if (h[18] != 0) {
          ok = (h[18] == (uint8_t)(h[17] >> 4));
        } else if (h[17] != 0) {
          ok = (h[17] == (uint8_t)(h[16] >> 4));
        } else if (h[16] != 0) {
          ok = (h[16] == (uint8_t)(h[15] >> 4));
        }
      } else {
        if (h[0] != 0) {
          ok = (h[0] == (uint8_t)(h[1] >> 4));
        } else if (h[1] != 0) {
          ok = (h[1] == (uint8_t)(h[2] >> 4));
        } else if (h[2] != 0) {
          ok = (h[2] == (uint8_t)(h[3] >> 4));
        } else if (h[3] != 0) {
          ok = (h[3] == (uint8_t)(h[4] >> 4));
        }
      }

      if (littleEndian) {
        if (h[19] != h0[19] || h[18] != h0[18] || h[17] != h0[17]) {
          wrong++;
        }
      } else {
        if (h[0] != h0[0] || h[1] != h0[1] || h[2] != h0[2] || h[3] != h0[3]) {
          wrong++;
        }
      }

      if (ok != outputPrefixPinned[(j * ITEM_SIZE32) + 1]) {
        wrong++;
      }
    }
  }

  if (wrong == 0)
    printf("Check OK\n");
  else
    printf("Check NOK (%d)\n", wrong);

  cudaFreeHost(outputPrefixPinned);
  cudaFreeHost(inputPrefixPinned);

  return (wrong == 0);
}

// ---------------------------------------------------------------------------------------

void GPUEngine::ComputeIndex(std::vector<int> &s, int depth, int n) {
  if (depth == 1) {
    for (int i = 0; i < n; i++) s.push_back(i);
  } else {
    int j = (depth & 1) ? 1 : 0;
    while (j < n) {
      ComputeIndex(s, depth - 1, n);
      s.push_back(j);
      j += 2;
    }
  }
}

// ---------------------------------------------------------------------------------------

void GPUEngine::Browse(FILE *f, int depth, int max, int s) {
  static std::vector<int> st;
  if (depth == 0) {
    fprintf(f, "  {%d,%d,%d,%d},\n", st[0], st[1], st[2], st[3]);

  } else {
    for (int i = 0; i <= max; i++) {
      st.push_back(i);
      Browse(f, depth - 1, s, s);
      st.pop_back();

      if (s > 0) {
        s--;
      } else if (max == 15) {
        s = 14;
      } else if (max == 31) {
        s = 30;
      }
    }
  }
}

// ---------------------------------------------------------------------------------------

void GPUEngine::GenerateCode(Secp256K1 *secp, int size) {
  FILE *f = fopen("GPU/GPUGroup.h", "w");
  fprintf(f, "// File generated by GPUEngine::GenerateCode()\n");
  fprintf(f, "// GROUP definitions\n");
  fprintf(f, "#define GRP_SIZE %d\n\n", size);
  fprintf(f, "// _2Gn = GRP_SIZE*G\n");
  fprintf(f, "__device__ __constant__ uint64_t _2Gnx[4] = {%s};\n",
          secp->GetPublicKeyHex(true, secp->Multiply(secp->G, GRP_SIZE)).c_str());
  fprintf(f, "__device__ __constant__ uint64_t _2Gny[4] = {%s};\n\n",
          secp->GetPublicKeyHex(true, secp->Multiply(secp->G, GRP_SIZE)).substr(69).c_str());

  fprintf(f, "// SecpK1 Generator table (Contains G,2G,3G,...,(GRP_SIZE/2 )G)\n");
  fprintf(f, "__device__ __constant__ uint64_t Gx[][4] = {\n");
  for (int i = 0; i < size / 2; i++) {
    string s = secp->GetPublicKeyHex(true, secp->Multiply(secp->G, i + 1));
    fprintf(f, "  {%s},\n", s.substr(0, 69).c_str());
  }
  fprintf(f, "};\n\n");

  fprintf(f, "__device__ __constant__ uint64_t Gy[][4] = {\n");
  for (int i = 0; i < size / 2; i++) {
    string s = secp->GetPublicKeyHex(true, secp->Multiply(secp->G, i + 1));
    fprintf(f, "  {%s},\n", s.substr(69).c_str());
  }
  fprintf(f, "};\n\n");

  fprintf(f, "// Browse index\n");
  fprintf(f, "__device__ __constant__ uint32_t BrowseIndex[] = {\n");
  std::vector<int> s;
  ComputeIndex(s, 5, 16);
  fprintf(f, "  ");
  for (int i = 0; i < (int)s.size(); i++) {
    fprintf(f, "%d", s[i]);
    if ((i & 3) == 3)
      fprintf(f, ",\n  ");
    else
      fprintf(f, ",");
  }
  fprintf(f, "\n};\n");

  fprintf(f, "// Browse index Compact\n");
  fprintf(f, "__device__ __constant__ uint32_t BrowseIndexCompact[] = {\n");
  s.clear();
  ComputeIndex(s, 4, 32);
  fprintf(f, "  ");
  for (int i = 0; i < (int)s.size(); i++) {
    fprintf(f, "%d", s[i]);
    if ((i & 3) == 3)
      fprintf(f, ",\n  ");
    else
      fprintf(f, ",");
  }
  fprintf(f, "\n};\n");

  fclose(f);
}

// ---------------------------------------------------------------------------------------
