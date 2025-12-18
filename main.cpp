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

#include <string.h>

#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "SECP256k1.h"
#include "Timer.h"
#include "Vanity.h"
#include "hash/sha256.h"
#include "hash/sha512.h"

#define RELEASE "1.19"

using namespace std;

// ------------------------------------------------------------------------------------------

void printUsage() {
  printf("VanitySeacrh [-check] [-v] [-u] [-b] [-c] [-gpu] [-stop] [-i inputfile]\n");
  printf("             [-gpuId gpuId1[,gpuId2,...]] [-g g1x,g1y,[,g2x,g2y,...]]\n");
  printf("             [-o outputfile] [-m maxFound] [-ps seed] [-s seed] [-t nbThread]\n");
  printf("             [-nosse] [-r rekey] [-check] [-kp] [-sp startPubKey]\n");
  printf("             [-rp privkey partialkeyfile] [prefix]\n\n");
  printf(" prefix: prefix to search (Can contains wildcard '?' or '*')\n");
  printf(" -v: Print version\n");
  printf(" -u: Search uncompressed addresses\n");
  printf(" -b: Search both uncompressed or compressed addresses\n");
  printf(" -c: Case unsensitive search\n");
  printf(" -gpu: Enable gpu calculation\n");
  printf(" -stop: Stop when all prefixes are found\n");
  printf(" -i inputfile: Get list of prefixes to search from specified file\n");
  printf(" -o outputfile: Output results to the specified file\n");
  printf(" -gpu gpuId1,gpuId2,...: List of GPU(s) to use, default is 0\n");
  printf(" -g g1x,g1y,g2x,g2y, ...: Specify GPU(s) kernel gridsize, default is 8*(MP number),128\n");
  printf(" -m: Specify maximun number of prefixes found by each kernel call\n");
  printf(" -s seed: Specify a seed for the base key, default is random\n");
  printf(" -ps seed: Specify a seed concatened with a crypto secure random seed\n");
  printf(" -t threadNumber: Specify number of CPU thread, default is number of core\n");
  printf(" -nosse: Disable SSE hash function\n");
  printf(" -l: List cuda enabled devices\n");
  printf(" -check: Check CPU and GPU kernel vs CPU\n");
  printf(" -cp privKey: Compute public key (privKey in hex hormat)\n");
  printf(" -ca pubKey: Compute address (pubKey in hex hormat)\n");
  printf(" -kp: Generate key pair\n");
  printf(" -rp privkey partialkeyfile: Reconstruct final private key(s) from partial key(s) info.\n");
  printf(" -sp startPubKey: Start the search with a pubKey (for private key splitting)\n");
  printf(" -r rekey: Rekey interval in MegaKey, default is disabled\n");
  exit(0);
}

// ------------------------------------------------------------------------------------------

int getInt(string name, char *v) {
  int r;

  try {
    r = std::stoi(string(v));

  } catch (std::invalid_argument &) {
    printf("Invalid %s argument, number expected\n", name.c_str());
    exit(-1);
  }

  return r;
}

// ------------------------------------------------------------------------------------------

void getInts(string name, vector<int> &tokens, const string &text, char sep) {
  size_t start = 0, end = 0;
  tokens.clear();
  int item;

  try {
    while ((end = text.find(sep, start)) != string::npos) {
      item = std::stoi(text.substr(start, end - start));
      tokens.push_back(item);
      start = end + 1;
    }

    item = std::stoi(text.substr(start));
    tokens.push_back(item);

  } catch (std::invalid_argument &) {
    printf("Invalid %s argument, number expected\n", name.c_str());
    exit(-1);
  }
}

// ------------------------------------------------------------------------------------------

void parseFile(string fileName, vector<string> &lines) {
  // Get file size
  FILE *fp = fopen(fileName.c_str(), "rb");
  if (fp == NULL) {
    printf("Error: Cannot open %s %s\n", fileName.c_str(), strerror(errno));
    exit(-1);
  }
  fseek(fp, 0L, SEEK_END);
  size_t sz = ftell(fp);
  size_t nbAddr = sz / 33; /* Upper approximation */
  bool loaddingProgress = sz > 100000;
  fclose(fp);

  // Parse file
  int nbLine = 0;
  string line;
  ifstream inFile(fileName);
  lines.reserve(nbAddr);
  while (getline(inFile, line)) {
    // Remove ending \r\n
    int l = (int)line.length() - 1;
    while (l >= 0 && isspace(line.at(l))) {
      line.pop_back();
      l--;
    }

    if (line.length() > 0) {
      lines.push_back(line);
      nbLine++;
      if (loaddingProgress) {
        if ((nbLine % 50000) == 0)
          printf("[Loading input file %5.1f%%]\r", ((double)nbLine * 100.0) / ((double)(nbAddr) * 33.0 / 34.0));
      }
    }
  }

  if (loaddingProgress) printf("[Loading input file 100.0%%]\n");
}

// ------------------------------------------------------------------------------------------

void generateKeyPair(Secp256K1 *secp, string seed, int searchMode, bool paranoiacSeed) {
  if (seed.length() < 8) {
    printf("Error: Use a seed of at least 8 characters to generate a key pair\n");
    printf("Ex: VanitySearch -s \"A Strong Password\" -kp\n");
    exit(-1);
  }

  if (paranoiacSeed) seed = seed + Timer::getSeed(32);

  if (searchMode == SEARCH_BOTH) {
    printf("Error: Use compressed or uncompressed to generate a key pair\n");
    exit(-1);
  }

  bool compressed = (searchMode == SEARCH_COMPRESSED);

  string salt = "VanitySearch";
  unsigned char hseed[64];
  pbkdf2_hmac_sha512(hseed, 64, (const uint8_t *)seed.c_str(), seed.length(), (const uint8_t *)salt.c_str(),
                     salt.length(), 2048);

  Int privKey;
  privKey.SetInt32(0);
  sha256(hseed, 64, (unsigned char *)privKey.bits64);
  Point p = secp->ComputePublicKey(&privKey);
  printf("Priv : %s\n", secp->GetPrivAddress(compressed, privKey).c_str());
  printf("Pub  : %s\n", secp->GetPublicKeyHex(compressed, p).c_str());
}

// ------------------------------------------------------------------------------------------

void computeAddress(Secp256K1 *secp, string pubKey) {
  bool compressed;
  Point p = secp->ParsePublicKeyHex(pubKey, compressed);
  printf("Address : %s\n", secp->GetPrivAddress(compressed, p).c_str());
}

// ------------------------------------------------------------------------------------------

void computePubKey(Secp256K1 *secp, string privKey) {
  Int key(privKey.c_str());
  printf("Compressed   : %s\n", secp->GetPublicKeyHex(true, secp->ComputePublicKey(&key)).c_str());
  printf("Uncompressed : %s\n", secp->GetPublicKeyHex(false, secp->ComputePublicKey(&key)).c_str());
}

// ------------------------------------------------------------------------------------------

int main(int argc, char *argv[]) {
  // Global Init
  Timer::Init();
  rseed(Timer::getSeed32());

  // Init SecpK1
  Secp256K1 *secp = new Secp256K1();
  secp->Init();

  // Browse arguments
  if (argc < 2) {
    printf("Error: No arguments (use -h for help)\n");
    exit(-1);
  }

  int a = 1;
  bool gpuEnable = false;
  bool stop = false;
  int searchMode = SEARCH_COMPRESSED;
  vector<int> gpuId = {0};
  bool gpuIdSpecified = false;
  vector<int> gridSize;
  string seed = "";
  vector<string> prefix;
  string outputFile = "";
  int nbCPUThread = Timer::getCoreNumber();
  bool tSpecified = false;
  bool sse = true;
  uint32_t maxFound = 65536;
  uint64_t rekey = 0;
  Point startPuKey;
  startPuKey.Clear();
  bool startPubKeyCompressed;
  bool caseSensitive = true;
  bool paranoiacSeed = false;

  while (a < argc) {
    if (strcmp(argv[a], "-gpu") == 0) {
      gpuEnable = true;
      a++;
    } else if (strcmp(argv[a], "-gpuId") == 0) {
      a++;
      getInts("gpuId", gpuId, string(argv[a]), ',');
      gpuIdSpecified = true;
      a++;
    } else if (strcmp(argv[a], "-stop") == 0) {
      stop = true;
      a++;
    } else if (strcmp(argv[a], "-c") == 0) {
      caseSensitive = false;
      a++;
    } else if (strcmp(argv[a], "-v") == 0) {
      printf("%s\n", RELEASE);
      exit(0);
    } else if (strcmp(argv[a], "-check") == 0) {
      Int::Check();
      secp->Check();

#ifdef WITHGPU
      if (gridSize.size() == 0) {
        gridSize.push_back(-1);
        gridSize.push_back(128);
      }
      GPUEngine g(gridSize[0], gridSize[1], gpuId[0], maxFound, false);
      g.SetSearchMode(searchMode);
      g.Check(secp);
#else
      printf("GPU code not compiled, use -DWITHGPU when compiling.\n");
#endif
      exit(0);
    } else if (strcmp(argv[a], "-l") == 0) {
#ifdef WITHGPU
      GPUEngine::PrintCudaInfo();
#else
      printf("GPU code not compiled, use -DWITHGPU when compiling.\n");
#endif
      exit(0);

    } else if (strcmp(argv[a], "-kp") == 0) {
      generateKeyPair(secp, seed, searchMode, paranoiacSeed);
      exit(0);
    } else if (strcmp(argv[a], "-sp") == 0) {
      a++;
      string pub = string(argv[a]);
      startPuKey = secp->ParsePublicKeyHex(pub, startPubKeyCompressed);
      a++;
    } else if (strcmp(argv[a], "-ca") == 0) {
      a++;
      string pub = string(argv[a]);
      bool isComp;
      Point p = secp->ParsePublicKeyHex(pub, isComp);
      printf("Address : %s\n", secp->GetPrivAddress(isComp, p).c_str());
      a++;
    } else if (strcmp(argv[a], "-cp") == 0) {
      a++;
      string priv = string(argv[a]);
      computePubKey(secp, priv);
      a++;
    } else if (strcmp(argv[a], "-rp") == 0) {
      string partialKeysFile;
      a++;
      if (a >= argc) {
        printUsage();
      }
      Int privKey(string(argv[a++]).c_str());
      partialKeysFile = string(argv[a++]);

      string partialPrivAddr = secp->GetPrivAddress(false, privKey);

      FILE *fp = fopen(partialKeysFile.c_str(), "r");
      if (fp == NULL) {
        printf("Error: Cannot open %s %s\n", partialKeysFile.c_str(), strerror(errno));
        exit(-1);
      }

      Int lambda;
      lambda.SetInt32(1);
      lambda.ModSubK1order(&secp->lambda1);

      Int lambda2;
      lambda2.SetInt32(1);
      lambda2.ModSubK1order(&secp->lambda2);

      char line[100];
      int i = 0;
      while (fgets(line, 100, fp)) {
        string addr = string(line);
        for (int i = 0; i < (int)addr.size(); i++) {
          if (addr[i] < 32 || addr[i] > 126) {
            addr.erase(addr.begin() + i);
            i--;
          }
        }

#define CHECK_ADDR()                                                                                        \
  for (int i = 0; i < (int)prefix.size(); i++) {                                                            \
    if ((addr.compare(0, prefix[i].size(), prefix[i]) == 0)) {                                              \
      if (found && (!privAddrFound || privAddr.compare(0, prefix[i].size(), prefix[i]) == 0)) {             \
        printf("Warning: Multiple solutions found\n");                                                      \
      }                                                                                                     \
      privAddrFound = true;                                                                                 \
      privAddr = addr;                                                                                      \
      printf("Final PrivKey : %s\n", secp->GetPrivAddress(true, privKey).c_str());                          \
      printf("         Addr : %s\n", secp->GetPrivAddress(true, secp->ComputePublicKey(&privKey)).c_str()); \
      found = true;                                                                                         \
    }                                                                                                       \
  }

        Int e;
        e.Set(&privKey);
        e.ModMulK1order(&lambda);
        e.ModSubK1order(&secp->order);
        CHECK_ADDR();

        e.Set(&privKey);
        e.ModMulK1order(&lambda2);
        e.Neg();
        e.Add(&secp->order);
        CHECK_ADDR();

        if (!found) {
          printf("Unable to reconstruct final key from partialkey line %d\n Addr: %s\n PartKey: %s\n", i, addr.c_str(),
                 partialPrivAddr.c_str());
        }
      }
    }
  }

  printf("VanitySearch v" RELEASE "\n");

#ifdef WITHGPU
  if (gpuEnable && !gpuIdSpecified) {
    vector<int> autoIds = GPUEngine::GetAutoGpuIds(8);
    if (!autoIds.empty()) gpuId = autoIds;
  }
#endif

  if (gridSize.size() == 0) {
    for (int i = 0; i < gpuId.size(); i++) {
      int gridX = -1;
      int gridY = 128;
#ifdef WITHGPU
      if (gpuEnable) {
        pair<int, int> autoGrid = GPUEngine::GetAutoGridSize(gpuId[i]);
        gridX = autoGrid.first;
        gridY = autoGrid.second;
      }
#endif
      gridSize.push_back(gridX);
      gridSize.push_back(gridY);
    }
  } else if (gridSize.size() != gpuId.size() * 2) {
    printf("Invalid gridSize or gpuId argument, must have coherent size\n");
    exit(-1);
  }

  // Let one CPU core free per gpu is gpu is enabled
  // It will avoid to hang the system
  if (!tSpecified && nbCPUThread > 1 && gpuEnable) nbCPUThread -= (int)gpuId.size();
  if (nbCPUThread < 0) nbCPUThread = 0;

  // If a starting public key is specified, force the search mode according to the key
  if (!startPuKey.isZero()) {
    searchMode = (startPubKeyCompressed) ? SEARCH_COMPRESSED : SEARCH_UNCOMPRESSED;
  }

  VanitySearch *v = new VanitySearch(secp, prefix, seed, searchMode, gpuEnable, stop, outputFile, sse, maxFound, rekey,
                                     caseSensitive, startPuKey, paranoiacSeed);
  v->Search(nbCPUThread, gpuId, gridSize);

  return 0;
}
