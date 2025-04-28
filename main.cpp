#include "Timer.h"
#include "Vanity.h"
#include "SECP256k1.h"
#include <fstream>
#include <string>
#include <string_view>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <span>
#include <charconv>
#include "hash/sha512.h"
#include "hash/sha256.h"

constexpr std::string_view RELEASE = "2.0";

using namespace std;

// ------------------------------------------------------------------------------------------

void printUsage() {
    cout << "VanitySearch [-check] [-v] [-u] [-b] [-c] [-gpu] [-stop] [-i inputfile]\n"
         << "             [-gpuId gpuId1[,gpuId2,...]] [-g g1x,g1y,[,g2x,g2y,...]]\n"
         << "             [-o outputfile] [-m maxFound] [-ps seed] [-s seed] [-t nbThread]\n"
         << "             [-nosse] [-r rekey] [-check] [-kp] [-sp startPubKey]\n"
         << "             [-rp privkey partialkeyfile] [prefix]\n\n"
         << " prefix: prefix to search (Can contain wildcard '?' or '*')\n"
         << " -v: Print version\n"
         << " -u: Search uncompressed addresses\n"
         << " -b: Search both uncompressed or compressed addresses\n"
         << " -c: Case insensitive search\n"
         << " -gpu: Enable GPU calculation\n"
         << " -stop: Stop when all prefixes are found\n"
         << " -i inputfile: Get list of prefixes to search from specified file\n"
         << " -o outputfile: Output results to the specified file\n"
         << " -gpu gpuId1,gpuId2,...: List of GPU(s) to use, default is 0\n"
         << " -g g1x,g1y,g2x,g2y, ...: Specify GPU(s) kernel gridsize\n"
         << " -m: Specify maximum number of prefixes found by each kernel call\n"
         << " -s seed: Specify a seed for the base key, default is random\n"
         << " -ps seed: Specify a seed concatenated with a crypto secure random seed\n"
         << " -t threadNumber: Specify number of CPU threads, default is number of cores\n"
         << " -nosse: Disable SSE hash function\n"
         << " -l: List CUDA enabled devices\n"
         << " -check: Check CPU and GPU kernel vs CPU\n"
         << " -cp privKey: Compute public key (privKey in hex format)\n"
         << " -ca pubKey: Compute address (pubKey in hex format)\n"
         << " -kp: Generate key pair\n"
         << " -rp privkey partialkeyfile: Reconstruct final private key(s) from partial key(s) info.\n"
         << " -sp startPubKey: Start the search with a pubKey (for private key splitting)\n"
         << " -r rekey: Rekey interval in MegaKey, default is disabled\n";
    exit(0);
}

// ------------------------------------------------------------------------------------------

int getInt(string_view name, string_view v) {
    int result;
    auto [ptr, ec] = from_chars(v.data(), v.data() + v.size(), result);
    
    if (ec != errc()) {
        cerr << "Invalid " << name << " argument, number expected\n";
        exit(EXIT_FAILURE);
    }
    
    return result;
}

// ------------------------------------------------------------------------------------------

void getInts(string_view name, vector<int>& tokens, string_view text, char sep) {
    tokens.clear();
    size_t start = 0;
    size_t end = text.find(sep);
    
    while (end != string::npos) {
        tokens.push_back(getInt(name, text.substr(start, end - start)));
        start = end + 1;
        end = text.find(sep, start);
    }
    
    tokens.push_back(getInt(name, text.substr(start)));
}

// ------------------------------------------------------------------------------------------

void parseFile(const string& fileName, vector<string>& lines) {
    ifstream inFile(fileName);
    if (!inFile) {
        cerr << "Error: Cannot open " << fileName << ": " << strerror(errno) << endl;
        exit(EXIT_FAILURE);
    }

    // Get file size for progress estimation
    inFile.seekg(0, ios::end);
    size_t sz = inFile.tellg();
    inFile.seekg(0, ios::beg);
    bool showProgress = sz > 100000;
    size_t nbAddr = sz / 33; // Upper approximation
    
    lines.reserve(nbAddr);
    string line;
    size_t nbLine = 0;
    
    while (getline(inFile, line)) {
        // Trim whitespace from end
        line.erase(find_if(line.rbegin(), line.rend(), [](unsigned char ch) {
            return !isspace(ch);
        }).base(), line.end());
        
        if (!line.empty()) {
            lines.push_back(move(line));
            nbLine++;
            
            if (showProgress && (nbLine % 50000 == 0)) {
                cout << format("[Loading input file {:.1f}%]\r", 
                    (static_cast<double>(nbLine) * 100.0) / (static_cast<double>(nbAddr) * 33.0 / 34.0));
                cout.flush();
            }
        }
    }
    
    if (showProgress) {
        cout << "[Loading input file 100.0%]\n";
    }
}

// ------------------------------------------------------------------------------------------

void generateKeyPair(Secp256K1* secp, const string& seed, int searchMode, bool paranoiacSeed) {
    if (seed.length() < 8) {
        cerr << "Error: Use a seed of at least 8 characters to generate a key pair\n"
             << "Ex: VanitySearch -s \"A Strong Password\" -kp\n";
        exit(EXIT_FAILURE);
    }

    if (paranoiacSeed) {
        string extendedSeed = seed + Timer::getSeed(32);
        seed = extendedSeed;
    }

    if (searchMode == SEARCH_BOTH) {
        cerr << "Error: Use compressed or uncompressed to generate a key pair\n";
        exit(EXIT_FAILURE);
    }

    bool compressed = (searchMode == SEARCH_COMPRESSED);
    string salt = "VanitySearch";
    
    array<uint8_t, 64> hseed;
    pbkdf2_hmac_sha512(hseed.data(), hseed.size(), 
                       reinterpret_cast<const uint8_t*>(seed.data()), seed.length(),
                       reinterpret_cast<const uint8_t*>(salt.data()), salt.length(),
                       2048);

    Int privKey;
    privKey.SetInt32(0);
    sha256(hseed.data(), hseed.size(), reinterpret_cast<uint8_t*>(privKey.bits64));
    
    Point p = secp->ComputePublicKey(&privKey);
    cout << "Priv : " << secp->GetPrivAddress(compressed, privKey) << "\n";
    cout << "Pub  : " << secp->GetPublicKeyHex(compressed, p) << "\n";
}

// ------------------------------------------------------------------------------------------

void outputAdd(const string& outputFile, int addrType, const string& addr, 
               const string& pAddr, const string& pAddrHex) {
    ofstream outFile;
    ostream* out = &cout;
    
    if (!outputFile.empty()) {
        outFile.open(outputFile, ios::app);
        if (!outFile) {
            cerr << "Cannot open " << outputFile << " for writing\n";
        } else {
            out = &outFile;
        }
    }

    *out << "\nPub Addr: " << addr << "\n";
    
    switch (addrType) {
        case P2PKH:  *out << "Priv (WIF): p2pkh:" << pAddr << "\n"; break;
        case P2SH:   *out << "Priv (WIF): p2wpkh-p2sh:" << pAddr << "\n"; break;
        case BECH32: *out << "Priv (WIF): p2wpkh:" << pAddr << "\n"; break;
    }
    
    *out << "Priv (HEX): 0x" << pAddrHex << "\n";
}

// ------------------------------------------------------------------------------------------

void reconstructAdd(Secp256K1* secp, const string& fileName, const string& outputFile, 
                   const string& privAddr) {
    bool compressed;
    Int lambda;
    Int lambda2;
    lambda.SetBase16("5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72");
    lambda2.SetBase16("ac9c52b33fa3cf1f5ad9e3fd77ed9ba4a880b9fc8ec739c2e0cfc810b51283ce");

    Int privKey = secp->DecodePrivateKey(privAddr.c_str(), &compressed);
    if (privKey.IsNegative()) {
        exit(EXIT_FAILURE);
    }

    vector<string> lines;
    parseFile(fileName, lines);

    for (size_t i = 0; i < lines.size(); i += 2) {
        if (i + 1 >= lines.size()) {
            cerr << "Invalid partialkey info file - missing line after line " << i << "\n";
            exit(EXIT_FAILURE);
        }

        if (!lines[i].starts_with("PubAddress: ")) {
            cerr << "Invalid partialkey info file at line " << i 
                 << " (\"PubAddress: \" expected)\n";
            exit(EXIT_FAILURE);
        }

        if (!lines[i+1].starts_with("PartialPriv: ")) {
            cerr << "Invalid partialkey info file at line " << i+1 
                 << " (\"PartialPriv: \" expected)\n";
            exit(EXIT_FAILURE);
        }

        string addr = lines[i].substr(12);
        string partialPrivAddr = lines[i+1].substr(13);
        
        int addrType;
        switch (addr[0]) {
            case '1': addrType = P2PKH; break;
            case '3': addrType = P2SH; break;
            case 'b': case 'B': addrType = BECH32; break;
            default:
                cerr << "Invalid partialkey info file at line " << i << "\n"
                     << addr << " Address format not supported\n";
                continue;
        }

        bool partialMode;
        Int partialPrivKey = secp->DecodePrivateKey(partialPrivAddr.c_str(), &partialMode);
        if (privKey.IsNegative()) {
            cerr << "Invalid partialkey info file at line " << i << "\n";
            exit(EXIT_FAILURE);
        }

        if (partialMode != compressed) {
            cerr << "Warning, Invalid partialkey at line " << i 
                 << " (Wrong compression mode, ignoring key)\n";
            continue;
        }

        // Reconstruct the address
        auto checkAddress = [&](const Int& e) -> bool {
            Int fullPriv;
            fullPriv.ModAddK1order(&e, &partialPrivKey);
            Point p = secp->ComputePublicKey(&fullPriv);
            string cAddr = secp->GetAddress(addrType, compressed, p);
            
            if (cAddr == addr) {
                string pAddr = secp->GetPrivAddress(compressed, fullPriv);
                string pAddrHex = fullPriv.GetBase16();
                outputAdd(outputFile, addrType, addr, pAddr, pAddrHex);
                return true;
            }
            return false;
        };

        Int e;
        bool found = false;
        
        // Try different combinations
        const array<function<void()>, 6> combinations = {
            [&] { e.Set(&privKey); }, // No sym, no endo
            [&] { e.Set(&privKey); e.ModMulK1order(&lambda); }, // No sym, endo 1
            [&] { e.Set(&privKey); e.ModMulK1order(&lambda2); }, // No sym, endo 2
            [&] { e.Set(&privKey); e.Neg(); e.Add(&secp->order); }, // sym, no endo
            [&] { e.Set(&privKey); e.ModMulK1order(&lambda); e.Neg(); e.Add(&secp->order); }, // sym, endo 1
            [&] { e.Set(&privKey); e.ModMulK1order(&lambda2); e.Neg(); e.Add(&secp->order); } // sym, endo 2
        };

        for (const auto& combo : combinations) {
            combo();
            if (checkAddress(e)) {
                found = true;
                break;
            }
        }

        if (!found) {
            cerr << format("Unable to reconstruct final key from partialkey line {}\nAddr: {}\nPartKey: {}\n",
                          i, addr, partialPrivAddr);
        }
    }
}

// ------------------------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    Timer::Init();
    rseed(Timer::getSeed32());

    unique_ptr<Secp256K1> secp = make_unique<Secp256K1>();
    secp->Init();

    if (argc < 2) {
        cerr << "Error: No arguments (use -h for help)\n";
        return EXIT_FAILURE;
    }

    // Configuration with defaults
    struct Config {
        bool gpuEnable = false;
        bool stop = false;
        int searchMode = SEARCH_COMPRESSED;
        vector<int> gpuId = {0};
        vector<int> gridSize;
        string seed;
        vector<string> prefix;
        string outputFile;
        int nbCPUThread = Timer::getCoreNumber();
        bool tSpecified = false;
        bool sse = true;
        uint32_t maxFound = 65536;
        uint64_t rekey = 0;
        Point startPuKey;
        bool startPubKeyCompressed = false;
        bool caseSensitive = true;
        bool paranoiacSeed = false;
    } config;

    // Parse arguments
    span<char*> args(argv, argc);
    for (size_t a = 1; a < args.size();) {
        string_view arg = args[a];
        
        auto getNextArg = [&]() -> string {
            if (++a >= args.size()) {
                cerr << "Missing argument for " << arg << "\n";
                exit(EXIT_FAILURE);
            }
            return args[a];
        };

        if (arg == "-gpu") {
            config.gpuEnable = true;
            a++;
        } else if (arg == "-gpuId") {
            getInts("gpuId", config.gpuId, getNextArg(), ',');
            a++;
        } else if (arg == "-stop") {
            config.stop = true;
            a++;
        } else if (arg == "-c") {
            config.caseSensitive = false;
            a++;
        } else if (arg == "-v") {
            cout << RELEASE << "\n";
            return 0;
        } else if (arg == "-check") {
            Int::Check();
            secp->Check();
#ifdef WITHGPU
            if (config.gridSize.empty()) {
                config.gridSize = {-1, 128};
            }
            GPUEngine g(config.gridSize[0], config.gridSize[1], config.gpuId[0], 
                      config.maxFound, false);
            g.SetSearchMode(config.searchMode);
            g.Check(secp.get());
#else
            cerr << "GPU code not compiled, use -DWITHGPU when compiling.\n";
#endif
            return 0;
        } else if (arg == "-l") {
#ifdef WITHGPU
            GPUEngine::PrintCudaInfo();
#else
            cerr << "GPU code not compiled, use -DWITHGPU when compiling.\n";
#endif
            return 0;
        } else if (arg == "-kp") {
            generateKeyPair(secp.get(), config.seed, config.searchMode, config.paranoiacSeed);
            return 0;
        } else if (arg == "-sp") {
            string pub = getNextArg();
            config.startPuKey = secp->ParsePublicKeyHex(pub, config.startPubKeyCompressed);
            a++;
        } else if (arg == "-ca") {
            string pub = getNextArg();
            bool isComp;
            Point p = secp->ParsePublicKeyHex(pub, isComp);
            cout << "Addr (P2PKH): " << secp->GetAddress(P2PKH, isComp, p) << "\n";
            cout << "Addr (P2SH): " << secp->GetAddress(P2SH, isComp, p) << "\n";
            cout << "Addr (BECH32): " << secp->GetAddress(BECH32, isComp, p) << "\n";
            return 0;
        } else if (arg == "-cp") {
            string priv = getNextArg();
            Int k;
            bool isComp = true;
            if (priv[0] == '5' || priv[0] == 'K' || priv[0] == 'L') {
                k = secp->DecodePrivateKey(priv.c_str(), &isComp);
            } else {
                k.SetBase16(priv.c_str());
            }
            Point p = secp->ComputePublicKey(&k);
            cout << "PrivAddr: p2pkh:" << secp->GetPrivAddress(isComp, k) << "\n";
            cout << "PubKey: " << secp->GetPublicKeyHex(isComp, p) << "\n";
            cout << "Addr (P2PKH): " << secp->GetAddress(P2PKH, isComp, p) << "\n";
            cout << "Addr (P2SH): " << secp->GetAddress(P2SH, isComp, p) << "\n";
            cout << "Addr (BECH32): " << secp->GetAddress(BECH32, isComp, p) << "\n";
            return 0;
        } else if (arg == "-rp") {
            string priv = getNextArg();
            string file = getNextArg();
            a++;
            reconstructAdd(secp.get(), file, config.outputFile, priv);
            return 0;
        } else if (arg == "-u") {
            config.searchMode = SEARCH_UNCOMPRESSED;
            a++;
        } else if (arg == "-b") {
            config.searchMode = SEARCH_BOTH;
            a++;
        } else if (arg == "-nosse") {
            config.sse = false;
            a++;
        } else if (arg == "-g") {
            getInts("gridSize", config.gridSize, getNextArg(), ',');
            a++;
        } else if (arg == "-s") {
            config.seed = getNextArg();
            a++;
        } else if (arg == "-ps") {
            config.seed = getNextArg();
            config.paranoiacSeed = true;
            a++;
        } else if (arg == "-o") {
            config.outputFile = getNextArg();
            a++;
        } else if (arg == "-i") {
            parseFile(getNextArg(), config.prefix);
            a++;
        } else if (arg == "-t") {
            config.nbCPUThread = getInt("nbCPUThread", getNextArg());
            config.tSpecified = true;
            a++;
        } else if (arg == "-m") {
            config.maxFound = getInt("maxFound", getNextArg());
            a++;
        } else if (arg == "-r") {
            config.rekey = getInt("rekey", getNextArg());
            a++;
        } else if (arg == "-h") {
            printUsage();
        } else if (a == args.size() - 1) {
            config.prefix.push_back(args[a]);
            a++;
        } else {
            cerr << "Unexpected argument: " << arg << "\n";
            return EXIT_FAILURE;
        }
    }

    cout << "VanitySearch v" << RELEASE << "\n";

    // Configure grid size
    if (config.gridSize.empty()) {
        for (size_t i = 0; i < config.gpuId.size(); i++) {
            config.gridSize.insert(config.gridSize.end(), {-1, 128});
        }
    } else if (config.gridSize.size() != config.gpuId.size() * 2) {
        cerr << "Invalid gridSize or gpuId argument, must have coherent size\n";
        return EXIT_FAILURE;
    }

    // Adjust CPU threads if GPU is enabled
    if (!config.tSpecified && config.nbCPUThread > 1 && config.gpuEnable) {
        config.nbCPUThread -= static_cast<int>(config.gpuId.size());
        config.nbCPUThread = max(config.nbCPUThread, 0);
    }

    // Set search mode based on start public key if specified
    if (!config.startPuKey.isZero()) {
        config.searchMode = config.startPubKeyCompressed ? SEARCH_COMPRESSED : SEARCH_UNCOMPRESSED;
    }

    auto vanity = make_unique<VanitySearch>(secp.get(), config.prefix, config.seed, 
        config.searchMode, config.gpuEnable, config.stop, config.outputFile, config.sse,
        config.maxFound, config.rekey, config.caseSensitive, config.startPuKey, 
        config.paranoiacSeed);

    vanity->Search(config.nbCPUThread, config.gpuId, config.gridSize);
    return 0;
}
