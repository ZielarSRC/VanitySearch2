#include "VanitySearch.h"
#include "SECP256k1.h"
#include "GPUEngine.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <openssl/sha.h>

using namespace std;
using namespace std::chrono;

const string VERSION = "2.0";

// Prototypy funkcji
void printHelp();
void generateKeyPair(const SECP256k1& secp, const string& seed, bool compressed);
vector<string> readPatternsFromFile(const string& fileName);

int main(int argc, char* argv[]) {

    // Inicjalizacja
    SECP256k1 secp;
    secp.Init();

    // Parsowanie argumentów
    vector<string> patterns;
    string seed;
    string outputFile;
    int threads = thread::hardware_concurrency();
    bool useGPU = false;
    int deviceId = 0;
    bool compressed = true;
    bool stopWhenFound = false;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            printHelp();
            return 0;
        } else if (arg == "-v" || arg == "--version") {
            cout << "VanitySearch v" << VERSION << endl;
            return 0;
        } else if (arg == "-gpu") {
            useGPU = true;
        } else if (arg == "-c") {
            compressed = false;
        } else if (arg == "-o") {
            if (++i >= argc) throw runtime_error("Brak pliku wyjściowego");
            outputFile = argv[i];
        } else if (arg == "-s") {
            if (++i >= argc) throw runtime_error("Brak wartości seed");
            seed = argv[i];
        } else if (arg == "-t") {
            if (++i >= argc) throw runtime_error("Brak liczby wątków");
            threads = stoi(argv[i]);
        } else if (arg == "-i") {
            if (++i >= argc) throw runtime_error("Brak pliku wejściowego");
            patterns = readPatternsFromFile(argv[i]);
        } else if (arg == "-stop") {
            stopWhenFound = true;
        } else {
            patterns.push_back(arg);
        }
    }

    if (patterns.empty()) {
        cerr << "Error: Podaj przynajmniej jeden wzorzec" << endl;
        printHelp();
        return 1;
    }

    // Konfiguracja wyszukiwania
    VanitySearch::Parameters params;
    params.patterns = patterns;
    params.useGPU = useGPU;
    params.deviceId = deviceId;
    params.threadCount = threads;
    params.compressed = compressed;
    params.stopWhenFound = stopWhenFound;
    params.startKey = SECP256k1::uint256_t(1);
    params.endKey = SECP256k1::uint256_t::Max();

    // Inicjalizacja wyszukiwarki
    VanitySearch vs(params);
    
    try {
        vs.Run();
        
        // Zapis wyników
        auto results = vs.GetResults();
        if (!outputFile.empty()) {
            ofstream fout(outputFile);
            for (const auto& res : results) {
                fout << "Address: " << res.address << endl
                     << "Private: " << res.privateKey.ToHex() << endl
                     << "Found: " << res.foundTime << endl << endl;
            }
        }
        
    } catch (const exception& ex) {
        cerr << "Error: " << ex.what() << endl;
        return 1;
    }

    return 0;
}

void printHelp() {
    cout << "VanitySearch v" << VERSION << " - Wyszukiwanie kryptowalutowych adresów\n\n"
         << "Użycie:\n"
         << "  VanitySearch [opcje] <wzorzec1> [wzorzec2...]\n\n"
         << "Opcje:\n"
         << "  -h, --help       Wyświetl pomoc\n"
         << "  -v, --version    Wyświetl wersję\n"
         << "  -gpu             Włącz obliczenia GPU\n"
         << "  -c               Wyszukuj adresy nieskompresowane\n"
         << "  -o <plik>        Zapisz wyniki do pliku\n"
         << "  -s <seed>        Ustaw seed generacji kluczy\n"
         << "  -t <threads>     Liczba wątków CPU\n"
         << "  -i <plik>        Wczytaj wzorce z pliku\n"
         << "  -stop            Zatrzymaj po znalezieniu pierwszego dopasowania\n";
}

vector<string> readPatternsFromFile(const string& fileName) {
    vector<string> patterns;
    ifstream fin(fileName);
    string line;
    
    while (getline(fin, line)) {
        if (!line.empty()) {
            patterns.push_back(line);
        }
    }
    
    return patterns;
}

void generateKeyPair(const SECP256k1& secp, const string& seed, bool compressed) {
    SECP256k1::uint256_t privKey = secp.GeneratePrivateKey(seed);
    SECP256k1::uint256_t pubX, pubY;
    
    secp.Multiply(privKey, pubX, pubY);
    string address = secp.GenerateAddress(pubX, pubY, compressed);
    
    cout << "Private key: " << privKey.ToHex() << endl
         << "Public address: " << address << endl;
}
