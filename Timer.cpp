#include "Timer.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <memory>

#ifdef _WIN32
#include <wincrypt.h>
LARGE_INTEGER Timer::perfTickStart;
double Timer::perfTicksPerSec;
LARGE_INTEGER Timer::qwTicksPerSec;
#else
#include <unistd.h>
time_t Timer::tickStart;
#endif

void Timer::Init() {
#ifdef _WIN32
    QueryPerformanceFrequency(&qwTicksPerSec);
    QueryPerformanceCounter(&perfTickStart);
    perfTicksPerSec = static_cast<double>(qwTicksPerSec.QuadPart);
#else
    tickStart = time(nullptr);
#endif
}

double Timer::get_tick() {
#ifdef _WIN32
    LARGE_INTEGER t, dt;
    QueryPerformanceCounter(&t);
    dt.QuadPart = t.QuadPart - perfTickStart.QuadPart;
    return static_cast<double>(dt.QuadPart) / perfTicksPerSec;
#else
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return static_cast<double>(tv.tv_sec - tickStart) + static_cast<double>(tv.tv_usec) / 1e6;
#endif
}

std::string Timer::getSeed(int size) {
    std::string ret;
    ret.reserve(size * 2); // Each byte becomes 2 hex chars

    auto buff = std::make_unique<unsigned char[]>(size);

#ifdef _WIN32
    HCRYPTPROV hCryptProv = 0;
    const char* UserName = "KeyContainer";

    if (!CryptAcquireContext(&hCryptProv, UserName, nullptr, PROV_RSA_FULL, 0)) {
        if (GetLastError() == NTE_BAD_KEYSET) {
            if (!CryptAcquireContext(&hCryptProv, UserName, nullptr, PROV_RSA_FULL, CRYPT_NEWKEYSET)) {
                throw std::runtime_error("CryptAcquireContext(): Could not create a new key container.");
            }
        } else {
            throw std::runtime_error("CryptAcquireContext(): A cryptographic service handle could not be acquired.");
        }
    }

    if (!CryptGenRandom(hCryptProv, size, buff.get())) {
        CryptReleaseContext(hCryptProv, 0);
        throw std::runtime_error("CryptGenRandom(): Error during random sequence acquisition.");
    }

    CryptReleaseContext(hCryptProv, 0);
#else
    FILE* f = fopen("/dev/urandom", "rb");
    if (f == nullptr) {
        throw std::runtime_error(std::string("Failed to open /dev/urandom: ") + strerror(errno));
    }

    if (fread(buff.get(), 1, size, f) != static_cast<size_t>(size)) {
        fclose(f);
        throw std::runtime_error(std::string("Failed to read from /dev/urandom: ") + strerror(errno));
    }
    fclose(f);
#endif

    char tmp[3];
    for (int i = 0; i < size; i++) {
        snprintf(tmp, sizeof(tmp), "%02X", buff[i]);
        ret += tmp;
    }

    return ret;
}

uint32_t Timer::getSeed32() {
    return static_cast<uint32_t>(std::strtoul(getSeed(4).c_str(), nullptr, 16));
}

std::string Timer::getResult(const char* unit, int nbTry, double t0, double t1) {
    char tmp[256];
    int pIdx = 0;
    double nbCallPerSec = static_cast<double>(nbTry) / (t1 - t0);
    
    while (nbCallPerSec > 1000.0 && pIdx < 6) {
        pIdx++;
        nbCallPerSec /= 1000.0;
    }
    
    snprintf(tmp, sizeof(tmp), "%.3f %s%s/sec", nbCallPerSec, prefix[pIdx], unit);
    return std::string(tmp);
}

void Timer::printResult(const char* unit, int nbTry, double t0, double t1) {
    printf("%s\n", getResult(unit, nbTry, t0, t1).c_str());
}

int Timer::getCoreNumber() {
#ifdef _WIN32
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return sysinfo.dwNumberOfProcessors;
#else
    return sysconf(_SC_NPROCESSORS_ONLN);
#endif
}

void Timer::SleepMillis(uint32_t millis) {
#ifdef _WIN32
    Sleep(millis);
#else
    usleep(millis * 1000);
#endif
}
