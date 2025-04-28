#ifndef TIMER_H
#define TIMER_H

#include <string>
#include <cstdint>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#include <time.h>
#endif

class Timer {
public:
    static void Init();
    static double get_tick();
    static void printResult(const char* unit, int nbTry, double t0, double t1);
    static std::string getResult(const char* unit, int nbTry, double t0, double t1);
    static int getCoreNumber();
    static std::string getSeed(int size);
    static uint32_t getSeed32();
    static void SleepMillis(uint32_t millis);

private:
#ifdef _WIN32
    static LARGE_INTEGER perfTickStart;
    static double perfTicksPerSec;
    static LARGE_INTEGER qwTicksPerSec;
#else
    static time_t tickStart;
#endif

    static constexpr const char* prefix[7] = {"", "Kilo", "Mega", "Giga", "Tera", "Peta", "Hexa"};
};

#endif // TIMER_H
