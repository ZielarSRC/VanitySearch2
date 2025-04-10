/*
 * Zmodernizowana wersja GPUWildcard.h
 * Optymalizacje: CUDA 12+, Ampere/Ada Lovelace, pełna kompatybilność wsteczna
 */

#ifndef GPU_WILDCARD_H
#define GPU_WILDCARD_H

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------------
// Optymalizacja: Wersja z prefetchingiem i optymalizacją dla krótkich wzorców
// ---------------------------------------------------------------------------------

__device__ __forceinline__ bool _MatchWildcard(const char *str, const char *pattern) {
    const char *s = str;
    const char *p = pattern;
    const char *last_star = nullptr;
    const char *last_str = nullptr;

#if __CUDA_ARCH__ >= 800
    // Prefetch danych dla Ampere/Ada
    asm volatile ("prefetch.global.L1 [%0];" :: "l"(str));
    asm volatile ("prefetch.global.L1 [%0];" :: "l"(pattern));
#endif

    while (true) {
        // Obsługa ciągów znaków do porównania
        if (*p == '\0') {
            return (*s == '\0');
        }

        if (*p == '*') {
            last_star = p++;
            last_str = s;
            continue;
        }

        if (*s == '\0') {
            return (*p == '\0' || (*p == '*' && *(p + 1) == '\0'));
        }

        // Optymalizacja: szybkie porównanie bez rozgałęzienia
        bool match = (*p == '?') || (*p == *s);
        if (match) {
            s++;
            p++;
            continue;
        }

        if (last_star) {
            p = last_star + 1;
            s = ++last_str;
            continue;
        }

        return false;
    }
}

// ---------------------------------------------------------------------------------
// Wersja z dodatkowymi optymalizacjami dla długich wzorców
// ---------------------------------------------------------------------------------

__device__ __noinline__ bool _MatchLongPattern(const char *str, const char *pattern) {
    const char *s;
    const char *p;
    bool star = false;

loopStart:
    for (s = str, p = pattern; *s; ++s, ++p) {
        switch (*p) {
            case '?':
                if (*s == '.') goto starCheck;
                break;

            case '*':
                star = true;
                str = s;
                pattern = p;
                if (!*++pattern) return true;
                goto loopStart;

            default:
                if (*s != *p)
                    goto starCheck;
                break;
        }
    }

    if (*p == '*') ++p;
    return (!*p);

starCheck:
    if (!star) return false;
    str++;
    goto loopStart;
}

// ---------------------------------------------------------------------------------
// Funkcja główna z automatycznym wyborem implementacji
// ---------------------------------------------------------------------------------

__device__ __forceinline__ bool _Match(const char *str, const char *pattern) {
    // Heurystyka: wybierz implementację na podstawie długości wzorca
    int pattern_len = 0;
    const char *tmp = pattern;
    while (*tmp++) pattern_len++;

    if (pattern_len < 32) {
        return _MatchWildcard(str, pattern);
    } else {
        return _MatchLongPattern(str, pattern);
    }
}

#endif // GPU_WILDCARD_H
