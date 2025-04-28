#include "Wildcard.h"
#include <cctype>
#include <algorithm>

bool Wildcard::match(std::string_view str, std::string_view pattern, bool caseSensitive) noexcept {
    const char* s = str.data();
    const char* p = pattern.data();
    const char* strPtr = s;
    const char* patternPtr = p;
    const char* starPos = nullptr;
    const char* strAfterStar = nullptr;

    while (*strPtr) {
        if (*patternPtr == '?' || 
            (caseSensitive ? *strPtr == *patternPtr 
                          : std::tolower(*strPtr) == std::tolower(*patternPtr))) {
            strPtr++;
            patternPtr++;
        } 
        else if (*patternPtr == '*') {
            starPos = patternPtr;
            strAfterStar = strPtr;
            patternPtr++;
        } 
        else if (starPos) {
            patternPtr = starPos + 1;
            strPtr = ++strAfterStar;
        } 
        else {
            return false;
        }
    }

    while (*patternPtr == '*') {
        patternPtr++;
    }

    return !*patternPtr;
}
