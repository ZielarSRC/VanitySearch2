#ifndef WILDCARD_H
#define WILDCARD_H

#include <string_view>

class Wildcard {
public:
    /**
     * Checks whether a string matches a given wildcard pattern.
     * @param str The input string to match
     * @param pattern The wildcard pattern (supports '?' and '*')
     * @param caseSensitive Whether the comparison should be case sensitive
     * @return true if the string matches the pattern, false otherwise
     */
    static bool match(std::string_view str, std::string_view pattern, bool caseSensitive) noexcept;
    
    // Overload for C-style strings
    static bool match(const char* str, const char* pattern, bool caseSensitive) noexcept {
        return match(std::string_view(str), std::string_view(pattern), caseSensitive);
    }
};

#endif // WILDCARD_H
