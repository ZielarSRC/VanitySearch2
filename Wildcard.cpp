#include "Wildcard.h"
#include <stdexcept>

Wildcard::Wildcard(const std::string& pattern) 
    : originalPattern(pattern),
      regexPattern(convertToRegex(pattern), std::regex::optimize) {}

bool Wildcard::match(const std::string& input) const {
    return std::regex_match(input, regexPattern);
}

bool Wildcard::isPattern(const std::string& str) {
    return str.find_first_of("*?") != std::string::npos;
}

std::string Wildcard::convertToRegex(const std::string& wildcardPattern) {
    std::string regexPattern;
    regexPattern.reserve(wildcardPattern.size() * 2);
    regexPattern.append("^");
    
    for (char c : wildcardPattern) {
        switch (c) {
            case '*': regexPattern.append(".*"); break;
            case '?': regexPattern.append("."); break;
            case '.': regexPattern.append("\\."); break;
            case '\\': regexPattern.append("\\\\"); break;
            default: regexPattern += c;
        }
    }
    
    regexPattern.append("$");
    return regexPattern;
}
