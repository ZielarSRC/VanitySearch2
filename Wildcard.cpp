#include "Wildcard.h"
#include <algorithm>

Wildcard::Wildcard(const std::string& pattern) 
    : originalPattern(pattern) {
    try {
        compileRegex();
    } catch (const std::regex_error& e) {
        throw std::runtime_error("Invalid wildcard pattern: " + pattern + " (" + e.what() + ")");
    }
}

void Wildcard::compileRegex() {
    std::string regexStr = convertToRegex(originalPattern);
    auto flags = std::regex::optimize | std::regex::nosubs;
    if (!isCaseSensitive) {
        flags |= std::regex::icase;
    }
    regexPattern = std::regex(regexStr, flags);
}

bool Wildcard::match(const std::string& input) const {
    try {
        return std::regex_match(input, regexPattern);
    } catch (const std::regex_error& e) {
        throw std::runtime_error("Wildcard matching failed: " + std::string(e.what()));
    }
}

bool Wildcard::isPattern(const std::string& str) {
    return str.find_first_of("*?") != std::string::npos;
}

std::string Wildcard::convertToRegex(const std::string& wildcardPattern) {
    std::string regexPattern;
    regexPattern.reserve(wildcardPattern.size() * 2 + 3);
    regexPattern.append("^");

    for (char c : wildcardPattern) {
        switch (c) {
            case '*': 
                regexPattern.append(".*");
                break;
            case '?': 
                regexPattern.append(".");
                break;
            case '.': 
                regexPattern.append("\\.");
                break;
            case '\\': 
                regexPattern.append("\\\\");
                break;
            case '+': 
                regexPattern.append("\\+");
                break;
            case '^': 
                regexPattern.append("\\^");
                break;
            case '$': 
                regexPattern.append("\\$");
                break;
            case '{': 
                regexPattern.append("\\{");
                break;
            case '}': 
                regexPattern.append("\\}");
                break;
            case '(': 
                regexPattern.append("\\(");
                break;
            case ')': 
                regexPattern.append("\\)");
                break;
            case '|': 
                regexPattern.append("\\|");
                break;
            case '[': 
                regexPattern.append("\\[");
                break;
            case ']': 
                regexPattern.append("\\]");
                break;
            default:
                regexPattern += c;
        }
    }

    regexPattern.append("$");
    return regexPattern;
}
