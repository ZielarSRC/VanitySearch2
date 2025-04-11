#pragma once
#include <string>
#include <vector>
#include <regex>

class Wildcard {
public:
    explicit Wildcard(const std::string& pattern);
    
    bool match(const std::string& input) const;
    static bool isPattern(const std::string& str);
    
private:
    std::regex regexPattern;
    std::string originalPattern;
    
    static std::string convertToRegex(const std::string& wildcardPattern);
};
