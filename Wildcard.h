#pragma once
#include <string>
#include <vector>
#include <regex>
#include <stdexcept>

class Wildcard {
public:
    explicit Wildcard(const std::string& pattern);
    Wildcard(const Wildcard&) = default;
    Wildcard(Wildcard&&) = default;
    Wildcard& operator=(const Wildcard&) = default;
    Wildcard& operator=(Wildcard&&) = default;
    ~Wildcard() = default;

    bool match(const std::string& input) const;
    static bool isPattern(const std::string& str);
    const std::string& getPattern() const { return originalPattern; }

private:
    std::regex regexPattern;
    std::string originalPattern;
    bool isCaseSensitive = true;
    
    static std::string convertToRegex(const std::string& wildcardPattern);
    void compileRegex();
};
