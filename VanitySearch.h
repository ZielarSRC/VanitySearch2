#ifndef VANITYSEARCH_H
#define VANITYSEARCH_H

#include "SECP256k1.h"
#include <vector>
#include <string>

class VanitySearch {
public:
    struct Result {
        SECP256k1::uint256_t privateKey;
        std::string address;
    };

    VanitySearch(const std::vector<std::string>& patterns);
    std::vector<Result> Search();

private:
    std::vector<std::string> patterns;
};

#endif