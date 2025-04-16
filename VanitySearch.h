#ifndef VANITYSEARCH_H
#define VANITYSEARCH_H

#include "SECP256k1.h"
#include <string>
#include <vector>

class VanitySearch {
public:
    struct Result {
        SECP256k1::uint256_t privateKey;
        std::string address;
        uint64_t balance;
    };

    explicit VanitySearch(const std::vector<std::string>& patterns);
    std::vector<Result> Search(int threads = 0);

private:
    std::vector<std::string> patterns;
};

#endif
