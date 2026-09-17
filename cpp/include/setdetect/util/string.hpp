#pragma once

#include <iterator>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace setdetect::util::string {

inline auto join(const std::vector<std::string>& vec, const std::string& delim) -> std::string {
    if (vec.empty()) {
        return std::string{};
    }
    return std::accumulate(std::next(vec.begin()), vec.end(), vec[0],
                           [&delim](std::string a, const std::string& b) { return std::move(a) + delim + b; });
}

} // namespace setdetect::util::string
