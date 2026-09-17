#pragma once

#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

namespace setdetect::apps {

struct Stats {
    double mean = 0.0;
    double stddev = 0.0;
};

// Mean / population-stddev summary of `samples`.
inline auto summarize(const std::vector<double>& samples) -> Stats {
    const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / static_cast<double>(samples.size());
    double var = 0.0;
    for (const double s : samples) {
        var += (s - mean) * (s - mean);
    }
    var /= static_cast<double>(samples.size());
    return {mean, std::sqrt(var)};
}

// Prints "<label padded to width> <mean*scale> +/- <stddev*scale> <unit>".
inline void print_stat(const char* label, const Stats& s, double unit_scale, const char* unit, int width = 28) {
    std::cout << std::left << std::setw(width) << label << std::setprecision(6) << s.mean * unit_scale << " +/- "
              << s.stddev * unit_scale << " " << unit << "\n";
}

} // namespace setdetect::apps
