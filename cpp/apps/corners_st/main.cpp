#include <CLI/CLI.hpp>
#include <apps/stats.hpp>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <random>
#include <setdetect/img_proc/corners_st.hpp>
#include <vector>

namespace ip = setdetect::img_proc;
using setdetect::apps::print_stat;
using setdetect::apps::Stats;
using setdetect::apps::summarize;

namespace {

constexpr int kDefaultRows = 240;
constexpr int kDefaultCols = 320;
constexpr int kDefaultRuns = 10;
constexpr unsigned kDefaultSeed = 42;
constexpr int kDefaultMaxCorners = 1000;
constexpr double kDefaultQualityLevel = 0.01;
constexpr double kDefaultMinDistance = 10.0;
constexpr int kDefaultBlockSize = 3;

// Random grayscale noise: texture everywhere, so every run exercises the full
// gradient, eigenvalue, NMS and min-distance pipeline.
auto make_noise_image(int rows, int cols, unsigned seed) -> cv::Mat {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 255);
    cv::Mat img(rows, cols, CV_8UC1);
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            img.ptr(y)[x] = static_cast<uint8_t>(dist(rng));
        }
    }
    return img;
}

auto profile(int rows, int cols, int runs, unsigned seed, int max_corners, double quality_level, double min_distance,
             int block_size) -> int {
    const cv::Mat img = make_noise_image(rows, cols, seed);

    std::vector<double> detect_s(runs);
    std::vector<size_t> corner_counts(runs);

    for (int run = 0; run < runs; ++run) {
        const auto t0 = std::chrono::steady_clock::now();
        const std::vector<ip::Corner> corners =
            ip::detect_corners_st(img, nullptr, max_corners, quality_level, min_distance, block_size);
        const auto t1 = std::chrono::steady_clock::now();
        detect_s[run] = std::chrono::duration<double>(t1 - t0).count();
        corner_counts[run] = corners.size();

        volatile double sink = 0.0;
        for (const ip::Corner& c : corners) {
            sink += c.score;
        }
    }

    const Stats detect = summarize(detect_s);
    const double avg_corners = static_cast<double>(std::accumulate(corner_counts.begin(), corner_counts.end(), 0ULL)) /
                               static_cast<double>(corner_counts.size());

    std::cout << "detect_corners_st profile (" << rows << "x" << cols << ", " << runs << " runs, seed " << seed
              << ", max_corners=" << max_corners << ", quality_level=" << quality_level
              << ", min_distance=" << min_distance << ", block_size=" << block_size << ")\n";
    print_stat("detect_corners_st", detect, 1e3, "ms");
    std::cout << std::left << std::setw(28) << "corners detected" << std::setprecision(6) << avg_corners << "\n";

    return 0;
}

} // namespace

auto main(int argc, char* argv[]) -> int {
    CLI::App app{"Functionality related to Shi-Tomasi corner detection."};

    int rows = kDefaultRows;
    int cols = kDefaultCols;
    int runs = kDefaultRuns;
    unsigned seed = kDefaultSeed;
    int max_corners = kDefaultMaxCorners;
    double quality_level = kDefaultQualityLevel;
    double min_distance = kDefaultMinDistance;
    int block_size = kDefaultBlockSize;

    auto* sub = app.add_subcommand("profile", "Profiling on a random noise image.");
    sub->add_option("--rows", rows, "Image height");
    sub->add_option("--cols", cols, "Image width");
    sub->add_option("--runs", runs, "Number of repeated runs for statistics");
    sub->add_option("--seed", seed, "RNG seed");
    sub->add_option("--max-corners", max_corners, "Maximum corners returned (0 = unlimited)");
    sub->add_option("--quality-level", quality_level, "Quality threshold as a fraction of the max score");
    sub->add_option("--min-distance", min_distance, "Minimum distance between corners (< 1 disables)");
    sub->add_option("--block-size", block_size, "Structure-tensor window size");
    sub->callback([&]() { profile(rows, cols, runs, seed, max_corners, quality_level, min_distance, block_size); });

    CLI11_PARSE(app, argc, argv);
    return 0;
}