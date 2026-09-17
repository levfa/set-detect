#include <CLI/CLI.hpp>
#include <algorithm>
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
#include <setdetect/img_proc/opt_flow_pyr_lk.hpp>
#include <thread>
#include <vector>

namespace ip = setdetect::img_proc;
using setdetect::apps::print_stat;
using setdetect::apps::Stats;
using setdetect::apps::summarize;

namespace {

// Defaults mirror the real card-tracking workload: frames are capped at 640px
// on the longest side and then run LK at 0.5x propagation scale, with the
// window/criteria of the production cv::calcOpticalFlowPyrLK call.
constexpr int kDefaultRows = 240;
constexpr int kDefaultCols = 320;
constexpr int kDefaultRuns = 10;
constexpr unsigned kDefaultSeed = 42;
constexpr int kDefaultNPoints = 200;
constexpr int kDefaultWinSize = 15;
constexpr int kDefaultMaxLevel = 3;

// Smooth fake texture with strong gradients everywhere, so most points track.
auto make_texture(int rows, int cols, unsigned seed) -> cv::Mat {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(-40, 40);
    cv::Mat img(rows, cols, CV_8UC1);
    for (int y = 0; y < rows; ++y) {
        uint8_t* row = img.ptr(y);
        for (int x = 0; x < cols; ++x) {
            const double v =
                120.0 * std::sin(0.05 * x + 1.3) + 70.0 * std::cos(0.07 * y + 0.7) + 40.0 * std::sin(0.2 * (x + y));
            const auto n = static_cast<double>(dist(rng));
            row[x] = static_cast<uint8_t>(std::clamp(128.0 + v + n, 0.0, 255.0));
        }
    }
    return img;
}

// A translated copy of `prev` by (dx, dy) whole pixels via its generating
// field, so the flow is exactly known.
auto shift_texture(const cv::Mat& prev, int dx, int dy, unsigned seed) -> cv::Mat {
    (void)prev;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(-40, 40);
    cv::Mat img(prev.rows, prev.cols, CV_8UC1);
    for (int y = 0; y < prev.rows; ++y) {
        uint8_t* row = img.ptr(y);
        for (int x = 0; x < prev.cols; ++x) {
            const double v = 120.0 * std::sin(0.05 * (x - dx) + 1.3) + 70.0 * std::cos(0.07 * (y - dy) + 0.7) +
                             40.0 * std::sin(0.2 * (x - dx + y - dy));
            const auto n = static_cast<double>(dist(rng));
            row[x] = static_cast<uint8_t>(std::clamp(128.0 + v + n, 0.0, 255.0));
        }
    }
    return img;
}

auto profile(int rows, int cols, int runs, unsigned seed, int n_points, int win_size, int max_level,
             int threads) -> int {
    const cv::Mat prev = make_texture(rows, cols, seed);
    const cv::Mat next = shift_texture(prev, 4, 3, seed + 1);

    std::mt19937 rng(seed + 2);
    std::uniform_int_distribution<int> px_dist(win_size / 2 + 4, cols - win_size / 2 - 4);
    std::uniform_int_distribution<int> py_dist(win_size / 2 + 4, rows - win_size / 2 - 4);
    std::vector<ip::Pt2f> prev_pts;
    prev_pts.reserve(static_cast<size_t>(n_points));
    for (int i = 0; i < n_points; ++i) {
        prev_pts.push_back({static_cast<float>(px_dist(rng)), static_cast<float>(py_dist(rng))});
    }

    ip::LkOptFlowParams params;
    params.win_size = win_size;
    params.max_level = max_level;

    // One persistent calculator: the pool is created once and reused across
    // runs, which is the realistic per-frame streaming pattern.
    const int hw = static_cast<int>(std::thread::hardware_concurrency());
    int pool_threads = threads > 0 ? threads : hw;
    if (pool_threads <= 0) {
        pool_threads = 1;
    }
    ip::OpticalFlowCalculator calculator(pool_threads);

    std::vector<double> flow_s(runs);
    std::vector<size_t> tracked_counts(runs);

    for (int run = 0; run < runs; ++run) {
        std::vector<ip::Pt2f> next_pts;
        std::vector<uint8_t> status;
        std::vector<float> err;
        const auto t0 = std::chrono::steady_clock::now();
        calculator.calc_optical_flow(prev, next, prev_pts, next_pts, status, err, params);
        const auto t1 = std::chrono::steady_clock::now();
        flow_s[run] = std::chrono::duration<double>(t1 - t0).count();

        size_t tracked = 0;
        for (uint8_t const s : status) {
            tracked += (s != 0) ? 1 : 0;
        }
        tracked_counts[run] = tracked;

        volatile double sink = 0.0;
        for (const auto& pt : next_pts) {
            sink += pt.x + pt.y;
        }
        (void)sink;
    }

    const Stats flow = summarize(flow_s);
    const double avg_tracked =
        static_cast<double>(std::accumulate(tracked_counts.begin(), tracked_counts.end(), 0ULL)) /
        static_cast<double>(tracked_counts.size());

    std::cout << "Settings: " << rows << "x" << cols << ", " << runs << " runs, seed " << seed
              << ", n_points=" << n_points << ", win_size=" << win_size << ", max_level=" << max_level
              << ", threads=" << threads << "\n";
    print_stat("calc_optical_flow", flow, 1e3, "ms", 35);
    std::cout << std::left << std::setw(35) << "points tracked" << std::setprecision(6) << avg_tracked << "\n";

    return 0;
}

} // namespace

auto main(int argc, char* argv[]) -> int {
    CLI::App app{"Functionality related to sparse pyramidal Lucas-Kanade optical flow."};

    int rows = kDefaultRows;
    int cols = kDefaultCols;
    int runs = kDefaultRuns;
    unsigned seed = kDefaultSeed;
    int n_points = kDefaultNPoints;
    int win_size = kDefaultWinSize;
    int max_level = kDefaultMaxLevel;
    int threads = -1;

    auto* sub = app.add_subcommand("profile", "Profiling on a synthetic moving texture.");
    sub->add_option("--rows", rows, "Image height");
    sub->add_option("--cols", cols, "Image width");
    sub->add_option("--runs", runs, "Number of repeated runs for statistics");
    sub->add_option("--seed", seed, "RNG seed");
    sub->add_option("--n-points", n_points, "Number of tracked points");
    sub->add_option("--win-size", win_size, "Integration window size");
    sub->add_option("--max-level", max_level, "Maximal pyramid level");
    sub->add_option("--threads", threads, "Worker threads (-1 = auto)");
    sub->callback([&]() { profile(rows, cols, runs, seed, n_points, win_size, max_level, threads); });

    CLI11_PARSE(app, argc, argv);
    return 0;
}
