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
#include <setdetect/img_proc/sparse_propagator.hpp>
#include <thread>
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
constexpr int kDefaultHistorySize = 60;
constexpr float kDefaultScale = 0.5F;

auto render(int x, int y) -> uint8_t {
    const double v =
        120.0 * std::sin(0.05 * x + 1.3) + 70.0 * std::cos(0.07 * y + 0.7) + 40.0 * std::sin(0.2 * (x + y));
    const unsigned h = static_cast<unsigned>(x) * 2654435761U ^ static_cast<unsigned>(y) * 2246822519U;
    const double n = static_cast<int>(h % 24) - 12;
    const double val = 128.0 + v + n;
    return static_cast<uint8_t>(std::clamp(val, 0.0, 255.0));
}

auto render_translated(int rows, int cols, int dx, int dy) -> cv::Mat {
    cv::Mat img(rows, cols, CV_8UC1);
    for (int y = 0; y < rows; ++y) {
        uint8_t* row = img.ptr(y);
        for (int x = 0; x < cols; ++x) {
            row[x] = render(x - dx, y - dy);
        }
    }
    return img;
}

auto profile_next_img(int rows, int cols, int runs, unsigned seed, int history_size, float scale, int seed_rows,
                      int seed_cols, int seed_max_corners) -> int {
    ip::SparsePropagatorParams params;
    params.history_size = history_size;
    params.scale = scale;
    params.seed_rows = seed_rows;
    params.seed_cols = seed_cols;
    params.seed_max_corners_per_cell = seed_max_corners;

    std::vector<double> next_img_s(runs);
    std::vector<size_t> tracked_counts(runs);
    std::vector<size_t> interpolator_counts(runs);

    for (int run = 0; run < runs; ++run) {
        ip::SparsePropagator propagator(params);
        const int n_frames = 10;

        for (int f = 0; f < n_frames; ++f) {
            const cv::Mat frame = render_translated(rows, cols, f * 2, f);
            const auto t0 = std::chrono::steady_clock::now();
            const ip::PropagationStep ps = propagator.next_img(frame);
            const auto t1 = std::chrono::steady_clock::now();
            next_img_s[run] += std::chrono::duration<double>(t1 - t0).count();

            size_t tracked = 0;
            for (uint8_t m : ps.pts_mask) {
                tracked += (m != 0) ? 1 : 0;
            }
            tracked_counts[run] = tracked;
            interpolator_counts[run] = (ps.interpolator != nullptr) ? 1 : 0;
        }
        next_img_s[run] /= n_frames;
    }

    const Stats next_img = summarize(next_img_s);
    const double avg_tracked =
        static_cast<double>(std::accumulate(tracked_counts.begin(), tracked_counts.end(), 0ULL)) /
        tracked_counts.size();
    const double interpolator_rate =
        static_cast<double>(std::accumulate(interpolator_counts.begin(), interpolator_counts.end(), 0ULL)) /
        interpolator_counts.size();

    std::cout << "next_img profile (" << rows << "x" << cols << ", " << runs << " runs, seed " << seed
              << ", history=" << history_size << ", scale=" << scale << ", seed=" << seed_rows << "x" << seed_cols
              << ", max_corners/cell=" << seed_max_corners << ")\n";
    print_stat("next_img (per frame)", next_img, 1e3, "ms", 35);
    std::cout << std::left << std::setw(35) << "tracked points/frame" << std::setprecision(6) << avg_tracked << "\n";
    std::cout << std::left << std::setw(35) << "interpolator rate" << std::setprecision(6) << interpolator_rate << "\n";

    return 0;
}

auto profile_propagate(int rows, int cols, int runs, unsigned seed, int history_size, float scale, int seed_rows,
                       int seed_cols, int seed_max_corners, int n_frames_propagate) -> int {
    ip::SparsePropagatorParams params;
    params.history_size = history_size;
    params.scale = scale;
    params.seed_rows = seed_rows;
    params.seed_cols = seed_cols;
    params.seed_max_corners_per_cell = seed_max_corners;

    std::vector<double> propagate_s(runs);

    for (int run = 0; run < runs; ++run) {
        ip::SparsePropagator propagator(params);

        const int n_total_frames = n_frames_propagate + 5;
        for (int f = 0; f < n_total_frames; ++f) {
            const cv::Mat frame = render_translated(rows, cols, f * 2, f);
            propagator.next_img(frame);
        }

        std::vector<ip::Pt2f> query_pts = {{static_cast<float>(cols / 2), static_cast<float>(rows / 2)}};
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = propagator.propagate(0, query_pts);
        const auto t1 = std::chrono::steady_clock::now();
        propagate_s[run] = std::chrono::duration<double>(t1 - t0).count();
        (void)ok;
    }

    const Stats propagate = summarize(propagate_s);

    std::cout << "propagate profile (" << rows << "x" << cols << ", " << runs << " runs, seed " << seed
              << ", history=" << history_size << ", scale=" << scale << ", seed=" << seed_rows << "x" << seed_cols
              << ", max_corners/cell=" << seed_max_corners << ", frames=" << n_frames_propagate << ")\n";
    print_stat("propagate", propagate, 1e6, "us", 35);

    return 0;
}

} // namespace

auto main(int argc, char* argv[]) -> int {
    CLI::App app{"SparsePropagator."};

    int rows = kDefaultRows;
    int cols = kDefaultCols;
    int runs = kDefaultRuns;
    unsigned seed = kDefaultSeed;
    int history_size = kDefaultHistorySize;
    float scale = kDefaultScale;
    int seed_rows = 8;
    int seed_cols = 8;
    int seed_max_corners = 8;
    int n_frames_propagate = 5;

    auto* sub_next = app.add_subcommand("profile-next-img", "Profile next_img per-frame latency.");
    sub_next->add_option("--rows", rows, "Image height");
    sub_next->add_option("--cols", cols, "Image width");
    sub_next->add_option("--runs", runs, "Number of repeated runs for statistics");
    sub_next->add_option("--seed", seed, "RNG seed");
    sub_next->add_option("--history", history_size, "History buffer size");
    sub_next->add_option("--scale", scale, "Downscale factor for tracking (0 < scale <= 1)");
    sub_next->add_option("--seed-rows", seed_rows, "Number of seed grid rows");
    sub_next->add_option("--seed-cols", seed_cols, "Number of seed grid columns");
    sub_next->add_option("--max-corners", seed_max_corners, "Max corners per grid cell");
    sub_next->callback([&]() {
        profile_next_img(rows, cols, runs, seed, history_size, scale, seed_rows, seed_cols, seed_max_corners);
    });

    auto* sub_prop = app.add_subcommand("profile-propagate", "Profile propagate latency over N frames.");
    sub_prop->add_option("--rows", rows, "Image height");
    sub_prop->add_option("--cols", cols, "Image width");
    sub_prop->add_option("--runs", runs, "Number of repeated runs for statistics");
    sub_prop->add_option("--seed", seed, "RNG seed");
    sub_prop->add_option("--history", history_size, "History buffer size");
    sub_prop->add_option("--scale", scale, "Downscale factor for tracking (0 < scale <= 1)");
    sub_prop->add_option("--seed-rows", seed_rows, "Number of seed grid rows");
    sub_prop->add_option("--seed-cols", seed_cols, "Number of seed grid columns");
    sub_prop->add_option("--max-corners", seed_max_corners, "Max corners per grid cell");
    sub_prop->add_option("--frames", n_frames_propagate, "Number of frames to propagate across");
    sub_prop->callback([&]() {
        profile_propagate(rows, cols, runs, seed, history_size, scale, seed_rows, seed_cols, seed_max_corners,
                          n_frames_propagate);
    });

    CLI11_PARSE(app, argc, argv);
    return 0;
}