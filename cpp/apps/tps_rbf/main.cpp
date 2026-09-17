#include <CLI/CLI.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <apps/stats.hpp>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <setdetect/interpolation/tps_rbf.hpp>
#include <vector>

namespace si = setdetect::interpolation;
using setdetect::apps::print_stat;
using setdetect::apps::Stats;
using setdetect::apps::summarize;

namespace {

constexpr int kDefaultNumPoints = 100;
constexpr int kDefaultNumQueries = 48;
constexpr int kDefaultNumRuns = 10;
constexpr unsigned kDefaultSeed = 42;

auto profile(int n_points, int n_queries, int n_runs, unsigned seed) -> int {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::vector<double> construction_s(n_runs);
    std::vector<double> interpolation_s(n_runs);

    for (int run = 0; run < n_runs; ++run) {
        std::vector<Eigen::VectorXd> inputs;
        std::vector<Eigen::VectorXd> outputs;
        inputs.reserve(n_points);
        outputs.reserve(n_points);
        for (int i = 0; i < n_points; ++i) {
            Eigen::VectorXd in(2);
            Eigen::VectorXd out(2);
            in << dist(rng), dist(rng);
            out << dist(rng), dist(rng);
            inputs.push_back(in);
            outputs.push_back(out);
        }

        auto t0 = std::chrono::steady_clock::now();
        si::ThinPlateSplineRBF const rbf(inputs, outputs);
        auto t1 = std::chrono::steady_clock::now();
        construction_s[run] = std::chrono::duration<double>(t1 - t0).count();

        auto q0 = std::chrono::steady_clock::now();
        volatile double sink = 0.0;
        for (int q = 0; q < n_queries; ++q) {
            Eigen::VectorXd point(2);
            point << dist(rng), dist(rng);
            sink += rbf.interpolate(point).norm();
        }
        auto q1 = std::chrono::steady_clock::now();
        interpolation_s[run] = std::chrono::duration<double>(q1 - q0).count();
    }

    Stats const c = summarize(construction_s);
    Stats const interp = summarize(interpolation_s);

    std::cout << "TPS-RBF profile (" << n_points << " points, " << n_queries << " queries, " << n_runs << " runs, seed "
              << seed << ")\n";
    print_stat("construction", c, 1e3, "ms");
    print_stat("interpolation (total)", interp, 1e3, "ms");
    print_stat("interpolation (per query)", interp, 1e6 / n_queries, "us");

    return 0;
}

} // namespace

auto main(int argc, char* argv[]) -> int {
    CLI::App app{"Functionality related to thin-plane-spline radial-basis-function (TPS-RBF) interpolation."};

    int n_points = kDefaultNumPoints;
    int n_queries = kDefaultNumQueries;
    int n_runs = kDefaultNumRuns;
    unsigned seed = kDefaultSeed;

    auto* sub = app.add_subcommand("profile", "Profiles TPS-RBF construction and interpolation.");
    sub->add_option("--n-points", n_points, "Number of interpolation points");
    sub->add_option("--n-queries", n_queries, "Number of query points per run");
    sub->add_option("--runs", n_runs, "Number of repeated runs for statistics");
    sub->add_option("--seed", seed, "RNG seed");
    sub->callback([&]() { profile(n_points, n_queries, n_runs, seed); });

    CLI11_PARSE(app, argc, argv);
    return 0;
}
