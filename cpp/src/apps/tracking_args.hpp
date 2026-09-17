#pragma once

#include <CLI/CLI.hpp>
#include <setdetect/img_proc/sparse_propagator.hpp>

namespace setdetect::apps {

struct TrackingArgs {
    img_proc::SparsePropagatorParams params;
};

// Registers --track-* flags (mirrors Python's card_tracking.py's --propagation-*/
// --seed-*/--move-* flags) on `sub`.
inline void add_tracking_args(CLI::App& sub, TrackingArgs& args) {
    sub.add_option("--track-history-size", args.params.history_size, "Propagation history ring-buffer size");
    sub.add_option("--track-scale", args.params.scale, "Working-resolution scale for propagation");
    sub.add_option("--track-seed-rows", args.params.seed_rows, "Seed grid rows");
    sub.add_option("--track-seed-cols", args.params.seed_cols, "Seed grid cols");
    sub.add_option("--track-seed-max-corners", args.params.seed_max_corners_per_cell, "Max corners per seed cell");
    sub.add_option("--track-seed-quality", args.params.seed_quality_level, "Seed corner quality level");
    sub.add_option("--track-seed-min-distance", args.params.seed_min_distance, "Seed corner min distance");
    sub.add_option("--track-lk-win-size", args.params.lk_win_size, "Optical flow window size");
    sub.add_option("--track-lk-max-level", args.params.lk_max_level, "Optical flow pyramid levels");
    sub.add_option("--track-lk-max-iter", args.params.lk_criteria_max_iter, "Optical flow max iterations");
    sub.add_option("--track-lk-eps", args.params.lk_criteria_eps, "Optical flow epsilon");
    sub.add_option("--track-fb-thresh", args.params.fb_thresh, "Forward-backward error threshold");
    sub.add_option("--track-rbf-min-points", args.params.rbf_min_points, "Min points to fit the RBF interpolator");
    sub.add_option("--track-rbf-max-points", args.params.rbf_max_points, "Max points used to fit the RBF interpolator");
    sub.add_option("--track-rbf-smoothing", args.params.rbf_smoothing, "RBF regularization (0 = exact interpolation)");
}

} // namespace setdetect::apps
