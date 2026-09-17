#pragma once

#include <cstdint>
#include <memory>
#include <opencv2/core.hpp>
#include <setdetect/img_proc/opt_flow_pyr_lk.hpp>
#include <setdetect/interpolation/tps_rbf.hpp>
#include <vector>

namespace setdetect::img_proc {

struct PropagationStep {
    int frame_id = -1;
    std::vector<Pt2f> pts;
    std::vector<uint8_t> pts_mask;
    std::vector<Pt2f> prev_pts;
    std::shared_ptr<interpolation::ThinPlateSplineRBF> interpolator;
};

struct SparsePropagatorParams {
    int history_size = 60;
    float scale = 0.5F;
    int seed_rows = 8;
    int seed_cols = 8;
    int seed_max_corners_per_cell = 8;
    double seed_quality_level = 0.01;
    double seed_min_distance = 10.0;
    int lk_win_size = 15;
    int lk_max_level = 3;
    int lk_criteria_max_iter = 10;
    double lk_criteria_eps = 0.03;
    float fb_thresh = 1.0F;
    int rbf_min_points = 6;
    int rbf_max_points = 100;
    double rbf_smoothing = 1.0;
};

class SparsePropagator {
  public:
    explicit SparsePropagator(const SparsePropagatorParams& params = {});
    ~SparsePropagator();

    SparsePropagator(const SparsePropagator&) = delete;
    auto operator=(const SparsePropagator&) -> SparsePropagator& = delete;
    SparsePropagator(SparsePropagator&&) noexcept;
    auto operator=(SparsePropagator&&) noexcept -> SparsePropagator&;

    void clear();

    [[nodiscard]] auto frame_id() const -> int;

    auto next_img(const cv::Mat& gray) -> PropagationStep;

    // Propagates pts from from_frame_id to the current frame.
    // Modifies pts in-place. Returns false if propagation fails (missing history or interpolator).
    auto propagate(int from_frame_id, std::vector<Pt2f>& pts) const -> bool;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace setdetect::img_proc