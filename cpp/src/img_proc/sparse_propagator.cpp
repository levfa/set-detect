#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <setdetect/img_proc/corners_st.hpp>
#include <setdetect/img_proc/opt_flow_pyr_lk.hpp>
#include <setdetect/img_proc/sparse_propagator.hpp>
#include <setdetect/interpolation/tps_rbf.hpp>
#include <vector>

namespace setdetect::img_proc {

namespace {

auto seed_pts_grid(const cv::Mat& gray, const SparsePropagatorParams& params) -> std::vector<Pt2f> {
    std::vector<Pt2f> pts;
    const int h = gray.rows;
    const int w = gray.cols;
    const int rows = std::max(1, params.seed_rows);
    const int cols = std::max(1, params.seed_cols);
    std::vector<int> row_edges(rows + 1);
    std::vector<int> col_edges(cols + 1);
    for (int ry = 0; ry <= rows; ++ry) {
        row_edges[ry] =
            static_cast<int>(std::round(static_cast<float>(ry) * static_cast<float>(h) / static_cast<float>(rows)));
    }
    for (int cx = 0; cx <= cols; ++cx) {
        col_edges[cx] =
            static_cast<int>(std::round(static_cast<float>(cx) * static_cast<float>(w) / static_cast<float>(cols)));
    }
    for (int ry = 0; ry < rows; ++ry) {
        const int y0 = row_edges[ry];
        const int y1 = row_edges[ry + 1];
        for (int cx = 0; cx < cols; ++cx) {
            const int x0 = col_edges[cx];
            const int x1 = col_edges[cx + 1];
            const int cell_h = y1 - y0;
            const int cell_w = x1 - x0;
            if (cell_h <= 0 || cell_w <= 0) {
                continue;
            }
            const cv::Mat cell = gray(cv::Rect(x0, y0, cell_w, cell_h));
            const std::vector<Corner> corners =
                detect_corners_st(cell, nullptr, params.seed_max_corners_per_cell, params.seed_quality_level,
                                  params.seed_min_distance, 3);
            for (const Corner& c : corners) {
                pts.push_back({static_cast<float>(c.x + x0), static_cast<float>(c.y + y0)});
            }
        }
    }
    return pts;
}

struct MovementResult {
    std::vector<Pt2f> pts;
    std::vector<uint8_t> pts_mask;
    std::unique_ptr<interpolation::ThinPlateSplineRBF> interpolator;

    explicit operator bool() const { return interpolator != nullptr; }
};

auto estimate_movement(const cv::Mat& prev_img, const std::vector<Pt2f>& prev_pts, const cv::Mat& img,
                       const SparsePropagatorParams& params, OpticalFlowCalculator& flow_calc) -> MovementResult {
    MovementResult result;
    if (prev_pts.empty()) {
        return result;
    }

    const std::vector<Pt2f>& prev_pts_col = prev_pts;

    LkOptFlowParams lk_params;
    lk_params.win_size = params.lk_win_size;
    lk_params.max_level = params.lk_max_level;
    lk_params.criteria.max_iter = params.lk_criteria_max_iter;
    lk_params.criteria.eps = params.lk_criteria_eps;

    // Each image's pyramid is built once and reused for both the forward and backward
    // LK passes, instead of calc_optical_flow_pyr_lk rebuilding both images' pyramids
    // on every call (4 builds total for 2 calls, when only 2 are ever needed).
    const OptFlowPyramid prev_pyr = build_opt_flow_pyramid(prev_img, lk_params.max_level, lk_params.win_size);
    const OptFlowPyramid next_pyr = build_opt_flow_pyramid(img, lk_params.max_level, lk_params.win_size);

    std::vector<Pt2f> fwd_pts(prev_pts.size());
    std::vector<uint8_t> fwd_status(prev_pts.size());
    std::vector<float> fwd_err;
    flow_calc.calc_optical_flow(prev_pyr, next_pyr, prev_pts_col, fwd_pts, fwd_status, fwd_err, lk_params);

    std::vector<Pt2f> bwd_pts(fwd_pts.size());
    std::vector<uint8_t> bwd_status(fwd_pts.size());
    std::vector<float> bwd_err;
    // Backward pass: track from img back to prev_img, so the pyramid order is
    // deliberately swapped relative to the forward call above.
    // NOLINTNEXTLINE(readability-suspicious-call-argument)
    flow_calc.calc_optical_flow(next_pyr, prev_pyr, fwd_pts, bwd_pts, bwd_status, bwd_err, lk_params);

    const size_t n = prev_pts.size();
    result.pts.resize(n);
    result.pts_mask.assign(n, 0);

    std::vector<Eigen::VectorXd> rbf_inputs;
    std::vector<Eigen::VectorXd> rbf_outputs;
    rbf_inputs.reserve(n);
    rbf_outputs.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        const float dx = prev_pts[i].x - bwd_pts[i].x;
        const float dy = prev_pts[i].y - bwd_pts[i].y;
        const float fb_error = std::sqrt(dx * dx + dy * dy);
        const bool valid = (fwd_status[i] == 1) && (bwd_status[i] == 1) && (fb_error < params.fb_thresh);
        result.pts[i] = fwd_pts[i];
        result.pts_mask[i] = valid ? 1 : 0;

        if (valid) {
            Eigen::VectorXd in(2);
            Eigen::VectorXd out(2);
            in << fwd_pts[i].x, fwd_pts[i].y;
            out << fwd_pts[i].x - prev_pts[i].x, fwd_pts[i].y - prev_pts[i].y;
            rbf_inputs.push_back(in);
            rbf_outputs.push_back(out);
        }
    }

    if (static_cast<int>(rbf_inputs.size()) >= params.rbf_min_points) {
        if (params.rbf_max_points > 0 && static_cast<int>(rbf_inputs.size()) > params.rbf_max_points) {
            const float step = static_cast<float>(rbf_inputs.size()) / static_cast<float>(params.rbf_max_points);
            std::vector<Eigen::VectorXd> subsampled_in;
            std::vector<Eigen::VectorXd> subsampled_out;
            subsampled_in.reserve(params.rbf_max_points);
            subsampled_out.reserve(params.rbf_max_points);
            for (int k = 0; k < params.rbf_max_points; ++k) {
                const auto idx = static_cast<size_t>(std::floor(static_cast<float>(k) * step));
                if (idx < rbf_inputs.size()) {
                    subsampled_in.push_back(rbf_inputs[idx]);
                    subsampled_out.push_back(rbf_outputs[idx]);
                }
            }
            rbf_inputs.swap(subsampled_in);
            rbf_outputs.swap(subsampled_out);
        }
        try {
            result.interpolator =
                std::make_unique<interpolation::ThinPlateSplineRBF>(rbf_inputs, rbf_outputs, params.rbf_smoothing);
        } catch (...) {
            result.interpolator = nullptr;
        }
    }

    return result;
}

struct RingBuffer {
    std::vector<PropagationStep> buf;
    int size = 0;

    explicit RingBuffer(int capacity) : buf(capacity), size(capacity) {
        for (auto& step : buf) {
            step.frame_id = -1;
        }
    }

    void put(PropagationStep step) {
        const int slot = step.frame_id % size;
        if (slot < 0) {
            return;
        }
        buf[slot] = std::move(step);
    }

    [[nodiscard]] auto get(int frame_id) const -> const PropagationStep* {
        const int slot = frame_id % size;
        if (slot < 0) {
            return nullptr;
        }
        const PropagationStep& step = buf[slot];
        if (step.frame_id != frame_id) {
            return nullptr;
        }
        return &step;
    }
};

// Downscales by scale using INTER_AREA. scale >= 1 is a plain clone: upscaling is
// not a supported use case here, and this keeps the zero-motion first-frame behavior
// exact (no interpolation artifacts from a no-op resize).
auto scale_down(const cv::Mat& src, float scale) -> cv::Mat {
    if (scale >= 1.0F) {
        return src.clone();
    }
    cv::Mat dst;
    cv::resize(src, dst, cv::Size(), scale, scale, cv::INTER_AREA);
    return dst;
}

} // namespace

struct SparsePropagator::Impl {
    SparsePropagatorParams params;
    int frame_id = -1;
    cv::Mat prev_img_scaled;
    std::vector<Pt2f> prev_pts_scaled;
    RingBuffer history;
    OpticalFlowCalculator flow_calc;

    explicit Impl(const SparsePropagatorParams& p) : params(p), history(p.history_size) {}
};

SparsePropagator::SparsePropagator(const SparsePropagatorParams& params) : impl_(std::make_unique<Impl>(params)) {}

SparsePropagator::~SparsePropagator() = default;

SparsePropagator::SparsePropagator(SparsePropagator&&) noexcept = default;
auto SparsePropagator::operator=(SparsePropagator&&) noexcept -> SparsePropagator& = default;

void SparsePropagator::clear() {
    impl_->frame_id = -1;
    impl_->prev_img_scaled = cv::Mat();
    impl_->prev_pts_scaled.clear();
    impl_->history = RingBuffer(impl_->params.history_size);
}

auto SparsePropagator::frame_id() const -> int { return impl_->frame_id; }

auto SparsePropagator::next_img(const cv::Mat& gray) -> PropagationStep {
    cv::Mat img_scaled = scale_down(gray, impl_->params.scale);
    impl_->frame_id += 1;

    const cv::Mat* prev_img_scaled = impl_->prev_img_scaled.empty() ? nullptr : &impl_->prev_img_scaled;
    // Copy previous points before they get overwritten by new seeding
    const std::vector<Pt2f> prev_pts_scaled_copy = impl_->prev_pts_scaled;

    MovementResult movement;
    if (prev_img_scaled != nullptr && !prev_pts_scaled_copy.empty()) {
        movement =
            estimate_movement(*prev_img_scaled, prev_pts_scaled_copy, img_scaled, impl_->params, impl_->flow_calc);
    }

    impl_->prev_img_scaled = std::move(img_scaled);
    impl_->prev_pts_scaled = seed_pts_grid(impl_->prev_img_scaled, impl_->params);

    const float inv_scale = (impl_->params.scale > 0.0F) ? (1.0F / impl_->params.scale) : 1.0F;

    PropagationStep ps;
    ps.frame_id = impl_->frame_id;
    ps.pts.reserve(movement.pts.size());
    ps.prev_pts.reserve(prev_pts_scaled_copy.size());
    for (const Pt2f& p : movement.pts) {
        ps.pts.push_back({p.x * inv_scale, p.y * inv_scale});
    }
    for (const Pt2f& p : prev_pts_scaled_copy) {
        ps.prev_pts.push_back({p.x * inv_scale, p.y * inv_scale});
    }
    ps.pts_mask = movement.pts_mask;
    ps.interpolator = std::shared_ptr<interpolation::ThinPlateSplineRBF>(std::move(movement.interpolator));
    impl_->history.put(ps);
    // Retrieve the stored step to return a valid copy
    const PropagationStep* stored = impl_->history.get(impl_->frame_id);
    if (stored != nullptr) {
        return *stored;
    }
    return PropagationStep{};
}

auto SparsePropagator::propagate(int from_frame_id, std::vector<Pt2f>& pts) const -> bool {
    if (pts.empty()) {
        return true;
    }
    const float scale = impl_->params.scale;
    for (int fid = from_frame_id + 1; fid <= impl_->frame_id; ++fid) {
        const PropagationStep* ps = impl_->history.get(fid);
        if (ps == nullptr || !ps->interpolator) {
            return false;
        }
        try {
            const auto& interpolator = *ps->interpolator;
            for (Pt2f& p : pts) {
                Eigen::VectorXd query(2);
                query << p.x * scale, p.y * scale;
                Eigen::VectorXd disp = interpolator.interpolate(query);
                p.x += static_cast<float>(disp[0]) / scale;
                p.y += static_cast<float>(disp[1]) / scale;
            }
        } catch (...) {
            return false;
        }
    }
    return true;
}

} // namespace setdetect::img_proc
