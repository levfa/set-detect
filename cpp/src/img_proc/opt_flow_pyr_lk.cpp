#include <cstddef>
#include <cstdint>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <setdetect/img_proc/opt_flow_pyr_lk.hpp>
#include <utility>
#include <vector>

namespace setdetect::img_proc {

namespace {

auto to_cv_pts(const std::vector<Pt2f>& pts) -> std::vector<cv::Point2f> {
    std::vector<cv::Point2f> out(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) {
        out[i] = {pts[i].x, pts[i].y};
    }
    return out;
}

auto invalid_params(const LkOptFlowParams& p) -> bool {
    return p.win_size < 3 || p.max_level < 0 || p.criteria.max_iter <= 0;
}

// Scopes cv::setNumThreads() for one call when the caller asked for a
// specific thread count; restores the previous global setting on exit.
class ScopedThreadOverride {
  public:
    explicit ScopedThreadOverride(int threads) : prev_(cv::getNumThreads()), active_(threads > 0) {
        if (active_) {
            cv::setNumThreads(threads);
        }
    }
    ~ScopedThreadOverride() {
        if (active_) {
            cv::setNumThreads(prev_);
        }
    }
    ScopedThreadOverride(const ScopedThreadOverride&) = delete;
    auto operator=(const ScopedThreadOverride&) -> ScopedThreadOverride& = delete;
    ScopedThreadOverride(ScopedThreadOverride&&) = delete;
    auto operator=(ScopedThreadOverride&&) -> ScopedThreadOverride& = delete;

  private:
    int prev_;
    bool active_;
};

// Runs cv::calcOpticalFlowPyrLK against either a raw cv::Mat pair or a
// pre-built pyramid pair (both are valid cv::InputArray types). Reads
// next_pts as the initial guess first when kUseInitialFlow is set, since
// OpenCV's nextPts is an in-out parameter: callers must not reset next_pts
// before calling this.
template <typename PrevT, typename NextT>
void run_lk(const PrevT& prev, const NextT& next, const std::vector<Pt2f>& prev_pts, std::vector<Pt2f>& next_pts,
            std::vector<uint8_t>& status, std::vector<float>& err, const LkOptFlowParams& params, int threads) {
    const bool use_initial = (params.flags & kUseInitialFlow) != 0;
    std::vector<cv::Point2f> cv_next;
    if (use_initial) {
        cv_next = to_cv_pts(next_pts);
    }
    const std::vector<cv::Point2f> cv_prev = to_cv_pts(prev_pts);
    std::vector<uint8_t> cv_status;
    std::vector<float> cv_err;

    const ScopedThreadOverride thread_override(threads);
    cv::calcOpticalFlowPyrLK(prev, next, cv_prev, cv_next, cv_status, cv_err,
                             cv::Size(params.win_size, params.win_size), params.max_level,
                             cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, params.criteria.max_iter,
                                              params.criteria.eps),
                             params.flags, params.min_eig_threshold);

    next_pts.resize(cv_next.size());
    for (size_t i = 0; i < cv_next.size(); ++i) {
        next_pts[i] = {cv_next[i].x, cv_next[i].y};
    }
    status = std::move(cv_status);
    err = std::move(cv_err);
}

// Resets outputs to the invalid/empty-input contract shared by every entry
// point below: next_pts all-zero, status all-zero, err all-zero.
void reset_outputs(size_t n, std::vector<Pt2f>& next_pts, std::vector<uint8_t>& status, std::vector<float>& err) {
    next_pts.assign(n, Pt2f{});
    status.assign(n, 0);
    err.assign(n, 0.0F);
}

} // namespace

auto build_opt_flow_pyramid(const cv::Mat& gray, int max_level, int win_size) -> OptFlowPyramid {
    OptFlowPyramid pyr;
    if (gray.empty() || gray.channels() != 1 || win_size < 3 || max_level < 0) {
        return pyr;
    }
    cv::buildOpticalFlowPyramid(gray, pyr.levels, cv::Size(win_size, win_size), max_level);
    return pyr;
}

auto calc_optical_flow_pyr_lk(const cv::Mat& prev_gray, const cv::Mat& next_gray, const std::vector<Pt2f>& prev_pts,
                              std::vector<Pt2f>& next_pts, std::vector<uint8_t>& status, std::vector<float>& err,
                              const LkOptFlowParams& params) -> void {
    if (invalid_params(params) || prev_gray.empty() || next_gray.empty() || prev_gray.channels() != 1 ||
        next_gray.channels() != 1 || prev_gray.rows != next_gray.rows || prev_gray.cols != next_gray.cols ||
        prev_pts.empty()) {
        reset_outputs(prev_pts.size(), next_pts, status, err);
        return;
    }
    run_lk(prev_gray, next_gray, prev_pts, next_pts, status, err, params, -1);
}

auto calc_optical_flow_on_pyramids(const OptFlowPyramid& prev_pyr, const OptFlowPyramid& next_pyr,
                                   const std::vector<Pt2f>& prev_pts, std::vector<Pt2f>& next_pts,
                                   std::vector<uint8_t>& status, std::vector<float>& err, const LkOptFlowParams& params,
                                   int threads) -> void {
    if (invalid_params(params) || prev_pyr.levels.empty() || next_pyr.levels.empty() || prev_pts.empty()) {
        reset_outputs(prev_pts.size(), next_pts, status, err);
        return;
    }
    run_lk(prev_pyr.levels, next_pyr.levels, prev_pts, next_pts, status, err, params, threads);
}

struct OpticalFlowCalculator::Impl {};

OpticalFlowCalculator::OpticalFlowCalculator(int /*max_workers*/) : impl_(std::make_unique<Impl>()) {}

OpticalFlowCalculator::~OpticalFlowCalculator() = default;

OpticalFlowCalculator::OpticalFlowCalculator(OpticalFlowCalculator&&) noexcept = default;

auto OpticalFlowCalculator::operator=(OpticalFlowCalculator&&) noexcept -> OpticalFlowCalculator& = default;

// NOLINTBEGIN(readability-convert-member-functions-to-static) kept as instance
// methods for API stability: callers hold an OpticalFlowCalculator across
// frames, and cv::calcOpticalFlowPyrLK needs no per-instance state
// to delegate to (OpenCV parallelizes internally), so there's nothing for
// impl_ to hold; a future change that does need per-instance state (e.g. a
// result cache) shouldn't have to change every call site's shape.
auto OpticalFlowCalculator::calc_optical_flow(const OptFlowPyramid& prev_pyr, const OptFlowPyramid& next_pyr,
                                              const std::vector<Pt2f>& prev_pts, std::vector<Pt2f>& next_pts,
                                              std::vector<uint8_t>& status, std::vector<float>& err,
                                              const LkOptFlowParams& params, int threads) -> void {
    calc_optical_flow_on_pyramids(prev_pyr, next_pyr, prev_pts, next_pts, status, err, params, threads);
}

auto OpticalFlowCalculator::calc_optical_flow(const cv::Mat& prev_gray, const cv::Mat& next_gray,
                                              const std::vector<Pt2f>& prev_pts, std::vector<Pt2f>& next_pts,
                                              std::vector<uint8_t>& status, std::vector<float>& err,
                                              const LkOptFlowParams& params, int threads) -> void {
    if (invalid_params(params) || prev_gray.empty() || next_gray.empty() || prev_gray.channels() != 1 ||
        next_gray.channels() != 1 || prev_gray.rows != next_gray.rows || prev_gray.cols != next_gray.cols ||
        prev_pts.empty()) {
        reset_outputs(prev_pts.size(), next_pts, status, err);
        return;
    }
    run_lk(prev_gray, next_gray, prev_pts, next_pts, status, err, params, threads);
}
// NOLINTEND(readability-convert-member-functions-to-static)

} // namespace setdetect::img_proc
