#pragma once

#include <cstdint>
#include <memory>
#include <opencv2/core.hpp>
#include <vector>

namespace setdetect::img_proc {

struct Pt2f {
    float x = 0.0F;
    float y = 0.0F;
};

// Iteration/epsilon termination, mirroring cv::TermCriteria.
struct LkTermCriteria {
    int max_iter = 10;
    double eps = 0.03;
};

// Flags mirroring the cv::OPTFLOW_* bit values exactly (passed straight
// through to cv::calcOpticalFlowPyrLK).
enum LkOptFlowFlags : std::uint8_t {
    kUseInitialFlow = 4,
    kGetMinEigenvals = 8,
};

struct LkOptFlowParams {
    int win_size = 15;
    int max_level = 3;
    LkTermCriteria criteria;
    int flags = 0;
    double min_eig_threshold = 1e-4;
};

// What cv::buildOpticalFlowPyramid produces: an image + Scharr-derivative
// pyramid, opaque to callers, passed straight into
// cv::calcOpticalFlowPyrLK's pyramid-accepting overload. Building it once per
// frame and reusing it for both the forward and backward LK pass (see
// sparse_propagator.cpp) avoids rebuilding it twice per direction.
//
// tryReuseInputImage means level 0 may be a zero-copy view over the cv::Mat
// build_opt_flow_pyramid() was built from -- that Mat must outlive this
// pyramid.
struct OptFlowPyramid {
    std::vector<cv::Mat> levels;
};

// Builds an image pyramid via cv::buildOpticalFlowPyramid. win_size must
// match the win_size passed to the tracking call this pyramid is used
// for -- it determines the pyramid's border padding.
auto build_opt_flow_pyramid(const cv::Mat& gray, int max_level, int win_size) -> OptFlowPyramid;

// Sparse pyramidal iterative Lucas-Kanade optical flow -- thin wrapper around
// cv::calcOpticalFlowPyrLK. prev_pts, next_pts must have the same size.
// next_pts receives the tracked positions, status[i] is 1 on success, and err
// (when non-empty) holds cv::calcOpticalFlowPyrLK's own per-point error. When
// params.flags has kUseInitialFlow set, next_pts provides the initial guess.
auto calc_optical_flow_pyr_lk(const cv::Mat& prev_gray, const cv::Mat& next_gray, const std::vector<Pt2f>& prev_pts,
                              std::vector<Pt2f>& next_pts, std::vector<uint8_t>& status, std::vector<float>& err,
                              const LkOptFlowParams& params = {}) -> void;

// Same as calc_optical_flow_pyr_lk but on caller-built pyramids; the levels
// are treated as read-only so a single pair can serve both flow directions.
// threads, when > 0, temporarily overrides cv::setNumThreads() for the
// duration of this call (cv::calcOpticalFlowPyrLK parallelizes internally via
// cv::parallel_for_); -1 (default) leaves the process-wide setting alone.
// cv::setNumThreads is a process-wide setting, not scoped to this call or
// thread: passing threads > 0 here while another thread concurrently runs
// any OpenCV parallel operation can transiently change that operation's
// parallelism too. No current caller in this codebase does that (this
// function's own callers all use the default), but it's worth knowing before
// reaching for threads > 0 from a multi-threaded caller.
auto calc_optical_flow_on_pyramids(const OptFlowPyramid& prev_pyr, const OptFlowPyramid& next_pyr,
                                   const std::vector<Pt2f>& prev_pts, std::vector<Pt2f>& next_pts,
                                   std::vector<uint8_t>& status, std::vector<float>& err,
                                   const LkOptFlowParams& params = {}, int threads = -1) -> void;

// Thin, stateless wrapper kept for API compatibility with existing callers.
// cv::calcOpticalFlowPyrLK needs no persistent worker pool or scratch state
// of its own (OpenCV parallelizes internally), so this class exists purely so
// callers that hold one across frames (avoiding per-call construction cost,
// negligible here) don't need to change.
class OpticalFlowCalculator {
  public:
    explicit OpticalFlowCalculator(int max_workers = -1);
    ~OpticalFlowCalculator();

    OpticalFlowCalculator(const OpticalFlowCalculator&) = delete;
    auto operator=(const OpticalFlowCalculator&) -> OpticalFlowCalculator& = delete;

    OpticalFlowCalculator(OpticalFlowCalculator&&) noexcept;
    auto operator=(OpticalFlowCalculator&&) noexcept -> OpticalFlowCalculator&;

    // Same semantics as calc_optical_flow_on_pyramids.
    auto calc_optical_flow(const OptFlowPyramid& prev_pyr, const OptFlowPyramid& next_pyr,
                           const std::vector<Pt2f>& prev_pts, std::vector<Pt2f>& next_pts, std::vector<uint8_t>& status,
                           std::vector<float>& err, const LkOptFlowParams& params = {}, int threads = -1) -> void;

    // Convenience wrapper that builds both pyramids then tracks.
    auto calc_optical_flow(const cv::Mat& prev_gray, const cv::Mat& next_gray, const std::vector<Pt2f>& prev_pts,
                           std::vector<Pt2f>& next_pts, std::vector<uint8_t>& status, std::vector<float>& err,
                           const LkOptFlowParams& params = {}, int threads = -1) -> void;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace setdetect::img_proc
