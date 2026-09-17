#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <setdetect/img_proc/opt_flow_pyr_lk.hpp>
#include <vector>

namespace ip = setdetect::img_proc;

namespace {

// Smooth synthetic texture with strong gradients almost everywhere: a sum of
// low- and mid-frequency sinusoids plus a small deterministic noise term, so
// no neighbourhood is perfectly flat.
auto render(int x, int y) -> uint8_t {
    const double v =
        120.0 * std::sin(0.05 * x + 1.3) + 70.0 * std::cos(0.07 * y + 0.7) + 40.0 * std::sin(0.2 * (x + y));
    const unsigned h = static_cast<unsigned>(x) * 2654435761U ^ static_cast<unsigned>(y) * 2246822519U;
    const double n = static_cast<int>(h % 24) - 12;
    const double val = 128.0 + v + n;
    return static_cast<uint8_t>(std::clamp(val, 0.0, 255.0));
}

// Renders a textured image that is `prev` translated by (dx, dy): the value at
// (x, y) equals the source field at (x - dx, y - dy).
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

auto make_flat(int rows, int cols, uint8_t value) -> cv::Mat {
    cv::Mat img(rows, cols, CV_8UC1);
    for (int y = 0; y < rows; ++y) {
        std::fill(img.ptr(y), img.ptr(y) + cols, value);
    }
    return img;
}

// Asserts a single point tracked with status=1 at the expected (ex, ey) and a
// valid error.
auto expect_tracked_point(const std::vector<ip::Pt2f>& out, const std::vector<uint8_t>& status,
                          const std::vector<float>& err, size_t i, float ex, float ey) -> void {
    EXPECT_EQ(status[i], 1) << "point " << i << " failed to track";
    EXPECT_NEAR(out[i].x, ex, 0.5);
    EXPECT_NEAR(out[i].y, ey, 0.5);
    EXPECT_GE(err[i], 0.0F);
}

// Asserts every point tracked with status=1 at pts[i] + (dx, dy).
auto expect_points_at(const std::vector<ip::Pt2f>& pts, const std::vector<ip::Pt2f>& out,
                      const std::vector<uint8_t>& status, const std::vector<float>& err, float dx, float dy) -> void {
    ASSERT_EQ(out.size(), pts.size());
    ASSERT_EQ(status.size(), pts.size());
    for (size_t i = 0; i < pts.size(); ++i) {
        expect_tracked_point(out, status, err, i, pts[i].x + dx, pts[i].y + dy);
    }
}

// Asserts no point tracked (all status zero).
auto expect_none_track(const std::vector<ip::Pt2f>& pts, const std::vector<uint8_t>& status) -> void {
    ASSERT_EQ(status.size(), pts.size());
    for (const uint8_t s : status) {
        EXPECT_EQ(s, 0);
    }
}

constexpr int kRows = 120;
constexpr int kCols = 160;
constexpr int kWinSize = 15;

} // namespace

TEST(OptFlowPyrLkTest, NoMotionYieldsZeroFlow) {
    const cv::Mat prev = render_translated(kRows, kCols, 0, 0);
    const cv::Mat& next = prev;
    const std::vector<ip::Pt2f> pts = {{50.0F, 60.0F}, {90.0F, 40.0F}, {70.0F, 90.0F}};
    std::vector<ip::Pt2f> next_pts;
    std::vector<uint8_t> status;
    std::vector<float> err;
    ip::calc_optical_flow_pyr_lk(prev, next, pts, next_pts, status, err);
    expect_points_at(pts, next_pts, status, err, 0.0F, 0.0F);
}

TEST(OptFlowPyrLkTest, IntegerTranslationRecovered) {
    constexpr int dx = 5;
    constexpr int dy = -3;
    const cv::Mat prev = render_translated(kRows, kCols, 0, 0);
    const cv::Mat next = render_translated(kRows, kCols, dx, dy);
    const std::vector<ip::Pt2f> pts = {{50.0F, 60.0F}, {90.0F, 40.0F}, {70.0F, 90.0F}};
    std::vector<ip::Pt2f> next_pts;
    std::vector<uint8_t> status;
    std::vector<float> err;
    ip::calc_optical_flow_pyr_lk(prev, next, pts, next_pts, status, err);
    expect_points_at(pts, next_pts, status, err, static_cast<float>(dx), static_cast<float>(dy));
}

TEST(OptFlowPyrLkTest, UniformRegionFails) {
    const cv::Mat flat = make_flat(64, 64, 128);
    const std::vector<ip::Pt2f> pts = {{32.0F, 32.0F}};
    std::vector<ip::Pt2f> next_pts;
    std::vector<uint8_t> status;
    std::vector<float> err;
    ip::calc_optical_flow_pyr_lk(flat, flat, pts, next_pts, status, err);
    EXPECT_EQ(status[0], 0);
}

TEST(OptFlowPyrLkTest, FarOutOfBoundsPointFails) {
    const cv::Mat prev = render_translated(kRows, kCols, 0, 0);
    const cv::Mat& next = prev;
    // cv::calcOpticalFlowPyrLK pads each pyramid level with a winSize border and
    // tracks points that hang partway off the image edge (e.g. (-5, 10) tracks
    // successfully); only a point far outside the image should fail.
    const std::vector<ip::Pt2f> pts = {{200.0F, 200.0F}};
    std::vector<ip::Pt2f> next_pts;
    std::vector<uint8_t> status;
    std::vector<float> err;
    ip::calc_optical_flow_pyr_lk(prev, next, pts, next_pts, status, err);
    expect_none_track(pts, status);
}

TEST(OptFlowPyrLkTest, SmallerWindowStillTracks) {
    constexpr int dx = 3;
    constexpr int dy = 2;
    const cv::Mat prev = render_translated(kRows, kCols, 0, 0);
    const cv::Mat next = render_translated(kRows, kCols, dx, dy);
    const std::vector<ip::Pt2f> pts = {{60.0F, 60.0F}};
    std::vector<ip::Pt2f> next_pts;
    std::vector<uint8_t> status;
    std::vector<float> err;
    ip::LkOptFlowParams params;
    params.win_size = 15;
    ip::calc_optical_flow_pyr_lk(prev, next, pts, next_pts, status, err, params);
    EXPECT_EQ(status[0], 1);
    EXPECT_NEAR(next_pts[0].x, pts[0].x + dx, 0.5);
    EXPECT_NEAR(next_pts[0].y, pts[0].y + dy, 0.5);
}

TEST(OptFlowPyrLkTest, PyramidReusableAcrossDirections) {
    constexpr int dx = 4;
    constexpr int dy = 1;
    const cv::Mat prev = render_translated(kRows, kCols, 0, 0);
    const cv::Mat next = render_translated(kRows, kCols, dx, dy);
    const ip::OptFlowPyramid p1 = ip::build_opt_flow_pyramid(prev, 3, kWinSize);
    const ip::OptFlowPyramid p2 = ip::build_opt_flow_pyramid(next, 3, kWinSize);

    const std::vector<ip::Pt2f> fwd = {{50.0F, 60.0F}};
    std::vector<ip::Pt2f> fwd_out;
    std::vector<uint8_t> fwd_status;
    std::vector<float> fwd_err;
    ip::calc_optical_flow_on_pyramids(p1, p2, fwd, fwd_out, fwd_status, fwd_err);
    EXPECT_EQ(fwd_status[0], 1);
    EXPECT_NEAR(fwd_out[0].x, fwd[0].x + dx, 0.5);
    EXPECT_NEAR(fwd_out[0].y, fwd[0].y + dy, 0.5);

    const std::vector<ip::Pt2f> bwd = {fwd_out[0]};
    std::vector<ip::Pt2f> bwd_out;
    std::vector<uint8_t> bwd_status;
    std::vector<float> bwd_err;
    ip::calc_optical_flow_on_pyramids(p2, p1, bwd, bwd_out, bwd_status, bwd_err);
    EXPECT_EQ(bwd_status[0], 1);
    EXPECT_NEAR(bwd_out[0].x, fwd[0].x, 0.5);
    EXPECT_NEAR(bwd_out[0].y, fwd[0].y, 0.5);
}

TEST(OptFlowPyrLkTest, InvalidInputsYieldNoTracking) {
    const std::vector<ip::Pt2f> pts = {{10.0F, 10.0F}};
    std::vector<ip::Pt2f> next_pts;
    std::vector<uint8_t> status;
    std::vector<float> err;

    // Empty images: outputs are sized to prev_pts but no point tracks.
    ip::calc_optical_flow_pyr_lk(cv::Mat{}, cv::Mat{}, pts, next_pts, status, err);
    ASSERT_EQ(next_pts.size(), pts.size());
    ASSERT_EQ(status.size(), pts.size());
    EXPECT_EQ(status[0], 0);

    // win_size < 3 is invalid: no tracking.
    ip::LkOptFlowParams params;
    params.win_size = 2;
    const cv::Mat prev = render_translated(kRows, kCols, 0, 0);
    ip::calc_optical_flow_pyr_lk(prev, prev, pts, next_pts, status, err, params);
    ASSERT_EQ(status.size(), pts.size());
    EXPECT_EQ(status[0], 0);

    // Multi-channel images are not supported: no tracking.
    const cv::Mat color(8, 8, CV_8UC3);
    ip::calc_optical_flow_pyr_lk(color, color, pts, next_pts, status, err);
    ASSERT_EQ(status.size(), pts.size());
    EXPECT_EQ(status[0], 0);
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity) gtest ASSERT macros inflate the count
TEST(OptFlowPyrLkTest, ThreadOverrideDoesNotChangeResults) {
    constexpr int dx = 5;
    constexpr int dy = -3;
    const cv::Mat prev = render_translated(kRows, kCols, 0, 0);
    const cv::Mat next = render_translated(kRows, kCols, dx, dy);
    const ip::OptFlowPyramid p1 = ip::build_opt_flow_pyramid(prev, 3, kWinSize);
    const ip::OptFlowPyramid p2 = ip::build_opt_flow_pyramid(next, 3, kWinSize);

    // Multiple points across several pyramid levels.
    const std::vector<ip::Pt2f> pts = {{50.0F, 60.0F}, {90.0F, 40.0F}, {70.0F, 90.0F}, {120.0F, 30.0F}};

    std::vector<ip::Pt2f> serial_out;
    std::vector<uint8_t> serial_status;
    std::vector<float> serial_err;
    ip::calc_optical_flow_on_pyramids(p1, p2, pts, serial_out, serial_status, serial_err, {}, 1);

    std::vector<ip::Pt2f> parallel_out;
    std::vector<uint8_t> parallel_status;
    std::vector<float> parallel_err;
    ip::calc_optical_flow_on_pyramids(p1, p2, pts, parallel_out, parallel_status, parallel_err, {}, 4);

    ASSERT_EQ(serial_out.size(), parallel_out.size());
    ASSERT_EQ(serial_status.size(), parallel_status.size());
    ASSERT_EQ(serial_err.size(), parallel_err.size());
    for (size_t i = 0; i < pts.size(); ++i) {
        EXPECT_EQ(serial_status[i], parallel_status[i]) << "status mismatch at point " << i;
        if (serial_status[i] == 1) {
            EXPECT_EQ(parallel_status[i], 1) << "point " << i << " should track in both";
            EXPECT_FLOAT_EQ(serial_out[i].x, parallel_out[i].x) << "x mismatch at point " << i;
            EXPECT_FLOAT_EQ(serial_out[i].y, parallel_out[i].y) << "y mismatch at point " << i;
            EXPECT_FLOAT_EQ(serial_err[i], parallel_err[i]) << "err mismatch at point " << i;
        }
    }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity) gtest ASSERT macros inflate the count
TEST(OptFlowPyrLkTest, CalculatorMatchesFreeFunction) {
    constexpr int dx = 5;
    constexpr int dy = -3;
    const cv::Mat prev = render_translated(kRows, kCols, 0, 0);
    const cv::Mat next = render_translated(kRows, kCols, dx, dy);
    const ip::OptFlowPyramid p1 = ip::build_opt_flow_pyramid(prev, 3, kWinSize);
    const ip::OptFlowPyramid p2 = ip::build_opt_flow_pyramid(next, 3, kWinSize);
    const std::vector<ip::Pt2f> pts = {{50.0F, 60.0F}, {90.0F, 40.0F}, {70.0F, 90.0F}, {120.0F, 30.0F}};

    std::vector<ip::Pt2f> free_out;
    std::vector<uint8_t> free_status;
    std::vector<float> free_err;
    ip::calc_optical_flow_on_pyramids(p1, p2, pts, free_out, free_status, free_err, {}, 4);

    ip::OpticalFlowCalculator calc(4);
    std::vector<ip::Pt2f> cls_out;
    std::vector<uint8_t> cls_status;
    std::vector<float> cls_err;
    calc.calc_optical_flow(p1, p2, pts, cls_out, cls_status, cls_err);

    ASSERT_EQ(free_out.size(), cls_out.size());
    ASSERT_EQ(free_status.size(), cls_status.size());
    ASSERT_EQ(free_err.size(), cls_err.size());
    for (size_t i = 0; i < pts.size(); ++i) {
        EXPECT_EQ(free_status[i], cls_status[i]) << "status mismatch at point " << i;
        if (free_status[i] == 1) {
            EXPECT_FLOAT_EQ(free_out[i].x, cls_out[i].x) << "x mismatch at point " << i;
            EXPECT_FLOAT_EQ(free_out[i].y, cls_out[i].y) << "y mismatch at point " << i;
            EXPECT_FLOAT_EQ(free_err[i], cls_err[i]) << "err mismatch at point " << i;
        }
    }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity) gtest ASSERT macros inflate the count
TEST(OptFlowPyrLkTest, CalculatorPoolReuse) {
    constexpr int dx = 5;
    constexpr int dy = -3;
    const cv::Mat prev = render_translated(kRows, kCols, 0, 0);
    const cv::Mat next = render_translated(kRows, kCols, dx, dy);
    const ip::OptFlowPyramid p1 = ip::build_opt_flow_pyramid(prev, 3, kWinSize);
    const ip::OptFlowPyramid p2 = ip::build_opt_flow_pyramid(next, 3, kWinSize);
    const std::vector<ip::Pt2f> pts = {{50.0F, 60.0F}, {90.0F, 40.0F}, {70.0F, 90.0F}, {120.0F, 30.0F}};

    ip::OpticalFlowCalculator calc(4);
    std::vector<ip::Pt2f> out1;
    std::vector<uint8_t> status1;
    std::vector<float> err1;
    calc.calc_optical_flow(p1, p2, pts, out1, status1, err1);

    // Second call on the same instance must be bit-identical.
    std::vector<ip::Pt2f> out2;
    std::vector<uint8_t> status2;
    std::vector<float> err2;
    calc.calc_optical_flow(p1, p2, pts, out2, status2, err2);

    ASSERT_EQ(out1.size(), out2.size());
    ASSERT_EQ(status1.size(), status2.size());
    ASSERT_EQ(err1.size(), err2.size());
    EXPECT_EQ(status1, status2);
    EXPECT_EQ(err1, err2);
    for (size_t i = 0; i < pts.size(); ++i) {
        if (status1[i] == 1) {
            EXPECT_FLOAT_EQ(out1[i].x, out2[i].x) << "x mismatch at point " << i;
            EXPECT_FLOAT_EQ(out1[i].y, out2[i].y) << "y mismatch at point " << i;
        }
    }
}
