#include <algorithm>
#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <setdetect/img_proc/corners_st.hpp>
#include <vector>

namespace ip = setdetect::img_proc;

namespace {

struct Pt {
    int x;
    int y;
};

auto make_checkerboard(int rows, int cols, int cell) -> cv::Mat {
    cv::Mat img(rows, cols, CV_8UC1);
    for (int y = 0; y < rows; ++y) {
        uint8_t* const row = img.ptr(y);
        for (int x = 0; x < cols; ++x) {
            row[x] = ((x / cell + y / cell) % 2 == 0) ? 255 : 0;
        }
    }
    return img;
}

auto make_uniform(int rows, int cols, uint8_t value) -> cv::Mat {
    cv::Mat img(rows, cols, CV_8UC1);
    for (int y = 0; y < rows; ++y) {
        std::fill(img.ptr(y), img.ptr(y) + cols, value);
    }
    return img;
}

// Mask that zeroes every pixel with x < split, leaves the rest enabled.
auto make_left_mask(int rows, int cols, int split) -> cv::Mat {
    cv::Mat img(rows, cols, CV_8UC1);
    for (int y = 0; y < rows; ++y) {
        uint8_t* const row = img.ptr(y);
        for (int x = 0; x < cols; ++x) {
            row[x] = x < split ? 0 : 255;
        }
    }
    return img;
}

auto dist2(Pt a, Pt b) -> int {
    const int dx = a.x - b.x;
    const int dy = a.y - b.y;
    return dx * dx + dy * dy;
}

constexpr int kTol2 = 4; // corners may sit up to 2 px from the ideal junction

auto corner_near(const ip::Corner& c, const Pt& j) -> bool { return dist2({c.x, c.y}, j) <= kTol2; }

// A 60x60 checkerboard with 20 px cells has exactly four interior grid
// junctions, at (20,20), (20,40), (40,20) and (40,40).
const std::array<Pt, 4> kJunctions = {Pt{20, 20}, Pt{20, 40}, Pt{40, 20}, Pt{40, 40}};

// Checks that every junction was detected.
auto expect_each_junction_detected(const std::vector<ip::Corner>& corners) -> void {
    for (const Pt& j : kJunctions) {
        const bool found =
            std::any_of(corners.begin(), corners.end(), [&](const ip::Corner& c) { return corner_near(c, j); });
        EXPECT_TRUE(found) << "junction (" << j.x << ", " << j.y << ") not detected";
    }
}

// Checks that every corner sits on a junction with a positive score.
auto expect_no_stray_corners(const std::vector<ip::Corner>& corners) -> void {
    for (const ip::Corner& c : corners) {
        const bool is_junction =
            std::any_of(kJunctions.begin(), kJunctions.end(), [&](const Pt& j) { return corner_near(c, j); });
        EXPECT_TRUE(is_junction) << "unexpected corner at (" << c.x << ", " << c.y << ")";
        EXPECT_GT(c.score, 0.0F) << "corner (" << c.x << ", " << c.y << ") has non-positive score";
    }
}

// Checks that the detected corners sit on the checkerboard junctions. With the
// >=-NMS used by cv, ties on the symmetric response plateau can surface as two
// adjacent corners per junction, so the size is only checked when exact.
auto check_junction_corners(const std::vector<ip::Corner>& corners, bool exact) -> void {
    if (exact) {
        ASSERT_EQ(corners.size(), kJunctions.size());
    }
    ASSERT_FALSE(corners.empty());
    expect_each_junction_detected(corners);
    expect_no_stray_corners(corners);
}

} // namespace

TEST(CornersStTest, CheckerboardJunctionsDetected) {
    const cv::Mat img = make_checkerboard(60, 60, 20);
    const std::vector<ip::Corner> corners = ip::detect_corners_st(img, nullptr, 1000, 0.01, 10.0);
    check_junction_corners(corners, true);
}

TEST(CornersStTest, MaxCornersLimitsResult) {
    const cv::Mat img = make_checkerboard(60, 60, 20);
    const std::vector<ip::Corner> corners = ip::detect_corners_st(img, nullptr, 2, 0.01, 10.0);
    EXPECT_EQ(corners.size(), 2);
}

TEST(CornersStTest, MinDistanceDropsCloseCorners) {
    const cv::Mat img = make_checkerboard(60, 60, 20);
    // Junctions sit 20 px apart horizontally/vertically and ~28.3 px on the
    // diagonal: radii below 20 keep all four, between 21 and 28 the two
    // opposing ones survive, and above that only one remains.
    EXPECT_EQ(ip::detect_corners_st(img, nullptr, 1000, 0.01, 10.0).size(), 4);
    EXPECT_EQ(ip::detect_corners_st(img, nullptr, 1000, 0.01, 25.0).size(), 2);
    EXPECT_EQ(ip::detect_corners_st(img, nullptr, 1000, 0.01, 50.0).size(), 1);
}

TEST(CornersStTest, SubPixelMinDistanceSkipsFilter) {
    const cv::Mat img = make_checkerboard(60, 60, 20);
    // max_corners == 0 means unlimited and min_distance < 1 disables the
    // distance filter, so every non-suppressed candidate is returned: at least
    // one per junction, possibly two where the response plateau ties.
    const std::vector<ip::Corner> corners = ip::detect_corners_st(img, nullptr, 0, 0.01, 0.5);
    check_junction_corners(corners, false);
}

TEST(CornersStTest, UniformImageHasNoCorners) {
    EXPECT_TRUE(ip::detect_corners_st(make_uniform(32, 32, 0), nullptr).empty());
    EXPECT_TRUE(ip::detect_corners_st(make_uniform(32, 32, 128), nullptr).empty());
}

TEST(CornersStTest, StraightEdgeHasNoCorners) {
    cv::Mat img(40, 40, CV_8UC1);
    for (int y = 0; y < 40; ++y) {
        uint8_t* const row = img.ptr(y);
        for (int x = 0; x < 40; ++x) {
            row[x] = x < 20 ? 0 : 255;
        }
    }
    EXPECT_TRUE(ip::detect_corners_st(img, nullptr).empty());
}

TEST(CornersStTest, MaskExcludesMaskedOutRegion) {
    const cv::Mat img = make_checkerboard(60, 60, 20);
    const cv::Mat mask = make_left_mask(60, 60, 30);
    const std::vector<ip::Corner> corners = ip::detect_corners_st(img, &mask, 1000, 0.01, 10.0);
    ASSERT_EQ(corners.size(), 2);
    for (const ip::Corner& c : corners) {
        EXPECT_GE(c.x, 30) << "corner (" << c.x << ", " << c.y << ") inside masked region";
    }
}

TEST(CornersStTest, MaskEmptyMeansNoCorners) {
    const cv::Mat img = make_checkerboard(60, 60, 20);
    const cv::Mat mask = make_uniform(60, 60, 0);
    EXPECT_TRUE(ip::detect_corners_st(img, &mask).empty());
}

TEST(CornersStTest, InvalidInputsReturnEmpty) {
    EXPECT_TRUE(ip::detect_corners_st(cv::Mat{}, nullptr).empty());
    EXPECT_TRUE(ip::detect_corners_st(cv::Mat(10, 10, CV_8UC3), nullptr).empty());
    const cv::Mat gray = make_uniform(10, 10, 128);
    EXPECT_TRUE(ip::detect_corners_st(gray, nullptr, -1).empty());
    EXPECT_TRUE(ip::detect_corners_st(gray, nullptr, 100, 0.0).empty());
    EXPECT_TRUE(ip::detect_corners_st(gray, nullptr, 100, 0.01, -1.0).empty());
    EXPECT_TRUE(ip::detect_corners_st(gray, nullptr, 100, 0.01, 10.0, 0).empty());
}
