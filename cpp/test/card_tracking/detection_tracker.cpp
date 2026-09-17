#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <setdetect/card_detection/detector.hpp>
#include <setdetect/card_tracking/detection_tracker.hpp>
#include <setdetect/img_proc/sparse_propagator.hpp>
#include <setdetect/set_game/game.hpp>
#include <vector>

namespace cd = setdetect::card_detection;
namespace ct = setdetect::card_tracking;
namespace ip = setdetect::img_proc;
namespace sg = setdetect::set_game;

namespace {

constexpr int kRows = 120;
constexpr int kCols = 160;

// Same deterministic textured-pattern renderer used by img_proc/sparse_propagator.cpp's
// tests, so SparsePropagator has real, trackable content to work with.
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

// matched: whether every head's argmax lands on a real class (index != 3/"other").
auto make_detected_card(double cx, double cy, bool matched) -> cd::DetectedCard {
    cd::DetectedCard dc;
    dc.corners = {cd::Point2d{cx - 5, cy - 5}, cd::Point2d{cx + 5, cy - 5}, cd::Point2d{cx + 5, cy + 5},
                  cd::Point2d{cx - 5, cy + 5}};
    dc.corner_visibility = {1.0, 1.0, 1.0, 1.0};
    dc.count_probs = {0.9, 0.03, 0.03, 0.04};
    dc.color_probs = {0.9, 0.03, 0.03, 0.04};
    dc.shape_probs = {0.9, 0.03, 0.03, 0.04};
    dc.fill_probs =
        matched ? std::array<double, 4>{0.9, 0.03, 0.03, 0.04} : std::array<double, 4>{0.03, 0.03, 0.03, 0.91};
    return dc;
}

auto permissive_params() -> ip::SparsePropagatorParams {
    ip::SparsePropagatorParams params;
    params.scale = 1.0F;
    params.seed_rows = 4;
    params.seed_cols = 4;
    params.seed_max_corners_per_cell = 5;
    params.seed_min_distance = 1.0;
    return params;
}

} // namespace

TEST(PropagateDetectionTest, EmptyDetectedCardsReturnsEmptyButKeepsArrangement) {
    const ip::SparsePropagator propagator;

    cd::Detection detection;
    detection.matches_arrangement.card_width = 0.5;
    detection.matches_arrangement.card_height = 0.75;

    const cd::Detection out = ct::propagate_detection(propagator, 0, detection);

    EXPECT_TRUE(out.detected_cards.empty());
    EXPECT_TRUE(out.matches.empty());
    EXPECT_DOUBLE_EQ(out.matches_arrangement.card_width, 0.5);
    EXPECT_DOUBLE_EQ(out.matches_arrangement.card_height, 0.75);
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity) gtest ASSERT macros inflate the count
TEST(PropagateDetectionTest, SuccessfulPropagationShiftsCornersAndRebuildsMatches) {
    ip::SparsePropagator propagator(permissive_params());

    const cv::Mat img0 = render_translated(kRows, kCols, 0, 0);
    propagator.next_img(img0);

    constexpr int dx = 3;
    constexpr int dy = 2;
    const cv::Mat img1 = render_translated(kRows, kCols, dx, dy);
    propagator.next_img(img1);

    cd::Detection detection;
    detection.detected_cards = {make_detected_card(60.0, 60.0, /*matched=*/true),
                                make_detected_card(90.0, 40.0, /*matched=*/false)};
    detection.matches_arrangement.card_width = 0.42;

    const cd::Detection out = ct::propagate_detection(propagator, 0, detection);

    ASSERT_EQ(out.detected_cards.size(), 2U);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t k = 0; k < 4; ++k) {
            EXPECT_NEAR(out.detected_cards[i].corners[k].x, detection.detected_cards[i].corners[k].x + dx, 4.0);
            EXPECT_NEAR(out.detected_cards[i].corners[k].y, detection.detected_cards[i].corners[k].y + dy, 4.0);
        }
        EXPECT_EQ(out.detected_cards[i].corner_visibility, detection.detected_cards[i].corner_visibility);
        EXPECT_EQ(out.detected_cards[i].count_probs, detection.detected_cards[i].count_probs);
        EXPECT_EQ(out.detected_cards[i].fill_probs, detection.detected_cards[i].fill_probs);
    }

    // Only the first (matched) card should show up in matches.
    ASSERT_EQ(out.matches.size(), 1U);
    EXPECT_NEAR(out.matches[0].first.corners[0].x, detection.detected_cards[0].corners[0].x + dx, 4.0);

    EXPECT_DOUBLE_EQ(out.matches_arrangement.card_width, 0.42);
}

TEST(PropagateDetectionTest, PropagationFailureDropsAllCardsButKeepsArrangement) {
    ip::SparsePropagatorParams params = permissive_params();
    params.history_size = 2;
    ip::SparsePropagator propagator(params);

    propagator.next_img(render_translated(kRows, kCols, 0, 0));
    propagator.next_img(render_translated(kRows, kCols, 3, 2));
    propagator.next_img(render_translated(kRows, kCols, 6, 4)); // evicts frame 0's history slot

    cd::Detection detection;
    detection.detected_cards = {make_detected_card(60.0, 60.0, /*matched=*/true)};
    detection.matches_arrangement.card_width = 0.33;

    const cd::Detection out = ct::propagate_detection(propagator, -1, detection);

    EXPECT_TRUE(out.detected_cards.empty());
    EXPECT_TRUE(out.matches.empty());
    EXPECT_DOUBLE_EQ(out.matches_arrangement.card_width, 0.33);
}

TEST(ScaleDetectionTest, EmptyDetectionPassthrough) {
    cd::Detection detection;
    detection.matches_arrangement.card_width = 0.5;

    const cd::Detection out = ct::scale_detection(detection, 2.0);

    EXPECT_TRUE(out.detected_cards.empty());
    EXPECT_TRUE(out.matches.empty());
    EXPECT_DOUBLE_EQ(out.matches_arrangement.card_width, 0.5);
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity) gtest ASSERT macros inflate the count
TEST(ScaleDetectionTest, ScalesCornersInDetectedCardsAndMatchesButNotArrangement) {
    cd::Detection detection;
    detection.detected_cards = {make_detected_card(60.0, 40.0, /*matched=*/true),
                                make_detected_card(90.0, 20.0, /*matched=*/false)};
    detection.matches = {{detection.detected_cards[0], sg::Card{}}};
    detection.matches_arrangement.card_width = 0.42;

    constexpr double kFactor = 2.5;
    const cd::Detection out = ct::scale_detection(detection, kFactor);

    ASSERT_EQ(out.detected_cards.size(), 2U);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t k = 0; k < 4; ++k) {
            EXPECT_DOUBLE_EQ(out.detected_cards[i].corners[k].x, detection.detected_cards[i].corners[k].x * kFactor);
            EXPECT_DOUBLE_EQ(out.detected_cards[i].corners[k].y, detection.detected_cards[i].corners[k].y * kFactor);
        }
        EXPECT_EQ(out.detected_cards[i].corner_visibility, detection.detected_cards[i].corner_visibility);
        EXPECT_EQ(out.detected_cards[i].count_probs, detection.detected_cards[i].count_probs);
    }

    ASSERT_EQ(out.matches.size(), 1U);
    EXPECT_DOUBLE_EQ(out.matches[0].first.corners[0].x, detection.matches[0].first.corners[0].x * kFactor);

    EXPECT_DOUBLE_EQ(out.matches_arrangement.card_width, 0.42);
}
