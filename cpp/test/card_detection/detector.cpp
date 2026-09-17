#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <numbers>
#include <optional>
#include <setdetect/card_detection/detector.hpp>
#include <vector>

namespace cd = setdetect::card_detection;

namespace {

constexpr double kPi = std::numbers::pi;

// Builds an axis-unrotated-then-rotated rectangle of the given size, centered at `center`,
// as a DetectedCard with fully visible corners. Corner order matches the canonical
// rectangle winding: top-left, top-right, bottom-right, bottom-left.
auto make_card(double center_x, double center_y, double width, double height, double angle_rad, double visibility = 1.0)
    -> cd::DetectedCard {
    const double half_w = width / 2.0;
    const double half_h = height / 2.0;
    const std::array<cd::Point2d, 4> local = {cd::Point2d{-half_w, -half_h}, cd::Point2d{half_w, -half_h},
                                              cd::Point2d{half_w, half_h}, cd::Point2d{-half_w, half_h}};
    const double c = std::cos(angle_rad);
    const double s = std::sin(angle_rad);

    cd::DetectedCard card;
    for (int i = 0; i < 4; ++i) {
        card.corners[i] = {center_x + (c * local[i].x) - (s * local[i].y),
                           center_y + (s * local[i].x) + (c * local[i].y)};
        card.corner_visibility[i] = visibility;
    }
    return card;
}

// One-hot-ish probability head: `hot_index` gets the mass, the rest get a small residual.
auto make_probs(int hot_index) -> std::array<double, 4> {
    std::array<double, 4> probs = {0.01, 0.01, 0.01, 0.01};
    probs[static_cast<size_t>(hot_index)] = 0.97;
    return probs;
}

auto make_detected_card(int count_idx, int color_idx, int shape_idx, int fill_idx) -> cd::DetectedCard {
    cd::DetectedCard card;
    card.count_probs = make_probs(count_idx);
    card.color_probs = make_probs(color_idx);
    card.shape_probs = make_probs(shape_idx);
    card.fill_probs = make_probs(fill_idx);
    return card;
}

} // namespace

TEST(MatchCardTest, ReturnsCardWhenAllHeadsConfident) {
    const cd::DetectedCard card = make_detected_card(0, 1, 2, 0);
    const std::optional<setdetect::set_game::Card> matched = cd::match_card(card);

    ASSERT_TRUE(matched.has_value());
    EXPECT_EQ(matched->count, setdetect::set_game::Count::ONE);
    EXPECT_EQ(matched->color, setdetect::set_game::Color::GREEN);
    EXPECT_EQ(matched->shape, setdetect::set_game::Shape::SQUIGGLE);
    EXPECT_EQ(matched->fill, setdetect::set_game::Fill::OPEN);
}

TEST(MatchCardTest, ReturnsNulloptWhenCountHeadIsOther) {
    EXPECT_FALSE(cd::match_card(make_detected_card(3, 0, 0, 0)).has_value());
}

TEST(MatchCardTest, ReturnsNulloptWhenColorHeadIsOther) {
    EXPECT_FALSE(cd::match_card(make_detected_card(0, 3, 0, 0)).has_value());
}

TEST(MatchCardTest, ReturnsNulloptWhenShapeHeadIsOther) {
    EXPECT_FALSE(cd::match_card(make_detected_card(0, 0, 3, 0)).has_value());
}

TEST(MatchCardTest, ReturnsNulloptWhenFillHeadIsOther) {
    EXPECT_FALSE(cd::match_card(make_detected_card(0, 0, 0, 3)).has_value());
}

TEST(CardsArrangementTest, EmptyInputReturnsEmptyArrangement) {
    const cd::CardsArrangement arrangement = cd::cards_arrangement({});
    EXPECT_TRUE(arrangement.card_poses.empty());
    EXPECT_TRUE(arrangement.card_z_orders.empty());
    EXPECT_DOUBLE_EQ(arrangement.card_width, 0.0);
    EXPECT_DOUBLE_EQ(arrangement.card_height, 0.0);
}

TEST(CardsArrangementTest, SingleCardCentersInNormalizedBoxWithDefaultInset) {
    const std::vector<cd::DetectedCard> cards = {make_card(100.0, 150.0, 120.0, 180.0, 0.4)};
    const cd::CardsArrangement arrangement = cd::cards_arrangement(cards);

    ASSERT_EQ(arrangement.card_poses.size(), 1U);
    // default inset = 0.02 -> inner = 0.96 -> a single card's bounding box is symmetric
    // about its own center, so the normalized center always lands at inner / 2.
    EXPECT_NEAR(arrangement.card_poses[0].center_x, 0.48, 1e-6);
    EXPECT_NEAR(arrangement.card_poses[0].center_y, 0.48, 1e-6);
    EXPECT_GT(arrangement.card_width, 0.0);
    EXPECT_GT(arrangement.card_height, 0.0);
    ASSERT_EQ(arrangement.card_z_orders.size(), 1U);
    EXPECT_EQ(arrangement.card_z_orders[0], 0);
}

TEST(CardsArrangementTest, SingleCardCentersInNormalizedBoxWithCustomInset) {
    const std::vector<cd::DetectedCard> cards = {make_card(50.0, 50.0, 200.0, 300.0, -0.7)};
    const cd::CardsArrangement arrangement = cd::cards_arrangement(cards, 0.1);

    ASSERT_EQ(arrangement.card_poses.size(), 1U);
    EXPECT_NEAR(arrangement.card_poses[0].center_x, 0.4, 1e-6);
    EXPECT_NEAR(arrangement.card_poses[0].center_y, 0.4, 1e-6);
}

TEST(CardsArrangementTest, InsetIsClampedToValidRange) {
    const std::vector<cd::DetectedCard> negative_inset_cards = {make_card(0.0, 0.0, 100.0, 150.0, 0.0)};
    const cd::CardsArrangement negative = cd::cards_arrangement(negative_inset_cards, -5.0);
    EXPECT_NEAR(negative.card_poses[0].center_x, 0.5, 1e-6);

    const std::vector<cd::DetectedCard> large_inset_cards = {make_card(0.0, 0.0, 100.0, 150.0, 0.0)};
    const cd::CardsArrangement large = cd::cards_arrangement(large_inset_cards, 10.0);
    EXPECT_NEAR(large.card_poses[0].center_x, 0.01, 1e-6);
}

TEST(CardsArrangementTest, PreservesRelativeOrderingUnderPureScaleProjection) {
    // No rotation/perspective in the synthetic "camera": image = 40 * world. Two
    // axis-aligned, non-overlapping cards recover axis-aligned poses with their world
    // left-to-right order preserved.
    constexpr double kScale = 40.0;
    constexpr double kCardW = 0.7; // matches the module's fixed card aspect ratio
    constexpr double kCardH = 1.0;

    auto world_card = [&](double cx, double cy) {
        return make_card(cx * kScale, cy * kScale, kCardW * kScale, kCardH * kScale, 0.0);
    };

    const std::vector<cd::DetectedCard> cards = {world_card(0.0, 0.0), world_card(5.0, 0.0)};
    const cd::CardsArrangement arrangement = cd::cards_arrangement(cards);

    ASSERT_EQ(arrangement.card_poses.size(), 2U);
    EXPECT_LT(arrangement.card_poses[0].center_x, arrangement.card_poses[1].center_x);
    EXPECT_NEAR(arrangement.card_poses[0].center_y, arrangement.card_poses[1].center_y, 1e-6);
    EXPECT_NEAR(arrangement.card_poses[0].angle, 0.0, 1e-6);
    EXPECT_NEAR(arrangement.card_poses[1].angle, 0.0, 1e-6);
    EXPECT_NEAR(arrangement.card_width / arrangement.card_height, kCardW / kCardH, 1e-6);
}

TEST(CardsArrangementTest, RecoversRotationForPureScaleProjection) {
    constexpr double kScale = 40.0;
    constexpr double kAngle = kPi / 6.0; // 30 degrees
    const std::vector<cd::DetectedCard> cards = {
        make_card(0.0, 0.0, 0.7 * kScale, 1.0 * kScale, kAngle),
    };
    const cd::CardsArrangement arrangement = cd::cards_arrangement(cards);

    ASSERT_EQ(arrangement.card_poses.size(), 1U);
    // The card's local aspect ratio matches the canonical one exactly, so its own quad
    // determines an exact similarity (not just projective) anchor homography, and the
    // recovered rotation exactly matches the input angle.
    EXPECT_NEAR(arrangement.card_poses[0].angle, kAngle, 1e-6);
}

TEST(CardsArrangementTest, OcclusionDeterminesZOrder) {
    // Card B (a large card at the origin) fully covers two corners of card A, which sits
    // half-underneath it; the other two corners of A are outside B and fully visible.
    const cd::DetectedCard card_b = make_card(0.0, 0.0, 100.0, 100.0, 0.0, 1.0);

    cd::DetectedCard card_a = make_card(60.0, 0.0, 100.0, 100.0, 0.0, 1.0);
    // Corners 0 (top-left) and 3 (bottom-left) of A fall inside B and are marked hidden.
    card_a.corner_visibility[0] = 0.0;
    card_a.corner_visibility[3] = 0.0;

    const std::vector<cd::DetectedCard> cards = {card_a, card_b};
    const cd::CardsArrangement arrangement = cd::cards_arrangement(cards);

    ASSERT_EQ(arrangement.card_z_orders.size(), 2U);
    EXPECT_GT(arrangement.card_z_orders[1], arrangement.card_z_orders[0]);
}

TEST(CardsArrangementTest, NoOcclusionEvidenceGivesDeterministicOrder) {
    const std::vector<cd::DetectedCard> cards = {
        make_card(0.0, 0.0, 100.0, 150.0, 0.0),
        make_card(300.0, 0.0, 100.0, 150.0, 0.0),
        make_card(600.0, 0.0, 100.0, 150.0, 0.0),
    };
    const cd::CardsArrangement arrangement = cd::cards_arrangement(cards);

    ASSERT_EQ(arrangement.card_z_orders.size(), 3U);
    std::vector<int> sorted_orders = arrangement.card_z_orders;
    std::sort(sorted_orders.begin(), sorted_orders.end());
    EXPECT_EQ(sorted_orders, (std::vector<int>{0, 1, 2}));
}
