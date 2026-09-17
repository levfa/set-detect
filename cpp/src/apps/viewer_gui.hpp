#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <setdetect/card_detection/detector.hpp>
#include <setdetect/set_game/game.hpp>
#include <string>
#include <vector>

// Keypress navigation codes and detection/arrangement drawing helpers, used by
// the show-boards viewer.
namespace setdetect::apps {

constexpr int kEscKey = 27;

enum class ArrowKey : std::uint8_t { kNone, kLeft, kUp, kRight, kDown };

// Maps backend-specific cv::waitKey return codes (X11/Wayland/Qt) to an arrow
// direction. Platform-fragile by nature — best-effort; ESC/'q'/"any other key"
// navigation must not depend on this.
inline auto arrow_direction(int key) -> ArrowKey {
    switch (key) {
    case 81:
    case 65361:
    case 16777234:
    case 2424832:
        return ArrowKey::kLeft;
    case 82:
    case 65362:
    case 16777235:
    case 2490368:
        return ArrowKey::kUp;
    case 83:
    case 65363:
    case 16777236:
    case 2555904:
        return ArrowKey::kRight;
    case 84:
    case 65364:
    case 16777237:
    case 2621440:
        return ArrowKey::kDown;
    default:
        return ArrowKey::kNone;
    }
}

// Draws `text` with a black outline behind a white fill so it stays readable over any
// background.
inline void draw_label(cv::Mat& img, const std::string& text, cv::Point org) {
    cv::putText(img, text, org, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 4);
    cv::putText(img, text, org, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
}

namespace detail {

inline auto corner_color(int idx, bool visible) -> cv::Scalar {
    if (!visible) {
        return {200, 200, 200};
    }
    switch (idx) {
    case 0:
        return {0, 0, 255};
    case 1:
        return {0, 255, 0};
    case 2:
        return {255, 0, 0};
    default:
        return {0, 255, 255};
    }
}

inline auto argmax4(const std::array<double, 4>& probs) -> int {
    return static_cast<int>(std::max_element(probs.begin(), probs.end()) - probs.begin());
}

inline auto card_label(const card_detection::DetectedCard& dc) -> std::string {
    std::string label;
    label += set_game::count_letter(static_cast<set_game::Count>(argmax4(dc.count_probs)));
    label += set_game::color_letter(static_cast<set_game::Color>(argmax4(dc.color_probs)));
    label += set_game::shape_letter(static_cast<set_game::Shape>(argmax4(dc.shape_probs)));
    label += set_game::fill_letter(static_cast<set_game::Fill>(argmax4(dc.fill_probs)));
    return label;
}

inline auto card_label(const set_game::Card& card) -> std::string {
    std::string label;
    label += set_game::count_letter(card.count);
    label += set_game::color_letter(card.color);
    label += set_game::shape_letter(card.shape);
    label += set_game::fill_letter(card.fill);
    return label;
}

} // namespace detail

// Draws one detected card's quad outline, per-corner dots (colored by index, gray
// when below the visibility threshold), and attribute label onto `vis`.
inline void draw_detection(cv::Mat& vis, const card_detection::DetectedCard& dc) {
    std::array<cv::Point, 4> pts{};
    double cx = 0.0;
    double cy = 0.0;
    for (size_t i = 0; i < 4; ++i) {
        pts[i] =
            cv::Point(static_cast<int>(std::lround(dc.corners[i].x)), static_cast<int>(std::lround(dc.corners[i].y)));
        cx += dc.corners[i].x;
        cy += dc.corners[i].y;
    }
    cv::polylines(vis, std::vector<cv::Point>(pts.begin(), pts.end()), true, cv::Scalar(0, 255, 0), 3);
    for (size_t i = 0; i < 4; ++i) {
        const bool visible = dc.corner_visibility[i] >= card_detection::kCornerVisibleThreshold;
        cv::circle(vis, pts[i], 6, detail::corner_color(static_cast<int>(i), visible), -1);
    }
    const cv::Point label_org(static_cast<int>(std::lround(cx / 4.0)),
                              static_cast<int>(std::lround((cy / 4.0) - 10.0)));
    draw_label(vis, detail::card_label(dc), label_org);
}

// Bare outline overlay for one matched card's quad: gold/amber, no corner dots
// or attribute label, thickness scaling mildly with frame size.
inline void draw_matched_outline(cv::Mat& vis, const std::array<card_detection::Point2d, 4>& corners) {
    std::array<cv::Point, 4> pts{};
    for (size_t i = 0; i < 4; ++i) {
        pts[i] = cv::Point(static_cast<int>(std::lround(corners[i].x)), static_cast<int>(std::lround(corners[i].y)));
    }
    const int thickness = std::max(2, std::min(vis.rows, vis.cols) / 480);
    cv::polylines(vis, std::vector<cv::Point>(pts.begin(), pts.end()), true, cv::Scalar(0, 200, 255), thickness);
}

// Draws a white canvas with each matched card as a colored rotated box (filled +
// outlined) plus a center dot and label, in `arrangement`'s z-order (deepest first).
// `labels[i]` must correspond to `arrangement.card_poses[i]`.
inline auto draw_arrangement(const card_detection::CardsArrangement& arrangement,
                             const std::vector<set_game::Card>& labels, int canvas_size = 1024) -> cv::Mat {
    cv::Mat canvas(canvas_size, canvas_size, CV_8UC3, cv::Scalar(255, 255, 255));

    const size_t n = arrangement.card_poses.size();
    std::vector<size_t> draw_order(n);
    std::iota(draw_order.begin(), draw_order.end(), 0);
    std::sort(draw_order.begin(), draw_order.end(),
              [&](size_t a, size_t b) { return arrangement.card_z_orders[a] < arrangement.card_z_orders[b]; });

    for (const size_t i : draw_order) {
        const card_detection::RotatedRect& pose = arrangement.card_poses[i];
        const cv::RotatedRect rect(cv::Point2f(static_cast<float>(pose.center_x * canvas_size),
                                               static_cast<float>(pose.center_y * canvas_size)),
                                   cv::Size2f(static_cast<float>(arrangement.card_width * canvas_size),
                                              static_cast<float>(arrangement.card_height * canvas_size)),
                                   static_cast<float>(pose.angle * 180.0 / M_PI));
        std::array<cv::Point2f, 4> box_pts{};
        rect.points(box_pts.data());
        std::vector<cv::Point> poly(4);
        for (size_t k = 0; k < 4; ++k) {
            poly[k] = box_pts[k];
        }

        cv::fillPoly(canvas, std::vector<std::vector<cv::Point>>{poly}, cv::Scalar(120, 200, 120));
        cv::polylines(canvas, poly, true, cv::Scalar(0, 0, 0), 2);
        cv::circle(canvas, rect.center, 5, cv::Scalar(0, 0, 255), -1);
        if (i < labels.size()) {
            draw_label(canvas, detail::card_label(labels[i]),
                       cv::Point(static_cast<int>(rect.center.x) - 30, static_cast<int>(rect.center.y) - 10));
        }
    }
    return canvas;
}

} // namespace setdetect::apps
