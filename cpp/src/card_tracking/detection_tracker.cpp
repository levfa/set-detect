#include <algorithm>
#include <cstddef>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <setdetect/card_detection/async_detector.hpp>
#include <setdetect/card_tracking/detection_tracker.hpp>
#include <setdetect/set_game/game.hpp>
#include <vector>

namespace setdetect::card_tracking {

auto propagate_detection(const img_proc::SparsePropagator& propagator, int from_frame_id,
                         const card_detection::Detection& detection) -> card_detection::Detection {
    card_detection::Detection out;
    out.matches_arrangement = detection.matches_arrangement;

    if (detection.detected_cards.empty()) {
        return out;
    }

    std::vector<img_proc::Pt2f> pts;
    pts.reserve(detection.detected_cards.size() * 4);
    for (const card_detection::DetectedCard& dc : detection.detected_cards) {
        for (const card_detection::Point2d& corner : dc.corners) {
            pts.push_back({static_cast<float>(corner.x), static_cast<float>(corner.y)});
        }
    }

    if (!propagator.propagate(from_frame_id, pts)) {
        return out;
    }

    out.detected_cards.reserve(detection.detected_cards.size());
    for (size_t i = 0; i < detection.detected_cards.size(); ++i) {
        card_detection::DetectedCard dc = detection.detected_cards[i];
        for (size_t k = 0; k < 4; ++k) {
            const img_proc::Pt2f& p = pts[(i * 4) + k];
            dc.corners[k] = {static_cast<double>(p.x), static_cast<double>(p.y)};
        }
        if (const std::optional<set_game::Card> card = card_detection::match_card(dc)) {
            out.matches.emplace_back(dc, *card);
        }
        out.detected_cards.push_back(dc);
    }
    return out;
}

auto scale_detection(const card_detection::Detection& detection, double factor) -> card_detection::Detection {
    card_detection::Detection out = detection;
    for (card_detection::DetectedCard& dc : out.detected_cards) {
        for (card_detection::Point2d& corner : dc.corners) {
            corner.x *= factor;
            corner.y *= factor;
        }
    }
    for (auto& [dc, card] : out.matches) {
        for (card_detection::Point2d& corner : dc.corners) {
            corner.x *= factor;
            corner.y *= factor;
        }
    }
    return out;
}

struct DetectionTracker::Impl {
    img_proc::SparsePropagator propagator;
    card_detection::AsyncCardDetector async_detector;
    int max_side;
    std::optional<card_detection::Detection> last_detection;
    int last_detection_frame_id = -1;

    Impl(const card_detection::CardDetector& detector, const img_proc::SparsePropagatorParams& params, int max_side_)
        : propagator(params), async_detector(detector), max_side(max_side_) {}
};

DetectionTracker::DetectionTracker(const card_detection::CardDetector& detector,
                                   const img_proc::SparsePropagatorParams& propagation_params, int max_side)
    : impl_(std::make_unique<Impl>(detector, propagation_params, max_side)) {}

DetectionTracker::~DetectionTracker() = default;

auto DetectionTracker::detect(const cv::Mat& img_rgb) -> std::optional<card_detection::Detection> {
    // Resize down to the working resolution once (only if larger); everything below
    // operates on `working`, so a fresh detection's and a propagated detection's corners
    // always live in the same coordinate space. INTER_AREA matches Python's
    // cli/card_tracking.py resolution-limiting resize.
    cv::Mat resized;
    const cv::Mat* working = &img_rgb;
    double to_orig = 1.0;
    const int longest = std::max(img_rgb.rows, img_rgb.cols);
    if (longest > impl_->max_side) {
        const auto scale = static_cast<float>(impl_->max_side) / static_cast<float>(longest);
        cv::resize(img_rgb, resized, cv::Size(), scale, scale, cv::INTER_AREA);
        working = &resized;
        to_orig = static_cast<double>(img_rgb.rows) / static_cast<double>(resized.rows);
    }

    cv::Mat gray;
    cv::cvtColor(*working, gray, cv::COLOR_RGB2GRAY);
    const img_proc::PropagationStep step = impl_->propagator.next_img(gray);

    impl_->async_detector.submit(step.frame_id, *working);

    if (const std::optional<card_detection::AsyncCardDetector::Result> result = impl_->async_detector.poll()) {
        impl_->last_detection = propagate_detection(impl_->propagator, result->frame_id, result->detection);
        impl_->last_detection_frame_id = step.frame_id;
    } else if (const std::optional<card_detection::Detection>& prev = impl_->last_detection; prev.has_value()) {
        impl_->last_detection = propagate_detection(impl_->propagator, impl_->last_detection_frame_id, *prev);
        impl_->last_detection_frame_id = step.frame_id;
    }

    if (const std::optional<card_detection::Detection>& last = impl_->last_detection; last.has_value()) {
        if (to_orig == 1.0) {
            return last;
        }
        return scale_detection(*last, to_orig);
    }
    return std::nullopt;
}

} // namespace setdetect::card_tracking
