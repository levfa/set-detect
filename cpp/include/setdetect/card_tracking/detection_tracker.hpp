#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include <optional>
#include <setdetect/card_detection/detector.hpp>
#include <setdetect/img_proc/sparse_propagator.hpp>

namespace setdetect::card_tracking {

// Propagates every one of detection's detected_cards' corners from from_frame_id to
// propagator's current frame, as a single batched call (propagate()'s only failure
// modes depend solely on from_frame_id, shared by every card, so this is behaviorally
// identical to propagating each card separately). On success, rebuilds detected_cards
// with the new corners and rebuilds matches via match_card() so each paired
// DetectedCard also carries the propagated corners. On failure or an empty detection,
// returns an empty Detection. matches_arrangement is always carried over unchanged;
// only a fresh detection recomputes it.
auto propagate_detection(const img_proc::SparsePropagator& propagator, int from_frame_id,
                         const card_detection::Detection& detection) -> card_detection::Detection;

// Returns a copy of detection with every corner (in detected_cards and matches) scaled
// by factor. matches_arrangement is left unchanged: cards_arrangement()'s output is
// normalized into a unit square and invariant to a uniform scale of its input corners.
auto scale_detection(const card_detection::Detection& detection, double factor) -> card_detection::Detection;

// Combines an async CardDetector (own background thread, drop-if-busy) with frame-to-
// frame optical-flow corner propagation (SparsePropagator), so detect() can be called
// every frame at video pace while the (slower) detector only completes every
// few frames: corners from the latest completed detection are caught up to the
// current frame via propagate_detection(). Cheap to construct (an AsyncCardDetector plus
// a SparsePropagator); construct a fresh instance per video rather than reusing one
// across videos, so a stale in-flight detection can never leak into the next video.
class DetectionTracker {
  public:
    // max_side: incoming frames are resized down so the longest side is at most
    // this, only if it's larger; an already-appropriately-sized frame is used as-is.
    // All internal state (SparsePropagator, the async detector) works in that working
    // resolution; only the Detection returned to the caller is scaled back up to the
    // input frame's own resolution, so callers never need to track a scale factor
    // themselves.
    explicit DetectionTracker(const card_detection::CardDetector& detector,
                              const img_proc::SparsePropagatorParams& propagation_params = {}, int max_side = 640);
    ~DetectionTracker();

    DetectionTracker(const DetectionTracker&) = delete;
    auto operator=(const DetectionTracker&) -> DetectionTracker& = delete;
    DetectionTracker(DetectionTracker&&) = delete;
    auto operator=(DetectionTracker&&) -> DetectionTracker& = delete;

    // img_rgb: current frame, RGB, must be called once per displayed frame in strict
    // sequence (next_img() needs strictly consecutive frames for correct optical flow).
    // Returns the current best-known detection, with corners in img_rgb's own
    // resolution, propagated onto this frame, or nullopt if no detection has
    // completed yet.
    auto detect(const cv::Mat& img_rgb) -> std::optional<card_detection::Detection>;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace setdetect::card_tracking
