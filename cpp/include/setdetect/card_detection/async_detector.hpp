#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include <optional>
#include <setdetect/card_detection/detector.hpp>

namespace setdetect::card_detection {

// Runs card detection on a dedicated background thread. Submitting work is
// non-blocking and drops any not-yet-consumed previous submission (single input
// slot); polling for a result is non-blocking and returns a finished result at most
// once (single output slot).
class AsyncCardDetector {
  public:
    struct Result {
        int frame_id = -1;
        Detection detection;
    };

    explicit AsyncCardDetector(const CardDetector& detector);
    ~AsyncCardDetector();

    AsyncCardDetector(const AsyncCardDetector&) = delete;
    auto operator=(const AsyncCardDetector&) -> AsyncCardDetector& = delete;
    AsyncCardDetector(AsyncCardDetector&&) = delete;
    auto operator=(AsyncCardDetector&&) -> AsyncCardDetector& = delete;

    // Submits img_rgb tagged with frame_id, overwriting any unconsumed previous
    // submission.
    void submit(int frame_id, const cv::Mat& img_rgb);

    // Non-blocking. Returns the most recently finished result, consumed once,
    // or nullopt if nothing has finished since the last call.
    auto poll() -> std::optional<Result>;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace setdetect::card_detection
