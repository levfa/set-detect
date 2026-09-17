#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace setdetect::img_proc {

struct Corner {
    int x = 0;
    int y = 0;
    float score = 0.0F;
};

// Shi-Tomasi corner detection: thin wrapper around
// cv::goodFeaturesToTrack(WithQuality). When mask is non-null it must be a
// single-channel Mat of the same size; pixels with value 0 are excluded.
// max_corners == 0 keeps every non-suppressed candidate; max_corners < 0,
// quality_level <= 0, min_distance < 0 or block_size <= 0 are invalid and
// return empty.
auto detect_corners_st(const cv::Mat& gray,           // 1-channel
                       const cv::Mat* mask = nullptr, // 0 = excluded, nullptr = no mask
                       int max_corners = 1000,        // 0 = unlimited
                       double quality_level = 0.01, double min_distance = 10.0,
                       int block_size = 3) -> std::vector<Corner>;

} // namespace setdetect::img_proc
