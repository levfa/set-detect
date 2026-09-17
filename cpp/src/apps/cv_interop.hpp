#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace setdetect::apps {

// Converts a BGR (or BGRA) cv::Mat, as loaded by cv::imread, into RGB, ready for
// detection. Drawing/display should stay on the original BGR cv::Mat instead of
// converting back -- only inference needs RGB.
inline auto bgr_to_rgb(const cv::Mat& bgr) -> cv::Mat {
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    return rgb;
}

} // namespace setdetect::apps
