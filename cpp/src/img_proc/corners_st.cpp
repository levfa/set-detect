#include <cmath>
#include <cstddef>
#include <opencv2/core.hpp>
#include <opencv2/features.hpp>
#include <setdetect/img_proc/corners_st.hpp>
#include <vector>

namespace setdetect::img_proc {

auto detect_corners_st(const cv::Mat& gray, const cv::Mat* mask, int max_corners, double quality_level,
                       double min_distance, int block_size) -> std::vector<Corner> {
    if (gray.empty() || gray.channels() != 1 || max_corners < 0 || quality_level <= 0.0 || min_distance < 0.0 ||
        block_size <= 0) {
        return {};
    }
    if (mask != nullptr &&
        (mask->empty() || mask->channels() != 1 || mask->rows != gray.rows || mask->cols != gray.cols)) {
        return {};
    }

    std::vector<cv::Point2f> corners;
    std::vector<float> qualities;
    cv::goodFeaturesToTrack(gray, corners, max_corners, quality_level, min_distance,
                            mask != nullptr ? *mask : cv::noArray(), qualities, block_size);

    std::vector<Corner> out;
    out.reserve(corners.size());
    for (size_t i = 0; i < corners.size(); ++i) {
        out.push_back(
            {static_cast<int>(std::lround(corners[i].x)), static_cast<int>(std::lround(corners[i].y)), qualities[i]});
    }
    return out;
}

} // namespace setdetect::img_proc
