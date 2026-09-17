#include <array>
#include <cstdlib>
#include <gtest/gtest.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <random>
#include <setdetect/card_detection/detector.hpp>
#include <string>

namespace cd = setdetect::card_detection;

namespace {

// Uses the same cache layout as this project's other model downloader, so both
// languages' tests can share one downloaded cache; populated by
// scripts/download_models.sh.
auto models_root() -> std::string {
    const char* env = std::getenv("SETDETECT_MODELS_ROOT");
    if (env != nullptr && *env != '\0') {
        return env;
    }
    const char* home = std::getenv("HOME");
    return std::string(home != nullptr ? home : "") + "/.cache/setdetect/models";
}

auto corner_model_path() -> std::string { return models_root() + "/card-corners/onnx/card_corners.onnx"; }

auto class_model_path() -> std::string { return models_root() + "/card-classification/onnx/model.onnx"; }

// Card classifier's fixed input size.
constexpr int kCardWidth = 224;
constexpr int kCardHeight = 320;

} // namespace

TEST(RealModelDetectorTest, CornerSessionLoadsAndRuns) {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "setdetect-test"};
    Ort::Session session = cd::create_session(env, corner_model_path(), /*device_id=*/-1);

    const cv::Mat img = cv::Mat::zeros(640, 480, CV_8UC3);
    const std::vector<cd::CornerDetection> detections = cd::detect_card_corners(img, session);

    for (const cd::CornerDetection& d : detections) {
        for (int i = 0; i < 4; ++i) {
            EXPECT_GE(d.visibility[static_cast<size_t>(i)], 0.0);
            EXPECT_LE(d.visibility[static_cast<size_t>(i)], 1.0);
        }
    }
}

TEST(RealModelDetectorTest, ClassifierSessionLoadsAndReturnsValidDistributions) {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "setdetect-test"};
    Ort::Session session = cd::create_session(env, class_model_path(), /*device_id=*/-1);

    std::mt19937 rng(0); // NOLINT(cert-msc32-c,cert-msc51-cpp) deterministic test data, not security-sensitive
    std::uniform_int_distribution<int> dist(0, 255);
    cv::Mat img(kCardHeight, kCardWidth, CV_8UC3);
    for (int y = 0; y < img.rows; ++y) {
        uint8_t* row = img.ptr(y);
        for (int x = 0; x < img.cols * 3; ++x) {
            row[x] = static_cast<uint8_t>(dist(rng));
        }
    }

    const std::array<cd::Point2d, 4> quad = {cd::Point2d{0.0, 0.0}, cd::Point2d{kCardWidth, 0.0},
                                             cd::Point2d{kCardWidth, kCardHeight}, cd::Point2d{0.0, kCardHeight}};
    const std::vector<cd::ClassProbs> results = cd::classify_cards(img, {quad}, session);

    ASSERT_EQ(results.size(), 1U);
    for (const std::array<double, 4>* head :
         {&results[0].count_probs, &results[0].color_probs, &results[0].shape_probs, &results[0].fill_probs}) {
        double sum = 0.0;
        for (double p : *head) {
            EXPECT_GE(p, 0.0);
            sum += p;
        }
        EXPECT_NEAR(sum, 1.0, 1e-5);
    }
}
