#pragma once

#include <array>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <optional>
#include <setdetect/set_game/game.hpp>
#include <string>
#include <utility>
#include <vector>

namespace setdetect::card_detection {

struct Point2d {
    double x = 0.0;
    double y = 0.0;
};

struct DetectedCard {
    std::array<Point2d, 4> corners{};          // corner pixel positions in the original image
    std::array<double, 4> corner_visibility{}; // per-corner visibility, in [0, 1]
    std::array<double, 4> count_probs{};       // softmax over [one, two, three, other]
    std::array<double, 4> color_probs{};       // softmax over [red, green, purple, other]
    std::array<double, 4> shape_probs{};       // softmax over [diamond, oval, squiggle, other]
    std::array<double, 4> fill_probs{};        // softmax over [open, solid, striped, other]
};

// (center_x, center_y, angle_radians)
struct RotatedRect {
    double center_x = 0.0;
    double center_y = 0.0;
    double angle = 0.0;
};

struct CardsArrangement {
    std::vector<RotatedRect> card_poses;
    std::vector<int> card_z_orders; // per-card rank into card_poses: 0 = deepest, larger = higher
    double card_width = 0.0;
    double card_height = 0.0;
};

constexpr double kCornerVisibleThreshold = 0.5;

// Jointly fits the shared-plane arrangement of a set of detected cards: per-card pose
// (position + rotation) on a common plane, normalized into the unit square with the
// given inset margin, plus a depth order inferred from corner occlusion.
auto cards_arrangement(const std::vector<DetectedCard>& detected_cards, double inset = 0.02) -> CardsArrangement;

struct CornerDetection {
    std::array<Point2d, 4> corners{};
    std::array<double, 4> visibility{}; // per-corner model output, in [0, 1]
};

struct ClassProbs {
    std::array<double, 4> count_probs{};
    std::array<double, 4> color_probs{};
    std::array<double, 4> shape_probs{};
    std::array<double, 4> fill_probs{};
};

struct Detection {
    std::vector<DetectedCard> detected_cards;
    std::vector<std::pair<DetectedCard, set_game::Card>> matches;
    CardsArrangement matches_arrangement;
};

// A set_game::Card only when every head's argmax is a real class (index != 3/"other").
auto match_card(const DetectedCard& dc) -> std::optional<set_game::Card>;

// Opens an inference session for `model_path`. Tries the given CUDA device when
// device_id >= 0, falling back to CPU (device_id < 0 skips CUDA entirely).
auto create_session(Ort::Env& env, const std::string& model_path, int device_id = 0) -> Ort::Session;

// Runs the corner-keypoint (YOLO-pose) model on an RGB image and returns one
// CornerDetection per surviving detection after confidence filtering and NMS.
auto detect_card_corners(const cv::Mat& img_rgb, Ort::Session& session, double conf = 0.25, double iou = 0.45,
                         int imgsz = 640) -> std::vector<CornerDetection>;

// Classifies each card quad in `corners_list` (cropped/warped from img_rgb), returning
// one ClassProbs per quad in the same order.
auto classify_cards(const cv::Mat& img_rgb, const std::vector<std::array<Point2d, 4>>& corners_list,
                    Ort::Session& session) -> std::vector<ClassProbs>;

// Runs the full detection + classification pipeline on a single image.
auto detect_cards(const cv::Mat& img_rgb, Ort::Session& corner_session, Ort::Session& class_session, double conf = 0.25,
                  double iou = 0.45, int imgsz = 640) -> Detection;

// Owns the corner and classification ONNX sessions for repeated synchronous detection.
class CardDetector {
  public:
    CardDetector(const std::string& corner_weights, const std::string& class_weights, int device_id = 0,
                 double conf = 0.25, double iou = 0.45, int imgsz = 640, bool warm_up = true);
    ~CardDetector();

    CardDetector(const CardDetector&) = delete;
    auto operator=(const CardDetector&) -> CardDetector& = delete;
    CardDetector(CardDetector&&) noexcept;
    auto operator=(CardDetector&&) noexcept -> CardDetector&;

    [[nodiscard]] auto detect(const cv::Mat& img_rgb) const -> Detection;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace setdetect::card_detection
