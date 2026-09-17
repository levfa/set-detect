#pragma once

#include <CLI/CLI.hpp>
#include <string>

namespace setdetect::apps {

struct DetectionArgs {
    std::string corner_weights;
    std::string class_weights;
    int device = 0;
    double conf = 0.25;
    int imgsz = 640;
};

// Registers the --det-* flags on `sub`, writing parsed values into `args`.
inline void add_detection_args(CLI::App& sub, DetectionArgs& args) {
    sub.add_option("--det-corner-weights", args.corner_weights, "Corner-keypoint ONNX model (YOLOv8-pose)")->required();
    sub.add_option("--det-class-weights", args.class_weights, "Classifier ONNX model")->required();
    sub.add_option("--det-device", args.device, "Corner detection and card classification GPU device (-1 for CPU)");
    sub.add_option("--det-conf", args.conf, "Corner detection confidence threshold");
    sub.add_option("--det-imgsz", args.imgsz, "Corner model input size");
}

} // namespace setdetect::apps
