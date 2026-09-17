#include <CLI/CLI.hpp>
#include <apps/cv_interop.hpp>
#include <apps/detection_args.hpp>
#include <apps/stats.hpp>
#include <apps/tracking_args.hpp>
#include <apps/video_args.hpp>
#include <apps/viewer_gui.hpp>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <optional>
#include <setdetect/card_detection/detector.hpp>
#include <setdetect/card_tracking/detection_tracker.hpp>
#include <setdetect/img_acq/frame_grabber.hpp>
#include <setdetect/set_game/game.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cd = setdetect::card_detection;
namespace ct = setdetect::card_tracking;
namespace ip = setdetect::img_proc;
namespace sg = setdetect::set_game;
namespace sd_ia = setdetect::img_acq;
namespace fs = std::filesystem;
using setdetect::apps::print_stat;
using setdetect::apps::Stats;
using setdetect::apps::summarize;

namespace {

constexpr int kDefaultProfileRuns = 200;

// Times DetectionTracker::detect() over a real video's actual consecutive frames (read
// directly via cv::VideoCapture, looping back to frame 0 if runs exceeds the frame
// count) -- unlike profile-detection's independent-image cycling, this must preserve
// frame-to-frame continuity for SparsePropagator's optical flow to mean anything. Timing
// happens entirely here in the app layer, wrapping the call with steady_clock --
// DetectionTracker itself carries no internal instrumentation.
auto profile_tracking(const std::string& video_root, const setdetect::apps::DetectionArgs& det_args,
                      const setdetect::apps::TrackingArgs& track_args, int runs) -> int {
    const std::vector<fs::path> video_paths = setdetect::apps::list_videos(video_root);
    if (video_paths.empty()) {
        throw std::runtime_error("no videos found under " + video_root);
    }

    const cd::CardDetector detector(det_args.corner_weights, det_args.class_weights, det_args.device, det_args.conf,
                                    /*iou=*/0.45, det_args.imgsz);
    ct::DetectionTracker tracker(detector, track_args.params, det_args.imgsz);

    cv::VideoCapture cap(video_paths.front().string());
    if (!cap.isOpened()) {
        throw std::runtime_error("could not open video: " + video_paths.front().string());
    }

    std::vector<double> detect_s;
    std::vector<size_t> card_counts;
    detect_s.reserve(static_cast<size_t>(runs));
    card_counts.reserve(static_cast<size_t>(runs));

    for (int i = 0; i < runs; ++i) {
        cv::Mat frame_bgr;
        if (!cap.read(frame_bgr)) {
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            if (!cap.read(frame_bgr)) {
                break;
            }
        }
        const cv::Mat img_rgb = setdetect::apps::bgr_to_rgb(frame_bgr);

        const auto t0 = std::chrono::steady_clock::now();
        const std::optional<cd::Detection> detection = tracker.detect(img_rgb);
        const auto t1 = std::chrono::steady_clock::now();

        detect_s.push_back(std::chrono::duration<double>(t1 - t0).count());
        card_counts.push_back(detection.has_value() ? detection->detected_cards.size() : 0);
    }

    const Stats detect = summarize(detect_s);
    const double avg_cards = static_cast<double>(std::accumulate(card_counts.begin(), card_counts.end(), size_t{0})) /
                             static_cast<double>(card_counts.size());

    std::cout << "DetectionTracker::detect() profile (" << video_paths.front().string() << ", " << detect_s.size()
              << " runs)\n";
    print_stat("tracker.detect()", detect, 1e3, "ms");
    std::cout << std::left << std::setw(28) << "cards tracked" << std::setprecision(6) << avg_cards << "\n";
    std::cout << std::left << std::setw(28) << "30fps frame budget" << (1000.0 / 30.0) << " ms\n";

    return 0;
}

// Plays one video with tracked detection overlaid on windows video_win/arrangement_win.
// tracker.detect() is called synchronously on the FrameGrabber's own paced thread (it
// must see every frame in strict sequence for correct optical-flow propagation); the
// slow ONNX detection call itself still runs on DetectionTracker's own internal
// AsyncCardDetector thread, so this stays cheap per frame. Returns true if the user
// requested quit.
auto play_video_with_tracking(const fs::path& video_path, const cd::CardDetector& detector,
                              const ip::SparsePropagatorParams& track_params, int max_side,
                              const std::string& video_win, const std::string& arrangement_win) -> bool {
    ct::DetectionTracker tracker(detector, track_params, max_side);

    std::mutex frame_lock;
    std::optional<cv::Mat> current_frame;
    std::optional<cd::Detection> latest_detection;

    auto on_frame = [&](const sd_ia::Frame& frame) {
        const cv::Mat img_rgb = setdetect::apps::bgr_to_rgb(frame.data);
        std::optional<cd::Detection> detection = tracker.detect(img_rgb);
        const std::lock_guard<std::mutex> guard(frame_lock);
        current_frame = frame.data.clone();
        latest_detection = std::move(detection);
    };

    sd_ia::FrameGrabber grabber(video_path.string(), on_frame, 30.0);
    grabber.start();

    bool quit_requested = false;
    while (true) {
        cv::Mat vis;
        std::optional<cd::Detection> detection_copy;
        bool have_frame = false;
        {
            const std::lock_guard<std::mutex> guard(frame_lock);
            if (current_frame.has_value()) {
                vis = std::move(*current_frame);
                current_frame.reset();
                detection_copy = latest_detection;
                have_frame = true;
            }
        }
        if (have_frame) {
            std::vector<sg::Card> labels;
            if (detection_copy.has_value()) {
                labels.reserve(detection_copy->matches.size());
                for (const auto& [dc, card] : detection_copy->matches) {
                    setdetect::apps::draw_matched_outline(vis, dc.corners);
                    labels.push_back(card);
                }
            }
            cv::imshow(video_win, vis);
            cv::imshow(arrangement_win, setdetect::apps::draw_arrangement(detection_copy.has_value()
                                                                              ? detection_copy->matches_arrangement
                                                                              : cd::CardsArrangement{},
                                                                          labels));
        } else if (!grabber.is_running()) {
            break;
        }

        const int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) {
            quit_requested = true;
            break;
        }
    }

    grabber.stop();
    return quit_requested;
}

// Shows tracked detections on the playing video (window "Video") alongside the
// estimated arrangement (window "Arrangement"), advancing automatically through val
// videos.
auto show_videos(const std::string& video_root, const setdetect::apps::DetectionArgs& det_args,
                 const setdetect::apps::TrackingArgs& track_args) -> int {
    const std::vector<fs::path> video_paths = setdetect::apps::list_videos(video_root);
    if (video_paths.empty()) {
        throw std::runtime_error("no videos found under " + video_root);
    }
    std::cout << "found " << video_paths.size() << " video(s)\n";

    const cd::CardDetector detector(det_args.corner_weights, det_args.class_weights, det_args.device, det_args.conf,
                                    /*iou=*/0.45, det_args.imgsz);

    const std::string video_win = "Video";
    const std::string arrangement_win = "Arrangement";
    cv::namedWindow(video_win, cv::WINDOW_NORMAL);
    cv::namedWindow(arrangement_win, cv::WINDOW_NORMAL);

    for (const fs::path& video_path : video_paths) {
        std::cout << "playing: " << video_path.string() << "\n";
        if (play_video_with_tracking(video_path, detector, track_args.params, det_args.imgsz, video_win,
                                     arrangement_win)) {
            break;
        }
    }

    cv::destroyAllWindows();
    return 0;
}

} // namespace

auto main(int argc, char* argv[]) -> int {
    CLI::App app{"Functionality related to card tracking."};

    setdetect::apps::DetectionArgs det_args;
    setdetect::apps::TrackingArgs track_args;
    std::string video_root;

    int profile_runs = kDefaultProfileRuns;
    auto* profile_sub =
        app.add_subcommand("profile-tracking", "Profiles DetectionTracker::detect() over a real video's frames.");
    setdetect::apps::add_detection_args(*profile_sub, det_args);
    setdetect::apps::add_tracking_args(*profile_sub, track_args);
    setdetect::apps::add_video_root_arg(*profile_sub, video_root);
    profile_sub->add_option("--runs", profile_runs,
                            "Number of consecutive video frames to time (loops if it runs out)");
    profile_sub->callback([&]() { profile_tracking(video_root, det_args, track_args, profile_runs); });

    auto* video_sub = app.add_subcommand(
        "show-videos", "Runs the game-state detection pipeline on all videos in a folder and displays overlaid cards.");
    setdetect::apps::add_detection_args(*video_sub, det_args);
    setdetect::apps::add_tracking_args(*video_sub, track_args);
    setdetect::apps::add_video_root_arg(*video_sub, video_root);
    video_sub->callback([&]() { show_videos(video_root, det_args, track_args); });

    try {
        CLI11_PARSE(app, argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
