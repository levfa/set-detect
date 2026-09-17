#include <CLI/CLI.hpp>
#include <algorithm>
#include <apps/cv_interop.hpp>
#include <apps/detection_args.hpp>
#include <apps/img_args.hpp>
#include <apps/stats.hpp>
#include <apps/video_args.hpp>
#include <apps/viewer_gui.hpp>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <optional>
#include <random>
#include <setdetect/card_detection/async_detector.hpp>
#include <setdetect/card_detection/detector.hpp>
#include <setdetect/img_acq/frame_grabber.hpp>
#include <setdetect/set_game/game.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace cd = setdetect::card_detection;
namespace sg = setdetect::set_game;
namespace sd_ia = setdetect::img_acq;
namespace fs = std::filesystem;
using setdetect::apps::print_stat;
using setdetect::apps::Stats;
using setdetect::apps::summarize;

namespace {

constexpr int kDefaultNumCards = 8;
constexpr int kDefaultRuns = 200;
constexpr unsigned kDefaultSeed = 42;
constexpr double kCardAspectRatio = 224.0 / 320.0; // matches the module's fixed card size

// Synthesizes n_cards detected cards scattered on a plane, seen through a fixed synthetic
// (mildly perspective) world->image homography, with randomized per-corner visibility so
// both the LM fit and the occlusion/z-order path are exercised.
auto make_synthetic_cards(int n_cards, std::mt19937& rng) -> std::vector<cd::DetectedCard> {
    std::uniform_real_distribution<double> pos_dist(0.0, 8.0);
    std::uniform_real_distribution<double> rot_dist(0.0, 2.0 * M_PI);
    std::uniform_real_distribution<double> vis_dist(0.3, 1.0);

    constexpr double kCardW = kCardAspectRatio;
    constexpr double kCardH = 1.0;
    const std::array<cd::Point2d, 4> local = {cd::Point2d{-kCardW / 2, -kCardH / 2},
                                              cd::Point2d{kCardW / 2, -kCardH / 2}, cd::Point2d{kCardW / 2, kCardH / 2},
                                              cd::Point2d{-kCardW / 2, kCardH / 2}};

    // clang-format off
    constexpr std::array<double, 9> kWorldToImg = {
        80.0,    10.0,   100.0,
        -5.0,    90.0,    60.0,
        0.0008, 0.0004,    1.0,
    };
    // clang-format on

    std::vector<cd::DetectedCard> cards(static_cast<size_t>(n_cards));
    for (auto& card : cards) {
        const double cx = pos_dist(rng);
        const double cy = pos_dist(rng);
        const double theta = rot_dist(rng);
        const double c = std::cos(theta);
        const double s = std::sin(theta);
        for (int k = 0; k < 4; ++k) {
            const double wx = cx + (c * local[static_cast<size_t>(k)].x) - (s * local[static_cast<size_t>(k)].y);
            const double wy = cy + (s * local[static_cast<size_t>(k)].x) + (c * local[static_cast<size_t>(k)].y);
            const double denom = (kWorldToImg[6] * wx) + (kWorldToImg[7] * wy) + kWorldToImg[8];
            const double px = ((kWorldToImg[0] * wx) + (kWorldToImg[1] * wy) + kWorldToImg[2]) / denom;
            const double py = ((kWorldToImg[3] * wx) + (kWorldToImg[4] * wy) + kWorldToImg[5]) / denom;
            card.corners[static_cast<size_t>(k)] = {px, py};
            card.corner_visibility[static_cast<size_t>(k)] = vis_dist(rng);
        }
    }
    return cards;
}

auto profile_arrangement_estimation(int n_cards, int runs, unsigned seed) -> int {
    std::mt19937 rng(seed);
    std::vector<double> fit_s(static_cast<size_t>(runs));

    for (int run = 0; run < runs; ++run) {
        const std::vector<cd::DetectedCard> cards = make_synthetic_cards(n_cards, rng);
        const auto t0 = std::chrono::steady_clock::now();
        const cd::CardsArrangement arrangement = cd::cards_arrangement(cards);
        const auto t1 = std::chrono::steady_clock::now();
        fit_s[static_cast<size_t>(run)] = std::chrono::duration<double>(t1 - t0).count();
        (void)arrangement;
    }

    const Stats fit = summarize(fit_s);
    std::cout << "cards_arrangement profile (" << n_cards << " cards, " << runs << " runs, seed " << seed << ")\n";
    print_stat("cards_arrangement", fit, 1e6, "us");

    return 0;
}

// "Val images" = every raw photo directly under <data-root>/raw/*.{jpg,jpeg,png} --
// mirrors the Python side's img_dir.list_images (a flat real board-photo dataset,
// e.g. img-real-web or img-real-boards; no more nested per-collection subfolders).
auto list_val_images(const std::string& data_root) -> std::vector<fs::path> {
    const fs::path raw_dir = fs::path(data_root) / "raw";
    std::vector<fs::path> images;
    if (fs::exists(raw_dir) && fs::is_directory(raw_dir)) {
        for (const auto& entry : fs::directory_iterator(raw_dir)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
                images.push_back(entry.path());
            }
        }
    }
    std::sort(images.begin(), images.end());
    return images;
}

// Shows detections on the real image (window "Original") alongside the estimated
// arrangement (window "Arrangement"), advancing on a keypress. Detection runs lazily,
// once per image, exactly when that image becomes current -- no upfront batch pass.
auto show_boards(const std::string& data_root, const setdetect::apps::DetectionArgs& det_args) -> int {
    const std::vector<fs::path> image_paths = list_val_images(data_root);
    if (image_paths.empty()) {
        throw std::runtime_error("no val images found under " + data_root);
    }
    std::cout << "found " << image_paths.size() << " image(s)\n";

    const cd::CardDetector detector(det_args.corner_weights, det_args.class_weights, det_args.device, det_args.conf,
                                    /*iou=*/0.45, det_args.imgsz);

    const std::string original_win = "Original";
    const std::string arrangement_win = "Arrangement";
    cv::namedWindow(original_win, cv::WINDOW_NORMAL);
    cv::namedWindow(arrangement_win, cv::WINDOW_NORMAL);

    size_t cur = 0;
    while (true) {
        const fs::path& path = image_paths[cur];
        const cv::Mat img_bgr = cv::imread(path.string());
        cv::Mat vis;
        cd::Detection detection;
        if (img_bgr.empty()) {
            std::cerr << "warning: failed to load " << path.string() << ", skipping\n";
            vis = cv::Mat::zeros(480, 640, CV_8UC3);
        } else {
            const cv::Mat img_rgb = setdetect::apps::bgr_to_rgb(img_bgr);
            detection = detector.detect(img_rgb);
            vis = img_bgr.clone();
            for (const cd::DetectedCard& dc : detection.detected_cards) {
                setdetect::apps::draw_detection(vis, dc);
            }
        }
        cv::imshow(original_win, vis);

        std::vector<sg::Card> labels;
        labels.reserve(detection.matches.size());
        for (const auto& [dc, card] : detection.matches) {
            labels.push_back(card);
        }
        cv::imshow(arrangement_win, setdetect::apps::draw_arrangement(detection.matches_arrangement, labels));

        const int key = cv::waitKey(0);
        const char ch = (key > 0 && key < 256) ? static_cast<char>(key) : '\0';
        if (key == setdetect::apps::kEscKey || ch == 'q') {
            break;
        }
        const setdetect::apps::ArrowKey dir = setdetect::apps::arrow_direction(key);
        if (dir == setdetect::apps::ArrowKey::kLeft || dir == setdetect::apps::ArrowKey::kUp) {
            cur = cur > 0 ? cur - 1 : 0;
        } else {
            cur = std::min(cur + 1, image_paths.size() - 1);
        }
    }

    cv::destroyAllWindows();
    return 0;
}

// Real ONNX inference is much slower than cards_arrangement()'s synthetic profile.
constexpr int kDefaultDetectionRuns = 10;

// Times the full detection pipeline (corners + classification + arrangement) on real
// val images, cycling through the available ones if runs exceeds their count.
auto profile_detection(const std::string& data_root, const setdetect::apps::DetectionArgs& det_args, int runs) -> int {
    const std::vector<fs::path> image_paths = list_val_images(data_root);
    if (image_paths.empty()) {
        throw std::runtime_error("no val images found under " + data_root);
    }

    const cd::CardDetector detector(det_args.corner_weights, det_args.class_weights, det_args.device, det_args.conf,
                                    /*iou=*/0.45, det_args.imgsz);

    std::vector<double> detect_s;
    std::vector<size_t> card_counts;
    detect_s.reserve(static_cast<size_t>(runs));
    card_counts.reserve(static_cast<size_t>(runs));

    for (int i = 0; i < runs; ++i) {
        const fs::path& path = image_paths[static_cast<size_t>(i) % image_paths.size()];
        const cv::Mat img_bgr = cv::imread(path.string());
        if (img_bgr.empty()) {
            std::cerr << "warning: failed to load " << path.string() << ", skipping\n";
            continue;
        }
        const cv::Mat img_rgb = setdetect::apps::bgr_to_rgb(img_bgr);

        const auto t0 = std::chrono::steady_clock::now();
        const cd::Detection detection = detector.detect(img_rgb);
        const auto t1 = std::chrono::steady_clock::now();

        detect_s.push_back(std::chrono::duration<double>(t1 - t0).count());
        card_counts.push_back(detection.detected_cards.size());
    }

    const Stats detect = summarize(detect_s);
    const double avg_cards = static_cast<double>(std::accumulate(card_counts.begin(), card_counts.end(), size_t{0})) /
                             static_cast<double>(card_counts.size());

    std::cout << "detect_cards profile (" << image_paths.size() << " val image(s) available, " << detect_s.size()
              << " runs)\n";
    print_stat("detect_cards", detect, 1e3, "ms");
    std::cout << std::left << std::setw(28) << "cards detected" << std::setprecision(6) << avg_cards << "\n";

    return 0;
}

// Plays one video with async detection overlaid on windows video_win/arrangement_win.
// Deliberately no tracking: the video plays at its own pace (a FrameGrabber thread
// updates the display frame every tick) while detection runs on a fully independent
// AsyncDispatcher thread, dropping frames it can't keep up with; the overlay always
// shows the latest *completed* detection on top of whatever frame is currently on
// screen, with no warping to compensate for camera motion since that detection was
// computed. Returns true if the user requested quit.
auto play_video_with_detection(const fs::path& video_path, const cd::CardDetector& detector,
                               const std::string& video_win, const std::string& arrangement_win) -> bool {
    std::mutex frame_lock;
    std::optional<cv::Mat> current_frame;

    cd::AsyncCardDetector async_detector(detector);

    // Runs at video pace; updates the display frame every tick and submits it (async,
    // drop-if-busy) to the detector.
    auto on_frame = [&](const sd_ia::Frame& frame) {
        {
            const std::lock_guard<std::mutex> guard(frame_lock);
            current_frame = frame.data.clone();
        }
        const cv::Mat img_rgb = setdetect::apps::bgr_to_rgb(frame.data);
        async_detector.submit(static_cast<int>(frame.frame_id), img_rgb);
    };

    sd_ia::FrameGrabber grabber(video_path.string(), on_frame, 30.0);
    grabber.start();

    bool quit_requested = false;
    std::optional<cd::Detection> latest_detection;
    while (true) {
        cv::Mat vis;
        bool have_frame = false;
        {
            const std::lock_guard<std::mutex> guard(frame_lock);
            if (current_frame.has_value()) {
                vis = std::move(*current_frame);
                current_frame.reset();
                have_frame = true;
            }
        }
        if (auto result = async_detector.poll()) {
            latest_detection = std::move(result->detection); // keep showing the last
        } // result until a newer one lands
        if (have_frame) {
            std::vector<sg::Card> labels;
            if (latest_detection.has_value()) {
                labels.reserve(latest_detection->matches.size());
                for (const auto& [dc, card] : latest_detection->matches) {
                    setdetect::apps::draw_matched_outline(vis, dc.corners);
                    labels.push_back(card);
                }
            }
            cv::imshow(video_win, vis);
            cv::imshow(arrangement_win, setdetect::apps::draw_arrangement(latest_detection.has_value()
                                                                              ? latest_detection->matches_arrangement
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

// Shows async detections on the playing video (window "Video") alongside the estimated
// arrangement (window "Arrangement"), advancing automatically through val videos.
auto show_videos(const std::string& video_root, const setdetect::apps::DetectionArgs& det_args) -> int {
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
        if (play_video_with_detection(video_path, detector, video_win, arrangement_win)) {
            break;
        }
    }

    cv::destroyAllWindows();
    return 0;
}

} // namespace

auto main(int argc, char* argv[]) -> int {
    CLI::App app{"Functionality related to card detection."};

    int n_cards = kDefaultNumCards;
    int runs = kDefaultRuns;
    unsigned seed = kDefaultSeed;

    auto* sub = app.add_subcommand("profile-arrangement-estimation", "Profiles cards_arrangement() latency.");
    sub->add_option("--n-cards", n_cards, "Number of synthetic cards per run");
    sub->add_option("--runs", runs, "Number of repeated runs for statistics");
    sub->add_option("--seed", seed, "RNG seed");
    sub->callback([&]() { profile_arrangement_estimation(n_cards, runs, seed); });

    setdetect::apps::DetectionArgs det_args;
    std::string data_root;
    auto* boards_sub =
        app.add_subcommand("show-boards", "Runs detection + classification on real board photos and shows the result.");
    setdetect::apps::add_detection_args(*boards_sub, det_args);
    setdetect::apps::add_img_root_arg(*boards_sub, data_root);
    boards_sub->callback([&]() { show_boards(data_root, det_args); });

    int profile_runs = kDefaultDetectionRuns;
    auto* profile_sub = app.add_subcommand(
        "profile-detection", "Profiles the full detection pipeline (corners + classification + arrangement).");
    setdetect::apps::add_detection_args(*profile_sub, det_args);
    setdetect::apps::add_img_root_arg(*profile_sub, data_root);
    profile_sub->add_option("--runs", profile_runs, "Number of images to time (cycles through available val images)");
    profile_sub->callback([&]() { profile_detection(data_root, det_args, profile_runs); });

    std::string video_root;
    auto* video_sub = app.add_subcommand(
        "show-videos", "Runs the game-state detection pipeline on all videos in a folder and displays overlaid cards.");
    setdetect::apps::add_detection_args(*video_sub, det_args);
    setdetect::apps::add_video_root_arg(*video_sub, video_root);
    video_sub->callback([&]() { show_videos(video_root, det_args); });

    try {
        CLI11_PARSE(app, argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
