#include <CLI/CLI.hpp>
#include <apps/video_args.hpp>
#include <functional>
#include <iostream>
#include <mutex>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <optional>
#include <setdetect/img_acq/frame_grabber.hpp>
#include <setdetect/util/string.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sd_str = setdetect::util::string;
namespace sd_ia = setdetect::img_acq;
using setdetect::apps::list_videos;

namespace {

auto show_videos(const std::string& data_root) -> int {
    auto videos = list_videos(data_root);
    if (videos.empty()) {
        throw std::runtime_error("no videos found under " + data_root);
    }

    std::cout << "found " << videos.size() << " video(s)\n";

    const std::string win_name = "Game State \u2014 Video";
    bool quit_requested = false;

    // Display slot: written by dispatcher worker, read by main thread
    std::mutex display_lock;
    std::optional<cv::Mat> display_frame;

    for (auto& vid_path : videos) {
        if (quit_requested) {
            break;
        }

        std::cout << "processing: " << vid_path.string() << "\n";

        cv::namedWindow(win_name, cv::WINDOW_NORMAL);

        // on_frame runs on the AsyncDispatcher worker thread
        auto on_frame = [&](const sd_ia::Frame& frame) {
            cv::Mat vis = frame.data.clone();
            {
                std::lock_guard<std::mutex> const guard(display_lock);
                display_frame = std::move(vis);
            }
        };

        sd_ia::AsyncDispatcher dispatcher(on_frame);
        sd_ia::FrameGrabber grabber(vid_path.string(), std::ref(dispatcher), 30.0);
        grabber.start();

        while (true) {
            cv::Mat vis;
            {
                std::lock_guard<std::mutex> const guard(display_lock);
                if (display_frame.has_value()) {
                    vis = std::move(*display_frame);
                    display_frame.reset();
                }
            }
            if (vis.empty()) {
                if (!grabber.is_running()) {
                    break;
                }
            } else {
                cv::imshow(win_name, vis);
            }
            int const key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 27) {
                quit_requested = true;
                break;
            }
        }

        dispatcher.stop();
        dispatcher.clear();
        grabber.stop();

        // Drain display slot
        {
            std::lock_guard<std::mutex> const guard(display_lock);
            display_frame.reset();
        }
    }

    cv::destroyAllWindows();
    return 0;
}

} // namespace

auto main(int argc, char* argv[]) -> int {
    CLI::App app{"Frame grabber functionality"};
    std::string data_root;
    auto* sub = app.add_subcommand("show-videos", "Shows videos in the specified folder (" +
                                                      sd_str::join(setdetect::apps::kVideoSuffixes, ", ") + ").");
    setdetect::apps::add_video_root_arg(*sub, data_root);
    sub->callback([&]() { show_videos(data_root); });

    CLI11_PARSE(app, argc, argv);
    return 0;
}
