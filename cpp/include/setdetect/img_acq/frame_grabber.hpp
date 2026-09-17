#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <thread>

namespace setdetect::img_acq {

struct Frame {
    cv::Mat data;
    double timestamp;
    int64_t frame_id;
};

using FrameCallback = std::function<void(const Frame&)>;

class FrameGrabber {
  public:
    FrameGrabber(const std::string& source, FrameCallback on_frame, double target_fps = 0.0);
    ~FrameGrabber();

    FrameGrabber(const FrameGrabber&) = delete;
    auto operator=(const FrameGrabber&) -> FrameGrabber& = delete;

    void start();
    void stop();
    [[nodiscard]] auto is_running() const -> bool { return !stop_.load(); }

    auto cap() -> cv::VideoCapture& { return cap_; }

  private:
    void run();

    cv::VideoCapture cap_;
    FrameCallback on_frame_;
    double target_fps_;
    std::atomic<bool> stop_{false};
    int64_t frame_id_{0};
    std::thread thread_;
};

class AsyncDispatcher {
  public:
    explicit AsyncDispatcher(FrameCallback handler);
    ~AsyncDispatcher();

    AsyncDispatcher(const AsyncDispatcher&) = delete;
    auto operator=(const AsyncDispatcher&) -> AsyncDispatcher& = delete;

    void operator()(const Frame& frame);
    void stop();
    void clear();

  private:
    void loop();

    FrameCallback handler_;
    std::mutex lock_;
    std::atomic<bool> has_slot_{false};
    Frame slot_;
    std::atomic<bool> stop_{false};
    std::thread worker_;
};

} // namespace setdetect::img_acq
