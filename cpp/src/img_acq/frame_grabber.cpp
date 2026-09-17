#include <chrono>
#include <mutex>
#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <setdetect/img_acq/frame_grabber.hpp>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

using namespace std::chrono;

namespace setdetect::img_acq {

FrameGrabber::FrameGrabber(const std::string& source, FrameCallback on_frame, double target_fps)
    : on_frame_(std::move(on_frame)), target_fps_(target_fps) {
    if (!cap_.open(source)) {
        throw std::runtime_error("Could not open source: " + source);
    }
    if (target_fps_ <= 0.0) {
        target_fps_ = cap_.get(cv::CAP_PROP_FPS);
    }
    if (target_fps_ <= 0.0) {
        target_fps_ = 30.0;
    }
}

FrameGrabber::~FrameGrabber() { stop(); }

void FrameGrabber::start() { thread_ = std::thread(&FrameGrabber::run, this); }

void FrameGrabber::stop() {
    stop_.store(true);
    if (thread_.joinable()) {
        thread_.join();
    }
    cap_.release();
}

void FrameGrabber::run() {
    auto period = duration<double>(1.0 / target_fps_);
    auto next_tick = steady_clock::now();

    while (!stop_.load()) {
        cv::Mat frame;
        if (!cap_.read(frame)) {
            break;
        }

        Frame const f{std::move(frame), duration<double>(steady_clock::now().time_since_epoch()).count(), frame_id_++};
        on_frame_(f);

        next_tick += duration_cast<steady_clock::duration>(period);
        auto now = steady_clock::now();
        if (next_tick > now) {
            std::this_thread::sleep_until(next_tick);
        } else {
            next_tick = now;
        }
    }
    stop_.store(true);
}

AsyncDispatcher::AsyncDispatcher(FrameCallback handler) : handler_(std::move(handler)) {
    worker_ = std::thread(&AsyncDispatcher::loop, this);
}

AsyncDispatcher::~AsyncDispatcher() { stop(); }

void AsyncDispatcher::operator()(const Frame& frame) {
    std::lock_guard<std::mutex> const guard(lock_);
    slot_ = frame;
    has_slot_.store(true);
}

void AsyncDispatcher::stop() {
    stop_.store(true);
    if (worker_.joinable()) {
        worker_.join();
    }
}

void AsyncDispatcher::clear() {
    std::lock_guard<std::mutex> const guard(lock_);
    has_slot_.store(false);
}

void AsyncDispatcher::loop() {
    while (!stop_.load()) {
        if (has_slot_.load()) {
            Frame frame;
            {
                std::lock_guard<std::mutex> const guard(lock_);
                frame = slot_;
                has_slot_.store(false);
            }
            handler_(frame);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

} // namespace setdetect::img_acq
