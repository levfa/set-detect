#include <atomic>
#include <chrono>
#include <mutex>
#include <opencv2/core.hpp>
#include <optional>
#include <setdetect/card_detection/async_detector.hpp>
#include <thread>
#include <utility>

namespace setdetect::card_detection {

struct AsyncCardDetector::Impl {
    const CardDetector* detector;

    std::mutex in_lock;
    bool has_input = false;
    int in_frame_id = -1;
    cv::Mat in_img;

    std::mutex out_lock;
    bool has_output = false;
    Result out_result;

    std::atomic<bool> stop{false};
    std::thread worker;

    explicit Impl(const CardDetector& det) : detector(&det) { worker = std::thread(&Impl::loop, this); }

    void loop() {
        while (!stop.load()) {
            std::optional<std::pair<int, cv::Mat>> job;
            {
                const std::lock_guard<std::mutex> guard(in_lock);
                if (has_input) {
                    job = std::make_pair(in_frame_id, std::move(in_img));
                    has_input = false;
                }
            }
            if (job) {
                Result result;
                result.frame_id = job->first;
                result.detection = detector->detect(job->second);
                const std::lock_guard<std::mutex> guard(out_lock);
                out_result = std::move(result);
                has_output = true;
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }
};

AsyncCardDetector::AsyncCardDetector(const CardDetector& detector) : impl_(std::make_unique<Impl>(detector)) {}

AsyncCardDetector::~AsyncCardDetector() {
    impl_->stop.store(true);
    if (impl_->worker.joinable()) {
        impl_->worker.join();
    }
}

void AsyncCardDetector::submit(int frame_id, const cv::Mat& img_rgb) {
    const std::lock_guard<std::mutex> guard(impl_->in_lock);
    impl_->in_frame_id = frame_id;
    impl_->in_img = img_rgb;
    impl_->has_input = true;
}

auto AsyncCardDetector::poll() -> std::optional<Result> {
    const std::lock_guard<std::mutex> guard(impl_->out_lock);
    if (!impl_->has_output) {
        return std::nullopt;
    }
    impl_->has_output = false;
    return std::move(impl_->out_result);
}

} // namespace setdetect::card_detection
