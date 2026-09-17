#pragma once

#include <CLI/CLI.hpp>
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <string>
#include <vector>

namespace setdetect::apps {

inline const std::vector<std::string> kVideoSuffixes = {".mp4", ".avi", ".mov", ".mkv"};

// All video files directly under `root` (non-recursive), sorted by path.
inline auto list_videos(const std::string& root) -> std::vector<std::filesystem::path> {
    namespace fs = std::filesystem;
    std::vector<fs::path> videos;
    if (fs::exists(root) && fs::is_directory(root)) {
        for (const auto& entry : fs::directory_iterator(root)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (std::find(kVideoSuffixes.begin(), kVideoSuffixes.end(), ext) != kVideoSuffixes.end()) {
                videos.push_back(entry.path());
            }
        }
    }
    std::sort(videos.begin(), videos.end());
    return videos;
}

// Registers a required -d/--data-root option on `sub` (mirrors Python's
// ui/data_args.py::add_video_root_arg), writing the parsed value into `video_root`.
inline void add_video_root_arg(CLI::App& sub, std::string& video_root) {
    sub.add_option("-d,--data-root", video_root, "Root directory of a video dataset")->required();
}

} // namespace setdetect::apps
