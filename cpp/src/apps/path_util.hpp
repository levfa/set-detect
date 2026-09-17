#pragma once

#include <cstdlib>
#include <string>

namespace setdetect::apps {

// Expands a leading "~" to $HOME; returns `path` unchanged otherwise (no $HOME, or no
// leading "~").
inline auto expand_tilde(const std::string& path) -> std::string {
    if (path.empty() || path[0] != '~') {
        return path;
    }
    // NOLINTNEXTLINE(concurrency-mt-unsafe) called once before any threads start
    const char* home = std::getenv("HOME");
    if (home == nullptr) {
        return path;
    }
    return std::string(home) + path.substr(1);
}

} // namespace setdetect::apps
