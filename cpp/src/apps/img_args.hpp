#pragma once

#include <CLI/CLI.hpp>
#include <string>

namespace setdetect::apps {

// Registers a required -d/--data-root option on `sub`, writing the parsed value
// into `data_root`.
inline void add_img_root_arg(CLI::App& sub, std::string& data_root) {
    sub.add_option("-d,--data-root", data_root,
                   "Root directory of a real board-photo dataset (raw/ + optional labels/)")
        ->required();
}

} // namespace setdetect::apps
