cmake_minimum_required(VERSION 3.25)
project(setdetect LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(SETDETECT_BUILD_APPS "Build desktop testing apps" ON)
option(SETDETECT_BUILD_INTEGRATION_TESTS
    "Build/register tests that download and run the real, published ONNX models (opt-in: network + slower)" OFF)

# Dependency: ONNX
if(NOT DEFINED ENV{ONNXRUNTIME_ROOT})
    message(FATAL_ERROR "ONNXRUNTIME_ROOT environment variable is not set.")
endif()
set(ONNXRUNTIME_ROOT $ENV{ONNXRUNTIME_ROOT})

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    PATHS "${ONNXRUNTIME_ROOT}/include"
    NO_DEFAULT_PATH
)
find_library(ONNXRUNTIME_LIB onnxruntime
    PATHS "${ONNXRUNTIME_ROOT}/lib"
    NO_DEFAULT_PATH
)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "Could not find ONNX Runtime headers/libs under ${ONNXRUNTIME_ROOT}")
endif()

add_library(onnxruntime::onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime::onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
)

# Dependency: cuDNN. Needed only at runtime, by ONNX Runtime's CUDA execution provider
# (libonnxruntime_providers_cuda.so), which is itself dlopen()'d by libonnxruntime.so --
# not linked by us. That provider .so carries no RPATH of its own, so RPATH on our own
# executables cannot help it find libcudnn: CUDNN_ROOT/lib must be on LD_LIBRARY_PATH in
# the environment that runs the binary (see cuDNN section of ~/.zshrc). This check just
# catches a misconfigured/missing CUDNN_ROOT early, at configure time. Only the versioned
# .so (e.g. libcudnn.so.9) is present -- no unversioned dev symlink -- so this is a plain
# existence check rather than find_library().
if(NOT DEFINED ENV{CUDNN_ROOT})
    message(FATAL_ERROR "CUDNN_ROOT environment variable is not set.")
endif()
set(CUDNN_ROOT $ENV{CUDNN_ROOT})

file(GLOB CUDNN_LIB_FILES "${CUDNN_ROOT}/lib/libcudnn.so*")
if(NOT CUDNN_LIB_FILES)
    message(FATAL_ERROR "Could not find libcudnn under ${CUDNN_ROOT}/lib")
endif()

# Dependency: Eigen
find_package(Eigen3 CONFIG REQUIRED)
find_package(Threads REQUIRED)

# Dependency: OpenCV. Used by the core library itself (corners_st.cpp,
# opt_flow_pyr_lk.cpp -- thin wrappers around cv::goodFeaturesToTrack and
# cv::calcOpticalFlowPyrLK, not our own reimplementations), not just desktop
# apps -- on Android this is a minimal core+imgproc+video+features build (see
# android/app/build.gradle.kts), but on desktop it's simplest to reuse the
# same full conda-forge install the desktop apps below already need.
if(NOT DEFINED ENV{OpenCV_ROOT})
    message(FATAL_ERROR "OpenCV_ROOT environment variable is not set. OpenCV can be installed using conda-forge.")
endif()
set(OpenCV_ROOT_PATH $ENV{OpenCV_ROOT})
find_package(OpenCV CONFIG REQUIRED)

# Core library
add_library(setdetect
    src/card_detection/async_detector.cpp
    src/card_detection/detector.cpp
    src/card_tracking/detection_tracker.cpp
    src/img_proc/corners_st.cpp
    src/img_proc/opt_flow_pyr_lk.cpp
    src/img_proc/sparse_propagator.cpp
    src/set_game/game.cpp
)
target_include_directories(setdetect PUBLIC include)
target_link_libraries(setdetect PUBLIC
    onnxruntime::onnxruntime
    Eigen3::Eigen
    Threads::Threads
    ${OpenCV_LIBS}
)
target_include_directories(setdetect PRIVATE ${OpenCV_INCLUDE_DIRS})

# Apps
if(SETDETECT_BUILD_APPS)
    # Dependency: CLI11
    find_package(CLI11 CONFIG REQUIRED)

    # Library for executables. OpenCV comes transitively via setdetect (linked
    # PUBLIC above), which now depends on it directly -- no need to relist it.
    add_library(setdetect_cv STATIC
        src/img_acq/frame_grabber.cpp)
    target_link_libraries(setdetect_cv PUBLIC setdetect)

    # App: frame-grabber
    add_executable(frame_grabber apps/frame_grabber/main.cpp)
    set_target_properties(frame_grabber PROPERTIES OUTPUT_NAME "frame-grabber")
    target_link_libraries(frame_grabber PRIVATE setdetect_cv CLI11::CLI11)
    target_include_directories(frame_grabber PRIVATE src)
    set_target_properties(frame_grabber PROPERTIES
        BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib;${OpenCV_ROOT_PATH}/lib"
        INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib;${OpenCV_ROOT_PATH}/lib"
    )

    # App: tps-rbf
    add_executable(tps_rbf apps/tps_rbf/main.cpp)
    set_target_properties(tps_rbf PROPERTIES OUTPUT_NAME "tps-rbf")
    target_link_libraries(tps_rbf PRIVATE setdetect CLI11::CLI11)
    target_include_directories(tps_rbf PRIVATE src)
    set_target_properties(tps_rbf PROPERTIES
        BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib;${OpenCV_ROOT_PATH}/lib"
        INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib;${OpenCV_ROOT_PATH}/lib"
    )

    # App: corners-st
    add_executable(corners_st apps/corners_st/main.cpp)
    set_target_properties(corners_st PROPERTIES OUTPUT_NAME "corners-st")
    target_link_libraries(corners_st PRIVATE setdetect CLI11::CLI11)
    target_include_directories(corners_st PRIVATE src)
    set_target_properties(corners_st PROPERTIES
        BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib;${OpenCV_ROOT_PATH}/lib"
        INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib;${OpenCV_ROOT_PATH}/lib"
    )

    # App: opt-flow-pyr-lk
    add_executable(opt_flow_pyr_lk apps/opt_flow_pyr_lk/main.cpp)
    set_target_properties(opt_flow_pyr_lk PROPERTIES OUTPUT_NAME "opt-flow-pyr-lk")
    target_link_libraries(opt_flow_pyr_lk PRIVATE setdetect CLI11::CLI11)
    target_include_directories(opt_flow_pyr_lk PRIVATE src)
    set_target_properties(opt_flow_pyr_lk PROPERTIES
        BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib;${OpenCV_ROOT_PATH}/lib"
        INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib;${OpenCV_ROOT_PATH}/lib"
    )

    # App: sparse-propagator
    add_executable(sparse_propagator apps/sparse_propagator/main.cpp)
    set_target_properties(sparse_propagator PROPERTIES OUTPUT_NAME "sparse-propagator")
    target_link_libraries(sparse_propagator PRIVATE setdetect CLI11::CLI11)
    target_include_directories(sparse_propagator PRIVATE src)
    set_target_properties(sparse_propagator PROPERTIES
        BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib;${OpenCV_ROOT_PATH}/lib"
        INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib;${OpenCV_ROOT_PATH}/lib"
    )

    # App: card-detection
    add_executable(card_detection apps/card_detection/main.cpp)
    set_target_properties(card_detection PROPERTIES OUTPUT_NAME "card-detection")
    target_link_libraries(card_detection PRIVATE setdetect_cv CLI11::CLI11)
    target_include_directories(card_detection PRIVATE src)
    set_target_properties(card_detection PROPERTIES
        BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib;${OpenCV_ROOT_PATH}/lib"
        INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib;${OpenCV_ROOT_PATH}/lib"
    )

    # App: card-tracking
    add_executable(card_tracking apps/card_tracking/main.cpp)
    set_target_properties(card_tracking PROPERTIES OUTPUT_NAME "card-tracking")
    target_link_libraries(card_tracking PRIVATE setdetect_cv CLI11::CLI11)
    target_include_directories(card_tracking PRIVATE src)
    set_target_properties(card_tracking PROPERTIES
        BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib;${OpenCV_ROOT_PATH}/lib"
        INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib;${OpenCV_ROOT_PATH}/lib"
    )
endif()

# Tests
include(CTest)
if(BUILD_TESTING)
    add_subdirectory(test)
endif()
