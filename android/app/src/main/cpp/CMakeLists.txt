cmake_minimum_required(VERSION 3.22.1)
project(setdetect_jni)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED NATIVE_SDK_DIR)
    set(NATIVE_SDK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../build/native-sdk)
endif()

message(STATUS "NATIVE_SDK_DIR: ${NATIVE_SDK_DIR}")

# ONNX Runtime
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${NATIVE_SDK_DIR}/onnx/jni/${ANDROID_ABI}/libonnxruntime.so"
    INTERFACE_INCLUDE_DIRECTORIES "${NATIVE_SDK_DIR}/onnx/headers")

# OpenCV: a minimal, size-scoped static build (core+imgproc+video+features --
# only what corners_st.cpp/opt_flow_pyr_lk.cpp actually call: cv::goodFeaturesToTrack
# and cv::calcOpticalFlowPyrLK/buildOpticalFlowPyramid), not the full ~32 MB
# Java-bindings AAR. Built by android/scripts/build_opencv_min.sh via the
# buildMinimalOpenCv Gradle task in app/build.gradle.kts, which runs before this
# CMake configure.
if(NOT DEFINED OPENCV_MIN_DIR)
    set(OPENCV_MIN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../build/native-sdk-opencv)
endif()
set(OPENCV_MIN_ABI_DIR ${OPENCV_MIN_DIR}/${ANDROID_ABI})
message(STATUS "OPENCV_MIN_ABI_DIR: ${OPENCV_MIN_ABI_DIR}")

set(OPENCV_MIN_LIBS
    opencv_video opencv_features opencv_imgproc opencv_core opencv_flann opencv_geometry
    kleidicv_hal kleidicv_thread kleidicv tegra_hal cpufeatures
)
foreach(lib ${OPENCV_MIN_LIBS})
    add_library(${lib} STATIC IMPORTED)
    set_target_properties(${lib} PROPERTIES IMPORTED_LOCATION "${OPENCV_MIN_ABI_DIR}/lib/lib${lib}.a")
endforeach()

# Eigen (header-only). setdetect's core library needs it (sparse_propagator.cpp,
# opt_flow_pyr_lk.hpp, detector.cpp), but desktop's find_package(Eigen3 CONFIG REQUIRED)
# is tied to a vcpkg x64-linux triplet install -- not usable from an NDK cross-compile,
# and not something this build should couple itself to (would require a desktop vcpkg
# build to already exist on disk). FetchContent pulls just the header tree, without
# invoking Eigen's own CMakeLists (unneeded build-time cost for a header-only lib).
include(FetchContent)
FetchContent_Declare(
    eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
)
FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
    FetchContent_Populate(eigen)
endif()

set(SETDETECT_CPP_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../cpp)

add_library(setdetect_jni SHARED
    native-lib.cpp
    ${SETDETECT_CPP_ROOT}/src/img_proc/corners_st.cpp
    ${SETDETECT_CPP_ROOT}/src/img_proc/opt_flow_pyr_lk.cpp
    ${SETDETECT_CPP_ROOT}/src/img_proc/sparse_propagator.cpp
    ${SETDETECT_CPP_ROOT}/src/card_detection/detector.cpp
    ${SETDETECT_CPP_ROOT}/src/card_detection/async_detector.cpp
    ${SETDETECT_CPP_ROOT}/src/card_tracking/detection_tracker.cpp
    ${SETDETECT_CPP_ROOT}/src/set_game/game.cpp
)

target_include_directories(setdetect_jni PRIVATE
    ${SETDETECT_CPP_ROOT}/include
    ${eigen_SOURCE_DIR}
    ${OPENCV_MIN_ABI_DIR}/include
)

# OpenCV's static libs are wrapped in --start-group/--end-group: core, imgproc,
# video and features cross-reference each other (verified when prototyping this
# build), so a single link-order pass isn't guaranteed to resolve everything.
target_link_libraries(setdetect_jni
    onnxruntime
    android
    log
    jnigraphics
    z
    "-Wl,--start-group"
    ${OPENCV_MIN_LIBS}
    "-Wl,--end-group"
)
