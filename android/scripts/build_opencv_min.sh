#!/usr/bin/env bash
# Builds a minimal, size-scoped static OpenCV for Android (core+imgproc+video+
# features: the only calls corners_st.cpp/opt_flow_pyr_lk.cpp make are
# cv::goodFeaturesToTrack and cv::calcOpticalFlowPyrLK/buildOpticalFlowPyramid)
# and installs it under OUT_DIR. Invoked by the buildMinimalOpenCv Gradle task
# in app/build.gradle.kts, gated on OUT_DIR already containing a stamp file so
# repeated builds are a no-op until OPENCV_VERSION or this script's own recipe
# changes.
#
# Usage: build_opencv_min.sh <ABI> <ANDROID_PLATFORM> <NDK_DIR> <OUT_DIR> <WORK_DIR>
set -euo pipefail

ABI="$1"
PLATFORM="$2"
NDK_DIR="$3"
OUT_DIR="$4"
WORK_DIR="$5"
OPENCV_VERSION="5.0.0"

STAMP="${OUT_DIR}/.stamp-${OPENCV_VERSION}"
if [[ -f "${STAMP}" ]]; then
    echo "Minimal OpenCV ${OPENCV_VERSION} for ${ABI} already built at ${OUT_DIR}, skipping."
    exit 0
fi

SRC_DIR="${WORK_DIR}/opencv-${OPENCV_VERSION}-src"
BUILD_DIR="${WORK_DIR}/opencv-${OPENCV_VERSION}-build-${ABI}"

if [[ ! -d "${SRC_DIR}" ]]; then
    git clone --depth 1 --branch "${OPENCV_VERSION}" https://github.com/opencv/opencv.git "${SRC_DIR}"
fi

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cmake -S "${SRC_DIR}" -B "${BUILD_DIR}" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="${NDK_DIR}/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="${ABI}" \
    -DANDROID_PLATFORM="${PLATFORM}" \
    -DANDROID_STL=c++_shared \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_LIST=core,imgproc,video,features \
    -DBUILD_JAVA=OFF \
    -DBUILD_ANDROID_PROJECTS=OFF \
    -DBUILD_ANDROID_EXAMPLES=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_DOCS=OFF \
    -DBUILD_opencv_apps=OFF \
    -DWITH_ITT=OFF \
    -DWITH_IPP=OFF \
    -DWITH_PROTOBUF=OFF \
    -DWITH_QUIRC=OFF \
    -DBUILD_PROTOBUF=OFF \
    -DWITH_JPEG=OFF -DWITH_PNG=OFF -DWITH_TIFF=OFF -DWITH_WEBP=OFF -DWITH_OPENJPEG=OFF -DWITH_JASPER=OFF -DWITH_OPENEXR=OFF \
    -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_V4L=OFF \
    -DBUILD_ZLIB=OFF -DWITH_1394=OFF

ninja -C "${BUILD_DIR}"

STRIP="${NDK_DIR}/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip"
LIB_OUT="${OUT_DIR}/lib"
INCLUDE_OUT="${OUT_DIR}/include"
rm -rf "${LIB_OUT}" "${INCLUDE_OUT}"
mkdir -p "${LIB_OUT}" "${INCLUDE_OUT}"

for lib in "${BUILD_DIR}"/lib/"${ABI}"/*.a "${BUILD_DIR}"/3rdparty/lib/"${ABI}"/*.a; do
    [[ -f "${lib}" ]] || continue
    cp "${lib}" "${LIB_OUT}/"
    "${STRIP}" --strip-debug "${LIB_OUT}/$(basename "${lib}")"
done

# Headers: the generated (per-build-config) ones plus every module's public
# include tree, mirroring what the real OpenCV Android SDK packages.
cp "${BUILD_DIR}"/cvconfig.h "${BUILD_DIR}"/cv_cpu_config.h "${BUILD_DIR}"/custom_hal.hpp "${INCLUDE_OUT}/" 2>/dev/null || true
cp -r "${BUILD_DIR}/opencv2" "${INCLUDE_OUT}/" 2>/dev/null || true
[[ -d "${BUILD_DIR}/carotene" ]] && cp -r "${BUILD_DIR}/carotene" "${INCLUDE_OUT}/"
for mod_include in "${SRC_DIR}"/modules/*/include; do
    cp -r "${mod_include}/." "${INCLUDE_OUT}/"
done

touch "${STAMP}"
echo "Minimal OpenCV ${OPENCV_VERSION} for ${ABI} installed to ${OUT_DIR}"
