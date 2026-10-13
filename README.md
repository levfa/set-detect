# Set Detect
Set Detect is an Android app that helps to find sets in the card game SET®.
The laid-out cards can be entered manually or scanned via the phone's camera.

<p>
  <img src="android-publishing/screen/screen-capture.gif" alt="Screen capture" height="300">
  <img src="android-publishing/screen/screenshot_0.png" alt="Scanning a board" height="300">
  <img src="android-publishing/screen/screenshot_1.png" alt="Detected cards" height="300">
  <img src="android-publishing/screen/screenshot_2.png" alt="Sets found" height="300">
</p>

This repository contains the Android frontend and the C++ computer vision backend for card scanning and real-time video processing. It also holds the Python framework for training the involved artificial neural networks.

## Disclaimer
Unofficial, fan-made project; not affiliated with, endorsed by, or sponsored by SET Enterprises, Inc.

## Pre-trained models
The pre-trained models for card detection and classification via the phone's camera are available at https://github.com/levfa/set-detect-models.

## Training and validation of vision models (Python)
The Python project is located in [py/](py/).

### Installation and general usage
The following commands have to be run from [py/](py/).
* Prerequisite: [uv](https://docs.astral.sh/uv/)
* Install and activate: `uv sync` and `source .venv/bin/activate`
* Test: `pytest`
* Lint: `ruff check` and `ruff format --check`

The executable scripts are located in [py/src/setdetect/cli/](py/src/setdetect/cli/) and are installed as `set-...` aliases. Each has a help page that can be shown via a `-h` flag.

### Example
The models for card detection and classification can be trained and used as follows (execution from the repository root):
* Create cards:
  ```
  set-card-cutouts extract-cards \
    --data-root examples/data/img-real-card-cutouts
  ```
* Download and select background textures:
  ```
  set-backgrounds-data download \
    --local-dir examples/data/huggingface/nyuuzyou/texturecan
  ```
  ```
  set-backgrounds-data build-manifest \
    --local-dir examples/data/huggingface/nyuuzyou/texturecan
  ```
* Train and export the card corner detection:
  * Generate a validation data set:
    ```
    set-card-corners synth-gen \
      --data-root examples/data/img-synth-boards-corners \
      --cards-root examples/data/img-real-card-cutouts \
      --textures-root examples/data/huggingface/nyuuzyou/texturecan
    ```
  * Train:
    ```
    set-card-corners synth-train \
      --data-root examples/data/img-synth-boards-corners \
      --runs-root examples/models/set-card-corners/runs \
      --cards-root examples/data/img-real-card-cutouts \
      --textures-root examples/data/huggingface/nyuuzyou/texturecan \
      --models-root examples/models/ultralytics-cache
    ```
  * Make the latest training result the default:
    ```
    set-card-corners promote \
      --runs-root examples/models/set-card-corners/runs
    ```
  * Export:
    ```
    set-card-corners export \
      --runs-root examples/models/set-card-corners/runs
    ```
* Train and export the quantized card classifier:
  * Generate a validation and quantization data set:
    ```
    set-card-classification synth-gen \
      --data-root examples/data/img-synth-boards-class \
      --cards-root examples/data/img-real-card-cutouts \
      --textures-root examples/data/huggingface/nyuuzyou/texturecan
    ```
  * Train:
    ```
    set-card-classification train \
      --data-root examples/data/img-synth-boards-class \
      --runs-root examples/models/set-card-classification/runs \
      --cards-root examples/data/img-real-card-cutouts \
      --textures-root examples/data/huggingface/nyuuzyou/texturecan
    ```
  * Make the latest training result the default:
    ```
    set-card-classification promote \
      --runs-root examples/models/set-card-classification/runs
    ```
  * Export and quantize:
    ```
    set-card-classification export \
      --runs-root examples/models/set-card-classification/runs
    ```
    ```
    set-card-classification quantize \
      --data-root examples/data/img-synth-boards-class \
      --runs-root examples/models/set-card-classification/runs
    ```
* Copy the exported models into the Android app's assets:
  ```
  set-card-corners prepare-android \
    --runs-root examples/models/set-card-corners/runs
  ```
  ```
  set-card-classification prepare-android \
    --runs-root examples/models/set-card-classification/runs
  ```
* Detect, rectify, and classify cards in an image:
  ```
  set-card-detection show-arrangement \
    --data-root examples/data/img-real-card-cutouts \
    --det-corner-weights examples/models/set-card-corners/runs/current/onnx/card_corners.onnx \
    --det-class-weights examples/models/set-card-classification/runs/current/onnx/model_quantized.onnx
  ```

## Computer vision backend for real-time video processing (C++)
The C++ implementation of the card detection and classification pipeline is located in [cpp/](cpp/).
It uses the ONNX-exports from the Python training pipeline.

### Building and development
* Prerequisites:
  * CMake
  * Ninja
  * vcpkg
  * ONNX runtime
  * cuDNN
  * OpenCV (can be installed via conda-forge)
  * pre-commit
  * The following environment variables must point to valid installations:
    * `VCPKG_ROOT`
    * `ONNXRUNTIME_ROOT`
    * `CUDNN_ROOT` (and `CUDNN_ROOT/lib` must be on `LD_LIBRARY_PATH`)
    * `OpenCV_ROOT`
* Build (in `cpp/`):
  ```
  ./configure
  ninja -C build
  ```
* Format (from the repository root): `pre-commit run clang-format --all-files`
* Code-quality check (from the repository root): `pre-commit run clang-tidy --all-files`
* Test (in `cpp/`): `ctest --test-dir build --output-on-failure`

### Usage
The header files are located in [cpp/include](cpp/include).
The sources of the example applications are located in [cpp/apps](cpp/apps).
After building, they are available as executables in `cpp/build/`, e.g. (execution from the repository root):
```
cpp/build/card-detection show-boards \
  --det-corner-weights examples/models/set-card-corners/runs/current/onnx/card_corners.onnx \
  --det-class-weights examples/models/set-card-classification/runs/current/onnx/model_quantized.onnx \
  --data-root examples/data/img-real-card-cutouts
```

## Android (Kotlin)

### Usage
[Download the latest release](https://github.com/levfa/set-detect/releases/latest) and install it on your Android device. A tutorial is included in the app.

### Development
The folder [android/](android/) can directly be opened as an Android Studio project.

## License
AGPL-3.0 (see [`LICENSE`](LICENSE)).
