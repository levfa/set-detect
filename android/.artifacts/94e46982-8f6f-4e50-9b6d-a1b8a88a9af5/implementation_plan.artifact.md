# Implementation Plan - Camera Arrangement View

Implement a new screen `CameraArrangementScreen` that displays the abstracted view of captured cards after a camera capture. This screen will allow users to review the detected cards, see how many sets were found, and decide whether to edit the selection, reveal the sets, or cancel.

## User Review Required

> [!IMPORTANT]
> This change introduces a new step in the camera-to-result flow. The user will now see an intermediate "Arrangement" view instead of being taken back to the manual selection screen immediately.

## Proposed Changes

### Domain & JNI

Expose the C++ `CardsArrangement` data to Kotlin and update the native detector to return it.

#### [MODIFY] [NativeSetDetector.kt](file:///home/fabian/repos/set-detect/android/app/src/main/java/com/fabianleven/setdetect/domain/NativeSetDetector.kt)
- Add `RotatedRect`, `CardsArrangement`, and `Detection` data classes (all `@Serializable`).
- Update `NativeSetDetector.detect` to return `Detection`.
- Update `detectNative` external function signature.

#### [MODIFY] [native-lib.cpp](file:///home/fabian/repos/set-detect/android/app/src/main/cpp/native-lib.cpp)
- Update `detectNative` to construct and return the new `Detection` object, including the `CardsArrangement` data.

---

### Navigation

Add the new screen to the navigation graph.

#### [MODIFY] [Routes.kt](file:///home/fabian/repos/set-detect/android/app/src/main/java/com/fabianleven/setdetect/navigation/Routes.kt)
- Add `CameraArrangement(val detection: Detection)` route.

#### [MODIFY] [SetDetectNavGraph.kt](file:///home/fabian/repos/set-detect/android/app/src/main/java/com/fabianleven/setdetect/navigation/SetDetectNavGraph.kt)
- Add entry for `Route.CameraArrangement`.
- Update `CameraScan` entry to navigate to `CameraArrangement` instead of returning to `ManualSelection`.

---

### UI Components

#### [NEW] [ArrangementView.kt](file:///home/fabian/repos/set-detect/android/app/src/main/java/com/fabianleven/setdetect/ui/components/ArrangementView.kt)
- Create a Composable that draws the abstracted card arrangement using the `CardsArrangement` data.

#### [NEW] [CameraArrangementScreen.kt](file:///home/fabian/repos/set-detect/android/app/src/main/java/com/fabianleven/setdetect/ui/CameraArrangementScreen.kt)
- Implement the new screen with the specified layout:
    - **Top**: Card count and set count.
    - **Middle**: `ArrangementView`.
    - **Bottom**: Buttons for "Edit", "Cancel", and "Reveal Sets".

#### [MODIFY] [CameraScanScreen.kt](file:///home/fabian/repos/set-detect/android/app/src/main/java/com/fabianleven/setdetect/ui/CameraScanScreen.kt)
- Update `onCardsDetected` callback to pass the full `Detection` object.
- Update UI to pass the `Detection` object when the capture button is clicked.

---

### Resources

#### [MODIFY] [strings.xml](file:///home/fabian/repos/set-detect/android/app/src/main/res/values/strings.xml)
- Add strings for the new screen (titles, button labels, plurals).

## Verification Plan

### Automated Tests
- Build the project to ensure JNI and Kotlin code are in sync.
- Run the app and perform a camera scan to verify the new screen appears and functions as expected.

### Manual Verification
- Verify that "Edit" takes the user to `ManualSelectionScreen` with the correct cards selected.
- Verify that "Cancel" takes the user back to the camera.
- Verify that "Reveal Sets" takes the user to the results screen.
- Check the visual layout of the `ArrangementView`.
