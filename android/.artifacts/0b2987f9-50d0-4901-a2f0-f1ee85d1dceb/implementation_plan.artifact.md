# Implementation Plan - Set Detect UI Refinement

Refine the UI for the 'Set Detect' app using Material 3 Adaptive components, finalize styling, and ensure edge-to-edge support.

## User Review Required

> [!IMPORTANT]
> I will be updating the navigation structure to use `ListDetailSceneStrategy`. This will change how the results are displayed on larger screens (tablets/foldables), where the list of found sets and the details of a selected set will be shown side-by-side.

## Proposed Changes

### Theme & Assets

#### [MODIFY] [colors.xml](file:///home/fabian/repos/set-detect/android/app/src/main/res/values/colors.xml)
- Update with Material 3 compliant colors. (Already partially done, will finalize).

#### [MODIFY] [Theme.kt](file:///home/fabian/repos/set-detect/android/app/src/main/java/com/fabianleven/setdetect/ui/theme/Theme.kt)
- Ensure dynamic color is correctly handled and fallback palette is expressive.

### Components & Polish

#### [MODIFY] [CardView.kt](file:///home/fabian/repos/set-detect/android/app/src/main/java/com/fabianleven/setdetect/ui/components/CardView.kt)
- Refine the `CardView` rendering for a more premium look.
- Improve the `getSquigglePath` for a smoother wavy shape.
- Add subtle border and shadow to cards.

### Navigation & Adaptive UI

#### [MODIFY] [Routes.kt](file:///home/fabian/repos/set-detect/android/app/src/main/java/com/fabianleven/setdetect/navigation/Routes.kt)
- Add a new route for set details if needed, or adjust existing ones to fit the `ListDetailSceneStrategy`.

#### [MODIFY] [SetDetectNavGraph.kt](file:///home/fabian/repos/set-detect/android/app/src/main/java/com/fabianleven/setdetect/navigation/SetDetectNavGraph.kt)
- Implement `ListDetailSceneStrategy` for the `Results` flow.
- Ensure the backstack management works correctly with the adaptive layout.

#### [MODIFY] [ResultsDashboardScreen.kt](file:///home/fabian/repos/set-detect/android/app/src/main/java/com/fabianleven/setdetect/ui/ResultsDashboardScreen.kt)
- Split into `ResultsListScreen` and `SetDetailScreen` to fit the adaptive layout.

### Edge-to-Edge

#### [MODIFY] [MainActivity.kt](file:///home/fabian/repos/set-detect/android/app/src/main/java/com/fabianleven/setdetect/MainActivity.kt)
- Verify `enableEdgeToEdge()` and handle system bar legibility.

## Verification Plan

### Automated Tests
- Run existing unit tests: `./gradlew :app:testDebugUnitTest`
- Verify build: `./gradlew :app:assembleDebug`

### Manual Verification
- Verify the adaptive layout on different emulator form factors (Phone, Tablet).
- Check edge-to-edge rendering and system bar transparency.
- Confirm the new app icon is correctly integrated.
