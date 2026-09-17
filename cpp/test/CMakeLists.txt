find_package(GTest CONFIG REQUIRED)

add_executable(setdetect_tests
    apps/utils.cpp
    card_detection/detector.cpp
    card_tracking/detection_tracker.cpp
    img_proc/corners_st.cpp
    img_proc/opt_flow_pyr_lk.cpp
    img_proc/sparse_propagator.cpp
    interpolation/tps_rbf.cpp
    set_game/game.cpp
    util/string.cpp
)
target_link_libraries(setdetect_tests
    PRIVATE
        setdetect
        GTest::gtest_main
)
# apps/path_util.hpp and apps/stats.hpp are app-private headers (not under the
# public setdetect include dir); test/apps/utils.cpp needs them directly.
target_include_directories(setdetect_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
# setdetect_tests only links `setdetect` (a static lib) directly; OpenCV comes in
# transitively through its PUBLIC ${OpenCV_LIBS} (plain absolute .so paths, not an
# IMPORTED target). Whether CMake's automatic build-tree RPATH computation picks up
# such transitively-inherited raw paths for the final executable isn't consistent
# across CMake versions (observed missing on CMake 4.x, present on 3.28); set it
# explicitly instead of relying on that, mirroring the apps' own BUILD_RPATH below.
set_target_properties(setdetect_tests PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib;${OpenCV_ROOT_PATH}/lib"
)

include(GoogleTest)
gtest_discover_tests(setdetect_tests)

# Real-model tests: opt-in (network + slower), kept out of the default `setdetect_tests`
# binary/target so a plain build+ctest never downloads or runs the actual ONNX models.
if(SETDETECT_BUILD_INTEGRATION_TESTS)
    add_executable(setdetect_integration_tests
        card_detection/detector_real_models.cpp
    )
    target_link_libraries(setdetect_integration_tests
        PRIVATE
            setdetect
            GTest::gtest_main
    )
    set_target_properties(setdetect_integration_tests PROPERTIES
        BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib;${OpenCV_ROOT_PATH}/lib"
    )

    # Runs before the real-model tests, downloading them into SETDETECT_MODELS_ROOT
    # (default ~/.cache/setdetect/models) if not already cached there.
    add_test(NAME DownloadRealModels COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/download_models.sh)
    set_tests_properties(DownloadRealModels PROPERTIES FIXTURES_SETUP RealModels)

    gtest_discover_tests(setdetect_integration_tests
        PROPERTIES FIXTURES_REQUIRED RealModels
    )
endif()

# CLI smoke tests: plain CTest entries invoking the already-built app binaries
# directly, no GTest needed. Only meaningful when the apps are built.
if(SETDETECT_BUILD_APPS)
    foreach(app_target IN ITEMS card_detection card_tracking corners_st opt_flow_pyr_lk sparse_propagator tps_rbf
                                 frame_grabber)
        string(REPLACE "_" "-" app_name ${app_target})
        add_test(NAME ${app_name}-help COMMAND ${app_target} --help)
    endforeach()

    add_test(NAME card-detection-show-boards-missing-weights COMMAND card_detection show-boards --data-root .)
    set_tests_properties(card-detection-show-boards-missing-weights PROPERTIES WILL_FAIL TRUE)
endif()
