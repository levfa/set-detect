#include <android/bitmap.h>
#include <android/log.h>
#include <exception>
#include <jni.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <setdetect/card_detection/detector.hpp>
#include <setdetect/card_tracking/detection_tracker.hpp>
#include <setdetect/set_game/game.hpp>
#include <string>

#define LOG_TAG "SetDetectJNI"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

using namespace setdetect;

namespace {

// Owns the sync detector plus a tracker wrapping it (tracker only holds a reference, so
// the detector must outlive it: keeping both together in one heap-allocated struct
// makes that lifetime trivially correct). Calling into the tracker stays fast per call:
// the ONNX inference itself runs on the tracker's own internal background thread, so
// this is safe to call once per analyzed camera frame without blocking the caller on
// inference latency.
struct NativeDetector {
    card_detection::CardDetector detector;
    card_tracking::DetectionTracker tracker;

    NativeDetector(const std::string& corner_weights, const std::string& class_weights)
        : detector(corner_weights, class_weights, /*device_id=*/-1), tracker(detector) {}
};

jobject color_to_jobject(JNIEnv* env, jclass color_class, set_game::Color c) {
    const char* name = nullptr;
    switch (c) {
    case set_game::Color::RED:
        name = "RED";
        break;
    case set_game::Color::GREEN:
        name = "GREEN";
        break;
    case set_game::Color::PURPLE:
        name = "PURPLE";
        break;
    default:
        return nullptr;
    }
    jfieldID id = env->GetStaticFieldID(color_class, name, "Lcom/fabianleven/setdetect/domain/CardColor;");
    return env->GetStaticObjectField(color_class, id);
}

jobject shape_to_jobject(JNIEnv* env, jclass shape_class, set_game::Shape s) {
    const char* name = nullptr;
    switch (s) {
    case set_game::Shape::DIAMOND:
        name = "DIAMOND";
        break;
    case set_game::Shape::SQUIGGLE:
        name = "SQUIGGLE";
        break;
    case set_game::Shape::OVAL:
        name = "OVAL";
        break;
    default:
        return nullptr;
    }
    jfieldID id = env->GetStaticFieldID(shape_class, name, "Lcom/fabianleven/setdetect/domain/CardShape;");
    return env->GetStaticObjectField(shape_class, id);
}

jobject count_to_jobject(JNIEnv* env, jclass number_class, set_game::Count c) {
    const char* name = nullptr;
    switch (c) {
    case set_game::Count::ONE:
        name = "ONE";
        break;
    case set_game::Count::TWO:
        name = "TWO";
        break;
    case set_game::Count::THREE:
        name = "THREE";
        break;
    default:
        return nullptr;
    }
    jfieldID id = env->GetStaticFieldID(number_class, name, "Lcom/fabianleven/setdetect/domain/CardNumber;");
    return env->GetStaticObjectField(number_class, id);
}

jobject fill_to_jobject(JNIEnv* env, jclass shading_class, set_game::Fill f) {
    const char* name = nullptr;
    switch (f) {
    case set_game::Fill::SOLID:
        name = "SOLID";
        break;
    case set_game::Fill::STRIPED:
        name = "STRIPED";
        break;
    case set_game::Fill::OPEN:
        name = "OPEN";
        break;
    default:
        return nullptr;
    }
    jfieldID id = env->GetStaticFieldID(shading_class, name, "Lcom/fabianleven/setdetect/domain/CardShading;");
    return env->GetStaticObjectField(shading_class, id);
}

} // namespace

extern "C" JNIEXPORT jlong JNICALL Java_com_fabianleven_setdetect_domain_NativeSetDetector_initNative(
    JNIEnv* env, jobject /* this */, jstring corner_model_path, jstring class_model_path) {

    const char* corner_path = env->GetStringUTFChars(corner_model_path, nullptr);
    const char* class_path = env->GetStringUTFChars(class_model_path, nullptr);

    jlong ptr = 0;
    try {
        auto* native = new NativeDetector(corner_path, class_path);
        ptr = reinterpret_cast<jlong>(native);
        LOGD("NativeDetector initialized successfully");
    } catch (const std::exception& e) {
        LOGE("Failed to initialize detector: %s", e.what());
    } catch (...) {
        LOGE("Failed to initialize detector: unknown error");
    }

    env->ReleaseStringUTFChars(corner_model_path, corner_path);
    env->ReleaseStringUTFChars(class_model_path, class_path);

    return ptr;
}

extern "C" JNIEXPORT void JNICALL Java_com_fabianleven_setdetect_domain_NativeSetDetector_closeNative(
    JNIEnv* /* env */, jobject /* this */, jlong native_ptr) {
    delete reinterpret_cast<NativeDetector*>(native_ptr);
}

extern "C" JNIEXPORT jobject JNICALL Java_com_fabianleven_setdetect_domain_NativeSetDetector_detectNative(
    JNIEnv* env, jobject /* this */, jlong native_ptr, jobject bitmap) {

    auto* native = reinterpret_cast<NativeDetector*>(native_ptr);
    if (native == nullptr) {
        return nullptr;
    }

    AndroidBitmapInfo info;
    void* pixels = nullptr;
    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0) {
        return nullptr;
    }
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        return nullptr;
    }
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) {
        return nullptr;
    }

    const cv::Mat rgba_mat(static_cast<int>(info.height), static_cast<int>(info.width), CV_8UC4, pixels, info.stride);
    cv::Mat img_rgb;
    cv::cvtColor(rgba_mat, img_rgb, cv::COLOR_RGBA2RGB);
    AndroidBitmap_unlockPixels(env, bitmap);

    const std::optional<card_detection::Detection> detection = native->tracker.detect(img_rgb);
    if (!detection.has_value()) {
        return nullptr;
    }

    jclass list_class = env->FindClass("java/util/ArrayList");
    jmethodID list_init = env->GetMethodID(list_class, "<init>", "()V");
    jmethodID list_add = env->GetMethodID(list_class, "add", "(Ljava/lang/Object;)Z");
    jobject det_cards_list = env->NewObject(list_class, list_init);

    jclass det_card_class = env->FindClass("com/fabianleven/setdetect/domain/DetectedCard");
    jmethodID det_card_init =
        env->GetMethodID(det_card_class, "<init>", "(Lcom/fabianleven/setdetect/domain/SetCard;Ljava/util/List;F)V");

    jclass set_card_class = env->FindClass("com/fabianleven/setdetect/domain/SetCard");
    jmethodID set_card_init = env->GetMethodID(
        set_card_class, "<init>",
        "(Lcom/fabianleven/setdetect/domain/CardColor;Lcom/fabianleven/setdetect/domain/CardShape;"
        "Lcom/fabianleven/setdetect/domain/CardNumber;Lcom/fabianleven/setdetect/domain/CardShading;)V");

    jclass color_class = env->FindClass("com/fabianleven/setdetect/domain/CardColor");
    jclass shape_class = env->FindClass("com/fabianleven/setdetect/domain/CardShape");
    jclass number_class = env->FindClass("com/fabianleven/setdetect/domain/CardNumber");
    jclass shading_class = env->FindClass("com/fabianleven/setdetect/domain/CardShading");

    jclass point_class = env->FindClass("com/fabianleven/setdetect/domain/PointF");
    jmethodID point_init = env->GetMethodID(point_class, "<init>", "(FF)V");

    for (const card_detection::DetectedCard& dc : detection->detected_cards) {
        jobject jcard = nullptr;
        if (const std::optional<set_game::Card> card = card_detection::match_card(dc)) {
            jobject jcolor = color_to_jobject(env, color_class, card->color);
            jobject jshape = shape_to_jobject(env, shape_class, card->shape);
            jobject jnumber = count_to_jobject(env, number_class, card->count);
            jobject jshading = fill_to_jobject(env, shading_class, card->fill);
            if (jcolor != nullptr && jshape != nullptr && jnumber != nullptr && jshading != nullptr) {
                jcard = env->NewObject(set_card_class, set_card_init, jcolor, jshape, jnumber, jshading);
            }
        }

        jobject jcorners = env->NewObject(list_class, list_init);
        for (const card_detection::Point2d& corner : dc.corners) {
            jobject jp =
                env->NewObject(point_class, point_init, static_cast<float>(corner.x), static_cast<float>(corner.y));
            env->CallBooleanMethod(jcorners, list_add, jp);
        }

        float conf = 0.0F;
        for (const double v : dc.corner_visibility) {
            conf += static_cast<float>(v);
        }
        conf /= 4.0F;

        jobject jdet = env->NewObject(det_card_class, det_card_init, jcard, jcorners, conf);
        env->CallBooleanMethod(det_cards_list, list_add, jdet);
    }

    const auto& arr = detection->matches_arrangement;
    jclass rect_class = env->FindClass("com/fabianleven/setdetect/domain/RotatedRect");
    jmethodID rect_init = env->GetMethodID(rect_class, "<init>", "(FFF)V");
    jobject poses_list = env->NewObject(list_class, list_init);
    for (const auto& pose : arr.card_poses) {
        jobject jpose = env->NewObject(rect_class, rect_init, static_cast<float>(pose.center_x),
                                       static_cast<float>(pose.center_y), static_cast<float>(pose.angle));
        env->CallBooleanMethod(poses_list, list_add, jpose);
    }

    jobject z_orders_list = env->NewObject(list_class, list_init);
    jclass integer_class = env->FindClass("java/lang/Integer");
    jmethodID integer_init = env->GetMethodID(integer_class, "<init>", "(I)V");
    for (const int z : arr.card_z_orders) {
        jobject jz = env->NewObject(integer_class, integer_init, z);
        env->CallBooleanMethod(z_orders_list, list_add, jz);
    }

    jclass arrangement_class = env->FindClass("com/fabianleven/setdetect/domain/CardsArrangement");
    jmethodID arrangement_init = env->GetMethodID(arrangement_class, "<init>", "(Ljava/util/List;Ljava/util/List;FF)V");
    jobject arrangement_obj = env->NewObject(arrangement_class, arrangement_init, poses_list, z_orders_list,
                                             static_cast<float>(arr.card_width), static_cast<float>(arr.card_height));

    jclass detection_class = env->FindClass("com/fabianleven/setdetect/domain/Detection");
    jmethodID detection_init = env->GetMethodID(
        detection_class, "<init>", "(Ljava/util/List;Lcom/fabianleven/setdetect/domain/CardsArrangement;)V");

    return env->NewObject(detection_class, detection_init, det_cards_list, arrangement_obj);
}
