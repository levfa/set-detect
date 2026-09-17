import java.util.Properties

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.google.devtools.ksp)
    alias(libs.plugins.jetbrains.kotlin.plugin.serialization)
    alias(libs.plugins.aboutlibraries)
}

// OpenCV and Eigen are vendored/built from source (scripts/build_opencv_min.sh,
// a CMake FetchContent) rather than resolved as Gradle dependencies, so
// AboutLibraries can't auto-detect them; these manual entries keep their
// attribution showing up in the Licenses screen anyway. ONNX Runtime stays a
// normal Gradle dependency (see the `onnx` configuration below) and needs no
// manual entry.
aboutLibraries {
    collect {
        configPath = file("config")
    }
}

android {
    namespace = "com.fabianleven.setdetect"
    compileSdk {
        version = release(37)
    }

    defaultConfig {
        applicationId = "com.fabianleven.setdetect"
        minSdk = 24
        targetSdk = 37
        // Overridden by the release workflow from the pushed tag; local/debug builds
        // fall back to these.
        versionCode = System.getenv("SETDETECT_VERSION_CODE")?.toIntOrNull() ?: 1
        versionName = System.getenv("SETDETECT_VERSION_NAME") ?: "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        externalNativeBuild {
            cmake {
                cppFlags("-std=c++20")
                val nativeSdkDir = layout.buildDirectory.get().asFile.absolutePath + "/native-sdk"
                val openCvMinDir = layout.buildDirectory.get().asFile.absolutePath + "/native-sdk-opencv"
                arguments("-DANDROID_STL=c++_shared", "-DNATIVE_SDK_DIR=$nativeSdkDir", "-DOPENCV_MIN_DIR=$openCvMinDir")
            }
        }
        ndk {
            abiFilters.add("arm64-v8a")
        }
    }

    signingConfigs {
        // Real signing key, provided via env vars by the release workflow (never
        // committed); falls back to the debug keystore for local/debug builds and the
        // existing CI test job, neither of which set these.
        create("release") {
            val releaseKeystore = System.getenv("SETDETECT_RELEASE_KEYSTORE")
            if (releaseKeystore != null) {
                storeFile = file(releaseKeystore)
                storePassword = System.getenv("SETDETECT_RELEASE_KEYSTORE_PASSWORD")
                keyAlias = System.getenv("SETDETECT_RELEASE_KEY_ALIAS")
                keyPassword = System.getenv("SETDETECT_RELEASE_KEYSTORE_PASSWORD") // PKCS12: one password
            } else {
                storeFile = File(System.getProperty("user.home"), ".android/debug.keystore")
                storePassword = "android"
                keyAlias = "androiddebugkey"
                keyPassword = "android"
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
            signingConfig = signingConfigs.getByName("release")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    buildFeatures {
        compose = true
        prefab = true
    }
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
    packaging {
        jniLibs {
            // libc++_shared.so is provided by both the ONNX Runtime AAR and
            // our own native build (via ANDROID_STL=c++_shared); pick one
            // rather than failing on the duplicate.
            pickFirsts += setOf("**/libc++_shared.so")
        }
    }
}

val onnxConfig = configurations.create("onnx")

dependencies {
    "onnx"(libs.onnxruntime)
}

val extractNativeLibs = tasks.register<Copy>("extractNativeLibs") {
    duplicatesStrategy = DuplicatesStrategy.EXCLUDE
    from(onnxConfig.map { zipTree(it) }) {
        into("onnx")
    }
    into(layout.buildDirectory.dir("native-sdk"))
}

// Resolves the NDK directory the same way AGP does when ndkVersion isn't
// pinned in this build (it isn't): the highest-versioned dir under
// <sdk.dir>/ndk. Read directly from local.properties rather than through
// AGP's variant APIs, which aren't reliably available at task-registration
// time in Kotlin DSL.
fun resolveNdkDir(): File {
    val localProps = Properties().apply {
        rootProject.file("local.properties").takeIf { it.exists() }?.inputStream()?.use { load(it) }
    }
    val sdkDir = File(localProps.getProperty("sdk.dir") ?: (System.getenv("ANDROID_HOME") ?: error("Neither sdk.dir (local.properties) nor ANDROID_HOME is set")))
    val ndkRoot = File(sdkDir, "ndk")
    val versions = ndkRoot.listFiles { f -> f.isDirectory } ?: error("No NDK found under $ndkRoot -- install one via the SDK Manager")
    return versions.maxByOrNull { it.name } ?: error("No NDK found under $ndkRoot")
}

// Builds a minimal, size-scoped static OpenCV (core+imgproc+video+features:
// the only calls corners_st.cpp/opt_flow_pyr_lk.cpp make) for each ABI
// this app ships, via scripts/build_opencv_min.sh. Declaring outputs.dir lets
// Gradle skip re-running once already built for a given ABI/platform; the
// script itself also stamp-checks OpenCV's own version so a change to it (or
// to the script's build recipe, inputs.file below) forces a rebuild.
val buildMinimalOpenCv = tasks.register<Exec>("buildMinimalOpenCv") {
    val abi = "arm64-v8a" // keep in sync with defaultConfig.ndk.abiFilters
    val platform = "android-24" // keep in sync with defaultConfig.minSdk
    val outDir = layout.buildDirectory.dir("native-sdk-opencv/$abi").get().asFile
    val workDir = layout.buildDirectory.dir("opencv-min-work").get().asFile
    val script = file("../scripts/build_opencv_min.sh")

    inputs.file(script)
    outputs.dir(outDir)
    workDir.mkdirs()

    commandLine("bash", script.absolutePath, abi, platform, resolveNdkDir().absolutePath, outDir.absolutePath, workDir.absolutePath)
}

tasks.matching { it.name.contains("externalNativeBuild", ignoreCase = true) || it.name.contains("generateJsonModel", ignoreCase = true) || it.name.contains("CMake", ignoreCase = true) }.configureEach {
    dependsOn(extractNativeLibs, buildMinimalOpenCv)
}

dependencies {
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.accompanist.permissions)
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.appcompat)
    implementation(libs.androidx.camera.camera2)
    implementation(libs.androidx.camera.core)
    implementation(libs.androidx.camera.lifecycle)
    implementation(libs.androidx.camera.view)
    implementation(libs.androidx.compose.material.icons.core)
    implementation(libs.androidx.compose.material.icons.extended)
    implementation(libs.androidx.compose.material3)
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.datastore.preferences)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.androidx.lifecycle.viewmodel.navigation3)
    implementation(libs.androidx.navigation3.runtime)
    implementation(libs.androidx.navigation3.ui)
    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.kotlinx.coroutines.core)
    implementation(libs.kotlinx.serialization.core)
    implementation(libs.onnxruntime)
    implementation(libs.aboutlibraries.compose.m3)
    testImplementation(libs.androidx.core)
    testImplementation(libs.androidx.junit)
    testImplementation(libs.junit)
    testImplementation(libs.kotlinx.coroutines.test)
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.compose.ui.test.junit4)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.runner)
    debugImplementation(libs.androidx.compose.ui.test.manifest)
    debugImplementation(libs.androidx.compose.ui.tooling)
}