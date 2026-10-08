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
// attribution showing up in the Licenses screen anyway.
aboutLibraries {
    collect {
        configPath = file("config")
    }
}

// Shared by AGP's own CMake build and buildMinimalOpenCv: both must link against
// the same libc++.
// Keep in sync with the "ndk;..." install in .github/workflows/android-*.yml.
val pinnedNdkVersion = "28.2.13676358"

android {
    namespace = "com.fabianleven.setdetect"
    ndkVersion = pinnedNdkVersion
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

// Resolves <sdk.dir>/ndk/<pinnedNdkVersion>, the NDK AGP uses for the app. Read
// directly from local.properties.
fun resolveNdkDir(): File {
    val localProps = Properties().apply {
        rootProject.file("local.properties").takeIf { it.exists() }?.inputStream()?.use { load(it) }
    }
    val sdkDir = File(localProps.getProperty("sdk.dir") ?: (System.getenv("ANDROID_HOME") ?: error("Neither sdk.dir (local.properties) nor ANDROID_HOME is set")))
    val ndkDir = File(sdkDir, "ndk/$pinnedNdkVersion")
    if (!ndkDir.isDirectory) error("NDK $pinnedNdkVersion not found at $ndkDir -- install it via the SDK Manager")
    return ndkDir
}

// Builds a minimal, size-scoped static OpenCV.
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