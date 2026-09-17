# Add project specific ProGuard rules here.
# By default, the flags in this file are appended to flags specified
# in /snap/android-studio/current/plugins/android/resources/proguard-android.txt
# You can edit the include path and order by changing the proguardFiles
# directive in build.gradle.kts.
#
# For more details, see
#   http://developer.android.com/guide/developing/tools/proguard.html

# Add any custom keep rules here that are specific to your project.

# Keep Room classes
-keep class androidx.room.RoomDatabase { *; }
-keep class * extends androidx.room.RoomDatabase
-keep class * extends androidx.room.Dao
-keep class * extends androidx.room.Entity

# Keep Moshi classes
-keep class com.squareup.moshi.** { *; }
-keep interface com.squareup.moshi.** { *; }
-keep @com.squareup.moshi.JsonQualifier interface *
-keepclassmembers class * {
    @com.squareup.moshi.FromJson *;
    @com.squareup.moshi.ToJson *;
}

# Keep native methods and classes used by JNI
-keepclasseswithmembernames class * {
    native <methods>;
}

# Keep specific domain classes if they are serialized or used by JNI
-keep class com.fabianleven.setdetect.domain.** { *; }
