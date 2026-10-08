package com.fabianleven.setdetect

import android.content.pm.ActivityInfo
import android.os.Bundle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import com.fabianleven.setdetect.navigation.SetDetectNavGraph
import com.fabianleven.setdetect.ui.theme.SetDetectTheme

// Extends AppCompatActivity (not just ComponentActivity) because the per-app
// language picker, AppCompatDelegate.setApplicationLocales(), silently fails
// to apply with Compose otherwise.
class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Portrait on phones only: large screens (tablets, foldables) must stay
        // rotatable, and Android 16 ignores orientation locks there anyway.
        if (resources.configuration.smallestScreenWidthDp < 600) {
            requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        }
        enableEdgeToEdge()
        setContent {
            SetDetectTheme {
                SetDetectNavGraph()
            }
        }
    }
}
