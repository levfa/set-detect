package com.fabianleven.setdetect

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
        enableEdgeToEdge()
        setContent {
            SetDetectTheme {
                SetDetectNavGraph()
            }
        }
    }
}
