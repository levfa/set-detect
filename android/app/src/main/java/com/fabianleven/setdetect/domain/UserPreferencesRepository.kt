package com.fabianleven.setdetect.domain

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

val Context.dataStore: DataStore<Preferences> by preferencesDataStore(name = "user_preferences")

class UserPreferencesRepository(private val context: Context) {
    private object PreferencesKeys {
        val SHOW_TUTORIAL_ON_STARTUP = booleanPreferencesKey("show_tutorial_on_startup")
    }

    // Defaults to true: the tutorial auto-starts on every launch until the
    // user turns this off, either in Settings or via the checkbox on the
    // tutorial's own last step. It's a standing preference, not a one-shot
    // "have I shown this before" flag.
    val showTutorialOnStartup: Flow<Boolean> = context.dataStore.data.map { preferences ->
        preferences[PreferencesKeys.SHOW_TUTORIAL_ON_STARTUP] ?: true
    }

    suspend fun setShowTutorialOnStartup(show: Boolean) {
        context.dataStore.edit { preferences ->
            preferences[PreferencesKeys.SHOW_TUTORIAL_ON_STARTUP] = show
        }
    }
}
