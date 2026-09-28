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
        val SHOW_TUTORIAL_HINT = booleanPreferencesKey("show_tutorial_hint")
    }

    // Defaults to true: a brief hint toward the Help icon is shown on every
    // launch until the user turns this off in Settings. It's a standing
    // preference, not a one-shot "have I shown this before" flag. The
    // tutorial itself never auto-opens; this only gates the hint.
    val showTutorialHint: Flow<Boolean> = context.dataStore.data.map { preferences ->
        preferences[PreferencesKeys.SHOW_TUTORIAL_HINT] ?: true
    }

    suspend fun setShowTutorialHint(show: Boolean) {
        context.dataStore.edit { preferences ->
            preferences[PreferencesKeys.SHOW_TUTORIAL_HINT] = show
        }
    }
}
