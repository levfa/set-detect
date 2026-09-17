package com.fabianleven.setdetect.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.CreationExtras
import com.fabianleven.setdetect.domain.UserPreferencesRepository
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

// Language isn't handled here: AppCompatDelegate is already the single
// source of truth for the chosen per-app locale (it self-persists), so
// SettingsScreen reads/writes it directly rather than routing it through
// this ViewModel and a second store.
class SettingsViewModel(
    private val userPreferencesRepository: UserPreferencesRepository
) : ViewModel() {

    val showTutorialOnStartup: StateFlow<Boolean> = userPreferencesRepository.showTutorialOnStartup
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), true)

    fun setShowTutorialOnStartup(show: Boolean) {
        viewModelScope.launch {
            userPreferencesRepository.setShowTutorialOnStartup(show)
        }
    }

    companion object {
        val Factory: ViewModelProvider.Factory = object : ViewModelProvider.Factory {
            @Suppress("UNCHECKED_CAST")
            override fun <T : ViewModel> create(
                modelClass: Class<T>,
                extras: CreationExtras
            ): T {
                val application = extras[ViewModelProvider.AndroidViewModelFactory.APPLICATION_KEY]
                    ?: error("SettingsViewModel requires an application context")
                return SettingsViewModel(UserPreferencesRepository(application.applicationContext)) as T
            }
        }
    }
}
