package com.fabianleven.setdetect.ui

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewmodel.CreationExtras
import com.fabianleven.setdetect.domain.Detection
import com.fabianleven.setdetect.domain.SetCard
import com.fabianleven.setdetect.domain.UserPreferencesRepository

const val BOARD_TAB_INDEX = 0
const val DECK_TAB_INDEX = 1

class SelectionViewModel(
    userPreferencesRepository: UserPreferencesRepository? = null
) : ViewModel() {

    val selectedCards = mutableStateListOf<SetCard>()

    // internal (not private) set: TutorialController lives in this same
    // module and drives this directly while scripting the tutorial (showing
    // its synthetic example scan, restoring the real one afterward, etc.).
    var scannedDetection by mutableStateOf<Detection?>(null)
        internal set

    var selectedTabIndex by mutableIntStateOf(BOARD_TAB_INDEX)

    // Owns the onboarding tutorial's own state and step choreography; see
    // TutorialController for why that isn't just inlined here.
    val tutorial = TutorialController(this, userPreferencesRepository)

    fun updateScannedDetection(detection: Detection) {
        scannedDetection = detection
        selectedTabIndex = BOARD_TAB_INDEX
        // A fresh scan replaces the previous selection entirely -- it doesn't
        // accumulate on top of whatever was selected before -- then
        // auto-selects everything the new scan detected.
        selectedCards.clear()
        val newCards = detection.detectedCards.mapNotNull { it.card }
        addCards(newCards)
    }

    fun toggleSelection(card: SetCard) {
        if (selectedCards.contains(card)) {
            selectedCards.remove(card)
        } else {
            selectedCards.add(card)
        }
    }

    fun addCards(cards: List<SetCard>) {
        cards.forEach { card ->
            if (!selectedCards.contains(card)) {
                selectedCards.add(card)
            }
        }
    }

    fun clearSelection() {
        selectedCards.clear()
        scannedDetection = null
    }

    companion object {
        val Factory: ViewModelProvider.Factory = object : ViewModelProvider.Factory {
            @Suppress("UNCHECKED_CAST")
            override fun <T : ViewModel> create(
                modelClass: Class<T>,
                extras: CreationExtras
            ): T {
                val application = extras[ViewModelProvider.AndroidViewModelFactory.APPLICATION_KEY]
                return SelectionViewModel(
                    application?.let { UserPreferencesRepository(it.applicationContext) }
                ) as T
            }
        }
    }
}
