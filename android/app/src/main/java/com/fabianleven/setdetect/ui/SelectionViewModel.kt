package com.fabianleven.setdetect.ui

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewmodel.CreationExtras
import com.fabianleven.setdetect.domain.Detection
import com.fabianleven.setdetect.domain.SetCard
import com.fabianleven.setdetect.domain.SlotChoice
import com.fabianleven.setdetect.domain.UserPreferencesRepository
import com.fabianleven.setdetect.domain.findDuplicateCards
import com.fabianleven.setdetect.domain.resolveBoard
import com.fabianleven.setdetect.domain.toggleBoardCard
import kotlinx.coroutines.flow.first

class SelectionViewModel(
    private val userPreferencesRepository: UserPreferencesRepository? = null
) : ViewModel() {

    // Cards the user added by hand, independent of the scan.
    val manualCards = mutableStateListOf<SetCard>()

    // Per scanned card (indexed like matchedCards); absent means Original.
    val slotChoices = mutableStateMapOf<Int, SlotChoice>()

    // Height of the manual-selection area, split from the scan arrangement by
    // a draggable divider. Lives here rather than as local Compose state in
    // BoardContent so the tutorial can reset it to a demo-friendly value
    // while playing and restore the user's chosen height afterward.
    var manualHeightDp by mutableFloatStateOf(DefaultManualHeight.value)

    private var detectionState by mutableStateOf<Detection?>(null)

    // The setter is internal so the tutorial controller can swap in its example
    // scan and restore the real one afterward.
    var scannedDetection: Detection?
        get() = detectionState
        internal set(value) {
            detectionState = value
            slotChoices.clear()
        }

    // Arrangement poses are indexed by matched card, so cards that failed
    // classification are dropped while preserving order.
    val matchedCards: List<SetCard>
        get() = detectionState?.detectedCards?.mapNotNull { it.card } ?: emptyList()

    val scanChoices: List<SlotChoice>
        get() = List(matchedCards.size) { slotChoices[it] ?: SlotChoice.Original }

    val boardCards: List<SetCard>
        get() = resolveBoard(matchedCards, scanChoices, manualCards)

    val duplicateCards: Set<SetCard>
        get() = findDuplicateCards(boardCards)

    // Owns the onboarding tutorial's state and step choreography.
    val tutorial = TutorialController(this)

    private var tutorialHintConsumed = false

    // Reads the "show tutorial hint" preference at most once per launch;
    // returns false on every call after the first, so navigating away (e.g.
    // to the camera screen) and back doesn't re-trigger the hint.
    suspend fun consumeTutorialHint(): Boolean {
        if (tutorialHintConsumed) return false
        tutorialHintConsumed = true
        return userPreferencesRepository?.showTutorialHint?.first() ?: true
    }

    fun updateScannedDetection(detection: Detection) {
        scannedDetection = detection
    }

    fun setSlotChoice(index: Int, choice: SlotChoice) {
        slotChoices[index] = choice
    }

    fun toggleCard(card: SetCard) {
        val edit = toggleBoardCard(matchedCards, scanChoices, manualCards, card)
        edit.choices.forEachIndexed { index, choice -> slotChoices[index] = choice }
        manualCards.clear()
        manualCards.addAll(edit.manualCards)
    }

    fun clearManual() {
        manualCards.clear()
    }

    fun clearScan() {
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
