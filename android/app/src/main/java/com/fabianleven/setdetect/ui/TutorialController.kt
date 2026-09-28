package com.fabianleven.setdetect.ui

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.viewModelScope
import com.fabianleven.setdetect.domain.Detection
import com.fabianleven.setdetect.domain.SetCard
import com.fabianleven.setdetect.domain.SlotChoice
import com.fabianleven.setdetect.domain.createExampleDetection
import com.fabianleven.setdetect.ui.components.TutorialStep
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

// Owns the onboarding tutorial's state and step choreography, operating on the
// host ViewModel's selection and scan state.
class TutorialController(
    private val viewModel: SelectionViewModel
) {
    var active by mutableStateOf(false)
        private set

    var currentStep by mutableStateOf(TutorialStep.INTRO)
        private set

    var isDemoPlaying by mutableStateOf(false)
        private set

    // The scanned-card slot the Scan Edit demo is currently editing, if any.
    // The arrangement reports that card's on-screen position so the tutorial
    // can draw a tap effect on it. Null outside that demo.
    var demoSlot by mutableStateOf<Int?>(null)
        private set

    // Bumped each time the Scan Edit demo needs a fresh tap-ripple on the demo
    // slot. Acts as an animation restart key, which a boolean can't do.
    var demoCardTapTrigger by mutableIntStateOf(0)
        private set

    private var savedSelection: List<SetCard> = emptyList()
    private var savedChoices: Map<Int, SlotChoice> = emptyMap()
    private var savedDetection: Detection? = null
    private var savedManualHeightDp: Float = DefaultManualHeight.value

    private var animationJob: Job? = null

    fun start() {
        animationJob?.cancel()
        isDemoPlaying = false
        savedSelection = viewModel.manualCards.toList()
        savedChoices = viewModel.slotChoices.toMap()
        savedDetection = viewModel.scannedDetection
        savedManualHeightDp = viewModel.manualHeightDp
        // Start with no scan; the example scan appears once the tutorial
        // explains scanning.
        viewModel.manualCards.clear()
        viewModel.scannedDetection = null
        // Reset the divider to a demo-friendly split so spotlighted steps
        // aren't cramped by whatever the user last dragged it to.
        viewModel.manualHeightDp = DefaultManualHeight.value
        currentStep = TutorialStep.INTRO
        active = true
    }

    fun next() {
        val steps = TutorialStep.entries
        val nextIndex = currentStep.ordinal + 1

        if (currentStep == TutorialStep.SCAN_EDIT && !isDemoPlaying) {
            playScanResultDemo()
            return
        }

        if (nextIndex < steps.size) {
            currentStep = steps[nextIndex]
            onStepEntered(currentStep)
        } else {
            dismiss()
        }
    }

    fun previous() {
        val steps = TutorialStep.entries
        val prevIndex = currentStep.ordinal - 1

        // Entering a step re-derives its scan and selection state.
        if (prevIndex >= 0) {
            currentStep = steps[prevIndex]
            onStepEntered(currentStep)
        }
    }

    fun skip() {
        animationJob?.cancel()
        isDemoPlaying = false
        demoSlot = null
        dismiss()
    }

    fun dismiss() {
        active = false
        viewModel.manualCards.clear()
        viewModel.scannedDetection = savedDetection
        viewModel.slotChoices.putAll(savedChoices)
        viewModel.manualCards.addAll(savedSelection)
        viewModel.manualHeightDp = savedManualHeightDp
    }

    // Populates the board with a synthetic example scan. The manual-selection
    // area is left empty so the tutorial can explain it from scratch.
    private fun setExampleScan() {
        val example = createExampleDetection()
        viewModel.scannedDetection = example
        viewModel.manualCards.clear()
    }

    private fun onStepEntered(step: TutorialStep) {
        animationJob?.cancel()
        isDemoPlaying = false
        when (step) {
            TutorialStep.SCAN_CARDS -> {
                viewModel.manualCards.clear()
                viewModel.scannedDetection = null
            }
            TutorialStep.SCAN_APPEARS,
            TutorialStep.SETS_AND_REVEAL,
            TutorialStep.SCAN_EDIT,
            TutorialStep.MANUAL_SELECTION -> setExampleScan()
            else -> {}
        }
    }

    // Demonstrates ignoring a scanned card, since the tutorial overlay
    // blocks real taps on the cards underneath. A tap-ripple plays on the
    // card immediately before each change, so it reads as caused by a tap.
    private fun playScanResultDemo() {
        val slot = if (viewModel.matchedCards.isEmpty()) null else 0
        demoSlot = slot

        animationJob = viewModel.viewModelScope.launch {
            isDemoPlaying = true

            delay(400)
            demoCardTapTrigger++
            delay(300)
            slot?.let { viewModel.setSlotChoice(it, SlotChoice.Excluded) }
            delay(1300)
            demoCardTapTrigger++
            delay(300)
            slot?.let { viewModel.setSlotChoice(it, SlotChoice.Original) }

            // Lets the restored card register before the demo ends.
            delay(900)
            isDemoPlaying = false
            demoSlot = null

            if (currentStep == TutorialStep.SCAN_EDIT) {
                val steps = TutorialStep.entries
                val nextIndex = currentStep.ordinal + 1
                if (nextIndex < steps.size) {
                    currentStep = steps[nextIndex]
                    onStepEntered(currentStep)
                }
            }
        }
    }
}
