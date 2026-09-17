package com.fabianleven.setdetect.ui

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.viewModelScope
import com.fabianleven.setdetect.domain.Detection
import com.fabianleven.setdetect.domain.SetCard
import com.fabianleven.setdetect.domain.UserPreferencesRepository
import com.fabianleven.setdetect.domain.createExampleDetection
import com.fabianleven.setdetect.ui.components.TutorialStep
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch

// Owns all onboarding-tutorial state and choreography for SelectionScreen,
// operating on the host ViewModel's selection/scan state (SelectionViewModel
// keeps `scannedDetection`'s setter `internal` specifically so this class can
// drive it). Kept as a separate object -- rather than folded directly into
// SelectionViewModel -- so the tutorial's fairly involved step-by-step
// scripting doesn't drown out the actual selection/scanning feature it's
// layered on top of.
class TutorialController(
    private val viewModel: SelectionViewModel,
    private val userPreferencesRepository: UserPreferencesRepository?
) {
    var active by mutableStateOf(false)
        private set

    var currentStep by mutableStateOf(TutorialStep.INTRO)
        private set

    var isDemoPlaying by mutableStateOf(false)
        private set

    // Non-null while a scripted tab switch is in progress -- the index of the
    // tab being "tapped". SelectionScreen hides the tutorial overlay
    // entirely and shows a tap ripple at that tab's position while this is
    // set (see simulateTabTap), so the switch reads as an actual interaction
    // instead of happening invisibly behind the overlay.
    var simulatedTapTarget by mutableStateOf<Int?>(null)
        private set

    // The card the Scan Edit demo (see playScanResultDemo) is currently
    // toggling, if any -- ArrangementView reports that specific card's
    // on-screen position back via TutorialTargets so the tutorial can draw a
    // tap effect exactly on it. Null outside that demo.
    var demoCard by mutableStateOf<SetCard?>(null)
        private set

    // Bumped each time playScanResultDemo wants a fresh tap-ripple drawn on
    // demoCard's position -- a plain "is a ripple showing" boolean can't
    // replay the same animation twice, so this acts as a restart key instead.
    var demoCardTapTrigger by mutableIntStateOf(0)
        private set

    // Set by SelectionScreen so this controller can drive the one demo
    // animation it has no direct access to: scrolling the Deck tab's grid
    // needs LazyGridState, which only exists in Compose. Performs the scroll
    // and returns whichever cards actually ended up centered on screen
    // afterward (computed from real layout there, not a hardcoded guess).
    var onScrollRequest: (suspend () -> List<SetCard>)? = null

    // Backs the "Show tutorial on startup" checkbox on the last step; seeded
    // from the persisted preference in start() and written straight through
    // on toggle (see setShowOnStartup), so it and the Settings screen's own
    // switch always agree since both read/write the same DataStore value.
    var showOnStartup by mutableStateOf(true)
        private set

    private var savedSelection: List<SetCard> = emptyList()
    private var savedDetection: Detection? = null
    private var savedTabIndex: Int = BOARD_TAB_INDEX

    private var animationJob: Job? = null

    // Cards the manual-selection demo actually added, so onStepEntered can
    // precisely undo them on a later re-entry and skip() can add them even if
    // skipped mid-animation. Populated by autoSelectSampleCards(); there's no
    // way to know this in advance since it depends on wherever the
    // two-stroke scroll (see SelectionScreen's onScrollRequest) actually
    // lands on the real device.
    private var demoAddedCards: List<SetCard> = emptyList()

    init {
        viewModel.viewModelScope.launch {
            // A standing preference, not a one-shot flag: this re-checks (and
            // may re-start the tutorial) on every launch, not just the first.
            if (userPreferencesRepository?.showTutorialOnStartup?.first() ?: true) {
                start()
            }
        }
    }

    fun start() {
        animationJob?.cancel()
        isDemoPlaying = false
        savedSelection = viewModel.selectedCards.toList()
        savedDetection = viewModel.scannedDetection
        savedTabIndex = viewModel.selectedTabIndex
        // Start from a clean "no scan yet" state -- the example scan is only
        // introduced once the tutorial actually explains scanning (see
        // onStepEntered), matching the real flow of tap-camera-then-see-results
        // instead of showing results before that's explained.
        viewModel.selectedCards.clear()
        viewModel.scannedDetection = null
        currentStep = TutorialStep.INTRO
        active = true
        viewModel.viewModelScope.launch {
            showOnStartup = userPreferencesRepository?.showTutorialOnStartup?.first() ?: true
        }
    }

    // Backs both the last tutorial step's checkbox and (indirectly, since
    // they share the same DataStore value) the Settings screen's switch.
    // Named "update", not "setShowOnStartup", to avoid clashing with the
    // showOnStartup property's own JVM-synthesized setter.
    fun updateShowOnStartup(enabled: Boolean) {
        showOnStartup = enabled
        viewModel.viewModelScope.launch {
            userPreferencesRepository?.setShowTutorialOnStartup(enabled)
        }
    }

    fun next() {
        val steps = TutorialStep.entries
        val nextIndex = currentStep.ordinal + 1

        if (currentStep == TutorialStep.MANUAL_SELECTION_INTRO && simulatedTapTarget == null) {
            switchToDeckTabDemo()
            return
        }

        if (currentStep == TutorialStep.MANUAL_SELECTION && !isDemoPlaying) {
            autoSelectSampleCards()
            return
        }

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

        // onStepEntered() re-derives the right scan/selection state for
        // whichever step we land back on (e.g. re-populating the example scan
        // when returning to a Board-tab step from MANUAL_SELECTION).
        if (prevIndex >= 0) {
            currentStep = steps[prevIndex]
            onStepEntered(currentStep)
        }
    }

    fun skip() {
        if (currentStep == TutorialStep.MANUAL_SELECTION) {
            animationJob?.cancel()
            // If the demo animation never ran, there's nothing to fall back
            // to -- we have no way to know what would've been centered
            // without actually running the scroll -- so this is a no-op then.
            viewModel.addCards(demoAddedCards)
            // Skipping bypasses the scripted tap back to Board, so switch
            // directly instead of leaving the user stranded on Deck.
            viewModel.selectedTabIndex = BOARD_TAB_INDEX
        }
        simulatedTapTarget = null
        currentStep = TutorialStep.RE_RUN_TUTORIAL
        onStepEntered(currentStep)
    }

    fun dismiss() {
        active = false
        viewModel.selectedCards.clear()
        viewModel.scannedDetection = savedDetection
        viewModel.addCards(savedSelection)
        viewModel.selectedTabIndex = savedTabIndex
        // Deliberately doesn't touch showTutorialOnStartup -- finishing (or
        // skipping) the tutorial once shouldn't silently turn off future
        // auto-starts; only the explicit checkbox/Settings switch should.
    }

    // Populates the Board tab with a synthetic example scan so its tutorial
    // steps have real content to point at without a camera capture. All
    // matched cards start selected -- exactly like a real scan, which
    // auto-selects everything it finds. The Manual Selection row is
    // deliberately left empty here -- it only gets populated once the
    // manual-selection demo actually adds cards from the Deck tab (see
    // autoSelectSampleCards), so the tutorial explains that row while it's
    // empty and then demonstrates filling it, instead of pre-filling it
    // before that's ever explained. The grayed-out "excluded" look is
    // demonstrated separately, by playScanResultDemo() toggling a card live.
    private fun setExampleScan() {
        val example = createExampleDetection()
        viewModel.scannedDetection = example
        viewModel.selectedCards.clear()
        viewModel.addCards(example.detectedCards.mapNotNull { it.card })
    }

    // Switches tabs as a scripted "tap" rather than an invisible state flip:
    // hides the tutorial overlay (see simulatedTapTarget) so the tab row is
    // fully visible, shows a tap ripple there, then performs the actual
    // switch and pauses (settleDelayMs) before the overlay is allowed to
    // return -- longer after landing back on the Board, so there's time to
    // actually take in the result before the next step's text covers it.
    private suspend fun simulateTabTap(targetTab: Int, settleDelayMs: Long = 500) {
        simulatedTapTarget = targetTab
        delay(600)
        viewModel.selectedTabIndex = targetTab
        delay(settleDelayMs)
        simulatedTapTarget = null
    }

    private fun onStepEntered(step: TutorialStep) {
        animationJob?.cancel()
        isDemoPlaying = false
        step.requiredTab?.let { viewModel.selectedTabIndex = it }
        when (step) {
            TutorialStep.SCAN_CARDS -> {
                viewModel.selectedCards.clear()
                viewModel.scannedDetection = null
            }
            TutorialStep.SCAN_APPEARS,
            TutorialStep.SETS_AND_REVEAL,
            TutorialStep.SCAN_EDIT,
            TutorialStep.MANUAL_SELECTION_INTRO -> setExampleScan()
            // Undo whatever the demo added on a previous visit to this step,
            // so re-entering it (Back, then Next again) always starts from
            // the same baseline instead of finding the demo cards already
            // selected and silently doing nothing the second time.
            TutorialStep.MANUAL_SELECTION -> {
                demoAddedCards.forEach { viewModel.selectedCards.remove(it) }
                demoAddedCards = emptyList()
            }
            else -> {}
        }
    }

    // Switches to the Deck tab as a scripted tap (see simulateTabTap)
    // before landing on MANUAL_SELECTION, so the transition itself reads as
    // an action rather than the Deck tab just silently already being
    // active once the step's text appears.
    private fun switchToDeckTabDemo() {
        animationJob = viewModel.viewModelScope.launch {
            simulateTabTap(DECK_TAB_INDEX)
            val steps = TutorialStep.entries
            val nextIndex = currentStep.ordinal + 1
            if (nextIndex < steps.size) {
                currentStep = steps[nextIndex]
                onStepEntered(currentStep)
            }
        }
    }

    // Demonstrates adding cards manually. Deliberately does *not* clear the
    // existing selection first -- the whole point is continuity with the scan
    // just shown on the Board tab: these are a few more cards added on top of
    // it, not an unrelated do-over. onScrollRequest performs the reset + two
    // natural downward scroll strokes (see SelectionScreen) and returns
    // whichever cards actually ended up centered on screen -- read from real
    // layout measurements there, not guessed here.
    private fun autoSelectSampleCards() {
        animationJob = viewModel.viewModelScope.launch {
            isDemoPlaying = true

            val centeredCards = onScrollRequest?.invoke() ?: emptyList()
            val alreadySelected = viewModel.selectedCards.toSet()
            val newCards = centeredCards.filter { it !in alreadySelected }
            demoAddedCards = newCards

            delay(500)
            newCards.forEach { delay(700); viewModel.toggleSelection(it) }

            delay(900)

            if (currentStep == TutorialStep.MANUAL_SELECTION) {
                // Close the loop: switch back to the Board tab (as a scripted
                // tap, not an invisible flip) so the cards just added are
                // seen landing in the previously-empty Manual Selection row.
                // A longer settle delay here (vs. the default) gives the
                // user a real moment to take in the board before moving on.
                // isDemoPlaying deliberately stays true through this whole
                // switch-and-settle stretch, not just the scroll/select part
                // above -- from the user's perspective it's all one demo, and
                // TutorialLayer's small badge (anchored to the Deck tab) uses
                // this flag to stay visible the entire time, including while
                // the main overlay is hidden for the scripted tap itself.
                simulateTabTap(BOARD_TAB_INDEX, settleDelayMs = 2500)
                isDemoPlaying = false
                val steps = TutorialStep.entries
                val nextIndex = currentStep.ordinal + 1
                if (nextIndex < steps.size) {
                    currentStep = steps[nextIndex]
                    onStepEntered(currentStep)
                }
            } else {
                isDemoPlaying = false
            }
        }
    }

    // Demonstrates excluding a scanned card (the gray-out), since the
    // tutorial overlay blocks real taps on the cards underneath -- without
    // this, nothing ever exercises that interaction during the tutorial. A
    // tap-ripple (see demoCardTapTrigger) plays right on the card itself
    // immediately before each toggle, so the gray-out reads as the result of
    // a tap rather than the card just changing on its own.
    private fun playScanResultDemo() {
        val card = viewModel.scannedDetection?.detectedCards
            ?.mapNotNull { it.card }
            ?.firstOrNull { it in viewModel.selectedCards }
        demoCard = card

        animationJob = viewModel.viewModelScope.launch {
            isDemoPlaying = true

            delay(400)
            demoCardTapTrigger++
            delay(300)
            card?.let { viewModel.toggleSelection(it) }
            delay(1300)
            demoCardTapTrigger++
            delay(300)
            card?.let { viewModel.toggleSelection(it) }

            // A beat longer than the pause before the gray-out above -- gives
            // a moment to actually register the card is back before the demo
            // ends and the next step's card appears.
            delay(900)
            isDemoPlaying = false
            demoCard = null

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
