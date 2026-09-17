package com.fabianleven.setdetect.ui.components

import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.PlayArrow
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import com.fabianleven.setdetect.R
import com.fabianleven.setdetect.ui.BOARD_TAB_INDEX
import com.fabianleven.setdetect.ui.DECK_TAB_INDEX
import com.fabianleven.setdetect.ui.TutorialController

// The screen positions (in window coordinates) the tutorial can spotlight or
// switch tabs toward. One holder instead of a dozen separate `remember`s in
// SelectionScreen, so it's visually obvious at a glance which bounds exist
// only for the tutorial's benefit; each field is still set right at the real
// UI element it tracks, via `onGloballyPositioned`, since that's the only
// place its true position is known.
class TutorialTargets {
    var selectionInfo by mutableStateOf(Rect.Zero)
    var deck by mutableStateOf(Rect.Zero)
    var tabRow by mutableStateOf(Rect.Zero)
    var boardTab by mutableStateOf(Rect.Zero)
    var deckTab by mutableStateOf(Rect.Zero)
    var arrangement by mutableStateOf(Rect.Zero)
    var manualSelection by mutableStateOf(Rect.Zero)
    var revealSetsButton by mutableStateOf(Rect.Zero)
    var fab by mutableStateOf(Rect.Zero)
    var helpButton by mutableStateOf(Rect.Zero)

    // Position of whichever card ArrangementView is currently highlighting
    // for the tutorial, used to draw a tap-ripple exactly on it during the
    // Scan Edit demo.
    var demoCardRect by mutableStateOf(Rect.Zero)
}

// Hosts everything the tutorial draws on top of the real screen: the
// spotlight overlay itself, and the tap-ripple shown during a scripted tab
// switch. Reads `tutorial`'s current step and `targets`' bounds to work out
// what to highlight; doesn't otherwise touch the rest of SelectionScreen.
@Composable
fun TutorialLayer(
    tutorial: TutorialController,
    targets: TutorialTargets,
    tabLabels: List<String>
) {
    // Small "still the tutorial" badge for scripted demos that either run
    // long (Manual Selection: scroll, auto-select, scripted tap back to
    // Board, then a beat to view the result, hiding the main overlay
    // entirely for that whole stretch), or where the main card's own
    // fade-out isn't quite enough on its own (Scan Edit). Positioned
    // per-step, not centered/spotlighted like the main overlay, so it never
    // covers the board/grid it's meant to be explaining.
    when {
        tutorial.active && tutorial.isDemoPlaying && tutorial.currentStep == TutorialStep.MANUAL_SELECTION ->
            DemoIndicatorBadge(anchor = targets.deckTab, corner = BadgeCorner.SCREEN_TOP_END)
        tutorial.active && tutorial.isDemoPlaying && tutorial.currentStep == TutorialStep.SCAN_EDIT ->
            DemoIndicatorBadge(anchor = targets.arrangement, corner = BadgeCorner.ANCHOR_BOTTOM_START)
    }

    // A tap-ripple drawn exactly on the card the Scan Edit demo is currently
    // toggling, so the gray-out reads as caused by a tap.
    if (tutorial.active && tutorial.currentStep == TutorialStep.SCAN_EDIT) {
        TapRippleIndicator(bounds = targets.demoCardRect, key = tutorial.demoCardTapTrigger)
    }

    // A scripted tab switch is in progress: hide the overlay entirely (not
    // just fade it) so the tab row is fully visible, and show a tap ripple
    // where the "tap" is landing.
    tutorial.simulatedTapTarget?.let { target ->
        val bounds = if (target == BOARD_TAB_INDEX) targets.boardTab else targets.deckTab
        TapRippleIndicator(bounds = bounds)
    }

    if (!tutorial.active || tutorial.simulatedTapTarget != null) return

    val currentTargetRects = when (tutorial.currentStep) {
        TutorialStep.INTRO -> emptyList()
        TutorialStep.SCAN_CARDS -> listOf(targets.fab)
        TutorialStep.SCAN_APPEARS -> listOf(targets.arrangement)
        // Highlights both the sets/cards indicator and the button at once:
        // the count is what motivates tapping Reveal, so showing them
        // together makes that connection clear.
        TutorialStep.SETS_AND_REVEAL -> listOf(targets.selectionInfo, targets.revealSetsButton)
        TutorialStep.SCAN_EDIT -> listOf(targets.arrangement)
        // Highlights the Manual Selection row and the Deck tab together:
        // both are ways of getting to the same manual-selection feature.
        TutorialStep.MANUAL_SELECTION_INTRO -> listOf(targets.manualSelection, targets.deckTab)
        TutorialStep.MANUAL_SELECTION -> listOf(targets.deck)
        TutorialStep.TABS_ARE_VIEWS -> listOf(targets.tabRow)
        TutorialStep.RE_RUN_TUTORIAL -> listOf(targets.helpButton)
    }

    val description = when (tutorial.currentStep) {
        TutorialStep.SETS_AND_REVEAL -> stringResource(
            tutorial.currentStep.descriptionRes,
            stringResource(R.string.selection_reveal_sets)
        )
        TutorialStep.MANUAL_SELECTION_INTRO -> stringResource(
            tutorial.currentStep.descriptionRes,
            tabLabels[DECK_TAB_INDEX]
        )
        TutorialStep.TABS_ARE_VIEWS -> stringResource(
            tutorial.currentStep.descriptionRes,
            tabLabels[BOARD_TAB_INDEX],
            tabLabels[DECK_TAB_INDEX]
        )
        else -> stringResource(tutorial.currentStep.descriptionRes)
    }

    TutorialOverlay(
        currentStepIndex = tutorial.currentStep.ordinal,
        totalStepsCount = TutorialStep.entries.size,
        title = stringResource(tutorial.currentStep.titleRes),
        description = description,
        targetRects = currentTargetRects,
        onNext = { tutorial.next() },
        onBack = { tutorial.previous() },
        onSkip = { tutorial.skip() },
        isFirstStep = tutorial.currentStep == TutorialStep.entries.first(),
        isLastStep = tutorial.currentStep == TutorialStep.entries.last(),
        isDemoPlaying = tutorial.isDemoPlaying,
        // Both of these steps' targets (the help icon, the tab row) sit right
        // at the top edge, which single-target placement would otherwise
        // anchor to the bottom; centered reads better for them.
        forceCenter = tutorial.currentStep == TutorialStep.RE_RUN_TUTORIAL ||
            tutorial.currentStep == TutorialStep.TABS_ARE_VIEWS,
        startupToggleChecked = if (tutorial.currentStep == TutorialStep.RE_RUN_TUTORIAL) tutorial.showOnStartup else null,
        onStartupToggleChanged = { tutorial.updateShowOnStartup(it) }
    )
}

// Where DemoIndicatorBadge sits relative to its anchor: pinned to the
// screen's own right edge (just below the anchor, for the Deck tab, since the
// badge's width varies by locale, so anchoring to the anchor's own edge
// there would drift), or tucked into the anchor's own bottom-left corner
// (for the arrangement/scan view, which already sits well inside the screen).
private enum class BadgeCorner { SCREEN_TOP_END, ANCHOR_BOTTOM_START }

// A small, unobtrusive pill, deliberately not centered or spotlighted like
// the main overlay card, since that would cover the board/grid the demo is
// showing off.
@Composable
private fun DemoIndicatorBadge(anchor: Rect, corner: BadgeCorner) {
    if (anchor == Rect.Zero) return
    val density = LocalDensity.current

    val boxAlignment: Alignment
    val offset: IntOffset
    with(density) {
        when (corner) {
            BadgeCorner.SCREEN_TOP_END -> {
                boxAlignment = Alignment.TopEnd
                offset = IntOffset(x = 0, y = (anchor.bottom + 8.dp.toPx()).toInt())
            }
            BadgeCorner.ANCHOR_BOTTOM_START -> {
                boxAlignment = Alignment.TopStart
                // No measured badge height to anchor its bottom edge exactly
                // against anchor.bottom, so a fixed upward shift comfortably
                // larger than the badge's own height reads as "bottom-left of
                // the view" closely enough without needing a second layout pass.
                offset = IntOffset(x = (anchor.left + 8.dp.toPx()).toInt(), y = (anchor.bottom - 44.dp.toPx()).toInt())
            }
        }
    }

    Box(modifier = Modifier.fillMaxSize()) {
        Surface(
            color = MaterialTheme.colorScheme.surfaceContainerHighest,
            shape = RoundedCornerShape(50),
            shadowElevation = 4.dp,
            modifier = Modifier
                .align(boxAlignment)
                .padding(end = if (corner == BadgeCorner.SCREEN_TOP_END) 16.dp else 0.dp)
                .offset { offset }
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp)
            ) {
                Icon(
                    imageVector = Icons.Rounded.PlayArrow,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.size(14.dp)
                )
                Spacer(Modifier.width(6.dp))
                Text(
                    text = stringResource(R.string.tut_demo_playing),
                    style = MaterialTheme.typography.labelSmall
                )
            }
        }
    }
}

// A brief expanding, fading circle centered on `bounds`, used to make a
// scripted tap (a tab switch, or the Scan Edit demo toggling a card) read
// as an actual tap, not an unexplained instant change. `key` restarts the
// animation on demand: needed when `bounds` stays the same across repeat
// taps on the same spot (e.g. the same card toggled off then back on),
// where `bounds` alone wouldn't change to naturally retrigger it.
@Composable
private fun TapRippleIndicator(bounds: Rect, modifier: Modifier = Modifier, key: Any = bounds) {
    if (bounds == Rect.Zero) return

    val color = MaterialTheme.colorScheme.primary
    val progress = remember(key) { Animatable(0f) }
    LaunchedEffect(key) {
        progress.snapTo(0f)
        progress.animateTo(1f, animationSpec = tween(900, easing = FastOutSlowInEasing))
    }

    Canvas(modifier = modifier.fillMaxSize()) {
        val maxRadius = 32.dp.toPx()
        drawCircle(
            color = color.copy(alpha = (1f - progress.value) * 0.5f),
            radius = maxRadius * progress.value,
            center = bounds.center
        )
    }
}
