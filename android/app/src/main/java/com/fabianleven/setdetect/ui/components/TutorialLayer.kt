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
import com.fabianleven.setdetect.ui.TutorialController

// The screen positions (in window coordinates) the tutorial can spotlight.
// Each field is set at the UI element it tracks via `onGloballyPositioned`.
class TutorialTargets {
    var selectionInfo by mutableStateOf(Rect.Zero)
    var arrangement by mutableStateOf(Rect.Zero)
    var manualSelection by mutableStateOf(Rect.Zero)
    var revealSetsButton by mutableStateOf(Rect.Zero)
    var fab by mutableStateOf(Rect.Zero)
    var editFab by mutableStateOf(Rect.Zero)
    var helpButton by mutableStateOf(Rect.Zero)

    // Position of the card the arrangement highlights for the tutorial, where
    // the Scan Edit demo draws its tap-ripple.
    var demoCardRect by mutableStateOf(Rect.Zero)
}

// Hosts everything the tutorial draws over the screen: the spotlight overlay
// and the tap-ripple shown during the Scan Edit demo.
@Composable
fun TutorialLayer(
    tutorial: TutorialController,
    targets: TutorialTargets
) {
    // "Still the tutorial" badge for the Scan Edit demo, tucked into the
    // arrangement's corner so it doesn't cover the board.
    if (tutorial.active && tutorial.isDemoPlaying && tutorial.currentStep == TutorialStep.SCAN_EDIT) {
        DemoIndicatorBadge(anchor = targets.arrangement)
    }

    // Tap-ripple on the card the Scan Edit demo is editing.
    if (tutorial.active && tutorial.currentStep == TutorialStep.SCAN_EDIT) {
        TapRippleIndicator(bounds = targets.demoCardRect, key = tutorial.demoCardTapTrigger)
    }

    if (!tutorial.active) return

    val currentTargetRects = when (tutorial.currentStep) {
        TutorialStep.INTRO -> emptyList()
        TutorialStep.SCAN_CARDS -> listOf(targets.fab)
        TutorialStep.SCAN_APPEARS -> listOf(targets.arrangement)
        // Highlights the sets/cards indicator and the button together, since
        // the count motivates tapping Reveal.
        TutorialStep.SETS_AND_REVEAL -> listOf(targets.selectionInfo, targets.revealSetsButton)
        TutorialStep.SCAN_EDIT -> listOf(targets.arrangement)
        TutorialStep.MANUAL_SELECTION -> listOf(targets.manualSelection, targets.editFab)
        TutorialStep.RE_RUN_TUTORIAL -> listOf(targets.helpButton)
    }

    val description = when (tutorial.currentStep) {
        TutorialStep.SETS_AND_REVEAL -> stringResource(
            tutorial.currentStep.descriptionRes,
            stringResource(R.string.selection_reveal_sets)
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
        // The help icon sits at the top edge, so center the card.
        forceCenter = tutorial.currentStep == TutorialStep.RE_RUN_TUTORIAL,
        startupToggleChecked = if (tutorial.currentStep == TutorialStep.RE_RUN_TUTORIAL) tutorial.showOnStartup else null,
        onStartupToggleChanged = { tutorial.updateShowOnStartup(it) }
    )
}

// A small pill in the anchor's bottom-left corner, so it doesn't cover the board.
@Composable
private fun DemoIndicatorBadge(anchor: Rect) {
    if (anchor == Rect.Zero) return
    val density = LocalDensity.current

    // The badge height isn't measured, so a fixed upward shift larger than it
    // approximates the bottom-left corner.
    val offset = with(density) {
        IntOffset(x = (anchor.left + 8.dp.toPx()).toInt(), y = (anchor.bottom - 44.dp.toPx()).toInt())
    }

    Box(modifier = Modifier.fillMaxSize()) {
        Surface(
            color = MaterialTheme.colorScheme.surfaceContainerHighest,
            shape = RoundedCornerShape(50),
            shadowElevation = 4.dp,
            modifier = Modifier
                .align(Alignment.TopStart)
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

// A brief expanding, fading circle centered on `bounds`, marking a scripted
// tap. `key` restarts the animation when `bounds` is unchanged across
// repeated taps on the same spot.
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
