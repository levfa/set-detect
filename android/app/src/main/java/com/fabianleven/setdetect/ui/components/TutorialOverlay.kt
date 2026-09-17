package com.fabianleven.setdetect.ui.components

import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.BlendMode
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.fabianleven.setdetect.R
import com.fabianleven.setdetect.ui.BOARD_TAB_INDEX
import com.fabianleven.setdetect.ui.DECK_TAB_INDEX

// requiredTab is null for steps whose target (top bar / bottom bar) is
// reachable from either tab; otherwise the tab that must be active for the
// step's target to be composed and visible.
enum class TutorialStep(
    val titleRes: Int,
    val descriptionRes: Int,
    val requiredTab: Int? = null
) {
    INTRO(R.string.tut_title, R.string.tut_intro),
    SCAN_CARDS(R.string.tut_title, R.string.tut_scan_cards, BOARD_TAB_INDEX),
    SCAN_APPEARS(R.string.tut_title, R.string.tut_scan_appears, BOARD_TAB_INDEX),
    SETS_AND_REVEAL(R.string.tut_title, R.string.tut_sets_and_reveal),
    SCAN_EDIT(R.string.tut_title, R.string.tut_scan_edit, BOARD_TAB_INDEX),
    MANUAL_SELECTION_INTRO(R.string.tut_title, R.string.tut_manual_selection_intro, BOARD_TAB_INDEX),
    MANUAL_SELECTION(R.string.tut_title, R.string.tut_manual_selection, DECK_TAB_INDEX),
    TABS_ARE_VIEWS(R.string.tut_title, R.string.tut_tabs_are_views),
    RE_RUN_TUTORIAL(R.string.tut_title, R.string.tut_re_run_tutorial)
}

@Composable
fun TutorialOverlay(
    currentStepIndex: Int,
    totalStepsCount: Int,
    title: String,
    description: String,
    targetRects: List<Rect>,
    onNext: () -> Unit,
    onBack: () -> Unit,
    onSkip: () -> Unit,
    isFirstStep: Boolean,
    isLastStep: Boolean,
    isDemoPlaying: Boolean = false,
    forceCenter: Boolean = false,
    // Non-null only on the last step -- lets the user turn off "show
    // tutorial on startup" right where they're told they can restart it
    // later, instead of only from Settings. Null hides the checkbox.
    startupToggleChecked: Boolean? = null,
    onStartupToggleChanged: (Boolean) -> Unit = {}
) {
    BoxWithConstraints(
        modifier = Modifier
            .fillMaxSize()
            .clickable(
                interactionSource = remember { MutableInteractionSource() },
                indication = null,
                onClick = {} // Capture clicks to prevent interaction with underlying UI
            )
            .graphicsLayer(alpha = 0.99f)
    ) {
        val density = LocalDensity.current
        val padding = with(density) { 8.dp.toPx() }
        val cornerRadius = with(density) { 8.dp.toPx() }
        val screenWidthPx = with(density) { maxWidth.toPx() }
        val screenHeightPx = with(density) { maxHeight.toPx() }

        val validRects = targetRects.filter { it != Rect.Zero && it.width > 0 && it.height > 0 }
        val hasTarget = validRects.isNotEmpty()
        val isMultiTarget = validRects.size > 1
        val isLargeArea = hasTarget && !isMultiTarget &&
            validRects[0].width > screenWidthPx * 0.7f && validRects[0].height > screenHeightPx * 0.5f

        // The text card must never sit on top of the element(s) it's explaining,
        // so instead of always centering it, anchor it to whichever half of the
        // screen the (single) target *isn't* in -- below a target in the top
        // half, above one in the bottom half. Falls back to centered when
        // there's no target (INTRO), the target is too large to leave anywhere
        // clear, there are multiple separate targets to highlight at once
        // (nothing to anchor relative to a single position), or the step
        // explicitly asks to always be centered regardless of target geometry.
        val cardAlignment = when {
            forceCenter || !hasTarget || isLargeArea || isMultiTarget -> Alignment.Center
            validRects[0].center.y < screenHeightPx / 2f -> Alignment.BottomCenter
            else -> Alignment.TopCenter
        }

        val infiniteTransition = rememberInfiniteTransition(label = "spotlight")
        val pulseOffsetPx by infiniteTransition.animateFloat(
            initialValue = 0f,
            targetValue = with(density) { 3.dp.toPx() }, // Tightened absolute 3dp pulse
            animationSpec = infiniteRepeatable(
                animation = tween(600, easing = LinearOutSlowInEasing),
                repeatMode = RepeatMode.Reverse
            ),
            label = "pulse"
        )

        Canvas(modifier = Modifier.fillMaxSize()) {
            drawRect(color = Color.Black.copy(alpha = 0.85f))

            validRects.forEach { targetRect ->
                val center = targetRect.center
                val isIcon = targetRect.width < 100.dp.toPx() && targetRect.height < 100.dp.toPx()
                val isCircularIcon = isIcon && (targetRect.width / targetRect.height in 0.9f..1.1f)

                // Safety margin from screen edges for the pulsing border
                val safetyMargin = 2.dp.toPx()
                val maxAllowedRadius = minOf(
                    center.x,
                    size.width - center.x,
                    center.y,
                    size.height - center.y
                ) - safetyMargin

                if (isCircularIcon) {
                    // Constant radius for the transparent hole (approx 0.38 of max dimension)
                    val baseRadius = (maxOf(targetRect.width, targetRect.height) * 0.38f)
                    // Ensure the pulse stays on screen
                    val radius = minOf(baseRadius, maxAllowedRadius - pulseOffsetPx).coerceAtLeast(0f)

                    drawCircle(
                        color = Color.Transparent,
                        center = center,
                        radius = radius,
                        blendMode = BlendMode.Clear
                    )
                    // Pulsing border centered on the icon, separate from the hole
                    drawCircle(
                        color = Color.White,
                        center = center,
                        radius = radius + pulseOffsetPx,
                        style = Stroke(width = 2.dp.toPx())
                    )
                } else {
                    val spotlightPadding = padding
                    // For rectangles, we keep it centered by symmetric clamping
                    val halfW = (targetRect.width / 2f + spotlightPadding).coerceAtMost(minOf(center.x, size.width - center.x) - safetyMargin)
                    val halfH = (targetRect.height / 2f + spotlightPadding).coerceAtMost(minOf(center.y, size.height - center.y) - safetyMargin)

                    val spotlightRect = Rect(center.x - halfW, center.y - halfH, center.x + halfW, center.y + halfH)

                    drawRoundRect(
                        color = Color.Transparent,
                        topLeft = spotlightRect.topLeft,
                        size = spotlightRect.size,
                        cornerRadius = CornerRadius(cornerRadius),
                        blendMode = BlendMode.Clear
                    )

                    // isLargeArea only ever applies to a single target, so it's
                    // safe to gate the pulsing border on it here too.
                    if (!isLargeArea) {
                        // Absolute pulsing border centered on the rectangle
                        val pW = minOf(pulseOffsetPx, minOf(center.x, size.width - center.x) - safetyMargin - halfW)
                        val pH = minOf(pulseOffsetPx, minOf(center.y, size.height - center.y) - safetyMargin - halfH)

                        drawRoundRect(
                            color = Color.White,
                            topLeft = Offset(spotlightRect.left - pW, spotlightRect.top - pH),
                            size = Size(spotlightRect.width + pW * 2, spotlightRect.height + pH * 2),
                            cornerRadius = CornerRadius(cornerRadius + maxOf(pW, pH)),
                            style = Stroke(width = 2.dp.toPx())
                        )
                    }
                }
            }
        }

        val alpha by animateFloatAsState(
            targetValue = if (isDemoPlaying) 0f else 1f,
            label = "alpha"
        )

        Card(
            modifier = Modifier
                .align(cardAlignment)
                // Keep clear of the status bar / gesture nav bar -- BottomCenter
                // and TopCenter alignment would otherwise push the card right up
                // against (or under) them.
                .windowInsetsPadding(WindowInsets.systemBars)
                .graphicsLayer(alpha = alpha)
                .padding(horizontal = 24.dp, vertical = 20.dp)
                .fillMaxWidth()
                .widthIn(max = 400.dp),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surface
            ),
            elevation = CardDefaults.cardElevation(defaultElevation = 8.dp),
            shape = RoundedCornerShape(16.dp)
        ) {
            Column(
                modifier = Modifier.padding(20.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = "${currentStepIndex + 1} / $totalStepsCount",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.outline
                )
                Spacer(Modifier.height(8.dp))
                Text(
                    text = title,
                    style = MaterialTheme.typography.titleLarge,
                    color = MaterialTheme.colorScheme.primary,
                    textAlign = TextAlign.Center
                )
                Spacer(Modifier.height(12.dp))
                Text(
                    text = description,
                    style = MaterialTheme.typography.bodyMedium,
                    textAlign = TextAlign.Center
                )

                if (startupToggleChecked != null) {
                    Spacer(Modifier.height(12.dp))
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier
                            .clickable(
                                interactionSource = remember { MutableInteractionSource() },
                                indication = null
                            ) { onStartupToggleChanged(!startupToggleChecked) }
                    ) {
                        Checkbox(checked = startupToggleChecked, onCheckedChange = onStartupToggleChanged)
                        Text(
                            text = stringResource(R.string.settings_show_tutorial_on_startup),
                            style = MaterialTheme.typography.bodySmall
                        )
                    }
                }

                Spacer(Modifier.height(20.dp))

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    if (isLastStep) {
                        Spacer(Modifier.width(1.dp))
                    } else {
                        TextButton(
                            onClick = onSkip,
                            enabled = !isDemoPlaying
                        ) {
                            Text(stringResource(R.string.tut_skip))
                        }
                    }
                    Row {
                        if (!isFirstStep) {
                            TextButton(
                                onClick = onBack,
                                enabled = !isDemoPlaying
                            ) {
                                Text(stringResource(R.string.tut_back))
                            }
                            Spacer(Modifier.width(8.dp))
                        }

                        Button(
                            onClick = onNext,
                            enabled = !isDemoPlaying
                        ) {
                            Text(
                                text = if (isLastStep) stringResource(R.string.tut_finish)
                                else stringResource(R.string.tut_next)
                            )
                        }
                    }
                }
            }
        }
    }
}
