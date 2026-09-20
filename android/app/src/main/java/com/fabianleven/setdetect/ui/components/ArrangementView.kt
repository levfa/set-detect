package com.fabianleven.setdetect.ui.components

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.foundation.gestures.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.RestartAlt
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.boundsInWindow
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.compose.ui.zIndex
import com.fabianleven.setdetect.R
import com.fabianleven.setdetect.domain.CardsArrangement
import com.fabianleven.setdetect.domain.SetCard
import com.fabianleven.setdetect.domain.SlotChoice
import com.fabianleven.setdetect.domain.resolveSlot

private const val PressedScale = 0.9f

@Composable
fun ArrangementView(
    arrangement: CardsArrangement,
    cards: List<SetCard>,
    choices: List<SlotChoice>,
    onCardClick: (Int) -> Unit,
    // Slot whose picker is opening or open; drawn pressed-in with an outline.
    activeSlot: Int? = null,
    modifier: Modifier = Modifier,
    // Set by the tutorial while demonstrating a card edit; that slot's
    // on-screen position is reported through onHighlightedCardPositioned.
    highlightedSlot: Int? = null,
    onHighlightedCardPositioned: (Rect) -> Unit = {},
    // Cards whose attributes match another card on the board (a repeated
    // card or a misclassification). Each gets a small badge.
    duplicateCards: Set<SetCard> = emptySet()
) {
    var manualRotation by remember(arrangement) { mutableFloatStateOf(0f) }
    var manualScale by remember(arrangement) { mutableFloatStateOf(1f) }

    BoxWithConstraints(
        // A graphicsLayer isn't clipped to layout bounds by default, so zoomed
        // cards could paint over neighboring content. clipToBounds() keeps the
        // drawing within this view regardless of scale or rotation.
        modifier = modifier
            .clipToBounds()
            .pointerInput(arrangement) {
                awaitEachGesture {
                    awaitFirstDown(requireUnconsumed = false)
                    do {
                        val event = awaitPointerEvent()
                        val canceled = event.changes.any { it.isConsumed }
                        if (!canceled) {
                            val zoomChange = event.calculateZoom()
                            val rotationChange = event.calculateRotation()
                            val panChange = event.calculatePan()
                            
                            if (event.changes.size > 1) {
                                // Multi-touch: standard zoom and rotation
                                manualScale = (manualScale * zoomChange).coerceIn(0.5f, 5f)
                                manualRotation += rotationChange
                            } else if (event.changes.size == 1 && panChange != Offset.Zero) {
                                // One-finger: circular rotation around the center
                                val centroid = event.calculateCentroid()
                                val viewCenter = Offset(size.width / 2f, size.height / 2f)
                                val oldPos = centroid - panChange
                                
                                val oldAngle = Math.atan2((oldPos.y - viewCenter.y).toDouble(), (oldPos.x - viewCenter.x).toDouble())
                                val newAngle = Math.atan2((centroid.y - viewCenter.y).toDouble(), (centroid.x - viewCenter.x).toDouble())
                                
                                manualRotation += Math.toDegrees(newAngle - oldAngle).toFloat()
                            }
                        }
                    } while (event.changes.any { it.pressed })
                }
            }
    ) {
        val viewWidth = maxWidth
        val viewHeight = maxHeight

        // The arrangement uses a [0, 1] coordinate system, scaled to fit the view
        // and shrunk by about 1/sqrt(2) so it stays in bounds when rotated.
        val baseScale = minOf(viewWidth.value, viewHeight.value) * 0.7f
        val baseOffsetX = (viewWidth.value - baseScale) / 2
        val baseOffsetY = (viewHeight.value - baseScale) / 2

        Box(
            modifier = Modifier
                .fillMaxSize()
                .graphicsLayer {
                    rotationZ = manualRotation
                    scaleX = manualScale
                    scaleY = manualScale
                }
        ) {
            arrangement.cardPoses.forEachIndexed { index, pose ->
                val scanned = cards.getOrNull(index) ?: return@forEachIndexed
                val choice = choices.getOrElse(index) { SlotChoice.Original }
                val resolved = resolveSlot(scanned, choice)
                val card = resolved ?: scanned
                val interaction = remember { MutableInteractionSource() }
                val pressed by interaction.collectIsPressedAsState()
                val pressScale by animateFloatAsState(
                    if (pressed || index == activeSlot) PressedScale else 1f,
                    label = "pressScale"
                )
                val zOrder = arrangement.cardZOrders.getOrNull(index) ?: 0
                
                val rotationDegrees = Math.toDegrees(pose.angle.toDouble()).toFloat()
                
                // The arrangement gives center and width in normalized space. Height
                // is derived from width via the card aspect ratio, not the
                // arrangement's own height, so the box always matches the card's
                // aspect ratio.
                val cardW = arrangement.cardWidth * baseScale
                val cardH = cardW / CardAspectRatio
                val cardX = pose.centerX * baseScale + baseOffsetX - (cardW / 2)
                val cardY = pose.centerY * baseScale + baseOffsetY - (cardH / 2)

                Box(
                    modifier = Modifier
                        .offset(x = cardX.dp, y = cardY.dp)
                        .size(width = cardW.dp, height = cardH.dp)
                        .zIndex(zOrder.toFloat())
                        .graphicsLayer {
                            rotationZ = rotationDegrees
                            scaleX = pressScale
                            scaleY = pressScale
                        }
                        .let {
                            if (index == highlightedSlot) {
                                it.onGloballyPositioned { coords -> onHighlightedCardPositioned(coords.boundsInWindow()) }
                            } else {
                                it
                            }
                        }
                ) {
                    CardView(
                        card = card,
                        isSelected = resolved != null,
                        isActive = resolved != null,
                        showBorder = false,
                        onClick = { onCardClick(index) },
                        interactionSource = interaction,
                        modifier = Modifier.fillMaxSize()
                    )
                    if (index == activeSlot) {
                        Box(
                            modifier = Modifier
                                .fillMaxSize()
                                .border(2.dp, MaterialTheme.colorScheme.primary, RoundedCornerShape(12))
                        )
                    }
                    if (resolved != null && resolved in duplicateCards) {
                        DuplicateBadge(
                            iconSize = 9.dp,
                            modifier = Modifier
                                .align(Alignment.TopEnd)
                                .padding(1.dp)
                        )
                    }
                }
            }
        }

        // Reset button only appears after the view was moved/zoomed
        if (manualRotation != 0f || manualScale != 1f) {
            IconButton(
                onClick = {
                    manualRotation = 0f
                    manualScale = 1f
                },
                modifier = Modifier
                    .align(Alignment.BottomStart)
                    .padding(16.dp)
                    .background(MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.8f), CircleShape)
            ) {
                Icon(
                    imageVector = Icons.Rounded.RestartAlt,
                    contentDescription = stringResource(R.string.selection_reset_view),
                    tint = MaterialTheme.colorScheme.onSecondaryContainer
                )
            }
        }
    }
}
