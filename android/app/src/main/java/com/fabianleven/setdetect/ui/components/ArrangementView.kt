package com.fabianleven.setdetect.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.RestartAlt
import androidx.compose.material.icons.rounded.Warning
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

@Composable
fun ArrangementView(
    arrangement: CardsArrangement,
    cards: List<SetCard?>,
    selectedCards: Set<SetCard>,
    onToggleCard: (SetCard) -> Unit,
    modifier: Modifier = Modifier,
    // Set by the tutorial while it's demonstrating excluding a card: reports
    // that specific card's on-screen position back via onHighlightedCardPositioned
    // so the tutorial can draw a tap effect exactly where it is, rather than
    // anywhere generic on the arrangement.
    highlightedCard: SetCard? = null,
    onHighlightedCardPositioned: (Rect) -> Unit = {},
    // Cards whose attributes exactly match another card currently on the board:
    // either the same physical card is there twice, or classification
    // misread one of them. Each gets a small badge; see BoardTab for the
    // corresponding screen-corner banner.
    duplicateCards: Set<SetCard> = emptySet()
) {
    var manualRotation by remember(arrangement) { mutableFloatStateOf(0f) }
    var manualScale by remember(arrangement) { mutableFloatStateOf(1f) }

    BoxWithConstraints(
        // Nothing clips a graphicsLayer's drawing to its layout bounds by
        // default in Compose, so a zoomed-in card could paint outside this
        // view, over whatever's above it in BoardTab's Column
        // (the Manual Selection section), since that's drawn earlier and so
        // sits underneath in paint order. clipToBounds() pins the drawn
        // content to exactly this view's own area on all four sides,
        // regardless of scale or rotation.
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

        // The arrangement is calculated in a coordinate system where cards fit
        // within a [0, 1] range, scaled here to fit the view while preserving
        // aspect ratio and shrunk by a factor (approx 1/sqrt(2)) so it stays
        // within bounds when rotated.
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
                val card = cards.getOrNull(index) ?: return@forEachIndexed
                val zOrder = arrangement.cardZOrders.getOrNull(index) ?: 0
                
                val rotationDegrees = Math.toDegrees(pose.angle.toDouble()).toFloat()
                
                // C++ arrangement gives center and width in normalized space. Height
                // is derived from width via the fixed card aspect ratio, not the
                // arrangement's own height, so the box always matches CardView's
                // aspect ratio exactly; an independently-estimated height could
                // letterbox or crop it.
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
                        }
                        .let {
                            if (card == highlightedCard) {
                                it.onGloballyPositioned { coords -> onHighlightedCardPositioned(coords.boundsInWindow()) }
                            } else {
                                it
                            }
                        }
                ) {
                    CardView(
                        card = card,
                        isSelected = selectedCards.contains(card),
                        isActive = selectedCards.contains(card),
                        showBorder = false,
                        onClick = { onToggleCard(card) },
                        modifier = Modifier.fillMaxSize()
                    )
                    if (card in duplicateCards) {
                        Icon(
                            imageVector = Icons.Rounded.Warning,
                            contentDescription = stringResource(R.string.selection_duplicate_card),
                            tint = MaterialTheme.colorScheme.onError,
                            modifier = Modifier
                                .align(Alignment.TopEnd)
                                .padding(1.dp)
                                .background(MaterialTheme.colorScheme.error, CircleShape)
                                .padding(1.dp)
                                .size(9.dp)
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
