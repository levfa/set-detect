package com.fabianleven.setdetect.ui.components

import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.BlendMode
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.dp
import com.fabianleven.setdetect.domain.DetectedCard
import com.fabianleven.setdetect.ui.CardOutlineColor
import com.fabianleven.setdetect.ui.expandQuad
import com.fabianleven.setdetect.ui.quadPath
import com.fabianleven.setdetect.ui.scaleQuad
import com.fabianleven.setdetect.ui.transformQuadCorners

private const val DimAlpha = 0.88f
private val OutlineWidth = 3.dp

// Pushed outward from each card's true edge by this much, so the ring sits
// in the dimmed background.
private val OutlineOffset = 4.dp

/**
 * Shown in place of the live camera feed during the post-scan review pause:
 * a frozen snapshot of what was detected, dimmed except for each card (cut
 * out at its true, undistorted shape), with a highlight ring around each one.
 *
 * @param entrance 0f (just started) to 1f (settled in): fades the dim and
 *   the highlight rings in together.
 * @param popScale the highlight rings' current scale, typically animated
 *   from slightly under 1f with a spring for a "pop" on entry, then held
 *   at 1f for the rest of the pause.
 */
@Composable
fun ScanReviewOverlay(
    cards: List<DetectedCard>,
    matrix: Matrix?,
    bitmap: Bitmap?,
    entrance: Float,
    popScale: Float,
    onCancel: () -> Unit,
    modifier: Modifier = Modifier
) {
    Box(modifier = modifier.fillMaxSize()) {
        if (bitmap != null) {
            Box(modifier = Modifier.fillMaxSize().background(Color.Black)) {
                Image(
                    bitmap = bitmap.asImageBitmap(),
                    contentDescription = null,
                    contentScale = ContentScale.Fit,
                    modifier = Modifier.fillMaxSize()
                )
            }
        }

        val quadCorners = remember(cards, matrix) {
            if (matrix == null) emptyList()
            else cards.filter { it.card != null && it.corners.size == 4 }
                .map { transformQuadCorners(it.corners, matrix) }
        }
        val density = LocalDensity.current
        val outlineWidthPx = with(density) { OutlineWidth.toPx() }
        val outlineOffsetPx = with(density) { OutlineOffset.toPx() }

        Canvas(
            modifier = Modifier
                .fillMaxSize()
                // Required for BlendMode.Clear below to punch a
                // transparent hole revealing what's drawn underneath.
                .graphicsLayer(alpha = 0.99f)
                .clickable(
                    interactionSource = remember { MutableInteractionSource() },
                    indication = null
                ) { onCancel() }
        ) {
            drawRect(color = Color.Black.copy(alpha = DimAlpha * entrance))
            quadCorners.forEach { points ->
                // The cutout stays at the card's true, undistorted shape;
                // only the decorative ring is pushed outward and scaled.
                drawPath(quadPath(points), color = Color.Transparent, blendMode = BlendMode.Clear)
                drawPath(
                    quadPath(scaleQuad(expandQuad(points, outlineOffsetPx), popScale)),
                    color = CardOutlineColor.copy(alpha = entrance),
                    style = Stroke(width = outlineWidthPx, join = StrokeJoin.Round)
                )
            }
        }
    }
}
