package com.fabianleven.setdetect.ui.components

import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.animateDpAsState
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Fill
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.clipPath
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.fabianleven.setdetect.domain.*

// Deselected cards use a solid muted gray (translucent white would vary in
// contrast with the background) and a faded symbol, so they read as
// deselected next to a selected card.
private val DeselectedContainerColor = Color(0xFFBDBDBD)
private const val DeselectedSymbolAlpha = 0.45f

// Width:height ratio a physical SET card is drawn at. Arranged card boxes use
// it too, so they match regardless of the detected card height.
const val CardAspectRatio = 5f / 7f

@Composable
fun CardView(
    card: SetCard,
    modifier: Modifier = Modifier,
    isSelected: Boolean = false,
    isActive: Boolean = true,
    showBorder: Boolean = true,
    onClick: () -> Unit = {},
    interactionSource: MutableInteractionSource? = null
) {
    val baseColor = when (card.color) {
        CardColor.RED -> Color(0xFFE91E63)
        CardColor.GREEN -> Color(0xFF4CAF50)
        CardColor.PURPLE -> Color(0xFF9C27B0)
    }
    val targetColor = if (isSelected) baseColor else baseColor.copy(alpha = DeselectedSymbolAlpha)
    val color by animateColorAsState(targetColor, label = "symbolColor")

    // Inactive cards (ignored scan cards) show no symbol, marking them as excluded.
    val showSymbol = isActive

    val containerColor by animateColorAsState(
        if (isSelected) Color.White else DeselectedContainerColor,
        label = "containerColor"
    )
    val elevation by animateDpAsState(
        if (isSelected) 4.dp else 0.dp,
        label = "elevation"
    )
    // Border width and color animate together so it fades instead of popping.
    // showBorder is false for the small, rotated cards in the scan arrangement,
    // where container color and symbol contrast suffice.
    val borderColor by animateColorAsState(
        if (isSelected && showBorder) MaterialTheme.colorScheme.primary else Color.Transparent,
        label = "borderColor"
    )
    val borderWidth by animateDpAsState(
        if (isSelected && showBorder) 3.dp else 0.dp,
        label = "borderWidth"
    )

    Card(
        onClick = onClick,
        interactionSource = interactionSource,
        modifier = modifier
            .aspectRatio(CardAspectRatio),
        shape = RoundedCornerShape(12),
        colors = CardDefaults.cardColors(containerColor = containerColor),
        elevation = CardDefaults.cardElevation(defaultElevation = elevation),
        border = BorderStroke(borderWidth, borderColor)
    ) {
        if (showSymbol) {
            BoxWithConstraints(
                modifier = Modifier.fillMaxSize()
            ) {
                val shapeWidthFraction = 0.8f
                val shapeAspectRatio = 2.5f
                val cardPadding = maxWidth * 0.08f

                val count = card.number.value
                val spacing = maxHeight * 0.05f

                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(cardPadding),
                    verticalArrangement = Arrangement.Center,
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    when (count) {
                        1 -> ShapeView(card.shape, card.shading, color, Modifier.fillMaxWidth(shapeWidthFraction).aspectRatio(shapeAspectRatio))
                        2 -> {
                            ShapeView(card.shape, card.shading, color, Modifier.fillMaxWidth(shapeWidthFraction).aspectRatio(shapeAspectRatio))
                            Spacer(Modifier.height(spacing))
                            ShapeView(card.shape, card.shading, color, Modifier.fillMaxWidth(shapeWidthFraction).aspectRatio(shapeAspectRatio))
                        }
                        3 -> {
                            ShapeView(card.shape, card.shading, color, Modifier.fillMaxWidth(shapeWidthFraction).aspectRatio(shapeAspectRatio))
                            Spacer(Modifier.height(spacing))
                            ShapeView(card.shape, card.shading, color, Modifier.fillMaxWidth(shapeWidthFraction).aspectRatio(shapeAspectRatio))
                            Spacer(Modifier.height(spacing))
                            ShapeView(card.shape, card.shading, color, Modifier.fillMaxWidth(shapeWidthFraction).aspectRatio(shapeAspectRatio))
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun ShapeView(
    shape: CardShape,
    shading: CardShading,
    color: Color,
    modifier: Modifier = Modifier
) {
    Canvas(modifier = modifier) {
        val path = when (shape) {
            CardShape.DIAMOND -> getDiamondPath(size)
            CardShape.SQUIGGLE -> getSquigglePath(size)
            CardShape.OVAL -> getOvalPath(size)
        }

        val strokeWidth = size.height * 0.1f

        when (shading) {
            CardShading.SOLID -> {
                drawPath(path, color, style = Fill)
            }
            CardShading.OPEN -> {
                drawPath(path, color, style = Stroke(width = strokeWidth))
            }
            CardShading.STRIPED -> {
                drawPath(path, color, style = Stroke(width = strokeWidth * 0.8f))
                drawStripes(path, color)
            }
        }
    }
}

private fun getDiamondPath(size: Size): Path {
    return Path().apply {
        moveTo(size.width / 2, 0f)
        lineTo(size.width, size.height / 2)
        lineTo(size.width / 2, size.height)
        lineTo(0f, size.height / 2)
        close()
    }
}

private fun getOvalPath(size: Size): Path {
    return Path().apply {
        addRoundRect(
            androidx.compose.ui.geometry.RoundRect(
                rect = androidx.compose.ui.geometry.Rect(0f, 0f, size.width, size.height),
                cornerRadius = androidx.compose.ui.geometry.CornerRadius(size.height / 2)
            )
        )
    }
}

private fun getSquigglePath(size: Size): Path {
    val w = size.width
    val h = size.height
    return Path().apply {
        moveTo(0.169f * w, 0.363f * h)
        cubicTo(
            0.199f * w, 0.179f * h,
            0.244f * w, 0.096f * h,
            0.333f * w, 0.067f * h
        )
        cubicTo(
            0.424f * w, 0.088f * h,
            0.516f * w, 0.221f * h,
            0.600f * w, 0.250f * h
        )
        cubicTo(
            0.672f * w, 0.262f * h,
            0.679f * w, 0.179f * h,
            0.802f * w, 0.033f * h
        )
        cubicTo(
            0.886f * w, 0.188f * h,
            0.847f * w, 0.446f * h,
            0.831f * w, 0.637f * h
        )
        cubicTo(
            0.801f * w, 0.821f * h,
            0.756f * w, 0.904f * h,
            0.667f * w, 0.933f * h
        )
        cubicTo(
            0.576f * w, 0.912f * h,
            0.484f * w, 0.779f * h,
            0.400f * w, 0.750f * h
        )
        cubicTo(
            0.328f * w, 0.738f * h,
            0.321f * w, 0.821f * h,
            0.198f * w, 0.967f * h
        )
        cubicTo(
            0.114f * w, 0.813f * h,
            0.153f * w, 0.554f * h,
            0.169f * w, 0.363f * h
        )
        close()
    }
}

private fun DrawScope.drawStripes(path: Path, color: Color) {
    clipPath(path) {
        val gap = size.width * 0.08f
        val strokeWidth = size.width * 0.02f
        var x = 0f
        while (x < size.width) {
            drawLine(
                color = color,
                start = Offset(x, 0f),
                end = Offset(x, size.height),
                strokeWidth = strokeWidth
            )
            x += gap
        }
    }
}

@Preview
@Composable
private fun CardViewPreview() {
    CardView(
        card = SetCard(CardColor.PURPLE, CardShape.SQUIGGLE, CardNumber.ONE, CardShading.STRIPED)
    )
}
