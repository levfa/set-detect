package com.fabianleven.setdetect.domain

import com.fabianleven.setdetect.ui.components.CardAspectRatio

// A hand-built example scan for the onboarding tutorial, giving its board steps
// content to point at without a camera capture. Twelve cards loosely arranged,
// not a perfect grid.
private const val EXAMPLE_CARD_WIDTH = 0.19f

private data class ExampleCardPlacement(val x: Float, val y: Float, val angleRad: Float)

private val examplePlacements = listOf(
    ExampleCardPlacement(0.13f, 0.16f, 0.04f),
    ExampleCardPlacement(0.38f, 0.14f, -0.03f),
    ExampleCardPlacement(0.63f, 0.16f, 0.06f),
    ExampleCardPlacement(0.88f, 0.14f, -0.05f),
    ExampleCardPlacement(0.13f, 0.44f, -0.06f),
    ExampleCardPlacement(0.38f, 0.42f, 0.02f),
    ExampleCardPlacement(0.63f, 0.44f, -0.04f),
    ExampleCardPlacement(0.88f, 0.42f, 0.07f),
    ExampleCardPlacement(0.13f, 0.72f, 0.03f),
    ExampleCardPlacement(0.38f, 0.74f, -0.02f),
    ExampleCardPlacement(0.63f, 0.72f, 0.05f),
    ExampleCardPlacement(0.88f, 0.74f, -0.06f)
)

// Indices into getAllCards() (81 cards total) used for the example scan:
// spread out and disjoint.
private val exampleMatchedIndices = listOf(1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66)

fun createExampleDetection(): Detection {
    val allCards = getAllCards()
    val matchedCards = exampleMatchedIndices.map { allCards[it] }

    val detectedCards = matchedCards.map { card ->
        DetectedCard(card = card, corners = emptyList(), confidence = 1f)
    }
    val arrangement = CardsArrangement(
        cardPoses = examplePlacements.map { RotatedRect(it.x, it.y, it.angleRad) },
        cardZOrders = examplePlacements.indices.toList(),
        cardWidth = EXAMPLE_CARD_WIDTH,
        cardHeight = EXAMPLE_CARD_WIDTH / CardAspectRatio
    )
    return Detection(detectedCards = detectedCards, arrangement = arrangement)
}
