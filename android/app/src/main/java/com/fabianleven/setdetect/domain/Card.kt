package com.fabianleven.setdetect.domain

import kotlinx.serialization.Serializable

@Serializable
enum class CardColor {
    RED, GREEN, PURPLE
}

@Serializable
enum class CardShape {
    DIAMOND, SQUIGGLE, OVAL
}

@Serializable
enum class CardNumber(val value: Int) {
    ONE(1), TWO(2), THREE(3)
}

@Serializable
enum class CardShading {
    SOLID, STRIPED, OPEN
}

@Serializable
data class SetCard(
    val color: CardColor,
    val shape: CardShape,
    val number: CardNumber,
    val shading: CardShading
)

fun isSet(c1: SetCard, c2: SetCard, c3: SetCard): Boolean {
    val colors = setOf(c1.color, c2.color, c3.color)
    val shapes = setOf(c1.shape, c2.shape, c3.shape)
    val numbers = setOf(c1.number, c2.number, c3.number)
    val shadings = setOf(c1.shading, c2.shading, c3.shading)

    return (colors.size == 1 || colors.size == 3) &&
            (shapes.size == 1 || shapes.size == 3) &&
            (numbers.size == 1 || numbers.size == 3) &&
            (shadings.size == 1 || shadings.size == 3)
}

fun findSets(cards: List<SetCard>): List<List<SetCard>> {
    val sets = mutableListOf<List<SetCard>>()
    for (i in 0 until cards.size) {
        for (j in i + 1 until cards.size) {
            for (k in j + 1 until cards.size) {
                if (isSet(cards[i], cards[j], cards[k])) {
                    sets.add(listOf(cards[i], cards[j], cards[k]))
                }
            }
        }
    }
    return sets
}

// Every SetCard is meant to be unique on a real board; two detected cards sharing all
// four attributes means either the same physical card really is on the table twice, or
// classification misread one of them -- either way, the UI should flag it rather than
// silently treat them as distinct.
fun findDuplicateCards(cards: List<SetCard>): Set<SetCard> {
    return cards.groupingBy { it }.eachCount().filterValues { it > 1 }.keys
}

fun getAllCards(): List<SetCard> {
    val allCards = mutableListOf<SetCard>()
    for (color in CardColor.entries) {
        for (shape in CardShape.entries) {
            for (number in CardNumber.entries) {
                for (shading in CardShading.entries) {
                    allCards.add(SetCard(color, shape, number, shading))
                }
            }
        }
    }
    return allCards
}
