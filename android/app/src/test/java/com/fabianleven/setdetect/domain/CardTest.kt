package com.fabianleven.setdetect.domain

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class CardTest {

    @Test
    fun `isSet returns true for a valid set`() {
        val c1 = SetCard(CardColor.RED, CardShape.DIAMOND, CardNumber.ONE, CardShading.SOLID)
        val c2 = SetCard(CardColor.RED, CardShape.DIAMOND, CardNumber.TWO, CardShading.SOLID)
        val c3 = SetCard(CardColor.RED, CardShape.DIAMOND, CardNumber.THREE, CardShading.SOLID)
        assertTrue(isSet(c1, c2, c3))
    }

    @Test
    fun `isSet returns true for another valid set`() {
        val c1 = SetCard(CardColor.RED, CardShape.DIAMOND, CardNumber.ONE, CardShading.SOLID)
        val c2 = SetCard(CardColor.GREEN, CardShape.SQUIGGLE, CardNumber.TWO, CardShading.STRIPED)
        val c3 = SetCard(CardColor.PURPLE, CardShape.OVAL, CardNumber.THREE, CardShading.OPEN)
        assertTrue(isSet(c1, c2, c3))
    }

    @Test
    fun `isSet returns false for an invalid set`() {
        val c1 = SetCard(CardColor.RED, CardShape.DIAMOND, CardNumber.ONE, CardShading.SOLID)
        val c2 = SetCard(CardColor.RED, CardShape.DIAMOND, CardNumber.TWO, CardShading.SOLID)
        val c3 = SetCard(CardColor.GREEN, CardShape.DIAMOND, CardNumber.THREE, CardShading.SOLID)
        assertFalse(isSet(c1, c2, c3))
    }

    @Test
    fun `findSets finds all sets in a small list`() {
        val cards = listOf(
            SetCard(CardColor.RED, CardShape.DIAMOND, CardNumber.ONE, CardShading.SOLID),
            SetCard(CardColor.RED, CardShape.DIAMOND, CardNumber.TWO, CardShading.SOLID),
            SetCard(CardColor.RED, CardShape.DIAMOND, CardNumber.THREE, CardShading.SOLID),
            SetCard(CardColor.GREEN, CardShape.DIAMOND, CardNumber.ONE, CardShading.SOLID)
        )
        val sets = findSets(cards)
        assertEquals(1, sets.size)
    }

    @Test
    fun `findDuplicateCards returns empty set when all cards are unique`() {
        val cards = listOf(
            SetCard(CardColor.RED, CardShape.DIAMOND, CardNumber.ONE, CardShading.SOLID),
            SetCard(CardColor.GREEN, CardShape.DIAMOND, CardNumber.ONE, CardShading.SOLID)
        )
        assertTrue(findDuplicateCards(cards).isEmpty())
    }

    @Test
    fun `findDuplicateCards finds a card repeated on the board`() {
        val duplicate = SetCard(CardColor.RED, CardShape.DIAMOND, CardNumber.ONE, CardShading.SOLID)
        val cards = listOf(
            duplicate,
            SetCard(CardColor.GREEN, CardShape.DIAMOND, CardNumber.ONE, CardShading.SOLID),
            duplicate
        )
        assertEquals(setOf(duplicate), findDuplicateCards(cards))
    }

    @Test
    fun `findDuplicateCards finds multiple distinct duplicated cards`() {
        val dup1 = SetCard(CardColor.RED, CardShape.DIAMOND, CardNumber.ONE, CardShading.SOLID)
        val dup2 = SetCard(CardColor.PURPLE, CardShape.OVAL, CardNumber.THREE, CardShading.OPEN)
        val unique = SetCard(CardColor.GREEN, CardShape.SQUIGGLE, CardNumber.TWO, CardShading.STRIPED)
        val cards = listOf(dup1, dup2, unique, dup1, dup2)
        assertEquals(setOf(dup1, dup2), findDuplicateCards(cards))
    }
}
