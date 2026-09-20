package com.fabianleven.setdetect.domain

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class BoardSlotsTest {
    private val a = SetCard(CardColor.RED, CardShape.OVAL, CardNumber.ONE, CardShading.SOLID)
    private val b = SetCard(CardColor.GREEN, CardShape.OVAL, CardNumber.ONE, CardShading.SOLID)
    private val c = SetCard(CardColor.PURPLE, CardShape.OVAL, CardNumber.ONE, CardShading.SOLID)

    @Test
    fun originalKeepsScannedCards() {
        val board = resolveBoard(listOf(a, b), listOf(SlotChoice.Original, SlotChoice.Original), emptyList())
        assertEquals(listOf(a, b), board)
    }

    @Test
    fun excludedSlotIsDropped() {
        val board = resolveBoard(listOf(a, b), listOf(SlotChoice.Excluded, SlotChoice.Original), emptyList())
        assertEquals(listOf(b), board)
    }

    @Test
    fun replacedSlotUsesReplacement() {
        val board = resolveBoard(listOf(a, b), listOf(SlotChoice.Replaced(c), SlotChoice.Original), emptyList())
        assertEquals(listOf(c, b), board)
    }

    @Test
    fun missingChoicesDefaultToOriginal() {
        assertEquals(listOf(a, b), resolveBoard(listOf(a, b), emptyList(), emptyList()))
    }

    @Test
    fun manualCardsAreAppended() {
        assertEquals(listOf(a, c), resolveBoard(listOf(a), listOf(SlotChoice.Original), listOf(c)))
    }

    @Test
    fun replacingWithScannedCardNormalizesToOriginal() {
        assertEquals(SlotChoice.Original, replacementChoice(a, a))
        assertEquals(SlotChoice.Replaced(b), replacementChoice(a, b))
    }

    @Test
    fun duplicatesAreKeptFlaggedAndFormNoSet() {
        val board = resolveBoard(listOf(a, b), listOf(SlotChoice.Original, SlotChoice.Replaced(a)), emptyList())
        assertEquals(listOf(a, a), board)
        assertEquals(setOf(a), findDuplicateCards(board))
        assertTrue(findSets(board + c).isEmpty())
    }

    @Test
    fun togglingAbsentCardAddsItManually() {
        val edit = toggleBoardCard(listOf(a), listOf(SlotChoice.Original), emptyList(), b)
        assertEquals(listOf(SlotChoice.Original), edit.choices)
        assertEquals(listOf(b), edit.manualCards)
    }

    @Test
    fun togglingScannedCardIgnoresItsSlot() {
        val edit = toggleBoardCard(listOf(a, b), listOf(SlotChoice.Original, SlotChoice.Original), emptyList(), a)
        assertEquals(listOf(SlotChoice.Excluded, SlotChoice.Original), edit.choices)
        assertTrue(edit.manualCards.isEmpty())
    }

    @Test
    fun togglingManualCardRemovesIt() {
        val edit = toggleBoardCard(listOf(a), listOf(SlotChoice.Original), listOf(c), c)
        assertEquals(listOf(SlotChoice.Original), edit.choices)
        assertTrue(edit.manualCards.isEmpty())
    }

    @Test
    fun togglingRemovesCardFromSlotsAndManualCards() {
        val edit = toggleBoardCard(listOf(a), listOf(SlotChoice.Replaced(b)), listOf(b), b)
        assertEquals(listOf(SlotChoice.Excluded), edit.choices)
        assertTrue(edit.manualCards.isEmpty())
    }

    @Test
    fun togglingRestoresIgnoredSlotOfThatScannedCard() {
        val edit = toggleBoardCard(listOf(a, b), listOf(SlotChoice.Excluded, SlotChoice.Original), emptyList(), a)
        assertEquals(listOf(SlotChoice.Original, SlotChoice.Original), edit.choices)
        assertTrue(edit.manualCards.isEmpty())
    }

    @Test
    fun togglingDoesNotRestoreSlotReplacedByAnotherCard() {
        val edit = toggleBoardCard(listOf(a), listOf(SlotChoice.Replaced(b)), emptyList(), a)
        assertEquals(listOf(SlotChoice.Replaced(b)), edit.choices)
        assertEquals(listOf(a), edit.manualCards)
    }
}
