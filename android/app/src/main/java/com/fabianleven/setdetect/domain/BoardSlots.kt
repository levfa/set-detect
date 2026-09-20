package com.fabianleven.setdetect.domain

sealed interface SlotChoice {
    data object Original : SlotChoice
    data object Excluded : SlotChoice
    data class Replaced(val card: SetCard) : SlotChoice
}

fun replacementChoice(original: SetCard, card: SetCard): SlotChoice =
    if (card == original) SlotChoice.Original else SlotChoice.Replaced(card)

fun resolveSlot(original: SetCard, choice: SlotChoice): SetCard? = when (choice) {
    SlotChoice.Original -> original
    SlotChoice.Excluded -> null
    is SlotChoice.Replaced -> choice.card
}

// The list may contain duplicates; a repeated card never contributes to a set.
fun resolveBoard(
    scanCards: List<SetCard>,
    choices: List<SlotChoice>,
    manualCards: List<SetCard>
): List<SetCard> {
    val scanned = scanCards.mapIndexedNotNull { index, card ->
        resolveSlot(card, choices.getOrElse(index) { SlotChoice.Original })
    }
    return scanned + manualCards
}

data class BoardEdit(val choices: List<SlotChoice>, val manualCards: List<SetCard>)

// A card counts as selected when it is anywhere on the board. Tapping a selected
// card ignores every scan slot showing it and drops it from the manual cards;
// tapping an absent card restores an ignored scan slot that was originally that
// card, or else adds it to the manual cards.
fun toggleBoardCard(
    scanCards: List<SetCard>,
    choices: List<SlotChoice>,
    manualCards: List<SetCard>,
    card: SetCard
): BoardEdit {
    if (card in resolveBoard(scanCards, choices, manualCards)) {
        return BoardEdit(
            choices = choices.mapIndexed { i, choice ->
                if (resolveSlot(scanCards[i], choice) == card) SlotChoice.Excluded else choice
            },
            manualCards = manualCards.filter { it != card }
        )
    }
    val restorable = scanCards.indices.firstOrNull { scanCards[it] == card && choices[it] == SlotChoice.Excluded }
    return if (restorable != null) {
        BoardEdit(choices.toMutableList().also { it[restorable] = SlotChoice.Original }, manualCards)
    } else {
        BoardEdit(choices, manualCards + card)
    }
}
