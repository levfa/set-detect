package com.fabianleven.setdetect.ui.components

import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.GridItemSpan
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.lazy.grid.rememberLazyGridState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.fabianleven.setdetect.R
import com.fabianleven.setdetect.domain.SetCard
import com.fabianleven.setdetect.domain.SlotChoice
import com.fabianleven.setdetect.domain.getAllCards
import com.fabianleven.setdetect.domain.replacementChoice
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

// Two special tiles and the divider precede the card grid.
private const val LeadingItemCount = 3
private val PickerCardShape = RoundedCornerShape(12)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CardPickerSheet(
    original: SetCard,
    current: SlotChoice,
    onSelect: (SlotChoice) -> Unit,
    onDismiss: () -> Unit
) {
    val allCards = remember { getAllCards() }
    val currentIndex = when (current) {
        is SlotChoice.Replaced -> LeadingItemCount + allCards.indexOf(current.card)
        else -> 0
    }
    val gridState = rememberLazyGridState(initialFirstVisibleItemIndex = currentIndex)
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)
    val scope = rememberCoroutineScope()
    var tapped by remember { mutableStateOf<SlotChoice?>(null) }
    val shown = tapped ?: current

    // Lets the tap feedback and the new outline render before the sheet slides away.
    fun choose(choice: SlotChoice) {
        if (tapped != null) return
        tapped = choice
        scope.launch {
            delay(TapFeedbackMillis)
            sheetState.hide()
            onSelect(choice)
        }
    }

    ModalBottomSheet(
        onDismissRequest = onDismiss,
        sheetState = sheetState
    ) {
        PickerSheetHeader(
            title = stringResource(R.string.picker_title),
            actionLabel = stringResource(R.string.picker_cancel),
            onAction = onDismiss
        )
        LazyVerticalGrid(
            columns = GridCells.Adaptive(minSize = 80.dp),
            state = gridState,
            contentPadding = PaddingValues(horizontal = 12.dp, vertical = 8.dp),
            modifier = Modifier.fillMaxWidth()
        ) {
            item {
                PickerTile(
                    label = stringResource(R.string.picker_ignored),
                    isCurrent = shown == SlotChoice.Excluded
                ) {
                    CardView(
                        card = original,
                        isSelected = false,
                        isActive = false,
                        onClick = { choose(SlotChoice.Excluded) }
                    )
                }
            }
            item {
                PickerTile(
                    label = stringResource(R.string.picker_scanned),
                    isCurrent = shown == SlotChoice.Original
                ) {
                    CardView(
                        card = original,
                        isSelected = true,
                        showBorder = false,
                        onClick = { choose(SlotChoice.Original) }
                    )
                }
            }
            item(span = { GridItemSpan(maxLineSpan) }) {
                HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))
            }
            items(allCards) { card ->
                PickerTile(
                    label = null,
                    isCurrent = (shown is SlotChoice.Replaced && shown.card == card) ||
                        (shown == SlotChoice.Original && card == original)
                ) {
                    CardView(
                        card = card,
                        isSelected = true,
                        showBorder = false,
                        onClick = { choose(replacementChoice(original, card)) }
                    )
                }
            }
        }
    }
}

@Composable
private fun PickerTile(
    label: String?,
    isCurrent: Boolean,
    content: @Composable () -> Unit
) {
    val outline = if (isCurrent) Modifier.border(3.dp, MaterialTheme.colorScheme.primary, PickerCardShape) else Modifier
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = Modifier.padding(4.dp)
    ) {
        Box(modifier = Modifier.fillMaxWidth()) {
            content()
            Box(modifier = Modifier.matchParentSize().then(outline))
        }
        if (label != null) {
            Text(
                text = label,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}
