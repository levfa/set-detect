package com.fabianleven.setdetect.ui

import androidx.compose.foundation.gestures.rememberDraggableState
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Edit
import androidx.compose.material.icons.rounded.PhotoCamera
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.layout.boundsInWindow
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.fabianleven.setdetect.R
import com.fabianleven.setdetect.ui.components.ArrangementView
import com.fabianleven.setdetect.ui.components.CardPickerSheet
import com.fabianleven.setdetect.ui.components.DividerHandle
import com.fabianleven.setdetect.ui.components.DuplicateBanner
import com.fabianleven.setdetect.ui.components.EmptyState
import com.fabianleven.setdetect.ui.components.ManualCardsGrid
import com.fabianleven.setdetect.ui.components.SectionHeader
import com.fabianleven.setdetect.ui.components.TapFeedbackMillis
import com.fabianleven.setdetect.ui.components.TutorialTargets
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

// Not private: SelectionViewModel and TutorialController also reference it,
// as the divider's default/demo height.
val DefaultManualHeight = 150.dp
private val MinManualHeight = 96.dp
// Keeps the scanned board's cards at a legible, tappable size even when the
// divider is dragged all the way down (ArrangementView scales the whole
// arrangement to fit whatever height it's given).
private val MinScanAreaHeight = 260.dp

// The manual selection on top and the scan arrangement below, split by a draggable divider.
@Composable
fun BoardContent(
    viewModel: SelectionViewModel,
    targets: TutorialTargets,
    modifier: Modifier = Modifier
) {
    val detection = viewModel.scannedDetection
    val matchedCards = viewModel.matchedCards
    val choices = viewModel.scanChoices
    val duplicateCards = viewModel.duplicateCards

    // The slot is marked active as soon as it is tapped and the picker opens
    // after a short delay, so the press feedback stays visible.
    var activeSlot by remember(detection) { mutableStateOf<Int?>(null) }
    var pickerSlot by remember(detection) { mutableStateOf<Int?>(null) }
    val scope = rememberCoroutineScope()

    val density = LocalDensity.current

    pickerSlot?.let { slot ->
        matchedCards.getOrNull(slot)?.let { original ->
            CardPickerSheet(
                original = original,
                current = choices[slot],
                onSelect = {
                    viewModel.setSlotChoice(slot, it)
                    pickerSlot = null
                    activeSlot = null
                },
                onDismiss = {
                    pickerSlot = null
                    activeSlot = null
                }
            )
        }
    }

    BoxWithConstraints(modifier = modifier) {
        val maxManualHeight = (maxHeight - MinScanAreaHeight).coerceAtLeast(MinManualHeight)
        val manualHeight = viewModel.manualHeightDp.dp.coerceIn(MinManualHeight, maxManualHeight)
        // Clamped immediately, not just at render, so a drag past the visual
        // limits can't leave the stored value stranded outside them (which
        // would otherwise cause a jump the next time maxManualHeight shrinks,
        // e.g. on rotation).
        val dragState = rememberDraggableState { delta ->
            viewModel.manualHeightDp = (viewModel.manualHeightDp + delta / density.density)
                .coerceIn(MinManualHeight.value, maxManualHeight.value)
        }

        Column(modifier = Modifier.fillMaxSize()) {
            SectionHeader(
                title = stringResource(R.string.selection_manual_selection),
                clearEnabled = viewModel.manualCards.isNotEmpty(),
                onClear = { viewModel.clearManual() }
            )
            Box(
                modifier = Modifier
                    .height(manualHeight)
                    .fillMaxWidth()
                    .onGloballyPositioned { targets.manualSelection = it.boundsInWindow() }
            ) {
                if (viewModel.manualCards.isEmpty()) {
                    EmptyState(
                        icon = Icons.Rounded.Edit,
                        text = stringResource(R.string.selection_manual_empty),
                        modifier = Modifier.fillMaxSize()
                    )
                } else {
                    ManualCardsGrid(
                        cards = viewModel.manualCards.toList(),
                        duplicateCards = duplicateCards,
                        onRemove = { viewModel.manualCards.remove(it) }
                    )
                }
            }
            DividerHandle(dragState = dragState)

            SectionHeader(
                title = stringResource(R.string.selection_scan),
                clearEnabled = detection != null,
                onClear = { viewModel.clearScan() }
            )
            Box(
                modifier = Modifier
                    .weight(1f)
                    .onGloballyPositioned { targets.arrangement = it.boundsInWindow() }
            ) {
                if (detection == null) {
                    EmptyState(
                        icon = Icons.Rounded.PhotoCamera,
                        text = stringResource(R.string.selection_scan_empty),
                        modifier = Modifier.fillMaxSize()
                    )
                } else {
                    ArrangementView(
                        arrangement = detection.arrangement,
                        cards = matchedCards,
                        choices = choices,
                        onCardClick = { slot ->
                            if (activeSlot == null) {
                                activeSlot = slot
                                scope.launch {
                                    delay(TapFeedbackMillis)
                                    pickerSlot = slot
                                }
                            }
                        },
                        activeSlot = activeSlot,
                        highlightedSlot = viewModel.tutorial.demoSlot,
                        onHighlightedCardPositioned = { targets.demoCardRect = it },
                        duplicateCards = duplicateCards,
                        modifier = Modifier.fillMaxSize().padding(16.dp)
                    )
                }
                if (duplicateCards.isNotEmpty()) {
                    DuplicateBanner(
                        modifier = Modifier
                            .align(Alignment.TopEnd)
                            .padding(16.dp)
                    )
                }
            }
        }
    }
}
