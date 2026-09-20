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
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
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

private val DefaultManualHeight = 150.dp
private val MinManualHeight = 96.dp
private val MinScanAreaHeight = 160.dp

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

    var manualHeightDp by rememberSaveable { mutableFloatStateOf(DefaultManualHeight.value) }
    val density = LocalDensity.current
    val dragState = rememberDraggableState { delta ->
        manualHeightDp += delta / density.density
    }

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
        val manualHeight = manualHeightDp.dp.coerceIn(MinManualHeight, maxManualHeight)

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
