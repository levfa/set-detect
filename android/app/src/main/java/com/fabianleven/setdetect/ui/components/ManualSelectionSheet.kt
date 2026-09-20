package com.fabianleven.setdetect.ui.components

import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.fabianleven.setdetect.R
import com.fabianleven.setdetect.domain.SetCard
import com.fabianleven.setdetect.domain.getAllCards

// Every card on the board (scanned or manual) shows as selected; tapping toggles it.
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ManualSelectionSheet(
    boardCards: List<SetCard>,
    onToggle: (SetCard) -> Unit,
    onDone: () -> Unit
) {
    val allCards = remember { getAllCards() }

    ModalBottomSheet(
        onDismissRequest = onDone,
        sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)
    ) {
        PickerSheetHeader(
            title = stringResource(R.string.manual_picker_title),
            actionLabel = stringResource(R.string.manual_picker_done),
            onAction = onDone
        )
        LazyVerticalGrid(
            columns = GridCells.Adaptive(minSize = 80.dp),
            contentPadding = PaddingValues(horizontal = 12.dp, vertical = 8.dp),
            modifier = Modifier.fillMaxWidth()
        ) {
            items(allCards) { card ->
                CardView(
                    card = card,
                    isSelected = card in boardCards,
                    onClick = { onToggle(card) },
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(4.dp)
                )
            }
        }
    }
}
