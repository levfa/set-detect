package com.fabianleven.setdetect.ui.components

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Close
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.fabianleven.setdetect.R
import com.fabianleven.setdetect.domain.SetCard

// Tapping a card removes it, which the corner "x" badge signals.
@Composable
fun ManualCardsGrid(
    cards: List<SetCard>,
    duplicateCards: Set<SetCard>,
    onRemove: (SetCard) -> Unit,
    modifier: Modifier = Modifier
) {
    LazyVerticalGrid(
        columns = GridCells.Adaptive(minSize = 60.dp),
        contentPadding = PaddingValues(horizontal = 16.dp, vertical = 8.dp),
        modifier = modifier.fillMaxSize()
    ) {
        items(cards) { card ->
            Box(modifier = Modifier.padding(2.dp)) {
                CardView(
                    card = card,
                    isSelected = true,
                    onClick = { onRemove(card) }
                )
                CornerBadge(
                    imageVector = Icons.Rounded.Close,
                    contentDescription = stringResource(R.string.selection_remove_card),
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    onColor = MaterialTheme.colorScheme.surface,
                    iconSize = 12.dp,
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .padding(2.dp)
                )
                if (card in duplicateCards) {
                    DuplicateBadge(
                        iconSize = 12.dp,
                        modifier = Modifier
                            .align(Alignment.TopStart)
                            .padding(2.dp)
                    )
                }
            }
        }
    }
}
